#!/usr/bin/env python3
"""
dump_step_p.py — Dump intermediate pressure fields for debugging.
Uses correct numba kernels (not the buggy inline version in dump_step_debug.py).
"""

import numpy as np
import numba as nb
from scipy.optimize import fsolve
import os

LX = LY = 30
K_ELASTIC = 2**14
gamma = 100.0
lam   = 0.7
nu    = (10.0 * K_ELASTIC)**0.5
rho   = 1.0
zeta  = float(K_ELASTIC)
A     = -K_ELASTIC / 81.0
C     =  K_ELASTIC / 81.0
S0    = 2.0**0.5
dt    = 1e-4
max_p_iters = 20
p_target_rel_change = 1e-4

D_EPI = 0.99
K_EPI = 2

def set_boundary_epitrochoid(Lx, Ly):
    """Build sim_points / bounds / boundary.
    Normals loaded from Rust bnd_*_normals.txt for exact bit-parity with Rust.
    """
    boundary = np.zeros((2, Lx, Ly, 2))
    sim_points = set()
    radius = Lx // 2 - 1
    for x in range(Lx):
        for y in range(Ly):
            def f(u, x=x, y=y):
                return np.arctan2(y-radius, x-radius) - np.arctan2(
                    (K_EPI+1)*np.sin(u)+D_EPI*np.sin((K_EPI+1)*u),
                    (K_EPI+1)*np.cos(u)+D_EPI*np.cos((K_EPI+1)*u))
            u_sol = fsolve(f, 0.1)[0]
            rhs = radius**2/(K_EPI+2)**2*((K_EPI+1)**2+D_EPI**2+2*(K_EPI+1)*D_EPI*np.cos(K_EPI*u_sol))
            if (x-radius)**2+(y-radius)**2 <= rhs:
                sim_points.add((x,y))
    _sdir = os.path.dirname(os.path.abspath(__file__))
    outer_flat = np.loadtxt(os.path.join(_sdir, 'bnd_outer_normals.txt'))
    inner_flat = np.loadtxt(os.path.join(_sdir, 'bnd_inner_normals.txt'))
    for x in range(Lx):
        for y in range(Ly):
            idx2 = (x * Ly + y) * 2
            boundary[1,x,y,0] = outer_flat[idx2]
            boundary[1,x,y,1] = outer_flat[idx2+1]
            boundary[0,x,y,0] = inner_flat[idx2]
            boundary[0,x,y,1] = inner_flat[idx2+1]
    bounds = np.zeros((len(sim_points), 2), dtype=int)
    for i, (x, y) in enumerate(sim_points):
        bounds[i] = [x, y]
    return boundary, bounds, sim_points

print("Building boundary...", flush=True)
boundary, bounds, sim_points = set_boundary_epitrochoid(LX, LY)
print(f"Interior cells: {len(sim_points)}", flush=True)

def flatten2(arr):
    Lx, Ly, _ = arr.shape
    out = np.zeros(Lx*Ly*2)
    for x in range(Lx):
        for y in range(Ly):
            out[(x*Ly+y)*2+0] = arr[x,y,0]
            out[(x*Ly+y)*2+1] = arr[x,y,1]
    return out

def flatten1(arr):
    Lx, Ly = arr.shape
    out = np.zeros(Lx*Ly)
    for x in range(Lx):
        for y in range(Ly):
            out[x*Ly+y] = arr[x,y]
    return out

# Correct numba kernels
@nb.njit(parallel=True, fastmath=True, nogil=True)
def Laplacian_vector(arr, out, bounds, coeff=1.):
    coeff /= 6
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; yup=(y+1)%Ly; ydn=y-1
        out[x,y,:] = coeff*(-20*arr[x,y,:]+4*(arr[xup,y,:]+arr[x,ydn,:]+arr[x,yup,:]+arr[xdn,y,:])+arr[xup,ydn,:]+arr[xup,yup,:]+arr[xdn,ydn,:]+arr[xdn,yup,:])

@nb.njit(parallel=True, fastmath=True, nogil=True)
def upwind_advective_term(u, arr, out, bounds, coeff=-1):
    coeff *= 0.5
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; xupup=(x+2)%Lx; xdndn=x-2
        tmp = coeff*u[x,y,0]
        if u[x,y,0]>0:
            out[x,y,:] += tmp*(3*arr[x,y,:]-4*arr[xdn,y,:]+arr[xdndn,y,:])
        else:
            out[x,y,:] += tmp*(-3*arr[x,y,:]+4*arr[xup,y,:]-arr[xupup,y,:])
        tmp = coeff*u[x,y,1]
        if u[x,y,1]>0:
            ydn2=y-1; ydndn=y-2
            out[x,y,:] += tmp*(3*arr[x,y,:]-4*arr[x,ydn2,:]+arr[x,ydndn,:])
        else:
            yup2=(y+1)%Ly; yupup=(y+2)%Ly
            out[x,y,:] += tmp*(-3*arr[x,y,:]+4*arr[x,yup2,:]-arr[x,yupup,:])

@nb.njit(parallel=True, fastmath=True, nogil=True)
def div_vector(arr, out, bounds):
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1
        out[x,y] = 0.5*((arr[xup,y,0]-arr[xdn,y,0])+(arr[x,(y+1)%Ly,1]-arr[x,y-1,1]))

@nb.njit(parallel=True, fastmath=True, nogil=True)
def H_S_from_Q(u, Q, H, S, A, C, K, lam, bounds):
    Laplacian_vector(Q, H, bounds, coeff=K)
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1
        dxux=0.5*(u[xup,y,0]-u[xdn,y,0]); dxuy=0.5*(u[xup,y,1]-u[xdn,y,1])
        dyux=0.5*(u[x,(y+1)%Ly,0]-u[x,y-1,0])
        omega_xy=0.5*(dxuy-dyux); trQsq=2*(Q[x,y,0]**2+Q[x,y,1]**2)
        lS=lam*np.sqrt(2*trQsq); H[x,y,:]-=(A+C*trQsq)*Q[x,y,:]
        TrQE=2*Q[x,y,0]*dxux+Q[x,y,1]*(dyux+dxuy)
        S[x,y,0]=lS*dxux-2*omega_xy*Q[x,y,1]-2*TrQE*Q[x,y,0]
        S[x,y,1]=lS*0.5*(dxuy+dyux)+2*omega_xy*Q[x,y,0]-2*TrQE*Q[x,y,1]

@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_Erickson_stress(Q, K, Pi_S, bounds):
    Lx, Ly = Q.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; yup=(y+1)%Ly; ydn=y-1
        dxQxx, dxQxy = 0.5*(Q[xup,y]-Q[xdn,y])
        dyQxx, dyQxy = 0.5*(Q[x,yup]-Q[x,ydn])
        Pi_S[x,y,0] -= K*((dxQxx)**2+(dxQxy)**2-(dyQxx)**2-(dyQxy)**2)
        Pi_S[x,y,1] -= 2*K*((dxQxy*dyQxy)+(dxQxx*dyQxx))

@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_TrQH_term(Q, H, Pi_S):
    TrQH=2*(Q[:,:,0]*H[:,:,0]+Q[:,:,1]*H[:,:,1])
    Pi_S[:,:,0]+=TrQH*Q[:,:,0]; Pi_S[:,:,1]+=TrQH*Q[:,:,1]

@nb.njit(parallel=True, fastmath=True, nogil=True)
def calculate_Pi(Pi_S, Pi_A, H, Q, lam, zeta, K, bounds):
    Pi_S[:]=-lam*H-zeta*Q
    get_Erickson_stress(Q, K, Pi_S, bounds); get_TrQH_term(Q, H, Pi_S)
    Pi_A[:]=2*(Q[:,:,0]*H[:,:,1]-H[:,:,0]*Q[:,:,1])

@nb.njit(parallel=True, fastmath=True, nogil=True)
def calculate_pressure_terms(u, rho, Pi_S, rhs, bounds):
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; yup=(y+1)%Ly; ydn=y-1
        dudx=0.5*(u[xup,y,0]-u[xdn,y,0]); dvdy=0.5*(u[x,yup,1]-u[x,ydn,1])
        rhs[x,y] += (
            (Pi_S[xup,y,0]+Pi_S[xdn,y,0]-Pi_S[x,yup,0]-Pi_S[x,ydn,0])
            + 0.5*(Pi_S[xup,yup,1]-Pi_S[xup,ydn,1]-Pi_S[xdn,yup,1]+Pi_S[xdn,ydn,1])
            - rho*(dudx*dudx+dvdy*dvdy+0.5*(u[x,yup,0]-u[x,ydn,0])*(u[xup,y,1]-u[xdn,y,1]))
        )

@nb.njit(parallel=True, fastmath=True, nogil=True)
def relax_pressure_inner_loop(p, p_aux, rhs, bounds):
    Lx, Ly = p.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; yup=(y+1)%Ly; ydn=y-1
        p[x,y]=0.05*(-6*rhs[x,y]+4*(p_aux[xup,y]+p_aux[x,yup]+p_aux[x,ydn]+p_aux[xdn,y])+p_aux[xup,yup]+p_aux[xup,ydn]+p_aux[xdn,yup]+p_aux[xdn,ydn])

@nb.njit
def apply_p_boundary_conditions(p, p_aux, boundary, u, rho, Pi_S, Pi_A, nu):
    Lx, Ly = p.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[l,x,y,0] or boundary[l,x,y,1]:
                    nx, ny = boundary[l,x,y]
                    a = 0 if nx==0 else round(nx/abs(nx))
                    b = 0 if ny==0 else round(ny/abs(ny))
                    Fx=a*(Pi_S[x,y,0]-Pi_S[x-a,y,0])+b*(Pi_S[x,y,1]+Pi_A[x,y]-Pi_S[x,y-b,1]-Pi_A[x,y-b])
                    Fy=a*(Pi_S[x,y,1]-Pi_A[x,y]-Pi_S[x-a,y,1]+Pi_A[x-a,y])-b*(Pi_S[x,y,0]-Pi_S[x,y-b,0])
                    F=np.array([Fx,Fy])
                    lapu=2*u[x,y]-2*(u[x-a,y]+u[x,y-b])+u[x-2*a,y]+u[x,y-2*b]
                    denom=a*nx+b*ny
                    p[x,y]=(np.dot(boundary[l,x,y],F+rho*nu*lapu)+a*nx*p_aux[x-a,y]+b*ny*p_aux[x,y-b])/denom

@nb.njit
def apply_H_boundary_conditions(H, gamma, Q, u, S, boundary):
    Lx, Ly = H.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[0,x,y,0] or boundary[0,x,y,1] or boundary[1,x,y,0] or boundary[1,x,y,1]:
                    nx, ny = boundary[l,x,y]
                    a = 0 if nx==0 else round(nx/abs(nx))
                    b = 0 if ny==0 else round(ny/abs(ny))
                    H[x,y,:]=gamma*(a*u[x,y,0]*(Q[x,y]-Q[x-a,y])+b*u[x,y,1]*(Q[x,y]-Q[x,y-b])-S[x,y,:])

# Warm up JIT
print("JIT warmup...", flush=True)
_u = np.zeros((LX,LY,2)); _Q=np.zeros((LX,LY,2)); _H=np.zeros((LX,LY,2)); _S=np.zeros((LX,LY,2))
_PS=np.zeros((LX,LY,2)); _PA=np.zeros((LX,LY)); _p=np.zeros((LX,LY)); _pa=np.zeros((LX,LY)); _r=np.zeros((LX,LY))
H_S_from_Q(_u,_Q,_H,_S,A,C,float(K_ELASTIC),lam,bounds)
apply_H_boundary_conditions(_H,gamma,_Q,_u,_S,boundary)
calculate_Pi(_PS,_PA,_H,_Q,lam,zeta,float(K_ELASTIC),bounds)
div_vector(_u,_r,bounds); _r*=rho/dt
calculate_pressure_terms(_u,rho,_PS,_r,bounds)
_pa[:]=_p; relax_pressure_inner_loop(_p,_pa,_r,bounds)
apply_p_boundary_conditions(_p,_pa,boundary,_u,rho,_PS,_PA,nu)
print("JIT warmup done.", flush=True)

# Load IC from step_Q_ic.txt etc
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_flat2(fname, Lx, Ly):
    """Load flat (x*Ly+y)*2+c -> (Lx,Ly,2) array"""
    data = np.loadtxt(fname)
    arr = np.zeros((Lx, Ly, 2))
    for x in range(Lx):
        for y in range(Ly):
            arr[x,y,0] = data[(x*Ly+y)*2+0]
            arr[x,y,1] = data[(x*Ly+y)*2+1]
    return arr

def load_flat1(fname, Lx, Ly):
    data = np.loadtxt(fname)
    arr = np.zeros((Lx, Ly))
    for x in range(Lx):
        for y in range(Ly):
            arr[x,y] = data[x*Ly+y]
    return arr

Q = load_flat2(os.path.join(script_dir, 'step_Q_ic.txt'), LX, LY)
u = load_flat2(os.path.join(script_dir, 'step_u_ic.txt'), LX, LY)
p = load_flat1(os.path.join(script_dir, 'step_p_ic.txt'), LX, LY)

# Run substeps and dump intermediates
H = np.zeros((LX,LY,2)); S = np.zeros((LX,LY,2))
Pi_S = np.zeros((LX,LY,2)); Pi_A = np.zeros((LX,LY))
rhs_arr = np.zeros((LX,LY)); p_aux = np.zeros((LX,LY))

H_S_from_Q(u, Q, H, S, A, C, float(K_ELASTIC), lam, bounds)
print(f"H_S_from_Q done: max|H|={np.max(np.abs(H)):.4e}", flush=True)

apply_H_boundary_conditions(H, gamma, Q, u, S, boundary)
print(f"apply_H_bc done: max|H|={np.max(np.abs(H)):.4e}", flush=True)

calculate_Pi(Pi_S, Pi_A, H, Q, lam, zeta, float(K_ELASTIC), bounds)
print(f"calculate_Pi done: max|Pi_S|={np.max(np.abs(Pi_S)):.4e}", flush=True)

# Pressure RHS (correct numba version)
div_vector(u, rhs_arr, bounds)
rhs_arr *= rho / dt
calculate_pressure_terms(u, rho, Pi_S, rhs_arr, bounds)
print(f"p_rhs done: max|rhs|={np.max(np.abs(rhs_arr)):.4e}", flush=True)
np.savetxt(os.path.join(script_dir, 'step_p_rhs.txt'), flatten1(rhs_arr))

# Pressure relaxation: dump after each iteration for comparison
for i in range(max_p_iters):
    p_aux[:] = p
    relax_pressure_inner_loop(p, p_aux, rhs_arr, bounds)
    apply_p_boundary_conditions(p, p_aux, boundary, u, rho, Pi_S, Pi_A, nu)
print(f"p_relax done: max|p|={np.max(np.abs(p)):.4e}", flush=True)
np.savetxt(os.path.join(script_dir, 'step_p_mid.txt'), flatten1(p))

# Save Pi arrays for Rust comparison
np.savetxt(os.path.join(script_dir, 'step_Pi_S.txt'), flatten2(Pi_S))
np.savetxt(os.path.join(script_dir, 'step_Pi_A.txt'), flatten1(Pi_A))
np.savetxt(os.path.join(script_dir, 'step_H.txt'), flatten2(H))
np.savetxt(os.path.join(script_dir, 'step_S.txt'), flatten2(S))

print("All saved.", flush=True)
