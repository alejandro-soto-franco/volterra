#!/usr/bin/env python3
"""
dump_step_debug.py — Dump intermediate arrays after each substep for debugging.
Uses same IC and boundary as dump_step.py (K=2, d=0.99, 30x30 grid, seed=42).
"""

import numpy as np
import numba as nb
from scipy.optimize import fsolve
import os, sys

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
K_EPI = 2   # matches Rust

def set_boundary_epitrochoid(Lx, Ly):
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
    outer_bound = set()
    for x, y in sim_points:
        if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} - sim_points:
            outer_bound.add((x,y))
            def f(u, x=x, y=y):
                return np.arctan2(y-radius, x-radius) - np.arctan2(
                    (K_EPI+1)*np.sin(u)+D_EPI*np.sin((K_EPI+1)*u),
                    (K_EPI+1)*np.cos(u)+D_EPI*np.cos((K_EPI+1)*u))
            u_sol = fsolve(f, 0.1)[0]
            norm = (1+D_EPI**2+2*D_EPI*np.cos(K_EPI*u_sol))**0.5
            boundary[1,x,y,0] = round((np.cos(u_sol)+D_EPI*np.cos((K_EPI+1)*u_sol))/norm, 4)
            boundary[1,x,y,1] = round((np.sin(u_sol)+D_EPI*np.sin((K_EPI+1)*u_sol))/norm, 4)
    for x, y in sim_points:
        if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
            def f(u, x=x, y=y):
                return np.arctan2(y-radius, x-radius) - np.arctan2(
                    (K_EPI+1)*np.sin(u)+D_EPI*np.sin((K_EPI+1)*u),
                    (K_EPI+1)*np.cos(u)+D_EPI*np.cos((K_EPI+1)*u))
            u_sol = fsolve(f, 0.1)[0]
            norm = (1+D_EPI**2+2*D_EPI*np.cos(K_EPI*u_sol))**0.5
            boundary[0,x,y,0] = round((np.cos(u_sol)+D_EPI*np.cos((K_EPI+1)*u_sol))/norm, 4)
            boundary[0,x,y,1] = round((np.sin(u_sol)+D_EPI*np.sin((K_EPI+1)*u_sol))/norm, 4)
    bounds = np.zeros((len(sim_points), 2), dtype=int)
    for i, (x, y) in enumerate(sim_points):
        bounds[i] = [x, y]
    return boundary, bounds, sim_points

print("Building boundary...", flush=True)
boundary, bounds, sim_points = set_boundary_epitrochoid(LX, LY)
print(f"Interior cells: {len(sim_points)}", flush=True)

# Dump boundary normals for comparison
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

script_dir = os.path.dirname(os.path.abspath(__file__))

# Dump outer and inner normals
outer_normals = np.zeros((LX, LY, 2))
inner_normals = np.zeros((LX, LY, 2))
for x in range(LX):
    for y in range(LY):
        if boundary[1,x,y,0] or boundary[1,x,y,1]:
            outer_normals[x,y] = boundary[1,x,y]
        if boundary[0,x,y,0] or boundary[0,x,y,1]:
            inner_normals[x,y] = boundary[0,x,y]

np.savetxt(os.path.join(script_dir, "dbg_outer_normals.txt"), flatten2(outer_normals))
np.savetxt(os.path.join(script_dir, "dbg_inner_normals.txt"), flatten2(inner_normals))

# Dump outer/inner bool masks as 0/1
outer_mask = np.zeros(LX*LY)
inner_mask = np.zeros(LX*LY)
inside_mask = np.zeros(LX*LY)
for x, y in sim_points:
    inside_mask[x*LY+y] = 1.0
for x in range(LX):
    for y in range(LY):
        if boundary[1,x,y,0] or boundary[1,x,y,1]:
            outer_mask[x*LY+y] = 1.0
        if boundary[0,x,y,0] or boundary[0,x,y,1]:
            inner_mask[x*LY+y] = 1.0
np.savetxt(os.path.join(script_dir, "dbg_inside_mask.txt"), inside_mask)
np.savetxt(os.path.join(script_dir, "dbg_outer_mask.txt"), outer_mask)
np.savetxt(os.path.join(script_dir, "dbg_inner_mask.txt"), inner_mask)
print(f"outer cells: {int(outer_mask.sum())}, inner cells: {int(inner_mask.sum())}", flush=True)

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
        rhs[x,y]+=(Pi_S[xup,y,0]+Pi_S[xdn,y,0]-Pi_S[x,yup,0]-Pi_S[x,ydn,0])
        +0.5*(Pi_S[xup,yup,1]-Pi_S[xup,ydn,1]-Pi_S[xdn,yup,1]+Pi_S[xdn,ydn,1])
        -rho*(dudx*dudx+dvdy*dvdy+0.5*(u[x,yup,0]-u[x,ydn,0])*(u[xup,y,1]-u[xdn,y,1]))

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
def apply_Q_boundary_conditions(Q, boundary, S0):
    Lx, Ly = Q.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[l,x,y,0] or boundary[l,x,y,1]:
                    nx, ny = boundary[l,x,y,0], boundary[l,x,y,1]
                    theta = np.arccos(nx)
                    if ny < 0: theta = 2*np.pi - theta
                    net_charge = 2/2
                    nnx, nny = np.cos(theta*net_charge), np.sin(theta*net_charge)
                    Q[x,y,0] = S0*(nny**2 - 0.5)
                    Q[x,y,1] = S0*(-nnx*nny)

@nb.njit
def apply_u_boundary_conditions(u, boundary):
    Lx, Ly = u.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[l,x,y,0] or boundary[l,x,y,1]:
                    nx, ny = boundary[l,x,y]
                    a = 0 if nx==0 else round(nx/abs(nx))
                    b = 0 if ny==0 else round(ny/abs(ny))
                    u[x,y]=np.array([ny,-nx])*(b*u[x,y-b,0]-a*u[x-a,y,1])/(a*nx+b*ny)
                    u[x,y,:] = 0

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

# Warmup
print("JIT warmup...", flush=True)
_u = np.zeros((LX, LY, 2)); _Q = np.zeros((LX, LY, 2))
_H = np.zeros((LX, LY, 2)); _S = np.zeros((LX, LY, 2))
_PS = np.zeros((LX, LY, 2)); _PA = np.zeros((LX, LY))
_p = np.zeros((LX, LY)); _pa = np.zeros((LX, LY)); _r = np.zeros((LX, LY))
H_S_from_Q(_u,_Q,_H,_S,A,C,float(K_ELASTIC),lam,bounds)
apply_H_boundary_conditions(_H,gamma,_Q,_u,_S,boundary)
calculate_Pi(_PS,_PA,_H,_Q,lam,zeta,float(K_ELASTIC),bounds)
div_vector(_u,_r,bounds); _r*=rho/dt
calculate_pressure_terms(_u,rho,_PS,_r,bounds)
_pa[:]=_p; relax_pressure_inner_loop(_p,_pa,_r,bounds)
apply_p_boundary_conditions(_p,_pa,boundary,_u,rho,_PS,_PA,nu)
print("JIT warmup done.", flush=True)

# Load IC
rng = np.random.default_rng(seed=42)
u = np.zeros((LX, LY, 2)); p = np.zeros((LX, LY)); Q = np.zeros((LX, LY, 2))
theta_ic = np.pi * rng.random((LX, LY))
theta_mask = np.zeros((LX, LY))
for x, y in sim_points: theta_mask[x, y] = theta_ic[x, y]
theta_ic = theta_mask
nx_ic = np.cos(theta_ic); ny_ic = np.sin(theta_ic)
Q[:,:,0] = S0*(nx_ic**2 - 0.5); Q[:,:,1] = S0*(nx_ic*ny_ic)
apply_Q_boundary_conditions(Q, boundary, S0)
apply_u_boundary_conditions(u, boundary)

dudt = np.zeros((LX,LY,2)); dQdt = np.zeros((LX,LY,2))
p_aux = np.zeros((LX,LY)); H = np.zeros((LX,LY,2)); S = np.zeros((LX,LY,2))
Pi_S = np.zeros((LX,LY,2)); Pi_A = np.zeros((LX,LY)); rhs_arr = np.zeros((LX,LY))

# Save IC
np.savetxt(os.path.join(script_dir,"dbg_Q_ic.txt"), flatten2(Q))
np.savetxt(os.path.join(script_dir,"dbg_u_ic.txt"), flatten2(u))
np.savetxt(os.path.join(script_dir,"dbg_p_ic.txt"), flatten1(p))

# Step 1: H_S_from_Q
H_S_from_Q(u, Q, H, S, A, C, float(K_ELASTIC), lam, bounds)
np.savetxt(os.path.join(script_dir,"dbg_H_after_HS.txt"), flatten2(H))
np.savetxt(os.path.join(script_dir,"dbg_S_after_HS.txt"), flatten2(S))
print(f"After H_S_from_Q: max|H|={np.max(np.abs(H)):.4e}, max|S|={np.max(np.abs(S)):.4e}", flush=True)

# Step 2: apply_H_bc
apply_H_boundary_conditions(H, gamma, Q, u, S, boundary)
np.savetxt(os.path.join(script_dir,"dbg_H_after_Hbc.txt"), flatten2(H))
print(f"After apply_H_bc: max|H|={np.max(np.abs(H)):.4e}", flush=True)

# Step 3: calculate_Pi
calculate_Pi(Pi_S, Pi_A, H, Q, lam, zeta, float(K_ELASTIC), bounds)
np.savetxt(os.path.join(script_dir,"dbg_Pi_S.txt"), flatten2(Pi_S))
np.savetxt(os.path.join(script_dir,"dbg_Pi_A.txt"), flatten1(Pi_A))
print(f"After calculate_Pi: max|Pi_S|={np.max(np.abs(Pi_S)):.4e}, max|Pi_A|={np.max(np.abs(Pi_A)):.4e}", flush=True)

# Step 4: pressure relaxation
rhs_arr[:] = 0
div_vector(u, rhs_arr, bounds); rhs_arr *= rho/dt
calculate_pressure_terms(u, rho, Pi_S, rhs_arr, bounds)
np.savetxt(os.path.join(script_dir,"dbg_p_rhs.txt"), flatten1(rhs_arr))
print(f"After p rhs: max|rhs|={np.max(np.abs(rhs_arr)):.4e}", flush=True)

for _ in range(max_p_iters):
    p_aux[:] = p
    relax_pressure_inner_loop(p, p_aux, rhs_arr, bounds)
    apply_p_boundary_conditions(p, p_aux, boundary, u, rho, Pi_S, Pi_A, nu)
np.savetxt(os.path.join(script_dir,"dbg_p_after_relax.txt"), flatten1(p))
print(f"After pressure relax: max|p|={np.max(np.abs(p)):.4e}", flush=True)

# Step 5: Q update
dQdt[:] = (1/gamma)*H[:] + S[:]
upwind_advective_term(u, Q, dQdt, bounds)
np.savetxt(os.path.join(script_dir,"dbg_dQdt.txt"), flatten2(dQdt))
print(f"After Q update: max|dQ|={np.max(np.abs(dQdt)):.4e}", flush=True)

# Step 6: u update (viscous + convective + p/pi)
Laplacian_vector(u, dudt, bounds, coeff=nu)
upwind_advective_term(u, u, dudt, bounds)
# u_update_p_pi
Lx2, Ly2 = u.shape[:2]
for point in range(len(bounds)):
    x, y = bounds[point, 0], bounds[point, 1]
    xup=(x+1)%Lx2; xdn=x-1; yup=(y+1)%Ly2; ydn=y-1
    dudt[x,y,0] += 0.5/rho*(-(p[xup,y]-p[xdn,y])+(Pi_S[xup,y,0]-Pi_S[xdn,y,0])+((Pi_S[x,yup,1]+Pi_A[x,yup])-(Pi_S[x,ydn,1]+Pi_A[x,ydn])))
    dudt[x,y,1] += 0.5/rho*(-(p[x,yup]-p[x,ydn])+((Pi_S[xup,y,1]-Pi_A[xup,y])-(Pi_S[xdn,y,1]-Pi_A[xdn,y]))-(Pi_S[x,yup,0]-Pi_S[x,ydn,0]))
np.savetxt(os.path.join(script_dir,"dbg_dudt.txt"), flatten2(dudt))
print(f"After u update: max|dudt|={np.max(np.abs(dudt)):.4e}", flush=True)

print("All debug files saved.", flush=True)
