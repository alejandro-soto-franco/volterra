#!/usr/bin/env python3
"""
dump_step_p2.py — Isolate pressure BC behavior.
Dumps p after N Jacobi sweeps WITHOUT BC, then p after N sweeps WITH BC,
so we can compare Rust vs Python BC application.
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
MAX_P_ITERS = 20

D_EPI = 0.99
K_EPI = 2

def set_boundary_epitrochoid(Lx, Ly):
    """Build sim_points / bounds / boundary.
    Normals loaded from Rust bnd_*_normals.txt for exact parity with Rust.
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

def flatten1(arr):
    Lx, Ly = arr.shape
    out = np.zeros(Lx*Ly)
    for x in range(Lx):
        for y in range(Ly):
            out[x*Ly+y] = arr[x,y]
    return out

def flatten2(arr):
    Lx, Ly, _ = arr.shape
    out = np.zeros(Lx*Ly*2)
    for x in range(Lx):
        for y in range(Ly):
            out[(x*Ly+y)*2+0] = arr[x,y,0]
            out[(x*Ly+y)*2+1] = arr[x,y,1]
    return out

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

# Warmup
print("JIT warmup...", flush=True)
_p=np.zeros((LX,LY)); _pa=np.zeros((LX,LY)); _r=np.zeros((LX,LY))
_u=np.zeros((LX,LY,2)); _PS=np.zeros((LX,LY,2)); _PA=np.zeros((LX,LY))
_pa[:]=_p; relax_pressure_inner_loop(_p,_pa,_r,bounds)
apply_p_boundary_conditions(_p,_pa,boundary,_u,rho,_PS,_PA,nu)
print("JIT warmup done.", flush=True)

# Load fields from step_p.py outputs
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_flat2(fname, Lx, Ly):
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

Pi_S = load_flat2(os.path.join(script_dir, 'step_Pi_S.txt'), LX, LY)
Pi_A = load_flat1(os.path.join(script_dir, 'step_Pi_A.txt'), LX, LY)
u    = load_flat2(os.path.join(script_dir, 'step_u_ic.txt'), LX, LY)
rhs_data = np.loadtxt(os.path.join(script_dir, 'step_p_rhs.txt'))
rhs  = np.zeros((LX, LY))
for x in range(LX):
    for y in range(LY):
        rhs[x,y] = rhs_data[x*LY+y]

# Run N sweeps WITHOUT BC
p_nobc = np.zeros((LX, LY))
p_aux = np.zeros((LX, LY))
for _ in range(MAX_P_ITERS):
    p_aux[:] = p_nobc
    relax_pressure_inner_loop(p_nobc, p_aux, rhs, bounds)
np.savetxt(os.path.join(script_dir, 'step_p_nobc.txt'), flatten1(p_nobc))
print(f"p without BC: max|p|={np.max(np.abs(p_nobc)):.4e}", flush=True)

# Run N sweeps WITH BC
p_withbc = np.zeros((LX, LY))
p_aux2 = np.zeros((LX, LY))
for i in range(MAX_P_ITERS):
    p_aux2[:] = p_withbc
    relax_pressure_inner_loop(p_withbc, p_aux2, rhs, bounds)
    apply_p_boundary_conditions(p_withbc, p_aux2, boundary, u, rho, Pi_S, Pi_A, nu)
    # Print BC effect on first iteration
    if i == 0:
        # Find a boundary cell and print its value
        for x in range(LX):
            for y in range(LY):
                if boundary[1,x,y,0] or boundary[1,x,y,1]:
                    print(f"  After iter 1 BC: p[{x},{y}] = {p_withbc[x,y]:.6e} (p_nobc={p_nobc[x,y]:.6e})", flush=True)
                    break
            else:
                continue
            break

np.savetxt(os.path.join(script_dir, 'step_p_withbc.txt'), flatten1(p_withbc))
print(f"p with BC: max|p|={np.max(np.abs(p_withbc)):.4e}", flush=True)

# Check one BC cell's value for debugging
print("\nSample boundary cell BC values:", flush=True)
for x in range(LX):
    for y in range(LY):
        for l in range(2):
            if boundary[l,x,y,0] or boundary[l,x,y,1]:
                nx, ny = boundary[l,x,y]
                a = 0 if nx==0 else round(nx/abs(nx))
                b = 0 if ny==0 else round(ny/abs(ny))
                Fx=a*(Pi_S[x,y,0]-Pi_S[x-a,y,0])+b*(Pi_S[x,y,1]+Pi_A[x,y]-Pi_S[x,y-b,1]-Pi_A[x,y-b])
                Fy=a*(Pi_S[x,y,1]-Pi_A[x,y]-Pi_S[x-a,y,1]+Pi_A[x-a,y])-b*(Pi_S[x,y,0]-Pi_S[x,y-b,0])
                denom=a*nx+b*ny
                lapu=2*u[x,y]-2*(u[x-a,y]+u[x,y-b])+u[x-2*a,y]+u[x,y-2*b]
                F_dot_n = nx*Fx + ny*Fy + rho*nu*(nx*lapu[0]+ny*lapu[1])
                p_nbr = a*nx*p_withbc[x-a,y] + b*ny*p_withbc[x,y-b]
                print(f"  l={l} [{x},{y}] n=({nx:.3f},{ny:.3f}) a={a} b={b} denom={denom:.4f} Fx={Fx:.3e} Fy={Fy:.3e} F_dot_n={F_dot_n:.3e} p_nbr={p_nbr:.3e} p_result={p_withbc[x,y]:.3e}", flush=True)
                break
    else:
        continue
    break

print("Done.", flush=True)
