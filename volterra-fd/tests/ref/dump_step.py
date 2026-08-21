#!/usr/bin/env python3
"""
dump_step.py — Generate reference data for the Task E step integration test.

Uses a fixed small nephroid grid (lx=ly=30, k=2 epitrochoid, d=0.99),
fixed physics constants matching the task spec, a seeded IC for Q (winding
director field from apply_Q_boundary_conditions), u=0, p=0.  Runs exactly
ONE update_step_inner and dumps the resulting Q, u, p.

Also stores the initial IC arrays (Q_ic, u_ic, p_ic) so the Rust test can
load them without re-deriving.

Output files (all in the same directory as this script):
  step_Q_ic.txt      — initial Q  (lx*ly*2 values, flat (x*ly+y)*2+c order)
  step_u_ic.txt      — initial u  (lx*ly*2)
  step_p_ic.txt      — initial p  (lx*ly)
  step_Q_ref.txt     — Q after 1 step
  step_u_ref.txt     — u after 1 step
  step_p_ref.txt     — p after 1 step
  step_interior_count.txt  — number of interior cells (for sanity check)

Usage:
  uv run --with numpy --with scipy --with numba python dump_step.py
"""

import numpy as np
import numba as nb
import scipy
from scipy.optimize import fsolve
import os, sys

# ── grid ──────────────────────────────────────────────────────────────────────
LX = LY = 30

# ── physics constants (task spec) ─────────────────────────────────────────────
K    = 2**14          # 16384
gamma = 100.0
lam  = 0.7            # λ  (flow-alignment)
nu   = (10.0 * K)**0.5  # η  (kinematic viscosity, same as eta in Params)
rho  = 1.0
chi  = 1.0            # not used in these kernels directly
zeta = K              # ζ = K / als^2  with als=1  → K
A    = -K / 81.0      # a_landau
C    =  K / 81.0      # c_landau
S0   = 2.0**0.5       # sqrt(-2*A/C) = sqrt(2)
dt   = 1e-4
max_p_iters    = 20
p_target_rel_change = 1e-4  # pressure stops at max_p_iters since it's low

# ── epitrochoid boundary (k=1, d=0.99 → nephroid) ─────────────────────────────
#  Python set_boundary uses k=1 for 'epitrochoid' (number of cusps = 2*(k)=2)
D_EPI = 0.99
K_EPI = 2   # matches Rust boundary.rs: K=2.0 in the epitrochoid formula
            # NOTE: Python flow-solver.py uses k=1 in set_boundary('epitrochoid'),
            # but the Rust crate was implemented with K=2 (Tasks A-D use K=2).
            # This dumper must match Rust so boundary cells agree exactly.

def set_boundary_epitrochoid(Lx, Ly):
    """Build sim_points / bounds / boundary.
    Normals are loaded from Rust-generated bnd_*_normals.txt so Python and
    Rust use byte-identical normal values (avoids fsolve vs Newton sign flip
    in near-zero components like cell [14,7] where nx ~ 1e-12).
    """
    boundary = np.zeros((2, Lx, Ly, 2))
    sim_points = set()
    radius = Lx // 2 - 1

    for x in range(Lx):
        for y in range(Ly):
            def f(u, x=x, y=y):
                return np.arctan2(y - radius, x - radius) - np.arctan2(
                    (K_EPI + 1)*np.sin(u) + D_EPI*np.sin((K_EPI+1)*u),
                    (K_EPI + 1)*np.cos(u) + D_EPI*np.cos((K_EPI+1)*u)
                )
            u_sol = fsolve(f, 0.1)[0]
            rhs_val = radius**2 / (K_EPI+2)**2 * (
                (K_EPI+1)**2 + D_EPI**2 + 2*(K_EPI+1)*D_EPI*np.cos(K_EPI*u_sol)
            )
            if (x-radius)**2 + (y-radius)**2 <= rhs_val:
                sim_points.add((x,y))

    outer_bound = set()
    for x, y in sim_points:
        if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} - sim_points:
            outer_bound.add((x,y))
    inner_bound = set()
    for x, y in sim_points:
        if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
            inner_bound.add((x,y))

    # Load normals from Rust reference files (same directory as this script)
    _sdir = os.path.dirname(os.path.abspath(__file__))
    outer_flat = np.loadtxt(os.path.join(_sdir, 'bnd_outer_normals.txt'))
    inner_flat = np.loadtxt(os.path.join(_sdir, 'bnd_inner_normals.txt'))
    for x in range(Lx):
        for y in range(Ly):
            idx2 = (x * Ly + y) * 2
            boundary[1, x, y, 0] = outer_flat[idx2]
            boundary[1, x, y, 1] = outer_flat[idx2 + 1]
            boundary[0, x, y, 0] = inner_flat[idx2]
            boundary[0, x, y, 1] = inner_flat[idx2 + 1]

    # Build bounds array
    bounds = np.zeros((len(sim_points), 2), dtype=int)
    sp = list(sim_points)
    for i, (x, y) in enumerate(sp):
        bounds[i] = [x, y]

    return boundary, bounds, sim_points

print("Building boundary...", flush=True)
boundary, bounds, sim_points = set_boundary_epitrochoid(LX, LY)
print(f"Interior cells: {len(sim_points)}", flush=True)

# ── physics kernels (copied verbatim from flow-solver.py) ─────────────────────

@nb.njit(parallel=True, fastmath=True, nogil=True)
def Laplacian_vector(arr, out, bounds, coeff=1.):
    coeff /= 6
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        out[x, y, :] = coeff * (
            -20 * arr[x, y, :]
            + 4 * (arr[xup, y, :] + arr[x, ydn, :] + arr[x, yup, :] + arr[xdn, y, :])
            + arr[xup, ydn, :] + arr[xup, yup, :] + arr[xdn, ydn, :] + arr[xdn, yup, :]
        )

@nb.njit(parallel=True, fastmath=True, nogil=True)
def upwind_advective_term(u, arr, out, bounds, coeff=-1):
    coeff *= 0.5
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        xupup = (x + 2) % Lx
        xdndn = (x - 2)
        tmp = coeff * u[x, y, 0]
        if u[x, y, 0] > 0:
            out[x, y, :] += tmp * (3*arr[x,y,:] - 4*arr[xdn,y,:] + arr[xdndn,y,:])
        else:
            out[x, y, :] += tmp * (-3*arr[x,y,:] + 4*arr[xup,y,:] - arr[xupup,y,:])
        tmp = coeff * u[x, y, 1]
        if u[x, y, 1] > 0:
            ydn2 = (y - 1)
            ydndn = (y - 2)
            out[x, y, :] += tmp * (3*arr[x,y,:] - 4*arr[x,ydn2,:] + arr[x,ydndn,:])
        else:
            yup2 = (y + 1) % Ly
            yupup = (y + 2) % Ly
            out[x, y, :] += tmp * (-3*arr[x,y,:] + 4*arr[x,yup2,:] - arr[x,yupup,:])

@nb.njit(parallel=True, fastmath=True, nogil=True)
def div_vector(arr, out, bounds):
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        out[x, y] = 0.5 * (
            (arr[xup, y, 0] - arr[xdn, y, 0])
            + (arr[x, (y+1)%Ly, 1] - arr[x, (y-1), 1])
        )

@nb.njit(parallel=True, fastmath=True, nogil=True)
def H_S_from_Q(u, Q, H, S, A, C, K, lam, bounds):
    Laplacian_vector(Q, H, bounds, coeff=K)
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        dxux = 0.5*(u[xup,y,0] - u[xdn,y,0])
        dxuy = 0.5*(u[xup,y,1] - u[xdn,y,1])
        dyux = 0.5*(u[x,(y+1)%Ly,0] - u[x,(y-1),0])
        omega_xy = 0.5*(dxuy - dyux)
        trQsq = 2*(Q[x,y,0]**2 + Q[x,y,1]**2)
        lS = lam * np.sqrt(2*trQsq)
        H[x,y,:] -= (A + C*trQsq)*Q[x,y,:]
        TrQE = 2*Q[x,y,0]*dxux + Q[x,y,1]*(dyux+dxuy)
        S[x,y,0] = lS*dxux - 2*omega_xy*Q[x,y,1] - 2*TrQE*Q[x,y,0]
        S[x,y,1] = lS*0.5*(dxuy+dyux) + 2*omega_xy*Q[x,y,0] - 2*TrQE*Q[x,y,1]

@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_Erickson_stress(Q, K, Pi_S, bounds):
    Lx, Ly = Q.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x+1)%Lx; xdn = x-1
        yup = (y+1)%Ly; ydn = y-1
        dxQxx, dxQxy = 0.5*(Q[xup,y]-Q[xdn,y])
        dyQxx, dyQxy = 0.5*(Q[x,yup]-Q[x,ydn])
        Pi_S[x,y,0] -= K*((dxQxx)**2+(dxQxy)**2-(dyQxx)**2-(dyQxy)**2)
        Pi_S[x,y,1] -= 2*K*((dxQxy*dyQxy)+(dxQxx*dyQxx))

@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_TrQH_term(Q, H, Pi_S):
    TrQH = 2*(Q[:,:,0]*H[:,:,0] + Q[:,:,1]*H[:,:,1])
    Pi_S[:,:,0] += TrQH*Q[:,:,0]
    Pi_S[:,:,1] += TrQH*Q[:,:,1]

@nb.njit(parallel=True, fastmath=True, nogil=True)
def calculate_Pi(Pi_S, Pi_A, H, Q, lam, zeta, K, bounds):
    Pi_S[:] = -lam*H - zeta*Q
    get_Erickson_stress(Q, K, Pi_S, bounds)
    get_TrQH_term(Q, H, Pi_S)
    Pi_A[:] = 2*(Q[:,:,0]*H[:,:,1] - H[:,:,0]*Q[:,:,1])

@nb.njit(parallel=True, fastmath=True, nogil=True)
def calculate_pressure_terms(u, rho, Pi_S, rhs, bounds):
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; yup=(y+1)%Ly; ydn=y-1
        dudx = 0.5*(u[xup,y,0]-u[xdn,y,0])
        dvdy = 0.5*(u[x,yup,1]-u[x,ydn,1])
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
        p[x,y] = 0.05*(
            -6*rhs[x,y]
            + 4*(p_aux[xup,y]+p_aux[x,yup]+p_aux[x,ydn]+p_aux[xdn,y])
            + p_aux[xup,yup]+p_aux[xup,ydn]+p_aux[xdn,yup]+p_aux[xdn,ydn]
        )

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
                    Fx = a*(Pi_S[x,y,0]-Pi_S[x-a,y,0]) + b*(Pi_S[x,y,1]+Pi_A[x,y]-Pi_S[x,y-b,1]-Pi_A[x,y-b])
                    Fy = a*(Pi_S[x,y,1]-Pi_A[x,y]-Pi_S[x-a,y,1]+Pi_A[x-a,y]) - b*(Pi_S[x,y,0]-Pi_S[x,y-b,0])
                    F = np.array([Fx, Fy])
                    lapu = 2*u[x,y]-2*(u[x-a,y]+u[x,y-b])+u[x-2*a,y]+u[x,y-2*b]
                    p[x,y] = (np.dot(boundary[l,x,y], F+rho*nu*lapu) + a*nx*p_aux[x-a,y] + b*ny*p_aux[x,y-b])/(a*nx+b*ny)

@nb.njit
def relax_pressure(u, rho, p, Pi_S, Pi_A, nu, p_aux, rhs, dt, target_rel_change, boundary, bounds, max_p_iters=-1):
    div_vector(u, rhs, bounds)
    rhs *= rho/dt
    calculate_pressure_terms(u, rho, Pi_S, rhs, bounds)
    p_iters = 0
    rel_change = target_rel_change + 1
    while (p_iters < max_p_iters or max_p_iters < 0) and rel_change > target_rel_change:
        p_aux[:] = p
        relax_pressure_inner_loop(p, p_aux, rhs, bounds)
        apply_p_boundary_conditions(p, p_aux, boundary, u, rho, Pi_S, Pi_A, nu)
        rel_change = np.sum(np.abs(p_aux-p)) / np.abs(1e-7 + np.sum(p_aux))
        p_iters += 1
    return p_iters

@nb.njit(parallel=True, fastmath=True, nogil=True)
def u_update_p_Pi_terms(dudt, p, rho, Pi_S, Pi_A, bounds):
    Lx, Ly = p.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup=(x+1)%Lx; xdn=x-1; yup=(y+1)%Ly; ydn=y-1
        dudt[x,y,0] += 0.5/rho*(
            -(p[xup,y]-p[xdn,y])
            +(Pi_S[xup,y,0]-Pi_S[xdn,y,0])
            +((Pi_S[x,yup,1]+Pi_A[x,yup])-(Pi_S[x,ydn,1]+Pi_A[x,ydn]))
        )
        dudt[x,y,1] += 0.5/rho*(
            -(p[x,yup]-p[x,ydn])
            +((Pi_S[xup,y,1]-Pi_A[xup,y])-(Pi_S[xdn,y,1]-Pi_A[xdn,y]))
            -(Pi_S[x,yup,0]-Pi_S[x,ydn,0])
        )

@nb.njit
def get_u_update(dudt, u, p, rho, Pi_S, Pi_A, nu, bounds):
    Laplacian_vector(u, dudt, bounds, coeff=nu)
    upwind_advective_term(u, u, dudt, bounds)
    u_update_p_Pi_terms(dudt, p, rho, Pi_S, Pi_A, bounds)

@nb.njit
def get_Q_update(dQ, Q, H, S, u, gamma, bounds):
    dQ[:] = (1/gamma)*H[:] + S[:]
    upwind_advective_term(u, Q, dQ, bounds)

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
                    u[x,y] = np.array([ny,-nx])*(b*u[x,y-b,0]-a*u[x-a,y,1])/(a*nx+b*ny)
                    u[x,y,:] = 0

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
                    Q[x,y,0] = S0*(nny**2 - 1/2)
                    Q[x,y,1] = S0*(-nnx*nny)

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
                    H[x,y,:] = gamma*(
                        a*u[x,y,0]*(Q[x,y]-Q[x-a,y])
                        + b*u[x,y,1]*(Q[x,y]-Q[x,y-b])
                        - S[x,y,:]
                    )

@nb.njit
def update_step_inner(arrs, consts, bounds, boundary):
    dt, nu, A, C, K, lam, zeta, gamma, rho, p_target_rel_change, max_p_iters = consts
    u, dudt, Q, dQdt, p, p_aux, rhs, S, H, Pi_S, Pi_A = arrs

    H_S_from_Q(u, Q, H, S, A, C, K, lam, bounds)
    apply_H_boundary_conditions(H, gamma, Q, u, S, boundary)
    calculate_Pi(Pi_S, Pi_A, H, Q, lam, zeta, K, bounds)
    relax_pressure(u, rho, p, Pi_S, Pi_A, nu, p_aux, rhs, dt, p_target_rel_change, boundary, bounds, max_p_iters=max_p_iters)
    get_Q_update(dQdt, Q, H, S, u, gamma, bounds)
    get_u_update(dudt, u, p, rho, Pi_S, Pi_A, nu, bounds)
    Q += dt*dQdt
    u += dt*dudt
    apply_Q_boundary_conditions(Q, boundary, S0)
    apply_u_boundary_conditions(u, boundary)

# ── initial conditions ─────────────────────────────────────────────────────────

rng = np.random.default_rng(seed=42)

u = np.zeros((LX, LY, 2))
p = np.zeros((LX, LY))
Q = np.zeros((LX, LY, 2))

# Set Q IC: use the same winding director as apply_Q_boundary_conditions
# (fill all interior cells from a seeded theta, then apply BC to set boundary)
theta_ic = np.pi * rng.random((LX, LY))
# mask to interior
theta_mask = np.zeros((LX, LY))
for x, y in sim_points:
    theta_mask[x, y] = theta_ic[x, y]
theta_ic = theta_mask

# initialise_Q_from_θ
nx_ic = np.cos(theta_ic)
ny_ic = np.sin(theta_ic)
Q[:,:,0] = S0*(nx_ic**2 - 0.5)
Q[:,:,1] = S0*(nx_ic*ny_ic)

# Apply Q BC so boundary anchoring is consistent from the start
apply_Q_boundary_conditions(Q, boundary, S0)
apply_u_boundary_conditions(u, boundary)

print("IC ready. Running warmup JIT...", flush=True)

# Prepare auxiliary arrays
dudt  = np.zeros((LX, LY, 2))
dQdt  = np.zeros((LX, LY, 2))
p_aux = np.zeros((LX, LY))
H     = np.zeros((LX, LY, 2))
S     = np.zeros((LX, LY, 2))
Pi_S  = np.zeros((LX, LY, 2))
Pi_A  = np.zeros((LX, LY))
rhs   = np.zeros((LX, LY))

arrs = (u, dudt, Q, dQdt, p, p_aux, rhs, S, H, Pi_S, Pi_A)
consts = (dt, nu, A, C, float(K), lam, float(zeta), gamma, rho, p_target_rel_change, max_p_iters)

# Save IC
Q_ic = Q.copy()
u_ic = u.copy()
p_ic = p.copy()

# Warm up JIT with a copy so IC is preserved for the real run
Q2 = Q.copy(); u2 = u.copy(); p2 = p.copy()
dudt2  = np.zeros_like(dudt); dQdt2 = np.zeros_like(dQdt)
p_aux2 = np.zeros_like(p_aux); H2 = np.zeros_like(H)
S2 = np.zeros_like(S); Pi_S2 = np.zeros_like(Pi_S)
Pi_A2 = np.zeros_like(Pi_A); rhs2 = np.zeros_like(rhs)
arrs2 = (u2, dudt2, Q2, dQdt2, p2, p_aux2, rhs2, S2, H2, Pi_S2, Pi_A2)
update_step_inner(arrs2, consts, bounds, boundary)
print("JIT warmup done.", flush=True)

# Real run from IC
Q[:] = Q_ic; u[:] = u_ic; p[:] = p_ic
dudt[:]  = 0; dQdt[:] = 0; p_aux[:] = 0
H[:] = 0; S[:] = 0; Pi_S[:] = 0; Pi_A[:] = 0; rhs[:] = 0

arrs = (u, dudt, Q, dQdt, p, p_aux, rhs, S, H, Pi_S, Pi_A)
print("Running one step...", flush=True)
update_step_inner(arrs, consts, bounds, boundary)
print("Step done.", flush=True)

# ── helpers to flatten (x*Ly+y)*2+c order ─────────────────────────────────────
def flatten2(arr):
    """arr shape (Lx,Ly,2) -> flat (x*Ly+y)*2+c"""
    Lx, Ly, _ = arr.shape
    out = np.zeros(Lx*Ly*2)
    for x in range(Lx):
        for y in range(Ly):
            out[(x*Ly+y)*2+0] = arr[x,y,0]
            out[(x*Ly+y)*2+1] = arr[x,y,1]
    return out

def flatten1(arr):
    """arr shape (Lx,Ly) -> flat x*Ly+y"""
    Lx, Ly = arr.shape
    out = np.zeros(Lx*Ly)
    for x in range(Lx):
        for y in range(Ly):
            out[x*Ly+y] = arr[x,y]
    return out

# ── save ───────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

np.savetxt(os.path.join(script_dir, "step_Q_ic.txt"),  flatten2(Q_ic))
np.savetxt(os.path.join(script_dir, "step_u_ic.txt"),  flatten2(u_ic))
np.savetxt(os.path.join(script_dir, "step_p_ic.txt"),  flatten1(p_ic))
np.savetxt(os.path.join(script_dir, "step_Q_ref.txt"), flatten2(Q))
np.savetxt(os.path.join(script_dir, "step_u_ref.txt"), flatten2(u))
np.savetxt(os.path.join(script_dir, "step_p_ref.txt"), flatten1(p))
np.savetxt(os.path.join(script_dir, "step_interior_count.txt"),
           np.array([len(sim_points)]))

print(f"Saved reference files to {script_dir}/")
print(f"Interior count: {len(sim_points)}")
print(f"|Q_ref| max: {np.max(np.abs(Q)):.6f}")
print(f"|u_ref| max: {np.max(np.abs(u)):.6e}")
print(f"|p_ref| max: {np.max(np.abs(p)):.6e}")
