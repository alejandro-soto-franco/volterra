"""
Reference dumper for Stokes / pressure-relaxation kernel tests.

Builds fixed pseudo-random fields on a 24x24 grid (seed 0, same as
dump_nematic.py), uses a rectangular interior (1-cell border excluded),
and exercises:

  1. calculate_pressure_terms + the initial RHS assembly (div·u scaled)
  2. N_ITERS fixed Jacobi sweeps of relax_pressure_inner_loop
  3. u_update_p_Π_terms to get dudt

The Python relax_pressure driver also calls apply_p_boundary_conditions,
but since our rectangular interior has NO boundary cells inside it (the
1-cell border is excluded from bounds), that call touches zero cells and
is a true no-op for this grid.  We replicate it faithfully: the BC call
runs but does nothing.

Outputs (one value per line, row-major (x, y) or (x, y, c)):
  p_rhs_ref.txt        scalar  lx*ly     pressure Poisson RHS
  p_after_N_ref.txt    scalar  lx*ly     p after exactly N Jacobi sweeps
  dudt_ref.txt         2-comp  lx*ly*2   velocity time-derivative

Constants:
  LX, LY = 24, 24
  rho    = 1.0
  dt     = 0.01
  nu     = sqrt(10 * 2**14)   (= params.eta / rho, used as kinematic viscosity)
  N_ITERS = 20                (exact sweep count; p_target_rel_change=1e-15)
"""

import sys
import os
import numpy as np

LX, LY = 24, 24
N = LX * LY

# ── constants ──────────────────────────────────────────────────────────────
K      = float(2**14)
rho    = 1.0
dt     = 0.01
nu     = (10.0 * K) ** 0.5          # kinematic viscosity = eta
lam    = 0.7
zeta   = K / 1.0
A      = -K / 81.0
C      =  K / 81.0
N_ITERS = 20                        # exact iteration count for the test

# ── random fields (same seed as dump_nematic.py) ───────────────────────────
rng = np.random.default_rng(0)
Q   = rng.uniform(-0.5, 0.5, (LX, LY, 2))
u   = rng.uniform(-0.1, 0.1, (LX, LY, 2))

# Pi_S and Pi_A: use small random fields (physically plausible scale)
# In a real step these come from calculate_Pi; here we use fixed randoms
# so both Python and Rust get identical inputs.
rng2  = np.random.default_rng(42)
Pi_S  = rng2.uniform(-1.0, 1.0, (LX, LY, 2)) * K * 0.01
Pi_A  = rng2.uniform(-1.0, 1.0, (LX, LY))    * K * 0.01
p     = np.zeros((LX, LY))

# ── interior mask (1-cell border excluded) ─────────────────────────────────
interior = []
for x in range(LX):
    for y in range(LY):
        if 1 <= x <= LX - 2 and 1 <= y <= LY - 2:
            interior.append((x, y))
bounds = np.array(interior, dtype=np.int32)   # shape (N_interior, 2)

# ── helper operators ────────────────────────────────────────────────────────

def div_vector(arr, bounds):
    """Centred divergence of a 2-component vector field."""
    Lx, Ly = arr.shape[:2]
    out = np.zeros((Lx, Ly))
    for (x, y) in bounds:
        xup = (x + 1) % Lx;  xdn = x - 1
        yup = (y + 1) % Ly;  ydn = y - 1
        out[x, y] = 0.5 * (
            (arr[xup, y, 0] - arr[xdn, y, 0])
            + (arr[x, yup, 1] - arr[x, ydn, 1])
        )
    return out


def calculate_pressure_terms(u, rho, Pi_S, rhs, bounds):
    """Accumulate ∇·F and −ρ·(∂ᵢuⱼ∂ⱼuᵢ) onto rhs (in-place)."""
    Lx, Ly = u.shape[:2]
    for (x, y) in bounds:
        xup = (x + 1) % Lx;  xdn = x - 1
        yup = (y + 1) % Ly;  ydn = y - 1
        dudx = 0.5 * (u[xup, y, 0] - u[xdn, y, 0])
        dvdy = 0.5 * (u[x, yup, 1] - u[x, ydn, 1])
        rhs[x, y] += (
            (Pi_S[xup, y, 0] + Pi_S[xdn, y, 0] - Pi_S[x, yup, 0] - Pi_S[x, ydn, 0])
            + 0.5 * (
                Pi_S[xup, yup, 1] - Pi_S[xup, ydn, 1]
                - Pi_S[xdn, yup, 1] + Pi_S[xdn, ydn, 1]
            )
            - rho * (
                dudx * dudx + dvdy * dvdy
                + 0.5 * (u[x, yup, 0] - u[x, ydn, 0]) * (u[xup, y, 1] - u[xdn, y, 1])
            )
        )


def relax_pressure_inner_loop(p, p_aux, rhs, bounds):
    """One Jacobi sweep: read p_aux, write p."""
    Lx, Ly = p.shape
    for (x, y) in bounds:
        xup = (x + 1) % Lx;  xdn = x - 1
        yup = (y + 1) % Ly;  ydn = y - 1
        p[x, y] = 0.05 * (
            -6.0 * rhs[x, y]
            + 4.0 * (p_aux[xup, y] + p_aux[x, yup] + p_aux[x, ydn] + p_aux[xdn, y])
            + p_aux[xup, yup] + p_aux[xup, ydn] + p_aux[xdn, yup] + p_aux[xdn, ydn]
        )


def apply_p_boundary_conditions_stub(p, p_aux, bounds):
    """No-op: rectangular interior has no boundary ring in bounds."""
    pass


def u_update_p_pi_terms(dudt, p, rho, Pi_S, Pi_A, bounds):
    """Accumulate −∇p/ρ + ∇·Π/ρ onto dudt (in-place)."""
    Lx, Ly = p.shape
    for (x, y) in bounds:
        xup = (x + 1) % Lx;  xdn = x - 1
        yup = (y + 1) % Ly;  ydn = y - 1
        dudt[x, y, 0] += 0.5 / rho * (
            -(p[xup, y] - p[xdn, y])
            + (Pi_S[xup, y, 0] - Pi_S[xdn, y, 0])
            + (Pi_S[x, yup, 1] + Pi_A[x, yup]) - (Pi_S[x, ydn, 1] + Pi_A[x, ydn])
        )
        dudt[x, y, 1] += 0.5 / rho * (
            -(p[x, yup] - p[x, ydn])
            + (Pi_S[xup, y, 1] - Pi_A[xup, y]) - (Pi_S[xdn, y, 1] - Pi_A[xdn, y])
            - (Pi_S[x, yup, 0] - Pi_S[x, ydn, 0])
        )


def laplacian_vector(arr, bounds, coeff=1.0):
    Lx, Ly = arr.shape[:2]
    out = np.zeros_like(arr)
    c = coeff / 6.0
    for (x, y) in bounds:
        xup = (x + 1) % Lx;  xdn = x - 1
        yup = (y + 1) % Ly;  ydn = y - 1
        out[x, y, :] = c * (
            -20 * arr[x, y, :]
            + 4 * (arr[xup, y, :] + arr[xdn, y, :] + arr[x, yup, :] + arr[x, ydn, :])
            + arr[xup, yup, :] + arr[xup, ydn, :] + arr[xdn, yup, :] + arr[xdn, ydn, :]
        )
    return out


def upwind_advective_term(u_field, arr, out, bounds, coeff=-1.0):
    """Second-order upwind advection: out += coeff*(u·∇)arr (in-place)."""
    Lx, Ly = arr.shape[:2]
    half = coeff * 0.5
    for (x, y) in bounds:
        xup   = (x + 1) % Lx;  xdn   = x - 1
        xupup = (x + 2) % Lx;  xdndn = x - 2
        yup   = (y + 1) % Ly;  ydn   = y - 1
        yupup = (y + 2) % Ly;  ydndn = y - 2
        ux = u_field[x, y, 0]
        uy = u_field[x, y, 1]
        tmp_x = half * ux
        if ux > 0:
            out[x, y, :] += tmp_x * (3*arr[x,y,:] - 4*arr[xdn,y,:] + arr[xdndn,y,:])
        else:
            out[x, y, :] += tmp_x * (-3*arr[x,y,:] + 4*arr[xup,y,:] - arr[xupup,y,:])
        tmp_y = half * uy
        if uy > 0:
            out[x, y, :] += tmp_y * (3*arr[x,y,:] - 4*arr[x,ydn,:] + arr[x,ydndn,:])
        else:
            out[x, y, :] += tmp_y * (-3*arr[x,y,:] + 4*arr[x,yup,:] - arr[x,yupup,:])


# ── 1. Build pressure Poisson RHS ──────────────────────────────────────────
rhs = div_vector(u, bounds)
rhs *= rho / dt
calculate_pressure_terms(u, rho, Pi_S, rhs, bounds)

p_rhs = rhs.copy()   # save before iteration

# ── 2. N_ITERS Jacobi sweeps ───────────────────────────────────────────────
p_work = p.copy()
p_target_rel_change = 1e-15   # tiny: won't trigger early exit in N_ITERS steps

for iteration in range(N_ITERS):
    p_aux = p_work.copy()
    relax_pressure_inner_loop(p_work, p_aux, rhs, bounds)
    apply_p_boundary_conditions_stub(p_work, p_aux, bounds)
    # convergence check (mirrors Python exactly; not used to stop here)
    sum_diff = np.sum(np.abs(p_aux - p_work))
    sum_old  = np.sum(p_aux)
    rel_change = sum_diff / (1e-7 + sum_old)

p_after_N = p_work.copy()

# ── 3. Compute dudt ────────────────────────────────────────────────────────
dudt = laplacian_vector(u, bounds, coeff=nu)     # viscous
upwind_advective_term(u, u, dudt, bounds, coeff=-1.0)  # convective
u_update_p_pi_terms(dudt, p_after_N, rho, Pi_S, Pi_A, bounds)

# ── 4. Dump fixtures ────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))

# Also save the inputs so the Rust test can load them
np.savetxt(os.path.join(out_dir, "stokes_u_input.txt"),    u.reshape(-1),        fmt="%.17g")
np.savetxt(os.path.join(out_dir, "stokes_Pi_S_input.txt"), Pi_S.reshape(-1),     fmt="%.17g")
np.savetxt(os.path.join(out_dir, "stokes_Pi_A_input.txt"), Pi_A.reshape(-1),     fmt="%.17g")
np.savetxt(os.path.join(out_dir, "p_rhs_ref.txt"),         p_rhs.reshape(-1),    fmt="%.17g")
np.savetxt(os.path.join(out_dir, "p_after_N_ref.txt"),     p_after_N.reshape(-1),fmt="%.17g")
np.savetxt(os.path.join(out_dir, "dudt_ref.txt"),          dudt.reshape(-1),     fmt="%.17g")

print(f"LX={LX}  LY={LY}  N_ITERS={N_ITERS}")
print(f"rho={rho}  dt={dt}  nu={nu:.6f}")
print(f"Interior cells: {len(bounds)}")
print(f"rhs    max abs: {np.max(np.abs(p_rhs)):.6e}")
print(f"p(N)   max abs: {np.max(np.abs(p_after_N)):.6e}")
print(f"dudt   max abs: {np.max(np.abs(dudt)):.6e}")
print("Rel change after last sweep:", rel_change)
print("Dumped: stokes_u_input.txt  stokes_Pi_S_input.txt  stokes_Pi_A_input.txt")
print("        p_rhs_ref.txt  p_after_N_ref.txt  dudt_ref.txt")
