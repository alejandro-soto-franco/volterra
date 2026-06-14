"""
Reference dumper for nematic kernel tests.

Builds a fixed pseudo-random Q and u field on a 24x24 grid (seed 0),
uses a simple rectangular interior (1-cell border excluded), calls the
Python/numba functions, and writes H, S, Pi_S, Pi_A to .txt files.

Constants:
    K     = 2**14
    gamma = 100       (not used for H/Pi, just for context)
    lam   = 0.7       (lambda / flow-alignment)
    zeta  = K / 1.0
    A     = -K / 81.0
    C     =  K / 81.0

Field layout (Python side): arr[x, y, c] with C-order numpy arrays.
Output files: one value per line, row-major over (x, y, c) — matches
Rust's flat index (x*ly + y)*2 + c.

Pi_A is scalar [x, y]: output row-major over (x, y).
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# numba-free pure-numpy implementations of the four kernels
# (identical arithmetic to the numba versions, but without JIT compilation
#  so there is no fastmath floating-point reassociation)
# ---------------------------------------------------------------------------

LX, LY = 24, 24

rng = np.random.default_rng(0)

# Q field: traceless symmetric 2D tensor → 2 independent components Q0=Qxx, Q1=Qxy
# Keep magnitudes physically plausible: O(1) relative to S0=sqrt(2)~1.41
Q = rng.uniform(-0.5, 0.5, (LX, LY, 2))

# u (velocity) field
u = rng.uniform(-0.1, 0.1, (LX, LY, 2))

# Constants
K    = float(2**14)
lam  = 0.7
zeta = K / 1.0
A    = -K / 81.0
C    =  K / 81.0

# Interior mask: all cells except the 1-cell border
interior = []
for x in range(LX):
    for y in range(LY):
        if 1 <= x <= LX - 2 and 1 <= y <= LY - 2:
            interior.append((x, y))
bounds = np.array(interior, dtype=np.int32)   # shape (N, 2)


# ---------------------------------------------------------------------------
# Laplacian_vector  (9-point isotropic, matching Python)
# ---------------------------------------------------------------------------
def laplacian_vector(arr, bounds, coeff=1.0):
    Lx, Ly = arr.shape[:2]
    out = np.zeros_like(arr)
    c = coeff / 6.0
    for (x, y) in bounds:
        xup = (x + 1) % Lx
        xdn = x - 1          # Python wrap: -1 → Lx-1
        yup = (y + 1) % Ly
        ydn = y - 1
        out[x, y, :] = c * (
            -20 * arr[x,   y,   :]
            + 4 * (arr[xup, y,   :]
                 + arr[xdn, y,   :]
                 + arr[x,   yup, :]
                 + arr[x,   ydn, :])
            + arr[xup, yup, :]
            + arr[xup, ydn, :]
            + arr[xdn, yup, :]
            + arr[xdn, ydn, :]
        )
    return out


# ---------------------------------------------------------------------------
# H_S_from_Q
# ---------------------------------------------------------------------------
def H_S_from_Q(u, Q, A, C, K, lam, bounds):
    Lx, Ly = u.shape[:2]
    H = laplacian_vector(Q, bounds, coeff=K)
    S = np.zeros_like(Q)
    for (x, y) in bounds:
        xup = (x + 1) % Lx
        xdn = x - 1
        yup = (y + 1) % Ly
        ydn = y - 1

        q0, q1 = Q[x, y, 0], Q[x, y, 1]
        trqsq = 2 * (q0**2 + q1**2)

        H[x, y, :] -= (A + C * trqsq) * Q[x, y, :]

        dxux = 0.5 * (u[xup, y, 0] - u[xdn, y, 0])
        dxuy = 0.5 * (u[xup, y, 1] - u[xdn, y, 1])
        dyux = 0.5 * (u[x, yup, 0] - u[x, ydn, 0])

        omega_xy = 0.5 * (dxuy - dyux)
        lambda_s = lam * np.sqrt(2 * trqsq)
        TrQE = 2 * q0 * dxux + q1 * (dyux + dxuy)

        S[x, y, 0] = lambda_s * dxux - 2 * omega_xy * q1 - 2 * TrQE * q0
        S[x, y, 1] = lambda_s * 0.5 * (dxuy + dyux) + 2 * omega_xy * q0 - 2 * TrQE * q1

    return H, S


# ---------------------------------------------------------------------------
# get_Erickson_stress
# ---------------------------------------------------------------------------
def get_Erickson_stress(Q, K, Pi_S, bounds):
    Lx, Ly = Q.shape[:2]
    for (x, y) in bounds:
        xup = (x + 1) % Lx
        xdn = x - 1
        yup = (y + 1) % Ly
        ydn = y - 1
        dxQxx, dxQxy = 0.5 * (Q[xup, y] - Q[xdn, y])
        dyQxx, dyQxy = 0.5 * (Q[x, yup] - Q[x, ydn])
        Pi_S[x, y, 0] -= K * (dxQxx**2 + dxQxy**2 - dyQxx**2 - dyQxy**2)
        Pi_S[x, y, 1] -= 2 * K * (dxQxy * dyQxy + dxQxx * dyQxx)


# ---------------------------------------------------------------------------
# get_TrQH_term  (no boundary mask — applied globally)
# ---------------------------------------------------------------------------
def get_TrQH_term(Q, H, Pi_S):
    TrQH = 2 * (Q[:, :, 0] * H[:, :, 0] + Q[:, :, 1] * H[:, :, 1])
    Pi_S[:, :, 0] += TrQH * Q[:, :, 0]
    Pi_S[:, :, 1] += TrQH * Q[:, :, 1]


# ---------------------------------------------------------------------------
# calculate_Pi
# ---------------------------------------------------------------------------
def calculate_Pi(H, Q, lam, zeta, K, bounds):
    Pi_S = -lam * H - zeta * Q
    get_Erickson_stress(Q, K, Pi_S, bounds)
    get_TrQH_term(Q, H, Pi_S)
    Pi_A = 2 * (Q[:, :, 0] * H[:, :, 1] - H[:, :, 0] * Q[:, :, 1])
    return Pi_S, Pi_A


# ---------------------------------------------------------------------------
# Run and dump
# ---------------------------------------------------------------------------
H, S = H_S_from_Q(u, Q, A, C, K, lam, bounds)
Pi_S, Pi_A = calculate_Pi(H, Q, lam, zeta, K, bounds)

out_dir = os.path.dirname(os.path.abspath(__file__))

# Save Q and u inputs so the Rust test can reproduce the same fields
np.savetxt(os.path.join(out_dir, "Q_input.txt"),   Q.reshape(-1),   fmt="%.17g")
np.savetxt(os.path.join(out_dir, "u_input.txt"),   u.reshape(-1),   fmt="%.17g")
np.savetxt(os.path.join(out_dir, "H_ref.txt"),     H.reshape(-1),   fmt="%.17g")
np.savetxt(os.path.join(out_dir, "S_ref.txt"),     S.reshape(-1),   fmt="%.17g")
np.savetxt(os.path.join(out_dir, "Pi_S_ref.txt"),  Pi_S.reshape(-1),fmt="%.17g")
np.savetxt(os.path.join(out_dir, "Pi_A_ref.txt"),  Pi_A.reshape(-1),fmt="%.17g")

print(f"LX={LX} LY={LY}")
print(f"K={K}  lambda={lam}  zeta={zeta}  A={A}  C={C}")
print(f"Interior cells: {len(bounds)}")
print(f"H   max abs: {np.max(np.abs(H)):.6e}")
print(f"S   max abs: {np.max(np.abs(S)):.6e}")
print(f"Pi_S max abs: {np.max(np.abs(Pi_S)):.6e}")
print(f"Pi_A max abs: {np.max(np.abs(Pi_A)):.6e}")
print("Dumped: Q_input.txt  u_input.txt  H_ref.txt  S_ref.txt  Pi_S_ref.txt  Pi_A_ref.txt")
