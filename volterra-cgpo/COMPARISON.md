# volterra-cgpo vs fsn_solver (flow-solver.py): runtime + numerical concurrence

**Date:** 2026-06-13. Confined nephroid (epitrochoid q=2/k=2) active Beris–Edwards +
relaxation-Stokes. `volterra-cgpo` is a faithful Rust port of the CGPO finite-difference
solver `~/Chaos-Generating-Periodic-Orbits/flow-solver.py` (Brandon et al.). Config for all
numbers below: **Lx=Ly=100, als=1, ncl=9 (paper silver point), dt=1e-4, max_p_iters=50**,
identical LdG/activity constants (K=2¹⁴, γ=100, η=√(10K), ρ=1, χ=1, λ=0.7, S₀=√2).

## 1. Numerical concurrence — the Rust port IS the same scheme

Per-kernel, element-wise vs the numba kernels on fixed inputs (the backbone test — if every
kernel matches to rounding, the discretizations are identical):

| Kernel | max abs diff vs Python |
|--------|------------------------|
| FD `Laplacian` (analytic) | rel err 6.6e-4 (O(dx²)) |
| FD `div_vector` (linear field) | 0 (exact) |
| FD `upwind_advective_term` | 1e-12 (element-wise vs numba) |
| `H_S_from_Q` → H | **0 (bit-identical)** |
| scalar order S | 1.7e-18 |
| stress Π_S | 4.5e-13 |
| stress Π_A | **0 (bit-identical)** |
| Stokes pressure (N sweeps) | < 1e-9 |
| velocity update dudt | < 1e-9 |

**Capstone — one full `update_step` from a fixed IC (with identical boundary normals on
both sides):**
- **max\|ΔQ\| = 1.1e-16** (machine epsilon)
- max\|Δu\| = 2.4e-15
- max\|Δp\| = 2.2e-11

The two solvers agree to machine precision per timestep. Numerical concurrence is therefore
**proven at the algorithm level** — not inferred from a chaotic long run. (Over a long run
the two diverge eventually, as any chaotic system must; that is expected and not a
discrepancy. A matched-IC run is available via `CGPO_THETA_IC`.)

Caveat: the epitrochoid boundary-normal inverse solve differs between Python (`scipy.fsolve`)
and Rust (scan + Newton) by ~1e-12 at the ~2 near-degenerate cells where a normal component
≈0; the sign flip there changes one Neumann-pressure stencil. For the kernel/one-step
concurrence both sides use identical (Rust-dumped) normals; in production each uses its own
(both are valid epitrochoid normals to rounding).

## 2. Runtime — single-threaded Rust vs 32-core numba

20000 steps, minimal I/O, 32-core machine:

| Solver | threading | steps/sec | note |
|--------|-----------|-----------|------|
| **volterra-cgpo (Rust)** | **single-thread** | **2300** | 20000 steps in 8.70 s |
| flow-solver.py (numba) | `parallel=True`, 32 cores | 504 | compute-only: 19800 steps in (77.94−38.68 compile)=39.26 s |

**Speedup ≈ 4.6× wall-clock**, and the Rust figure is **single-threaded** against a
32-core-parallel numba — so the per-core advantage is large. Both solvers are bottlenecked
by the serial pressure-relaxation iteration (which `parallel=True` can't parallelize across
sweeps), which is why numba's 32 cores don't run away. Parallelizing the Rust kernels with
rayon (the per-sweep relaxation, the stress/H kernels) is open headroom to widen the gap.
Rust also has no JIT warm-up (numba pays ~39 s compile per process).

## 3. Status
- **Phase 1 (this): Rust FD port — COMPLETE & VALIDATED** (19 tests; machine-epsilon
  per-step concurrence; 4.6× runtime). Crate `volterra-cgpo`, runner `bin/cgpo_fd`
  (env-configurable, writes flow-solver-format Q/u snapshots → reuses `physical/concurrence.py`).
- **Phase 2 (deferred): volterra DEC-native run** — the gated no-slip DEC Stokes on the
  epitrochoid mesh; a different discretization, for statistical/topological concurrence.

## Reproduce
```bash
cd ~/volterra && cargo test -p volterra-cgpo            # concurrence tests (19)
cargo build --release -p volterra-cgpo --bin cgpo_fd
CGPO_LX=100 CGPO_ALS=1 CGPO_NCL=9 CGPO_MAX_P_ITERS=50 \
  CGPO_MAX_STEPS=20000 CGPO_SAVE_EVERY=20000 CGPO_OUT=/tmp/r ./target/release/cgpo_fd  # Rust timing
# Python baseline: flow_solver_nephroid.py on ASF-EX1 venv, same env knobs.
```
