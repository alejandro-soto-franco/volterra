# Numerical concurrence: volterra DEC solver vs CGPO flow-solver

Compares the two solvers at the paper's known-optimal phase points
(arXiv:2503.10880):
**cardioid → golden** `(ℓ̃a,ℓ̃c)=(0.0139,0.0903)`, 200×200; **nephroid → silver**
`(0.0128,0.0766)`, 100×100; both 1.5×10⁶ steps.

Concurrence is defined on two regimes (the system is chaotic, so fields diverge
pointwise): **short-time pointwise** (matched IC) and **long-time
statistical/topological**.

## Status

| Piece | State |
|-------|-------|
| SP0: paper-concordant Burau (SI.11–SI.17) | **landed** (`volterra-braid` entropy.rs; `paper_burau_*`) |
| Braid word + entropy concurrence vs braid_tracker.py | **landed earlier** (`oracle/compare_cgpo.py`) |
| SP3: dynamical observables (⟨ω⟩ + gyre count, line-stretching entropy) | **landed** (`observables.py`, self-test passes) |
| SP2: CGPO reference runner (single point, persist Q+u, dump IC, no rm) | **landed** (`patch_flow_solver.py` → `flow_solver_run.py`); runs at 1.93 ms a step at L=100, 8.2 at L=200 |
| SP1: volterra confined DEC runner (no-slip Stokes + Dirichlet Q) | **landed** (`volterra.ConfinedRun`, clamped wall); 1.18 ms a step on 1435 vertices |
| Topological concurrence at the published point | **landed** (`compare_solvers.py`); complement and net charge agree, braid window still short on the lattice |
| SP3: DEC↔Cartesian interpolation + matched-IC pointwise norms | **blocked on SP1/SP2 output** |
| SP4: viscometric concurrence | **deferred** |

Machine-checking of the paper's *analytical* claims (braid algebra, Burau,
metallic identities, Beris-Edwards functional derivative, free-energy sign) is a
separate, **complete** deliverable in the private `cgpo-review` repo (SymPy +
Cadabra2 + Lean), which surfaced two manuscript-level findings.

## Topological concurrence, measured

Nephroid at the published pair `(l_a, l_c) = (0.0128, 0.0766)`, which is
`als = 2`, `ncl = 11.7` at `Lx = 200`. Both walls impose a charge of `+1`: the
lattice mask at `d = 0.99`, and the mesh at `d = 0.9`. Same constants, same
`dt = 1e-4`, same 500-step cadence, 120000 steps, 240 frames.

| | complement | net charge | braid entropy |
|---|---|---|---|
| reference lattice | `(4, 2)` in 212 frames | `+1` | 0.000000 |
| conforming mesh | `(4, 2)` in 216 frames | `+1` | 1.762747, silver |

The complement and the charge agree. The braid does not yet: the lattice's four
positive cores have not begun to exchange by `t = 12`, where the mesh reaches
the silver braid by `t = 1.65`. Closing that needs the published 1.5e6 steps,
about 3.4 hours on the lattice.

Read the reference's frames with `braid_detect_defects_winding`. The other
detector thresholds the saddle-splay density rather than an angle, and an
angle-sized threshold returns nothing at all on a settled field.

## What runs now

```bash
# Dynamical observables, self-test on synthetic flows (no solver needed):
uv run observables.py            # double-gyre -> 2 gyres; extensional -> h=lambda; rotation -> h=0

# Generate the CGPO reference runner, then a short smoke (validates Q+u+IC output):
uv run patch_flow_solver.py
FD_LX=60 FD_LY=60 FD_MAX_STEPS=2000 uv run --with numpy --with numba flow_solver_run.py
```

## Remaining critical path (the field-level concurrence)

1. **SP1 confined DEC runner**, which the other two wait on. Highest risk: no-slip DEC
   Stokes on a bounded flat domain (pressure Neumann SI.5), validated against an
   analytic confined Stokes flow before wiring the runner. Then Dirichlet-Q
   anchoring (SI.2/SI.3) + the epitrochoid mesh, producing Q/u snapshots.
2. **SP2 run**. Fix the integer `(als,ncl)` map from `(ℓ̃a,ℓ̃c)` + resolution +
   domain area, then the full 1.5×10⁶-step golden/nephroid runs (archive to
   ASF-EX2).
3. **SP3 full harness**. DEC↔Cartesian interpolation; matched-IC short-time
   `‖Q_v−Q_cgpo‖` + divergence time `t*`; convergence order; then the long-time
   observables above on both solvers' output.

See the spec for equations, parameters, and the metric ladder.
