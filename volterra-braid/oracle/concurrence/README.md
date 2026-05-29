# Numerical concurrence: volterra DEC solver vs CGPO flow-solver

Implements the comparison specified in
`docs/superpowers/specs/2026-05-29-numerical-concurrence-volterra-cgpo-design.md`,
at the paper's known-optimal phase points (arXiv:2503.10880):
**cardioid → golden** `(ℓ̃a,ℓ̃c)=(0.0139,0.0903)`, 200×200; **nephroid → silver**
`(0.0128,0.0766)`, 100×100; both 1.5×10⁶ steps.

Concurrence is defined on two regimes (the system is chaotic, so fields diverge
pointwise): **short-time pointwise** (matched IC) and **long-time
statistical/topological**.

## Status

| Piece | State |
|-------|-------|
| SP0 — paper-concordant Burau (SI.11–SI.17) | **landed** (`volterra-braid` entropy.rs; `paper_burau_*`) |
| Braid word + entropy concurrence vs braid_tracker.py | **landed earlier** (`oracle/compare_cgpo.py`) |
| SP3 — dynamical observables (⟨ω⟩ + gyre count, line-stretching entropy) | **landed** (`observables.py`, self-test passes) |
| SP2 — CGPO reference runner (single point, persist Q+u, dump IC, no rm) | **scaffolded** (`patch_flow_solver.py` → `flow_solver_run.py`); full run is hours, deferred |
| SP1 — volterra confined DEC runner (no-slip Stokes + Dirichlet Q) | **gated** (research-grade no-slip DEC Stokes; see spec §SP1) |
| SP3 — DEC↔Cartesian interpolation + matched-IC pointwise norms | **gated on SP1/SP2 output** |
| SP4 — viscometric concurrence | **deferred** |

Machine-checking of the paper's *analytical* claims (braid algebra, Burau,
metallic identities, Beris-Edwards functional derivative, free-energy sign) is a
separate, **complete** deliverable in the private `cgpo-review` repo (SymPy +
Cadabra2 + Lean), which surfaced two manuscript-level findings.

## What runs now

```bash
# Dynamical observables, self-test on synthetic flows (no solver needed):
uv run observables.py            # double-gyre -> 2 gyres; extensional -> h=lambda; rotation -> h=0

# Generate the CGPO reference runner, then a short smoke (validates Q+u+IC output):
uv run patch_flow_solver.py
CGPO_LX=60 CGPO_LY=60 CGPO_MAX_STEPS=2000 uv run --with numpy --with numba flow_solver_run.py
```

## Remaining critical path (the field-level concurrence)

1. **SP1 confined DEC runner** — the gating piece. Highest risk: no-slip DEC
   Stokes on a bounded flat domain (pressure Neumann SI.5), validated against an
   analytic confined Stokes flow before wiring the runner. Then Dirichlet-Q
   anchoring (SI.2/SI.3) + the epitrochoid mesh, producing Q/u snapshots.
2. **SP2 run** — fix the integer `(als,ncl)` map from `(ℓ̃a,ℓ̃c)` + resolution +
   domain area, then the full 1.5×10⁶-step golden/nephroid runs (archive to
   ASF-EX2).
3. **SP3 full harness** — DEC↔Cartesian interpolation; matched-IC short-time
   `‖Q_v−Q_cgpo‖` + divergence time `t*`; convergence order; then the long-time
   observables above on both solvers' output.

See the spec for equations, parameters, and the metric ladder.
