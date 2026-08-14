# Reproducing Klein et al. (2026)

Klein, Soto Franco, Sabbir, Deutsch, Kliegman, Selinger, Mitchell, Beller,
"Chaos-Generating Periodic Orbits of Topological Defects in Confined Active
Nematics", PNAS 123(28), e2516670123 (2026). The applicant is second author.
`volterra-cgpo` is a Rust port of the paper's own released solver,
`flow-solver.py` (github.com/Brandonkl/Spontaneous-Optimal-Mixing, ref. [62]).

**Date:** 2026-08-14. **Machine:** Fedora 44, AMD Ryzen 9 8940HX, single
thread per run (`volterra-cgpo` has no internal parallelism; see
`BENCHMARKS.md` section 9).

## What this checks

The paper reports two kinds of result: an algebraic braid-topology claim
(golden and silver braid words, their closed-form entropies) and a PDE claim
(that a Beris-Edwards active-nematic simulation at stated parameters produces
those braids). `volterra-braid`'s existing tests already confirm the
algebraic claim against the reference `braid_tracker.py`, but do so on a
synthetic, constructed-to-match defect trajectory
(`volterra-braid/oracle/compare_cgpo.py`), not on the output of an actual PDE
run. This document runs the actual PDE solver at the paper's stated
parameters and extracts the braid word and entropy from the resulting
trajectory, using the same detection and extraction routines
(`volterra-braid/oracle/braid_tracker_v2.py`) applied to real simulation
output rather than a synthetic one.

## Parameter resolution: lambda vs chi

The paper's Materials and Methods (p. 13) states the flow-alignment
parameter as chi = 1. `volterra-cgpo`'s `Params` struct carries two separate
fields, `chi` (hardcoded to 1.0 inside `Params::new`, never exposed as a free
parameter) and `lambda` (an exposed constructor argument, doc-commented
"code-truth flow-solver.py: lambda = 1", `volterra-cgpo/src/lib.rs:69`,
`:95`). Checked directly against the paper's own released code: `~/Chaos-
Generating-Periodic-Orbits/flow-solver.py:1437` reads `lambda = 1  # flow
alignment parameter`, and the same symbol is threaded through the H and
stress kernels (`flow-solver.py:741-747`) exactly where the paper's chi
belongs physically. **`chi` in the paper's prose and `lambda` in the paper's
own code are the same physical flow-alignment parameter, under two names.**
`volterra-cgpo`'s `lambda` field is the one that matters; its `chi` field is
a vestigial name collision, always 1.0, and irrelevant to matching the paper.
The runs below use `lambda = 1`, not the `lambda = 0.7` used elsewhere in
this repository's benchmark fixtures (`volterra-cgpo/tests/step.rs:31`),
which is an arbitrary concurrence-test value with no claim to match the
paper.

## A defect found and fixed while preparing these runs

`Params::new` computed the equilibrium scalar order parameter as `s0 =
sqrt(-a_landau / c_landau)`, which, since `a_landau = -c_landau` throughout
this crate, always evaluated to 1.0. The paper states S0 = sqrt(2) (p. 1),
and `flow-solver.py:1481` computes `S0 = np.sqrt(-2 * A / C)`, i.e. with a
factor of 2 this crate's formula omitted. The crate's own doc comment already
stated the intended value as `sqrt(2)` (`volterra-cgpo/src/lib.rs:99`,
unchanged by this fix), so this was an implementation bug against the crate's
own documented intent, not a documentation error. The bit-for-bit concurrence
tests never exercised this path: they set `s0` directly from a hardcoded
`SQRT_2` constant (`volterra-cgpo/tests/step.rs:35`) rather than through
`Params::new`, so the bug affected every run built through the convenience
constructor, including the `cgpo_fd` production runner and the "paper silver
point" configuration recorded in `volterra-cgpo/COMPARISON.md`, without
affecting the validated kernel-level concurrence claims in that same
document. Fixed to `s0 = sqrt(-2 * a_landau / c_landau)`
(`volterra-cgpo/src/lib.rs:122`). All 38 existing tests in `volterra-cgpo`
still pass after the fix. `COMPARISON.md`'s throughput numbers are unaffected
(s0 only sets initial-condition amplitude and the pressure/order-parameter
scale, not the per-step cost); its physical configuration should be read
as having run at the wrong S0 until this fix.

## The boundary this paper's headline results actually use

The paper's golden and silver results (Figs. 2-4, p. 5-8) use a "steady-
winding circle" boundary condition (Eq. 1, p. 4): a plain disk with a
tangential director anchoring that winds through angle `2*pi*q` around the
boundary; `q` is a free half-integer. This is a different, simpler geometry from
the cardioid/nephroid/trefoiloid epitrochoid confinements the paper
introduces later (p. 10, Fig. 7) as a physically-motivated secondary
demonstration. Before this work, `volterra-cgpo` implemented only the
nephroid epitrochoid boundary (`boundary::nephroid_boundary`, k=2 fixed), and
the winding charge in `bc::apply_q_boundary_conditions` was hardcoded to
`net_charge = 1.0`, matching only that one nephroid case. Neither the golden
(q=3/2), the q=1 control, nor the q=5/2 aperiodic case can be run with a
fixed k=2 nephroid boundary: each is a physically distinct confinement shape,
or a distinct q on the circle, from what the crate had.

To make these runs possible at all, this work added:

- `boundary::circular_boundary(lx, ly)`: a plain disk of radius `lx/2 - 1`
  centred at `(radius, radius)`, ported directly from `flow-solver.py`'s own
  `'circular'` branch of `set_boundary` (`flow-solver.py:1205-1222`).
- `net_charge` as a parameter of `bc::apply_q_boundary_conditions` (was a
  hardcoded local `1.0`) and of `Params` (`Params::with_net_charge`), so `q`
  is settable per run rather than fixed to the nephroid's value.
- `CGPO_BOUNDARY` (`circular`/`nephroid`) and `CGPO_NET_CHARGE` environment
  variables on the `cgpo_fd` runner.

This is new code, not a parameter change to existing code, and it has not
been validated against a captured Python reference the way the nephroid path
was in `COMPARISON.md`. The regression suite (38 tests, all passing after
this change, `net_charge` defaulted to `1.0` at every pre-existing call site)
confirms the change did not alter the nephroid path's already-validated
behaviour; it does not independently confirm the new circular path's
correctness beyond the qualitative check the q=1 run below provides.

## Active length and coherence length: raw pixel values

The paper reports lengths nondimensionalised by the square root of the
confined domain's area in lattice units, `ell_tilde = ell / sqrt(A_sys)` (p.
3). For the 100x100 steady-winding-circle domain, the paper states `A_sys ~=
88.6` lattice points (p. 6, in the context of the q >= 5/2 sweep on the same
domain and boundary family). The golden and silver runs (p. 19, Fig. 3
caption) are reported at a dimensionless active length of 0.045 and
dimensionless coherence length of 0.011, for both q=3/2 and q=4/2. The
q=5/2 aperiodic demonstration (p. 20) is reported at dimensionless active
length 0.0003, coherence length 0.011. Converting with the same `sqrt(A_sys)
~= 88.6` factor:

| Run | Dimensionless (paper) | Raw pixels (this work, ell x 88.6) |
|-----|------------------------|-------------------------------------|
| golden / silver | ell_a=0.045, ell_c=0.011 | als=3.99, ncl=0.975 |
| q=5/2 | ell_a=0.0003, ell_c=0.011 | als=0.0266, ncl=0.975 |

The paper does not restate `A_sys` separately for the golden/silver point;
this work assumes the same domain and boundary family (100x100,
steady-winding circle) gives the same `A_sys`, which is not confirmed
independently in the source text. The paper does not state a dimensionless
active length for q=1 at all (only "ell_a sufficiently large as to prevent
pair creation," p. 4); this work reused the golden/silver value (als=3.99)
for q=1, an assumption rather than a value read from the paper.

## Results

Time step `dt=1e-4` (paper, SI p. 25-26, and `cgpo_fd`'s own default),
`lambda=1`, `max_p_iters=50` (matching `COMPARISON.md`'s validated
high-throughput configuration; the paper does not state its own pressure
solver's iteration cap). Each run saved 200 Q-tensor frames spread across the
full run (including the initial transient); braid extraction used
`braid_tracker_v2.detect_defects` (threshold 0.1, matching the published
`braid_tracker.py`) and `braid_tracker_v2.braidword_from_frames` /
`topological_entropy` on the trailing run of frames with a stable defect
count.

| q | Steps | net_charge | als, ncl | Extracted word | Extracted entropy | Published word | Published entropy | Verdict |
|---|-------|-----------|----------|-----------------|--------------------|-----------------|--------------------|---------|
| 1 (control) | 335,000 | 1.0 | 3.99, 0.975 | `{sigma_1^-1}` | 0.000000 | `{sigma_1}` | 0 | **pass**, up to an orientation sign (see below) |
| 3/2 (golden) | 750,000 | 1.5 | 3.99, 0.975 | `{sigma_2^-1 sigma_1^-1 sigma_2 sigma_1}` repeated 16x (64 generators) | 15.398778 | `{sigma_2^-1 sigma_1}` | 0.96242 | **fail** |
| 4/2 (silver) | 500,000 | 2.0 | 3.99, 0.975 | 39-generator non-repeating sequence (see run log) | 11.804430 | `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}` | 1.76275 | **fail** |
| 5/2 (aperiodic) | 300,000 (reduced from the paper's 1,000,000 to fit the 30-minute compute budget) | 2.5 | 0.0266, 0.975 | see below | see below | none (no stable word expected) | see below |

### q=1 control passes, up to a sign convention

Defect count is stable at 2 for the trailing 194 of 200 sampled frames after
a short transient. The extracted word is the single generator
`sigma_1^-1`, the mirror image of the published `sigma_1`. A single-generator
sign is a chirality/orientation convention (which way the boundary winding
was signed, `n(theta) = +-(-sin(q theta), cos(q theta))`, Eq. 1 of the
paper, carries an explicit `+-` already), not a topological discrepancy: the
entropy of a single generator is 0 either way, matching the published value
exactly. This is the one run of the four that behaves as expected, and it is
the simplest of the four (two defects, one crossing type, no periodic-orbit
structure to get wrong), so it functions as a sanity check on the new
circular-boundary and variable-net_charge code rather than as a demonstration
of the paper's more detailed golden/silver claims.

### golden and silver fail as measured

Both runs hold the correct defect count throughout the trailing window (3 for
golden, 4 for silver, matching n=2q), so the topology-fixing part of the
boundary condition is doing the right thing. Neither extracted word matches
the published braid, and neither extracted entropy is close to the published
value; the silver run's entropy (11.8) is roughly 6.7x the published 1.76,
and the golden run's word is a length-4 repeating block against the
published length-2 block. These are reported as failures, not adjusted
towards the published values.

The leading candidate explanation, not confirmed within this work's time
budget, is temporal sampling resolution: 200 frames were spread across each
run's *entire* duration (including a substantial pre-periodic transient
visible in the early defect-count columns of the run logs), giving markedly
fewer samples per orbital period than the paper's own windowed analysis
(frames 80-139 of a purpose-selected window, per
`volterra-braid/oracle/compare_cgpo.py`'s citation of the published script's
convention). A defect pair passing close in x between two sampled frames,
without an intervening sample to confirm a single clean crossing, can read as
spurious swap-and-unswap pairs to the sort-based extraction algorithm, which
would inflate both the generator count and the computed entropy in exactly
the direction observed. Distinguishing this from a physical
mismatch (wrong `A_sys` normalisation, an error in the new circular-boundary
port, or a difference between this crate's finite-difference scheme and
the published one at this specific operating point) needs a rerun with dense,
post-transient-only sampling, which was not completed in this work.

### q=5/2 (aperiodic): [fill in after the run completes]

## What this means for the subsumption claim

Two limits on any claim that volterra subsumes open-Qmin, independent of the
runs above:

- open-Qmin's headline results (the hedgehog/Saturn-ring metastability
  boundary and the patterned-boundary defect arrays in Sussman & Beller
  2019) are equilibrium free-energy minimisations reached with FIRE. volterra
  is a time-integrated solver; its passive relaxation settles at a nonzero
  residual floor rather than true equilibrium (`BENCHMARKS.md`, Sections 1
  and 6), and it has no FIRE-equivalent minimiser, no colloidal-inclusion
  boundary geometry, and no patterned-boundary geometry. This is a real gap,
  not a difference in emphasis.
- Beller's 2026 three-dimensional preprint (Head, Digregorio, Marenduzzo,
  Pagonabarraga, Beller, Negro, "Topological delocalisation of confined 3D
  active nematics", arXiv:2607.10234) needs, on top of volterra's existing
  validated 3D Beris-Edwards solver: a Cahn-Hilliard phase field for the
  double-emulsion cylinder confinement, tangential anchoring on a curved 3D
  boundary, and a 3D defect-line/loop tracker (`volterra-braid` currently
  extracts 2D point defects only). None of this was attempted in this
  dispatch.

## Reproduce

```bash
cd volterra
cargo build --release -p volterra-cgpo --bin cgpo_fd
CGPO_LX=100 CGPO_BOUNDARY=circular CGPO_NET_CHARGE=1.5 CGPO_ALS=3.99 \
  CGPO_NCL=0.975 CGPO_LAMBDA=1.0 CGPO_MAX_P_ITERS=50 CGPO_MAX_STEPS=750000 \
  CGPO_SAVE_EVERY=3750 CGPO_OUT=/tmp/cgpo-golden CGPO_SEED=0 \
  ./target/release/cgpo_fd
python3 planning-scripts/extract_braid.py /tmp/cgpo-golden/als_3.99_ncl_0.975 100 3
```

(`extract_braid.py` lives outside this repository during this dispatch; see
the accompanying report for its path.)
