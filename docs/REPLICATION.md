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

## Velocity boundary condition, checked and ruled out as a cause

`volterra-cgpo/src/step.rs:258` labels its call to `apply_u_boundary_conditions`
"no-slip". `flow-solver.py`'s own markdown (lines 533-547) derives a Lions
slip condition (zero normal velocity, zero tangential shear at the wall) and
contrasts it explicitly with true no-slip, which raised the possibility that
the comment was stale and the port applies the wrong physics at the wall.
Checked directly against `flow-solver.py:556-567`, `apply_u_boundary_conditions`:
the function computes the Lions-slip value on one line
(`u[x, y] = np.array([ny, -nx]) * (...) / (...)`) and then, on the very next
line, unconditionally overwrites it: `u[x, y, :] = 0`. The Lions-slip
derivation in the markdown is never wired into the code that runs: every
call leaves `u=0` at the boundary regardless of the Lions computation. This
is dead code in the reference script itself, the same shape as the seed
defect below.

`volterra-cgpo/src/bc.rs:96-98`'s own doc comment already states this
precisely ("Python code... first computes a Lions slip BC, then immediately
overwrites with `u[x,y,:] = 0`. The net effect is: set u=0..."), and the Rust
implementation skips the discarded Lions computation and sets `u=0` directly.
**The comment at `step.rs:258` is accurate, not stale, and the boundary
condition is not the cause of the golden/silver mismatch.** `volterra-cgpo`
matches `flow-solver.py`'s executed behaviour (plain no-slip) rather than its
markdown derivation (Lions slip); the two happen to coincide here because the
reference script never executes the derivation it documents.

## The reference script's initial condition is not reproducible from a seed

`flow-solver.py:1494` draws its per-run `seed` from the unseeded global
`np.random.rand()`, then seeds a generator with it (`:1524`). That seeded
generator is used exactly once, to set a single scalar broadcast to every
site (`theta_initial = pi * rng.random() * np.ones((Lx, Ly))`, `:1539`), and
the very next line discards it: `theta_initial = 1.0 * pi *
np.random.random((Lx, Ly))` (`:1541`) draws the actual per-site field from
the unseeded global generator. Neither the initial condition nor, therefore,
any specific trajectory the reference script produces is reproducible from
its own seed, including by the paper's own authors. This work's runs used a
fixed `CGPO_SEED` for internal reproducibility (so the dense re-sampling
re-run below is guaranteed to replay the same trajectory as the coarse run),
but this cannot be understood as reproducing "the" trajectory behind any
published figure: no such fixed trajectory is recoverable from the reference
code. What is reproducible, in the reference code and in this work, is the
statistical or topological outcome (a braid word, an entropy), never a
specific realisation.

A related, separate point: `flow-solver.py`'s own default run configuration
(`bc_label = 'epitrochoid'`, `:1533`; `Lx = Ly = 200`, `:1431-1432`) is the
epitrochoid confinement at 200x200, not the steady-winding circle at 100x100
the golden and silver results use. The script as checked out is not itself
configured for the paper's headline runs; its defaults should not be read as
those parameters.

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

| q | Steps | Frames | net_charge | als, ncl | Extracted word | Extracted entropy | Published word | Published entropy | Verdict |
|---|-------|--------|-----------|----------|-----------------|--------------------|-----------------|--------------------|---------|
| 1 (control) | 335,000 | 200 | 1.0 | 3.99, 0.975 | `{sigma_1^-1}` | 0.000000 | `{sigma_1}` | 0 | **pass**, up to an orientation sign (see below) |
| 3/2 (golden), coarse | 750,000 | 200 | 1.5 | 3.99, 0.975 | `{sigma_2^-1 sigma_1^-1 sigma_2 sigma_1}` repeated 16x (64 generators) | 15.398778 | `{sigma_2^-1 sigma_1}` | 0.96242 | **fail** |
| 3/2 (golden), dense re-run | 750,000 (same trajectory, same seed) | 1000 | 1.5 | 3.99, 0.975 | `{sigma_2^-1 sigma_1^-1 sigma_2 sigma_1}` repeated 16x (64 generators) | 15.398778 | `{sigma_2^-1 sigma_1}` | 0.96242 | **fail**, identical to the coarse run |
| 4/2 (silver), coarse | 500,000 | 200 | 2.0 | 3.99, 0.975 | 39-generator repeating sequence (see run log) | 11.804430 | `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}` | 1.76275 | **fail** |
| 4/2 (silver), dense re-run | 500,000 (same trajectory, same seed) | 1000 | 2.0 | 3.99, 0.975 | 39-generator repeating sequence (see run log) | 11.804430 | `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}` | 1.76275 | **fail**, identical to the coarse run |
| 5/2 (aperiodic) | 300,000 (reduced from the paper's 1,000,000 to fit the 30-minute compute budget) | 200 | 2.5 | 0.0266, 0.975 | trivial (16 defects, zero crossings) | 0.000000 | none (no stable word expected) | n/a | see below |

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

### golden and silver fail as measured, and the failure is not a sampling artefact

Both runs hold the correct defect count throughout the trailing window (3 for
golden, 4 for silver, matching n=2q), so the topology-fixing part of the
boundary condition is doing the right thing. Neither extracted word matches
the published braid, and neither extracted entropy is close to the published
value; the silver run's entropy (11.8) is roughly 6.7x the published 1.76,
and the golden run's word is a length-4 repeating block against the
published length-2 block. These are reported as failures, not adjusted
towards the published values.

**Tested and ruled out: temporal sampling resolution.** The initial
candidate explanation was that 200 frames spread across each run's entire
duration (including a pre-periodic transient) undersamples each orbital
period relative to the paper's own windowed analysis (frames 80-139 of a
purpose-selected window), so a defect pair passing close in x between two
samples could read as a spurious swap-and-unswap pair to the sort-based
extractor. Two further checks addressed this directly:

- **The velocity boundary condition was checked against the reference and
  ruled out separately** (see above): both `volterra-cgpo` and the reference
  script's actually-executed code apply plain no-slip, so this is not a
  source of the mismatch either.
- **Both runs were repeated at the same total step count with 5x denser
  saving** (golden: `CGPO_SAVE_EVERY` 3750 to 750, 200 to 1000 frames;
  silver: 2500 to 500, 200 to 1000 frames), same seed, same `lambda`, `als`,
  `ncl`, `dt`, boundary, and extraction routine and threshold. **Both dense
  re-runs reproduce the coarse run's extracted word and entropy exactly**,
  to six decimal places on the entropy and generator-for-generator on the
  word. Since the underlying trajectory is deterministic (fixed seed) and
  finer sampling changed nothing, the coarse run was not missing crossings
  between samples: there is nothing in the trajectory a denser sample would
  have caught that the coarse sample did not already catch. **The sampling
  hypothesis is ruled out.**

The discrepancy is therefore physical or in the new code, not a measurement
artefact. The remaining candidates, in the order they should be checked
next:

1. **The `A_sys ~= 88.6` normalisation was assumed to carry over from the
   q>=5/2 sweep to the golden/silver point**, since the paper does not
   restate it there; if the true value differs, `als` and `ncl` are wrong
   for this run, changing the activity and coherence length actually
   simulated.
2. **The new `circular_boundary` and variable-`net_charge` code has not
   been validated against a captured Python reference the way the nephroid
   path was** in `COMPARISON.md`. The q=1 pass is not sufficient evidence
   for this: it is the simplest possible case (two defects, one crossing
   type, no periodic-orbit structure), and a boundary or charge error could
   easily be invisible there while still corrupting the golden/silver
   dynamics. The highest-value next check is a field-by-field comparison of
   `circular_boundary` against `flow-solver.py`'s `'circular'` branch
   (`flow-solver.py:1205-1222`) on a single step, the same way the nephroid
   port was validated.
3. **A scheme difference** between this crate's finite-difference
   Beris-Edwards implementation and the published one, specific to this
   operating point, distinct from the general kernel-level concurrence
   already established for the nephroid configuration in `COMPARISON.md`.

None of these three was completed within this dispatch.

### q=5/2 aperiodic run shows a different failure mode, also reported as measured

Run reduced to 300,000 steps (from the paper's 1,000,000) to fit the
30-minute compute budget, after the unreduced attempt was measured at
roughly 270-310 steps/second (far slower than the ~1,500-1,800 steps/second
of the golden/silver runs, since the much smaller `als=0.0266` drives a
stiffer, more active system) and killed before completion.

The defect count does not stay near the topologically-required minimum of
n=2q=5: it climbs during the transient and stabilises at **16** defects for
essentially the entire post-transient window, not 5. Despite 16 defects
persisting, the extraction finds **zero crossings** across that whole
window: the braid word is trivial and the entropy is 0. This is a different
failure mode from "aperiodic, no stable word" (the paper's own description
of this regime): a frozen 16-defect configuration is neither the periodic
few-defect orbit of golden/silver nor the erratic many-swap motion the
paper describes for q>=5/2. The most likely reading is that `als=0.0266`
(derived from the paper's stated dimensionless active length 0.0003 under
the same `A_sys` assumption used for golden and silver, itself unconfirmed)
drives activity far past the regime the paper examines, nucleating many more
defect pairs than the boundary requires and jamming them in place, rather
than reproducing the paper's own aperiodic-but-still-five-to-ten-defect
dynamics. This is reported as the measured outcome, not adjusted towards the
paper's qualitative description.

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

The dense re-run (the one reported above as the definitive golden/silver
result):

```bash
cd volterra
cargo build --release -p volterra-cgpo --bin cgpo_fd
CGPO_LX=100 CGPO_BOUNDARY=circular CGPO_NET_CHARGE=1.5 CGPO_ALS=3.99 \
  CGPO_NCL=0.975 CGPO_LAMBDA=1.0 CGPO_MAX_P_ITERS=50 CGPO_MAX_STEPS=750000 \
  CGPO_SAVE_EVERY=750 CGPO_OUT=/tmp/cgpo-golden-dense CGPO_SEED=0 \
  ./target/release/cgpo_fd
python3 extract_braid.py /tmp/cgpo-golden-dense/als_3.99_ncl_0.975 100 3
```

Swap `CGPO_NET_CHARGE=2.0`, `CGPO_MAX_STEPS=500000`, `CGPO_SAVE_EVERY=500`
for silver. `extract_braid.py` lives outside this repository during this
dispatch; see the accompanying report for its path. See `docs/SUBSUMPTION.md`
for how this boundary condition and its validation status are recorded in
the coverage matrix.
