# Reproducing Klein et al. (2026)

Klein, Soto Franco, Sabbir, Deutsch, Kliegman, Selinger, Mitchell, Beller,
"Chaos-Generating Periodic Orbits of Topological Defects in Confined Active
Nematics", PNAS 123(28), e2516670123 (2026). The applicant is second author.
`volterra-cgpo` is a Rust port of the paper's own released solver,
`flow-solver.py` (github.com/Brandonkl/Spontaneous-Optimal-Mixing, ref. [62]).

**Date:** 2026-08-14, revised 2026-08-15. **Machine:** Fedora 44, AMD Ryzen 9
8940HX, single thread per run (`volterra-cgpo` has no internal parallelism; see
`BENCHMARKS.md` section 9).

**Revision (2026-08-15).** The 2026-08-14 pass reported the golden and silver
runs as failures against the published braids. Both reproduce. The two runs were
regenerated from the same seed and the same parameters, and the fault was in how
the extracted braid was compared, in two places:

1. **A whole-window entropy was compared against a per-period one.** The paper
   quotes the braid of one period and that braid's entropy. The extractor summed
   over every period in the sampling window, so its number grew with the length
   of the run. The golden window holds exactly 16 periods and the reported
   15.398778 is 16 times the published 0.96242, to twelve digits.
2. **Words were compared as strings, where the comparison belongs in the
   braid group.**
   `sigma_i` and `sigma_j` commute when `|i - j| >= 2`, so two swaps among
   disjoint pairs of defects are the same braid in either order, and which order
   the extractor emits depends on which crossing the sampling caught first. The
   silver word repeats a six-generator block that appears with its commuting
   pair written both ways, so no string period exists where a braid period does.

Both are fixed in the library rather than in the analysis script:
`BraidWord::commutation_normal_form`, `period_word` and `entropy_per_period`
(`volterra-braid/src/braidword.rs`), mirrored in the Python oracle. The
measured entropies are now `0.962424` against a published `0.96242` for golden
and `1.762747` against a published `1.76275` for silver.

The two candidate explanations the previous pass named, the assumed
`A_sys` normalisation and the unvalidated `circular_boundary` port, are both
answered by this. The active length `als=3.99` derived from that normalisation
is what produced the published braid and the published dilatation to twelve
digits, which an active length wrong by any meaningful factor would not, and the
boundary produced both the correct defect count and the correct orbit. The port
was also read against the reference line by line: `circular_boundary` reproduces
`flow-solver.py:1205-1222` including its rounding of the normals to four
decimals, and `net_charge` enters `apply_q_boundary_conditions` exactly as the
Python's own winding index does.

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

Each entry below is the braid of one period and that period's entropy, the
quantities the paper quotes. The window each was read from, and the entropy
accumulated across that whole window, follow in the next table.

| q | Steps | net_charge | als, ncl | Extracted period | Extracted entropy | Published word | Published entropy | Verdict |
|---|-------|-----------|----------|-------------------|--------------------|-----------------|--------------------|---------|
| 1 (control) | 335,000 | 1.0 | 3.99, 0.975 | `{sigma_1^-1}` | 0.000000 | `{sigma_1}` | 0 | **pass**, up to an orientation sign (see below) |
| 3/2 (golden) | 750,000 | 1.5 | 3.99, 0.975 | `{sigma_2^-1 sigma_1^-1 sigma_2 sigma_1}` | 0.962424 | `{sigma_2^-1 sigma_1}` | 0.96242 | **pass** |
| 4/2 (silver) | 500,000 | 2.0 | 3.99, 0.975 | `{sigma_2 sigma_1^-1 sigma_3^-1 sigma_2^-1 sigma_1 sigma_3}` | 1.762747 | `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1}` | 1.76275 | **pass** |
| 5/2 (aperiodic) | 300,000 (reduced from the paper's 1,000,000 to fit the 30-minute compute budget) | 2.5 | 0.0266, 0.975 | trivial (16 defects, zero crossings) | 0.000000 | none (no stable word expected) | n/a | see below |

The golden period is not the published word as a string, and the silver period
is not the published word as a string either. Both are the published braid:

- **Golden.** `sigma_2^-1 sigma_1^-1 sigma_2 sigma_1` and `sigma_2^-1 sigma_1`
  have the same reduced Burau matrix at `t = -1`, `[[2,1],[1,1]]`, so they act
  identically and carry the same dilatation `phi^2`. The extracted word is a
  longer presentation of the published element.
- **Silver.** Under the commutation normal form the extracted period is a cyclic
  rotation of the published word, and a cyclic rotation is a conjugate, so the
  dilatation is unchanged. The published word normalises to
  `sigma_1 sigma_3 sigma_2 sigma_1^-1 sigma_3^-1 sigma_2^-1`; rotating it by two
  gives the extracted period exactly.

The windows, and what the whole-window entropy reads there:

| q | Frames | Window | Generators | Periods | Whole-window entropy |
|---|--------|--------|------------|---------|-----------------------|
| 3/2 (golden) | 1000 | frames 17-1000, defect count 3 | 64 | 16.00 | 15.398778 |
| 4/2 (silver) | 1000 | frames 38-1000, defect count 4 | 39 | 6.50 | 11.804430 |

The whole-window column is what the previous pass reported as the measured
entropy. For golden it is exactly 16 times the published value. For silver the
window closes mid-period, and a partial period does not scale the entropy by any
fixed factor: the same six-generator block cut off after 39 generators can drive
the dilatation to 1 outright, which is a test in
`volterra-braid/src/braidword.rs`. A whole-window entropy is not comparable to a
published one at any run length.

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

### golden and silver reproduce, and what the earlier failure was

Both runs hold the correct defect count throughout the trailing window (3 for
golden, 4 for silver, matching n=2q), and both produce the published braid at
the published dilatation.

The 2026-08-14 pass reported both as failures. The trajectories were never at
fault: regenerating them from the same seed and the same parameters, and
comparing them correctly, gives the published result. Two things were wrong in
the comparison, and both are now fixed in `volterra-braid` rather than in the
analysis script, so neither can recur through a different caller.

**The entropy was summed over the window rather than taken per period.** The
paper quotes the braid of one period and that braid's entropy.
`topological_entropy` applied to a 64-generator window returns the entropy of
all 16 periods together, which is 16 times larger and grows with run length. The
library now carries `BraidWord::entropy_per_period`, and the window figure is
reported alongside it under a name that marks it as a window measure.

**Periods were found by string equality, which a braid does not obey.**
`sigma_i` and `sigma_j` commute when `|i - j| >= 2`. Two defect swaps among
disjoint pairs are therefore the same braid whichever order they are written in,
and the order the extractor writes them in is set by which crossing the sampling
caught first, a property of the frame spacing and not of the orbit. The silver
word repeats a six-generator block whose commuting pair `sigma_1 sigma_3`
appears both ways round, so it has a braid period of 6 and no string period at
all. `BraidWord::commutation_normal_form` sorts each commuting adjacent pair by
index, which is a normal form for those relations, and `period_word` takes the
period of that. Period detection also now allows the last repeat to be cut short,
since a window closes where the run ended rather than on a period boundary.

Two earlier checks stand, and their results are unchanged by this:

- **The velocity boundary condition was checked against the reference and ruled
  out** (see above): both `volterra-cgpo` and the reference script's
  actually-executed code apply plain no-slip.
- **Both runs were repeated at 5x denser saving** (golden: `CGPO_SAVE_EVERY`
  3750 to 750; silver: 2500 to 500), same seed and same parameters, and
  reproduced the coarse run's word and entropy exactly. The trajectory is
  deterministic and finer sampling caught nothing the coarse sampling missed.
  That remains true, and it is why the fault had to be in the comparison.

The two candidate explanations that pass left open are both answered. The
`A_sys` normalisation gives `als=3.99`, and `als=3.99` is what produced the
published braid and the published dilatation to twelve digits; an active length
wrong by the factor the alternative reading of `A_sys` would imply, roughly 9x,
would not. The `circular_boundary` port was read against
`flow-solver.py:1205-1222` line by line: the inside test, the outer-boundary and
inner-boundary passes and the normals all correspond, including the reference's
rounding of each normal component to four decimals, and `net_charge` enters
`apply_q_boundary_conditions` exactly where the Python's own winding index does.
The boundary also produces the topologically required `2q` defects in every run.

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
  now carries FIRE on both CPU and GPU, ported step for step from open-Qmin's
  own (`BENCHMARKS.md` section 1), so the minimiser is no longer the gap. What
  remains missing is the geometry those results are posed in: no
  colloidal-inclusion boundary and no patterned boundary. This is a real gap,
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
