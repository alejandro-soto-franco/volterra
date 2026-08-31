# Reproducing Mitchell et al. (2024) and Mitchell et al. (2025)

Two papers on the same solver, the same group, and the same parameter set:

- **M24.** Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, "Maximally mixing
  active nematics", Phys. Rev. E **109**, 014606 (2024). A 2D active nematic on a
  square with periodic boundaries settles, when confined tightly enough, into a
  periodic four-defect orbit whose `+1/2` pair executes the maximal mixing braid,
  and that periodic motion mixes harder than the chaotic state.
- **NL.** Mitchell, Sabbir, Klein and Beller, "Modelling active nematics via the
  nematic locking principle", Soft Matter (2025), arXiv:2506.20996. A microtubule
  bundle cannot rotate unless its neighbourhood rotates with it; standard
  Beris-Edwards violates that throughout the bulk, and a switch on the mobility
  of one term repairs it.

**Date:** 2026-08-29. **Machine:** Fedora 44, AMD Ryzen 9 8940HX. Each run
single-threaded (`FD_FORCE_SERIAL=1`); `volterra-fd`'s rayon path is 7x slower
than serial at `100 x 100`, as `par_gate` documents.

**Runner:** `volterra-fd/examples/periodic_active_nematic.rs`, analysed by
`examples/analyse_periodic.py`.

---

## 1. Parameter mapping

M24 states five dimensionless groups and no dimensional constants: `lambda = 1`,
`Re = K/(rho nu^2) = 0.01`, `gamma_tilde = gamma nu / K = 50`,
`C_tilde = C/zeta = (ell_a/ell_n)^2 = 9`, and `ell_a/L` with `L = 100`. NL states
the constants themselves: `gamma = 5*256`, `C = 256^2`, `K = 256^2`,
`eta = 2560`, `zeta = (256/3)^2`, on `200 x 200`.

`Params::from_dimensionless` maps the groups to constants, and at
`K = 256^2, ell_a = 3` it returns NL's five numbers digit for digit
(`tests/test_locking.rs::from_dimensionless_reproduces_the_reference_constants`).
`K` is the only free scale, and it only sets the time unit: at fixed groups a
rescaling `K -> sK` rescales every rate by `sqrt(s)` and changes nothing else.

**One thing the two papers state differently.** M24 sets `C = -A`, so
`S_eq = sqrt(-2A/C) = sqrt 2`. NL sets `C = -2A`, so `S_eq = 1`. Both write the
same free energy. `Dimensionless` therefore takes `s_eq` explicitly, and NL's
switch width `sigma = 0.2` is given in units of `S_eq` rather than as a raw
number. Both conventions were run; Sect. 4 reports the control.

## 2. Nematic locking

`nl_lock0` and `nl_lock1`: `200 x 200` periodic, `ell_a = 3`, `ell_n = 1`,
`S_eq = 1`, identical seed and initial field, run to `t = 10` (2837 `t_a`),
statistics over the last 100 recorded frames, `t` in `[6.26, 9.98]`.

| | NL, standard | here, standard | NL, enhanced | here, enhanced |
|---|---|---|---|---|
| RMS `omega_A` | 0.263 | 0.491 | 0.249 | 0.512 |
| RMS `omega_F` | 0.158 | 0.339 | 0.605 | 0.299 |
| median `\|omega_A\|` | 0.1490 | 0.336 | 0.168 | 0.329 |
| median `\|omega_F\|` | 0.0687 | 0.155 | 7.14e-7 | 1.44e-6 |
| **median ratio `\|omega_F\|/\|omega_A\|`** | **0.461** | **0.460** | **4.25e-6** | **4.39e-6** |
| mean `+1/2` defects | 6.4 | 3.7 | 9.6 | 11.8 |

The result is the last two rows. Under the standard model the median fracturing
rate is 46 per cent of the median advective rate, and the paper's summary of that
number, "roughly half that of advection", is reproduced to three decimal places.
Under enhanced locking it falls by five orders of magnitude, to a few parts in a
million of the advective rate, again to within 3 per cent of the published ratio.
The defect count rises under enhanced locking in both.

Both absolute rates here run 1.9x the published ones, uniformly: the ratio is
unaffected, so the discrepancy is a scale on the rate rather than a difference in
the mechanism. `omega` is an inverse time, `K` sets the time unit, and `K` was
taken from NL's own list, so the residual factor is not explained here and is
recorded rather than absorbed.

The distributional statement also reproduces. Under enhanced locking the RMS of
`omega_F` stays the same order while its median collapses, which is the paper's
own reading: fracturing has not been removed, it has been concentrated into a
small part of the domain. Here the RMS falls only from 0.339 to 0.299 while the
median falls by 10^5.

## 3. Chaotic regime of M24

`m2024_ella1.0`: `100 x 100` periodic, `ell_a = 1`, random director field, run to
`t = 25` (63936 `t_a`). Four material lines seeded at `t = 10` and advected;
`h` is the slope of `log(length)` against time.

| | M24 | here |
|---|---|---|
| `h` (reciprocal integration time) | 3.19 +- 0.03 | 3.52 +- 0.05 |
| `h_tilde = h t_a` | 1.25e-3 (line stretching) | 1.376e-3 |
| | 1.184e-3 (E-tec) | |
| `u_rms` | 50 to 90 (their Fig. 3a) | 74 +- 10 |
| defects | chaotic, continuously created and destroyed | `+1/2` count 17 +- 3, aperiodic |

The topological entropy agrees to 10 per cent and the velocity scale and defect
behaviour agree qualitatively. The measurement is independent of the solver: a
material line advected by the recovered flow, refined by midpoint insertion, with
minimum-image segment lengths on the torus
(`volterra-fd/src/stretching.rs`, validated against a uniform-strain field where
the exact rate is recovered to 1e-3).

## 4. Periodic orbit of M24

M24's headline claim is that at `ell_a = 3` on `L = 100` the system settles into a
periodic orbit of two `+1/2` and two `-1/2` defects, with
`h_tilde = 1.66e-3`, about 33 per cent above the chaotic value. **That state was
not reached here under standard Beris-Edwards, by any protocol tried.**

What was tried, all at `L = 100`, `dt = 7.5e-5`, run to `t = 25` except where
the row says otherwise:

| Protocol | Outcome |
|---|---|
| Random director, `ell_a = 3`, seed 1 | Reaches exactly `2 +1/2` and `2 -1/2` at `t = 3.7` and keeps them to `t = 9.5`, with `u_rms` decaying monotonically from 15.6 to 8.7, then annihilates |
| Random director, `ell_a = 3`, seeds 2 to 7 (to `t = 12`) | 0 or 1 `+1/2` remaining in all six |
| Random director, `ell_a` in `{1.5, 2, 2.5, 3, 3.5, 4, 4.5}` | 0 defects at every value; only `ell_a = 1` stays chaotic |
| Four-defect seeded field (Sect. 5), `ell_a` in `{2, 2.5, 3, 3.5, 4}` | Annihilates in every case by `t = 25`; at `ell_a` of 3, 3.5 and 4 the endpoint is stationary to a relative `u_rms` fluctuation of `7e-4` or less |
| Four-defect seeded field, `ell_a` in `{1.2, 1.4, 1.6, 1.8}` (to `t = 14.2`) | Chaotic, with 9.6, 7.3, 4.1 and 1.1 `+1/2` defects; aperiodic at every value |
| Four-defect seeded field, `ell_a = 3`, `S_eq = 1` (NL's convention, to `t = 14.7`) | 0 defects from `t = 6.1` |

The endpoint at `ell_a = 3` is a **stationary state with constant nonzero
flow**: `u_rms = 8.755 +- 0.003` over `t` in `[12, 25]`, a
relative fluctuation of `3e-4`, with `mean S` constant to six figures and no
defects. Under the same parameters M24 reports a periodic orbit.

The mixing follows the defects. Material lines in these stationary states give
`h_tilde` between `1.8e-4` and `6.0e-4`, against `1.38e-3` measured in the
chaotic state at `ell_a = 1` and the `1.66e-3` M24 reports for its periodic
orbit. A steady flow with hyperbolic stagnation points still stretches a
material line exponentially, which is why the value is nonzero rather than zero.

This is the class of state NL's own Sect. VIII is about: "we argue that such
active states must break nematic locking and are hence unphysical for
microtubule-based active nematics... The standard simulation converges to a
stationary state whereas the modified version produces active turbulence." The
two papers are in tension at this parameter point, and the run here lands on the
side NL describes.

**The control that closes the loop.** `m2024_ella3.0_benl`: the same seed, the
same parameters, the same initial field, with enhanced nematic locking switched
on.

| over `t` in `[12, 25]` | standard | enhanced |
|---|---|---|
| mean `+1/2` defects | 0.00 +- 0.00 | 3.63 +- 1.58 |
| `u_rms` | 8.755 +- 0.003 | 14.65 +- 2.39 |
| relative fluctuation in `u_rms` | 3e-4 | 0.163 |

Enhanced locking sustains defects and sustained flow at exactly the parameter
point where the standard model here falls into a stationary state. That is NL's
prediction, tested on M24's own configuration.

**What was ruled out as the cause of the discrepancy.**

- *Pressure convergence.* The Jacobi solve is converged: `u_rms` after 2000 steps
  agrees to seven figures between 50, 200 and 1000 iterations, and
  `max |div u|` is unchanged at 0.154. The residual divergence is the scheme's
  own inconsistency, since centred `div grad` is not the 9-point Laplacian the
  pressure is solved against, and the reference `flow-solver.py` shares it.
- *Timestep.* `dt = 1e-4` and `dt = 5e-5` agree to `7e-5` relative in `u_rms` at
  equal `t`. The production `dt = 7.5e-5` is 51 per cent of the explicit viscous
  limit `0.375/nu`.
- *Advection scheme.* `flow-solver.py`'s upwind term, which this crate ports, is
  second order (`(3 f_i - 4 f_{i-1} + f_{i-2})/2`), not first, so it is not a
  source of the first-order numerical diffusion that would smear a marginally
  resolved core.
- *Parameter mapping.* Verified against NL's own stated constants (Sect. 1), and
  independently against M24's reported velocity scales: `u_rms/v_a` is 0.026 to
  0.031 here at `ell_a = 1` against 0.02 to 0.035 read off their Fig. 3a, and
  0.018 here at `ell_a = 3` against about 0.02 from their Fig. 3b.
- *Initial condition.* Both a random director field and an analytically seeded
  four-defect field in M24's own Fig. 2 arrangement.
- *Order-parameter convention.* Both `S_eq = sqrt 2` (M24's stated `C = -A`) and
  `S_eq = 1` (NL's stated `C = -2A`); the seeded four-defect field annihilates in
  both.
- *A shifted active length.* Sweeping `ell_a` from a seeded four-defect field in
  steps of 0.2 across the transition, `1.2` to `1.8`, and in steps of 0.5 from
  `2` to `4.5`, finds no value at which two `+1/2` defects persist with a
  periodic `u_rms`. The count falls monotonically through the transition, from
  9.6 at `ell_a = 1.2` to 1.1 at `ell_a = 1.8` to 0 above `ell_a = 2`. The
  strongest autocorrelation of `u_rms` over any late window in the sweep is 0.32,
  at `ell_a = 1.2`, and every other value is below 0.06; a periodic orbit returns
  a value near one, which is the threshold `analyse_periodic.py` reports on.

What remains unexplored is the possibility that the orbit is stable only in a
narrower parameter window than the one swept, or that it depends on a
discretisation detail neither paper states. M24 reaches its own Fig. 5 blue band
by continuation from a state it already has, "an initial Q-field taken from the
periodic state at `ell_a = 3`", and does not say how that state was first found.

## 4b. What the search had wrong

**Date:** 2026-08-30. Four things were established, three of them by reading the
published version and the paper's own figure files rather than the arXiv text.

**The dimensional constants are in the published version, not the preprint.**
Sect. II of Phys. Rev. E **109**, 014606 states them outright: `gamma = 5*256`,
`K = 256^2`, `rho = 1`, `nu = 2560`, `C = 256^2 * (3/ell_a)^2`,
`alpha = (256/ell_a)^2`, with `L = 100`. arXiv:2308.08657v1 gives only the five
dimensionless groups. Every constant matches what `from_dimensionless` returns,
so the parameter mapping of Sect. 1 is confirmed against M24 directly and not
only against NL.

**The crate is an exact port of the reference solver.** Starting
`flow-solver.py` and `volterra-fd` from one initial `Q` field, on a `100 x 100`
torus at `ell_a = 3`, `dt = 1e-4`:

| `t` | `flow-solver.py` | `volterra-fd` |
|---|---|---|
| 0.05 | 1.005339e1 | 1.00533819e1 |
| 0.10 | 9.214165e0 | 9.21412152e0 |
| 0.20 | 1.019333e1 | 1.01932095e1 |
| 0.45 | 1.422178e1 | 1.42219866e1 |

Five to six figures at every checkpoint, and the two codes reach the same
defect-free stationary state, `u_rms = 8.753` against `8.755`. The
discretisation was the last untested hypothesis of Sect. 4 and it is closed:
the two schemes agree to the precision of the comparison.

**M24 was produced with the full stress, not the reduced one its text quotes.**
The paper states `Pi^E = -lambda H + [Q, H]`, following Giomi. The code that
produced its figures, `flow-solver.py`, assembles `Pi_S` with the Ericksen term
and the `2 Tr[QH] Q` term as well, which is Eq. (11) of the CGPO paper and what
this crate ports. The two are told apart by one number: M24's own Fig. 3(b)
gives `u_rms = 10.05` at `t = 0.05`, and so does the reference solver at the
stated parameters. The `fig5` and `paper/rnd_*` sweeps recorded above were run
with `VP_STRESS=giomi`, on the strength of the paper's equations, so they are on
the wrong branch of the very comparison they were built for.

**The target state is now measured, from the paper's own figure files.**
`urms.pdf` and `lambdavsla.pdf` in the arXiv source are vector, so their curves
can be read as data rather than by eye. From Fig. 3(b) at `ell_a = 3`:

| | value |
|---|---|
| `u_rms` in the periodic state | 16.89, between 16.35 and 17.43 |
| ripple | 3.2 per cent |
| ripple period | 1.994 |
| orbit period `T` | 7.98, four ripples an orbit |
| `T_tilde` | 2269 |
| `h_tilde_max = log(phi + sqrt phi)/(T_tilde/4)` | 1.87e-3 |
| run length | `t = 45` |
| transient | over by `t = 3` |

The four ripples an orbit are the four encounters, which is the braid's own
signature read off the velocity trace. `h_tilde_max = 1.87e-3` against the
`1.66e-3` M24 measures is the "slightly larger value" its text reports, so the
period is right. From Fig. 6(b, c), the `+1/2` orbit is a circle of radius
`L/2` centred on a `-1/2` defect and the `-1/2` orbits are squares of side about
`0.3 L`, which is the construction of Fig. 2(a) at the measured scale. Fig. 2(b)
itself, digitised from its raster, gives the same: the `-1/2` defects sit on the
checkerboard `(0, 0)`, `(L/2, L/2)` to within a per cent, and the `+1/2` ring
radius is `L/2`.

**Two corrections to the search itself.** Fig. 5's legend names the black
curve's initial condition "Periodic", a snapshot of the periodic state, and its
blue band runs `2.77 <= ell_a <= 4.25`. Fig. 6(a) puts the Lyapunov exponent at
`-0.11` at `ell_a = 3`, so the headline value sits near the band's unstable
edge and the orbit is most robust nearer `ell_a = 3.5`. The search above
concentrated at `ell_a = 3`.

**The orbit is not what a random draw reaches.** Three numpy seeds through
`flow-solver.py` at the stated parameters, run to `t = 25`, end with no defects
at `u_rms` of 8.75, 8.88 and 7.13. The paper's own bistability is the reading:
its Fig. 5 red curve, from a nearly uniform field, has no bump either.

## 4c. Reproducing the maximal mixing braid

**Date:** 2026-08-30. With the stress corrected to the full Beris-Edwards form
of Sect. 4b, a seed screen at `ell_a = 3` kept two to three `+1/2` defects at
`u_rms` between 15 and 18 where the Giomi branch had reached zero. Sixteen runs
were then taken to M24's own length of `t = 45`, eight seeds each at
`ell_a = 3` and at `ell_a = 3.5`, the latter because Fig. 6(a) puts the orbit
further inside its stability band there. One survived: `ell_a = 3.5`, seed 10.

| | M24, `ell_a = 3` | here, `ell_a = 3.5` |
|---|---|---|
| cast | 2 `+1/2`, 2 `-1/2` | 2 `+1/2`, 2 `-1/2`, 78 per cent of frames exactly |
| `u_rms` | 16.89 | 14.96 +- 0.77 |
| orbit period `T` | 7.98 | 9.35 (recurrence quality 0.962) |
| `u_rms` cycles an orbit | 4 | 4.02 |
| passes an orbit | 4 | 4.02, and 4.25 counting the window's leading transient |
| passes of one sense | all | 16 of 17 |
| `T_tilde` | 2269 | 1954 |
| `h_tilde_max` | 1.87e-3 | 2.17e-3 |

The passes are evenly spaced: from `t = 16.4` to `t = 44.3` there are thirteen
consecutive passes at intervals of `2.325 +- 0.02`, which is `T/4`. The
minimum-image separation of the two `+1/2` strands has its own dominant period
at 2.33, so the four passes an orbit are measured three independent ways, from
the velocity trace, from the separation series, and from the passes themselves.
`figure2.pdf` in the run directory reproduces M24's Fig. 2: circular `+1/2`
orbits tiling the plane with the `-1/2` orbits as small squares on the
checkerboard.

The orbit sits at `ell_a = 3.5` rather than at M24's headline 3.0, and it runs
about 17 per cent slower than the period their Fig. 5 blue curve implies there.
Both are recorded rather than reconciled.

**Five corrections to the braid reader, each found by checking it against the
geometry measured in Sect. 4b.** They are listed because four of the five made
the tool reject the very state it was written for.

- *The bounded test rejected the target.* It required a radius of gyration below
  `0.5 L`, and the orbit is a circle of radius `L/2`, whose radius of gyration
  is exactly `L/2`. The criterion sat on the answer. The bound is now `0.75 L`
  and a test pins the ideal orbit's gyration to the ring radius.
- *Winding was measured per window rather than per orbit*, so the test tightened
  as a run lengthened and rejected a state periodic up to a slow drift. It is
  now per revolution.
- *The period came from the RMS velocity*, which cycles four times an orbit
  because it peaks at every pass. That divided `T_tilde` by four and multiplied
  the braid prediction by four. `TorusWorldlines::recurrence_period` measures it
  from the worldlines instead, taking the deepest return rather than the first,
  since a ring has a shallow dip at half a revolution where the strand sits
  antipodally.
- *Passes arrived in clusters.* Defect positions are quantised to the detection
  grid, so one approach plateaus and every frame of the plateau read as a pass:
  six on consecutive frames of one approach. `encounters_apart` keeps the
  deepest minimum in each refractory window of an eighth of the orbit
  period.
- *The sense fraction reported the margin*, `|net|/n`, which is 0.88 where 16 of
  17 passes agree. It now reports the majority's share, and the test is a
  supermajority rather than unanimity, because the alternative it has to exclude
  is the Ceilidh dance, which alternates. The controls sit at 53, 60 and 0 per
  cent against this run's 94.

The unit test on `ideal_figure_2a` now asserts each of the five, so tightening
any of them back onto the target fails the suite, and the reader still rejects
five states that are not the braid, on four different grounds.

**A defect in every PDF figure this repository has produced.** With
`text.usetex` on, matplotlib's PDF backend subsets the Type 1 Computer Modern
fonts and the CMSY minus does not survive: a tick at `-100` sets as `100` and a
label of `$-1/2$` as a gap. The Agg path is unaffected, so a PNG of the same
figure looks right and hides it. `plot_braid.py`, `plot_fig5.py` and
`plot_paper_figures.py` now render through the PGF backend, which runs LaTeX
over the figure and keeps the glyph.

## 4d. The paper's figures

Four of M24's five figures are reproduced, from the continuation family of
Sect. 4c and from a chaotic run at `ell_a = 1`.

**Fig. 2, the orbits.** `plot_braid.py` draws the published construction beside
the run's own lifted worldlines, tiled over the plane. Circular `+1/2` orbits on
the checkerboard with the `-1/2` orbits as small squares, which is the published
panel.

**Fig. 3, the RMS velocity.** `plot_paper_figures.py fig3`, with two director
snapshots one orbit period apart rather than at arbitrary positions in the run,
so the pair shows the same field. That is the paper's own reading of its panel:
"two snapshots taken at the same phase of the motion show essentially the same
director field and defect locations".

**Fig. 4, the line growth.** Four advected curves an axis, the fitted slope, and
the final curve inset.

| | M24 | here |
|---|---|---|
| `h`, `ell_a = 1` | 3.19 +- 0.03 | 3.523 +- 0.054 |
| `h`, periodic state | 0.475 +- 0.003 at `ell_a = 3` | 0.428 +- 0.008 at `ell_a = 3.5` |

**Fig. 5, the entropy across the active length.** Both of the paper's initial
conditions, twenty-three runs. The black curve continues from a state on the
orbit; the red starts from a nearly uniform director field, which `VP_IC=uniform`
adds.

| `ell_a` | `h_tilde`, periodic start | `h_tilde`, uniform start | braid prediction |
|---|---|---|---|
| 1.50 | 1.85e-4 | 1.16e-4 | |
| 2.00 | 2.64e-4 | 2.06e-4 | |
| 2.25 | 3.31e-4 | | |
| 2.50 | 1.56e-3 | 9.14e-5 | |
| 2.75 | 2.05e-3 | 4.05e-4 | 2.31e-3 |
| 3.00 | 2.17e-3 | 3.22e-5 | 2.30e-3 |
| 3.25 | 2.05e-3 | 4.66e-4 | 2.25e-3 |
| 3.50 | 1.62e-3 | 5.59e-4 | |
| 3.75 | 1.73e-3 | 5.53e-4 | |
| 4.00 | 1.68e-3 | 4.91e-4 | |
| 4.25 | 1.25e-3 | 4.01e-4 | |
| 4.50 | 2.33e-4 | 3.98e-4 | |

The shape is the published one: a plateau over the band, a collapse either side,
and a uniform-start curve that stays low across the whole range with no bump,
which is the bistability M24 reports. Two `+1/2` defects survive from
`ell_a = 2.75` to `4.5` against the paper's `2.77` to `4.25`, so the lower edge
agrees to 0.02 and the upper runs high. The entropy itself runs about 25 per
cent above the published values throughout, and the braid prediction sits above
the measurement at every active length where the strands return, by 6 to 11 per
cent against the paper's 11.

The braid prediction is drawn only where the recurrence quality of the
worldlines exceeds a half. Above `ell_a = 3.25` the continuation keeps two
defects while the orbit itself stops closing, which is the instability the paper
describes at the band's right edge, and a period taken from a lag the strands
never return to would put the prediction several times too high.

**A defect in `plot_paper_figures.py`, from before this session.** Its Fig. 4
inset globbed `line_*.csv`, which also takes `line_lengths.csv`, and that sorts
last, so the inset drew the length table as a curve. Its entropy label built
`10^{-3}` by substituting into a formatted string and closed the brace after the
closing `$`, which halts LaTeX outright.

## 4e. The orbit over eighty revolutions of the active time

`braidfilm_ella3.5`: the same seed and parameters as Sect. 4c, run to `t = 90`,
twice M24's own length, with a `32 x 32` tracer lattice and a frame every 0.2
time units.

| over `t` in `[13.5, 90]`, 8.2 revolutions | |
|---|---|
| cast | 2 `+1/2` and 2 `-1/2`, 78.0 per cent of frames exactly |
| orbit period `T` | 9.35, recurrence quality 0.988 |
| `u_rms` cycles a revolution | 4.02, autocorrelation 0.972 |
| passes | 33, **4.00 a revolution**, **100 per cent of one sense** |
| gyration | 57.07 on both strands, windings exact mirrors |
| `h_tilde` | 1.975e-3 +- 2.5e-5 |
| `h_tilde_max` | 2.173e-3 |
| verdict | the maximal mixing braid of Fig. 2(a) |

Thirty-three passes of one sense with no exception is the paper's "defects
always pass each other in the same sense" without a qualifier, and the rate is
four an orbit to two decimal places. The entropy sits 9 per cent below the braid
prediction, against the 11 per cent M24 reports at its own `ell_a`.

## 4f. Films

Two layouts, both fixed so runs can be put side by side.

`panels_braid_video.py`, nine panels, row 3 the paper's own three measurements
rather than the two rotation rates of the locking comparison: the braid diagram
after their Fig. 1(c), the advected material lines with their length growth
inset after Fig. 4, and the defect count with the RMS velocity after Fig. 3.
`panels_periodic_video.py` keeps the rotation-rate layout, and each layout
belongs to the paper it was built for.

The braid panel projects each strand on a LATTICE axis and wraps it to the box.
An oblique axis has no period on the torus and its own wrap reads as a crossing:
on the first run filmed that counted 55 passes where the analysis finds 17. One
projection also sees only the passes that happen along it, 13 of 17 in `x` here,
so the pass marks come from `braid.json` and the film and the report count the
same events.

`torus3d_braid_video.py`, Fig. 2 animated flat beside the same instant on an
embedded torus, which takes the remaining two by two as one panel. The
construction's rods run at the measured orbit period, the run's own orbits trace
over the plane, and the same orbits trace on the surface over the director
field.

Three things the surface needed. The director is drawn as the PUSHFORWARD of a
flat segment rather than in unit tangents, so it stretches on the outside of the
tube and compresses on the inside: the embedding is a picture and not an
isometry, and unit tangents would hide that. Everything whose outward normal
faces away from the camera is dropped, since a 3D axes composites in painter's
order and a glyph on the far side would otherwise sit over the near face. And
the surface is shaded by hand: `shade=True` drove one side of the tube almost to
black and the order parameter, which is what the colour is for, stopped reading.

## 5. New capabilities

- `boundary::periodic_boundary`: a flat torus. Every cell interior, neither
  boundary ring populated, so the four boundary-condition passes in
  `update_step_inner` visit nothing and the domain closes through the modular
  indexing every stencil already used.
- `locking`: the enhanced-locking term, the `Q`/`U` decomposition of `H`, the
  `omega_A` and `omega_F` diagnostics, and their RMS and median. Switched on
  through `Params::locking`, which is `None` everywhere by default and
  `#[serde(default)]`, so the golden concurrence tests and every earlier config
  are untouched.
- `stretching`: material-line topological entropy.
- `ic`: analytically seeded disclinations, `theta = sum_k q_k arg(r - r_k)`
  summed over periodic images, refused unless the charges cancel. This closes the
  "defect-seeded initial condition" row of `SUBSUMPTION.md`, which no reference
  code ships either.
- `Params::from_dimensionless` and `Dimensionless`, plus `active_length`,
  `coherence_length` and `active_time`.

- `examples/panels_periodic_video.py`, a nine-panel film of a run, and the
  fields the runner writes for it: `p`, the two rotation-rate fields, the
  material-line vertices, and a lattice of passive tracers.

147 tests pass across 32 suites; `cargo clippy --all-targets` is clean.

## 6. Films

Three runs on a `100 x 100` torus, `dt = 7.5e-5` to `t = 12`, filmed at 321
frames each in one fixed nine-panel layout. The layout is the same in all three,
so the second and the third can be put side by side: they differ in the locking
switch and in nothing else, down to the seed and the initial condition.

| Panel | Row 1 | Row 2 | Row 3 |
|---|---|---|---|
| left | director field | vorticity field | advective rotation rate |
| centre | `Q` isocontours | RMS vorticity | fracturing rotation rate |
| right | velocity field | passive tracers | defect count and RMS velocity |

The two rotation rates share one colour scale, taken from the advective one, so
enhanced locking draws the fracturing panel blank rather than rescaling its own
noise. Every other scale is fixed across a film from a prepass over its frames.

| Run | `ell_a` | Locking | Initial condition | `n_{+1/2}` | `u_rms` | median `|omega_A|` | median `|omega_F|` |
|---|---|---|---|---|---|---|---|
| `panels_film_chaotic` | 1.0 | off | random | 15.8 +- 2.0 | 72.3 +- 9.0 | 3.652 | 1.465 |
| `panels_film_ella3_be` | 3.0 | off | four-defect | 1.00 +- 0.00 | 11.8 +- 0.6 | 0.369 | 0.244 |
| `panels_film_ella3_benl` | 3.0 | on | four-defect | 3.28 +- 0.75 | 17.2 +- 1.2 | 0.414 | 1.66e-6 |

Statistics are over the last third of each run; the rotation rates are over the
last 100 frames, which is the window arXiv:2506.20996 averages its own over.

The third row of the last two films is the measurement of the nematic locking
paper as a picture. Switching the term on moves the advective median by 12 per
cent, from 0.369 to 0.414, and divides the fracturing median by `1.5e5`. The
reference reports `0.149 -> 0.168` and `0.0687 -> 7.14e-7` on its own `200 x 200`
domain, which is a factor of `9.6e4`. The absolute rates here run high by the same
uniform factor recorded in Sect. 2, and the ratio is what transfers.

At `ell_a = 3` the standard model keeps one `+1/2` pair and settles to a nearly
steady flow, `u_rms` fluctuating by 5.2 per cent. With the switch on the same
configuration sustains 3.3 `+1/2` defects and a flow that never settles. Neither
is the periodic orbit of M24, and Sect. 4 records the search that did not find
it.

## 7. Braid on the torus

M24's central claim is that the four-defect periodic orbit's two `+1/2` defects
write the *maximal mixing braid*: bounded circular paths, four encounters an
orbit, every pass in the same sense. Its entropy per operation is
`log(phi + sqrt phi) = 1.0613`, quoted from Smith and Dunn, so with the measured
period the braid predicts `h_tilde_max = log(phi + sqrt phi) / (T_tilde / 4)`,
the blue curve of its Fig. 5.

`volterra_braid::torus` reads that braid off a run rather than assuming it.
Worldlines are lifted to the universal cover, so a bounded orbit is a closed loop
and a defect that winds is visibly not one. Encounters are found over the image
lattice, which is where the four an orbit come from: in Fig. 2a each rod runs on
the circle of radius `L/2` about a `-1/2` defect, the two circles meet exactly at
the rods' own sites, and counting periodic images each rod meets the other's
track four times a revolution. `ideal_figure_2a` reconstructs that motion and the
reader returns 4.00 same-sense encounters an orbit on it.

`examples/braid_report` applies the reader to a run over the longest window in
its developed half whose defect census never changes. The window must come from
the developed state: the longest such window over a whole run lands in the
quench, where the census happens to sit still and the defects wind rather than
orbit, and a braid read there is the transient's.

### Stress

The two papers use different stresses, and each states its own.

Klein, Soto Franco, Mitchell and Beller, Eq. (11), take the full Beris-Edwards
form,

```text
F_i = d_j [ -H_ij - zeta Q_ij + [Q, H]_ij + 2 Tr(QH) Q_ij - K d_i Q_kl d_j Q_kl ],
```

which is `flow-solver.py`'s force and therefore this crate's. M24 takes
`Pi = Pi_E + Pi_A` with `Pi_E = -lambda H + [Q, H]` and `Pi_A = -alpha Q`, "as
given in Ref. 10", which is Giomi, Phys. Rev. X 5, 031003 (2015). Giomi states
the omission: the Ericksen stress "has been neglected because of higher order in
the derivatives of `Q_ij`", a simplification "known not to have appreciable
consequences in the fluid mechanics of two-dimensional active nematics".

`Params::stress` selects between them, `Full` by default so nothing earlier
moves. The difference is large at this confinement. With the stress as the only
change, averaged over the late quarter of a run to `t = 22`:

| `ell_a` | Full, `n_{+1/2}` | Giomi, `n_{+1/2}` | Full fluctuation | Giomi fluctuation |
|---|---|---|---|---|
| 2.0 | 0.18 | 3.32 | 3.4 % | 16.2 % |
| 2.5 | 1.03 | 2.53 | 14.0 % | 10.2 % |
| 3.0 | 0.00 | 1.93 | 0.0 % | 24.7 % |
| 3.5 | 1.00 | 1.22 | 3.7 % | 18.3 % |
| 3.0, random start | 0.00 | 1.64 | 4.4 % | 23.1 % |

At `ell_a = 3` the full stress gives a defect-free stationary state whose flow
does not fluctuate at all, and the reduced one sustains defects. Giomi's
simplification is not appreciable for the bulk turbulence statistics he measured,
and it is decisive here.

### Search for the orbit

Under M24's own stress, at `ell_a` in `{3.0, 3.25, 3.5, 3.75, 4.0}`, inside the
window its Fig. 6a opens at `ell_c = 2.92` and its Fig. 5 closes near 4.2, from
both its random-start protocol and the seeded four-defect field, no run reaches
the orbit. Every settled window holds at most one `+1/2` defect and nine of
sixteen end defect-free. The closest approaches are two windows under a time unit
long with two `+1/2` defects and every pass of one sense, at `ell_a = 3.5` and
3.75 from a random start, which fail on the encounter rate.

Ruled out across this and Sect. 4: seven random seeds, ten values of `ell_a`,
both order-parameter conventions, two seeded geometries, eight initial director
orientations, with and without pre-relaxation, both stresses, to `t = 25`.

What remains different from the paper is the numerical scheme. The equations, the
five dimensionless groups, `lambda = 1` against its stated threshold of 0.6, the
confinement window and the initial-condition protocol now all match. The 9-point
Laplacian, the second-order upwind advection and the projection-method pressure
solve do not, and the paper reports the orbit as bistable, unstable below
`ell_c`, and unstable on a doubled domain without control points. A
discretisation difference is a sufficient explanation for an attractor that
delicate.

### Entropy

`entropy.json` and `line_lengths.csv` are written at every observation rather
than at the end. A material-line entropy is a fit to a history that is complete
the moment the line saturates, so a run stopped part way is exactly as
informative about mixing as one that reaches its last step.

Over the batch above, `h_tilde` runs from `2.5e-5` to `1.63e-3`. The largest
comes from the only fit whose line saturated, at `ell_a = 3.25`, and sits close
to M24's `1.66e-3` for the periodic orbit; its state is a fluctuating 1.4
defects rather than the orbit, so the proximity is a coincidence of two different states.
