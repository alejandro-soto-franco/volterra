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
