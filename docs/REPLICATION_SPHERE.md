# Active nematics on the sphere

Replication target: Zhu, Saintillan and Chern, "Active nematic fluids on
Riemannian 2-manifolds" (`zhu-2024-act-nem-flu`, arXiv:2405.06044; published as
Proc. R. Soc. A **481**, 2311, doi:10.1098/rspa.2024.0418).

## Claim being reproduced

At low activity the sphere settles into a fixed cast of four `+1/2` defects in
periodic orbits; at high activity the system is turbulent, with defect pairs
created and annihilated. The two cases are quoted at `Pe = 1` and `Pe = 10^4`
(p. 26).

The defect count is forced by Poincaré-Hopf: the charges on a closed surface sum
to the Euler characteristic, `+2` on a sphere, which is four `+1/2` defects. The
periodicity is not forced, and is the part this replication targets.

## Péclet number

The reference non-dimensionalises with the domain radius `r` as the length and
`tau = mu |alpha|^-1` as the time (p. 15). Its system is

    (Delta + K) u - grad p + div_grad Q = 0,    div u = 0,
    [q]. + P^L_lambda L_u [q] = Pe^-1 ( Delta^L [q] - varepsilon^-2 (|[q]|^2 - 1) [q] ),

with `Pe = |alpha| eta r^2 mu^-1` and `varepsilon = eps / r` the core size
against the system size. Two features of that scaling matter for a comparison.

The active stress in the momentum equation has coefficient one: activity is
absorbed into the time unit, so `Pe` sets the advection against the relaxation
and nothing else, and `Pe^-1` multiplies the elastic diffusion and the
Ginzburg-Landau term together, so a single number moves both.

This solver instead fixes the relaxation and scales the active stress, with
`zeta = Pe_flag K`. The two agree once the flag is calibrated, which is done by
measuring rather than by converting conventions:

    Pe = u_rms r / (gamma_r K)

is written to `stats.csv` on every run as `pe_measured`. Measured from the
relaxed tetrahedron at refinement 5:

| `--pe` | `u_rms` | `pe_measured` |
|---|---|---|
| 1 | 0.001124 | 0.112 |
| 10 | 0.011141 | 1.114 |

The response is linear, so `Pe_flag ~ 9` is the reference's `Pe = 1`, the
periodic-orbiting case, and its `Pe = 10^4` sits near `Pe_flag ~ 9 x 10^4`.

## Surface Stokes operator

The reference states that the Stokes operator has the Killing fields in its
kernel, and removes that component to take the solution of minimal `L^2` norm
(p. 26). Checking that property against this code found the operator wrong.

The momentum equation has `(Delta_B + K) u` on the vector field, with the
Bochner Laplacian. Rewriting a divergence-free `u = J grad psi` through its
stream function turns that into the scalar Laplace-Beltrami operator shifted by
**twice** the curvature, so steady Stokes on a surface is

    Delta (Delta + 2K) psi = curl f,

rather than the flat biharmonic `Delta^2 psi = curl f` the production solver was
using. The stream-function path had no curvature term at all.

The factor is what makes a Killing field free. On the unit sphere a rigid
rotation about `z` has stream function `z`, an `l = 1` harmonic with
`Delta z = -2 z`, so `(Delta + 2K) z = 0` while `(Delta + K) z = -z`. Applying
the composite `(Delta + cK) Delta` in the forward direction to `z` returns
`2 (2 - c) z`.

Measured on the discrete operators, with an `l = 2` harmonic as the control that
must never be annihilated:

| refinement | rotation at `c = 1` | rotation at `c = 2` | `l = 2` at `c = 2` |
|---|---|---|---|
| 3 (642 v) | 1.9919 | -0.0050 | 23.58 |
| 4 (2562 v) | 1.9980 | -0.0013 | 23.89 |
| 5 (10242 v) | 1.9995 | -0.0003 | 23.97 |

The `c = 2` column converges to zero at second order and the `c = 1` column
converges to the analytic `2`. The control converges to `6 x 4 = 24`, so the
test discriminates.

Three further checks pin the corrected operator down.

The kernel the solve detects numerically has dimension 3 at `2K` and dimension 0
at both `K` and no shift, which is the same factor arrived at a second way.

A manufactured solution converges at order **2.033**. With `psi = Y_lm` on the
unit sphere the source that returns it is `mu (mu - 2) / Er`, derived from the
corrected factorisation; a wrong operator does not converge to the manufactured
stream function at all. `l = 1` is excluded because `mu (mu - 2)` vanishes
there, those being the rigid rotations.

The Gaussian curvature the solver builds for itself, by angle defect, satisfies
discrete Gauss-Bonnet to `1e-9` at three refinements.

### Rigid-spin content of the flow

A rigid spin takes the defects around the sphere without moving them relative to
one another, so any of it in the velocity is added to every defect trajectory
and contributes nothing to the braid they write. The fraction of `|u|^2` in the
three-dimensional rotation subspace measures it, in the dual-area inner product.

The measure needs its own controls, since a blind measure also reads zero: it
returns `1.000000` on a field that is a rigid rotation, and `0.4998` on the
solver's answer plus an equal-energy spin.

On that measure the solver read `1.0e-5` before the correction and `4.0e-7`
after. **The uncorrected solver was not driving a rigid spin**, so the missing
curvature term is a quantitative error in the `l >= 2` response rather than the
reason the low-activity runs wrote no braid. At `l = 2` the two operators differ
by `(-6)(-6 + 2) = 24` against `(-6)(-6) = 36`, so the response differs by a
factor of 1.5.

### Consequences for the earlier runs

Every sphere result taken before this correction is superseded, since the flow
driving the defects came from the wrong operator at every wavelength.

## Solver consolidation

The correction was first written into `CurvedStokesSolver`, which turned out not
to be the code the sphere runs use. Three overlapping Stokes paths existed:

| type | lines | users | curvature | Killing kernel | warm start |
|---|---|---|---|---|---|
| `StokesSolverDec` | 2379 | 23, every production example | none | no | yes |
| `CurvedStokesSolver` | 595 | 6, mostly tests | yes | yes | no |
| `KillingOperatorSolver` | - | trait only | via cartan-dec | yes | n/a |

The physics was in one and the machinery in the other, and the production path
was the one missing the physics. They also disagreed on the sign of the source
and on the order of the two Poisson factors.

They are now one solver, `SurfaceStokes`, which computes its own curvature from
the mesh so the constructor signature is unchanged and every caller gains the
corrected operator. On a flat mesh the angle defect vanishes, the shift is zero,
and the operator is the one those callers had before.

`PoissonSolver` also had two solve paths, one of which skipped the kernel
projection. That is why the first corrected run came back 70 per cent rigid
spin. `solve` is now a wrapper over `solve_from`.

Names dropped the `Dec` suffix throughout: `SurfaceStokes`, `VelocityField`,
`QField`, and the modules `stokes` and `qfield`.

## Performance

At refinement 5 the Poisson iteration was 87 per cent of the run, by profile:
`apply_a` 35.8, `IChol::apply` 30.8, `solve` 21.0.

| change | effect | how it was checked |
|---|---|---|
| `dt` 0.002 to 0.05 | 25x | defect trajectories agree to 3 decimals across the range |
| warm-started solve | 1.8x | the routine existed and the example called the cold one |
| tolerance 1e-8 to 1e-6 | 1.5x | 143 to 72 CG iterations a step, `<S>` to 4 digits |

The timestep was 50 times inside the stability limits the file itself documents.


## Mixing rate

Topological entropy of the four-defect braid turned out to be the wrong
instrument. On four strands over one period the value is close to binary, and
it inherits a projection axis and a window that the flow knows nothing about.
Pooled properly over 67 windows and 12 axes it reads `0.051 +/- 0.223` with a
median of zero, while the fluid is demonstrably chaotic.

Two measurements replace it, and they answer different questions.

### Lyapunov exponent

Tracer pairs advected by the run's own velocity field under the Benettin
algorithm, renormalised each step so the separation stays linear and the log
growth accumulates. This is the exponent of the physical measure, `h_KS` by
Pesin's identity.

| measured `Pe` | `lambda` | e-folding time | drift over the last half | seeds stretching |
|---|---|---|---|---|
| 1.03 | 1.97e-3 +/- 9.7e-4 | 507 | 1.0 per cent | 97.5 per cent |
| 2.15 | 4.21e-3 +/- 1.5e-3 | 238 | 2.8 per cent | 100 per cent |
| 5.15 | 1.13e-2 +/- 2.7e-3 | 89 | - | 100 per cent |

The null control is the passive run, which has no flow and returns
`-5.6e-9 +/- 3.4e-13`.

### Ensemble topological entropy

A band wrapped around an ensemble of tracers and transported on a triangulation
whose vertices are those tracers, after E-tec (`roberts-2019-ens-top-ent`) with
the surface train-track formulation of `smith-2022-top-ent-sur`. The band is
stored as its intersection number with every edge and transported by
`E' = max(A + C, B + D) - E` at each flip, so the edge COUNT is fixed while the
weights grow. Advecting the curve itself would need exponentially many points.

| measured `Pe` | rate | drift | steps with a turned-over face |
|---|---|---|---|
| 0.49 | 2.900e-3 | 1.3 per cent | 0 |
| 1.02 | 6.276e-3 | 0.5 per cent | 0 |
| 2.13 | 1.312e-2 | 0.3 per cent | 0 |
| 5.20 | 3.256e-2 | 0.1 per cent | 0 |

All four are read at four hundred tracers. The reading settles with the
ensemble, and repeats bit for bit:

| measured `Pe` | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| 1.02 | 5.875e-3 | 6.077e-3 | 6.276e-3 | 6.247e-3 | 6.212e-3 |
| 2.13 | 1.252e-2 | 1.281e-2 | 1.312e-2 | 1.318e-2 | 1.319e-2 |
| 5.20 | 3.074e-2 | 3.195e-2 | 3.256e-2 | 3.251e-2 | 3.232e-2 |

Four hundred tracers is within one per cent of sixteen hundred at every
activity, over a sixteenfold range of ensemble size. The incremental
construction above is what makes the largest of these affordable.

### Scaling with activity

Five activities from the same relaxed tetrahedron, spanning a factor of 19 in
the measured Peclet number. `T` is the shape period of the defect
configuration, which is rotation invariant by construction and so survives the
precession that defeats a positional period.

| measured `Pe` | `T` | `h` | `lambda` | `h/lambda` | `hT` |
|---|---|---|---|---|---|
| 0.49 | 318 | 2.900e-3 | 1.066e-3 | 2.72 | 0.922 |
| 1.02 | 142 | 6.276e-3 | 1.999e-3 | 3.14 | 0.891 |
| 2.13 | 66 | 1.312e-2 | 4.196e-3 | 3.13 | 0.866 |
| 5.20 | 26 | 3.256e-2 | 1.142e-2 | 2.85 | 0.846 |
| 9.40 | 14 | 6.041e-2 | 2.190e-2 | 2.76 | 0.846 |

Both rates are linear in the activity and the period is inversely linear:

```
h       ~ Pe^+1.021        lambda ~ Pe^+1.031
T       ~ Pe^-1.051        hT     ~ Pe^-0.030
```

The product `hT` is the stretching a material line takes over one orbit of the
defects, and it moves from 0.922 to 0.846 across the whole range, an eight per
cent decline against a nineteenfold change in activity. Activity therefore sets
the clock and little else: the defects orbit faster and the fluid mixes faster
in the same proportion, and each orbit stretches a material line by about
`e^0.87`, which is a factor near 2.4.

The decline is monotone and so is a real weak trend rather than scatter. It
flattens at the top of the range, where the last two activities agree to
`0.846`.

The exponents do not rest on the weakest measurement. `Pe = 0.49` has the least
convincing period, at a repeat quality of 0.710 against 0.79 to 0.92 elsewhere,
and dropping it moves `h` to `Pe^+1.010` and `hT` to `Pe^-0.031`.

`output/scaling.png` draws the three panels, from
`volterra-fd/examples/sphere_scaling.py`.

### Defect braid against ensemble

The defect braid understates the mixing, and by more as the activity falls. At
`Pe = 0.49` the four defects precess on a near-periodic orbit whose braid has
median entropy zero, with only six per cent of windows mixing, so the scan
declines to quote a rate at all. The tracer ensemble in the same flow mixes at
`2.900e-3`. Four trajectories are too few to see what four hundred see, which
is the reason the ensemble measure replaced the braid word.

### Variational principle

The band's rate exceeds the pair separation rate by about a factor of three at
every activity. The two measure different quantities. A generic material line
grows at the TOPOLOGICAL entropy, which is what the band measures, and a typical
parcel separates at the exponent of the physical measure, and `h_top >= h_KS`
always.
The gap is complexity living on invariant sets of small measure rather than in
the bulk of the flow.

The defect braid sits below both, at `0.0505` over a period of `142`, so
`3.6e-4` per unit time. Reading the three together: four trajectories see least,
an ensemble of four hundred sees more, and neither is the same quantity as the
stretching a typical parcel feels.

## Coordinate freedom of the band

The algorithm asks the domain two questions and no others: whether a triangle is
positively oriented, and whether a point falls inside another triangle's
circumcircle. Both are signs. `Domain` in `volterra-braid` is that interface, and
a bounded region of the plane and a sphere differ ONLY in those two predicates.
On the sphere the orientation is the triple product, which is the area form read
through the ambient space, and the circumcircle test is the convex-hull side
test, since every point already lies on the sphere.

Braiding entropy is two-dimensional, and not by convention. The fundamental
group of the configuration space of `n` points in `R^3` is the symmetric group,
so there are no braids in three dimensions and no entropy to take from point
trajectories there. A three-dimensional flow is treated through two-dimensional
sections, or through the growth of material surfaces, which is a different
construction from this one.

### Construction of the initial triangulation

The band needs a Delaunay triangulation of the tracers before it can start, and
the definition alone gives a quartic algorithm: every triple of points tested
against every other point. That is what `delaunay_small` does, and at four
hundred tracers it takes 0.68 s, which caps the ensemble well below where the
entropy has settled.

`delaunay_sphere` builds the same triangulation by insertion. Points of a sphere
are the vertices of their own convex hull, so a point lies inside a face's
circumcircle exactly when it lies outside that face's plane, and inserting a
point is removing the faces it sees and joining it to the horizon they leave.
The work is quadratic in the point count.

`delaunay_plane` reuses that construction rather than repeating it. Inverse
stereographic projection sends circles of the plane to circles of the sphere and
the plane's point at infinity to the pole, so the planar triangulation is the
sphere's with the pole's own faces dropped, and the faces dropped are the convex
hull's edges. One algorithm therefore serves both domains, on the same footing
as the two predicates that support the rest of the band.

| points | insertion | brute force | ratio |
|---|---|---|---|
| 200 | 0.31 ms | 54.5 ms | 177 |
| 400 | 1.01 ms | 679 ms | 675 |
| 800 | 3.26 ms | - | - |
| 2000 | 17.8 ms | - | - |
| 5000 | 104 ms | - | - |

`delaunay_small` stays as the reference the other two are checked against.

### Validation

| check | result |
|---|---|
| the flip rule is involutive | exact, four cases |
| a sphere triangulation closes | `2V - 4` faces and `3V - 6` edges at 12, 40 and 120 points |
| the insertion is Delaunay | `2V - 4` faces and no point inside a circumcircle at 4, 12, 40, 120 and 501 points |
| the insertion agrees with the brute force | identical face sets at 12, 40, 120, 200 and 400 points |
| the planar lift agrees with the brute force | identical face sets at 8, 30 and 90 points |
| the planar lift covers the hull | `2n - 2 - h` faces at 8, 30, 90 and 400 points |
| a rigid rotation grows no band | `4.4e-16` over six turns, with zero flips |
| the golden braid returns its entropy | **0.962423 against the exact 0.962424** |

The last is the one that matters. `sigma_1 sigma_2^-1` on three strands is
pseudo-Anosov with dilatation `(3 + sqrt 5) / 2`, so its entropy
`log((3 + sqrt 5) / 2)` is exact and owes nothing to this implementation. The
per-word sequence locks onto it after the transient the band needs to align with
the unstable foliation: 0.606, 0.934, 0.958, 0.962, 0.962.
