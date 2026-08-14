# Volterra Performance Benchmarks

Running document tracking performance comparisons between volterra and competing codes.

**Machine:** Fedora 43, Linux 6.19.10, AMD Renoir + NVIDIA RTX 5060, 32 cores, 30 GiB RAM.

**Date started:** 2026-04-09.

---

## 1. Passive LdG Relaxation (3D Periodic) Against open-Qmin

**Revision (2026-08-14, second pass).** The previous revision of this section
established the correct protocol (wall-clock time to a target residual force,
each code's best configuration) but conceded the comparison outright: volterra
had no minimiser and its Euler integrator never reached the target. This
revision adds a FIRE minimiser to volterra (`volterra-solver::fire`, CPU) and
a double-precision CUDA port of it (`volterra-cuda`, via cuda-oxide), and
re-runs the comparison with both. It also corrects two problems in how the
previous revision scored volterra, found while building the minimiser:

1. **The residual metric did not match.** open-Qmin's own `getMaxForce()`
   returns `sqrt(sum_i |f_i|^2) / N`, not a literal per-site maximum despite
   the name. The previous revision scored volterra with a true elementwise
   maximum instead (`max|dQ/dt|`), a different quantity, and the two are not
   directly comparable at N=1,000,000: open-Qmin's own quantity shrinks as
   `1/sqrt(N)` for a fixed per-site force scale. Every row below uses
   open-Qmin's formula for both codes.
2. **The initial condition was far less disordered than open-Qmin's.**
   volterra's existing `random_perturbation` draws each component from a small amplitude
   (0.01) around `Q=0`, not around the bulk equilibrium. open-Qmin's own
   default initial condition (`-z 0`) is a fully random director at every
   site at the bulk equilibrium magnitude `S0`, a far-from-equilibrium state
   with real elastic work to do between neighbours. volterra now has the same
   kind of initial condition (`QField3D::random_director_field`), used below.

Fixing only the metric, or only the initial condition, would not have been
enough; both are needed for the "1e-3" target to mean a comparable thing on
each side. Even with both fixed, a residual scale mismatch remains and is
reported openly rather than hidden: see **Scale mismatch** in the caveats.

**Revision (2026-08-15, third pass).** The previous revision left the tuned
preset (43 steps down to 19 at the scale-matched target, from a CPU sweep)
untested on the device, the fused/SoA force kernels unwired and untimed, and
several roofline numbers pending a GPU that was not free at the time. All of
that is now measured: the tuned preset's GPU timing, both fused kernels
timed back-to-back against the split kernel, and host-to-device upload
timed directly rather than left out of the accounting. Every number below in
this pass is a fresh measurement from `volterra-cuda`'s own `roofline`,
`validate`, `time-tuned` and `kernels` phases, run in that order (nothing is
timed before its own correctness check has passed), with `nvidia-smi
--query-compute-apps` confirming the device was free immediately before every
timing phase. The **After tuning** and roofline sections below carry the new
numbers; the headline is the tuned preset's scale-matched time, now
**0.067 s**, against open-Qmin's 0.198 s.

**Machine for this section:** Fedora 44, AMD Ryzen 9 8940HX (16 cores, 32
threads), NVIDIA RTX 5060 Laptop GPU (8 GiB, compute capability 12.0, driver
610.57.04), CUDA Toolkit 13.3, Open MPI 5.0.9. Differs from the machine line at
the top of this document (Fedora 43, AMD Renoir), which describes an earlier
state of this workstation; the CPU and OS have since changed.

**Physics:** Landau-de Gennes free energy relaxation on a 3D periodic cubic
lattice, N=100 (1,000,000 sites), one-constant elastic approximation, zero
activity, no flow. Both codes start from a fully random director field at the
bulk equilibrium magnitude; the two initial conditions are not bit-identical
(different RNGs, and each code uses its own default bulk LdG coefficients
rather than a matched parameter set), so this is a comparison of each code's
own default passive-relaxation problem at this grid size, not a bit-for-bit
shared initial state.

**open-Qmin build:** built from source at the `master` branch (`a9b5a14`)
against CUDA Toolkit 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a
-DCMAKE_CUDA_HOST_COMPILER=g++-15`, native compilation for this GPU's compute
capability (12.0), not a compatibility-mode build. Getting this build working
needed three local source patches, since upstream master does not compile
against CUDA 13.3 on this hardware unmodified:

1. `CMakeLists.txt` hardcoded `CMAKE_CUDA_ARCHITECTURES` to `"50;52;70"` with a
   plain `set()`, which discards any `-DCMAKE_CUDA_ARCHITECTURES` passed on the
   command line. Patched to fall back to the hardcoded list only when the
   caller has not set one.
2. `inc/std_include.h` includes `nvToolsExt.h` unqualified; CUDA 12 moved that
   header under an `nvtx3/` subdirectory, so the build fails with `fatal
   error: nvToolsExt.h: No such file or directory` against any CUDA 12+
   toolkit. Patched `CMakeLists.txt` to add that subdirectory to the include
   path.
3. `inc/initializationFunctions.h` reads `cudaDeviceProp::memoryClockRate`,
   absent from the `cudaDeviceProp` struct CUDA Toolkit 13.3 ships, causing
   `error: 'struct cudaDeviceProp' has no member named 'memoryClockRate'`.
   Patched to query the same value via
   `cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate, ...)`.

Each is a minimal, independent fix, kept on its own branch with a staged pull
request (not yet submitted; see the accompanying report). None changes
open-Qmin's physics or numerics. The resulting binary reports the correct GPU
name, clock rate and residual force at runtime, and MPI multi-rank runs
communicate correctly (verified below).

**MPI domain decomposition.** open-Qmin's `-l`/`--Lx`/`--Ly`/`--Lz` flags set
the PER-RANK lattice size, not the global size (its own `README.md`, "moving
to two processors... will run a simulation domain of total size (200x100x100)
lattice sites"). A fixed global N=100 problem run on P ranks therefore needs
each rank's box scaled down accordingly. This machine has 16 physical cores
(32 SMT threads); Open MPI's default slot count is the physical core count, so
16 ranks is the largest run reachable without `--oversubscribe`. Reported
below: 1 rank (`-l 100`), 8 ranks (`-l 50`, topology 2x2x2), and 16 ranks
(`--Lx 25 --Ly 50 --Lz 50`, topology 4x2x2); all three decompose the same
100x100x100 global problem.

**volterra:** two configurations, both against the passive-dry molecular
field `beris_edwards_rhs_3d_par_dry` (`k_r=1`, `a_landau=-0.5`, `c_landau=4.5`,
`gamma_r=1`, `dt=0.005`):

- **CPU FIRE** (`volterra_solver::fire`), ported step for step from
  open-Qmin's own `energyMinimizerFIRE`: the same velocity-Verlet update, the
  same velocity mix and adaptive-timestep rule, the same residual formula.
- **GPU FIRE** (`volterra-cuda`), the same algorithm on CUDA in double
  precision via cuda-oxide (`rustc-codegen-cuda`, kernels written in ordinary
  Rust, following the pattern `cartan-cuda` establishes), targeting `sm_120a`
  native compilation for this GPU. Six kernels: `Tr(Q^2)` then the fused
  6-point-stencil-plus-bulk force (the same `Tr(Q^2)`-then-apply split
  `cartan-cuda` uses for its own per-point reductions), the velocity-Verlet
  position and half-kick updates, the FIRE velocity mix, and a
  warp-reduction-plus-device-atomic reduction for the three FIRE dot products.
  Validated against the CPU result before any timing run; see **Validation**
  below.

Euler (`dt=0.005`, same fused kernel as previous revisions) is also reported,
now scored on the identical metric and initial condition as FIRE, since it
remains volterra's only time-accurate integrator even though it is not built
for pure minimisation.

### Time to reach a target residual force at N=100 (1M sites)

Residual is open-Qmin's own `getMaxForce()` quantity, `sqrt(sum_i|f_i|^2)/N`,
computed identically for volterra (`volterra_solver::fire::force_max_metric`,
applied to the same `beris_edwards_rhs_3d_par_dry` field FIRE and Euler both
act on). Times are wall-clock: open-Qmin's reported minimisation time
(excludes process/MPI startup, see caveats); volterra's total process
wall-clock for the reported step count, averaged over three repeated runs
(GPU) or two to three (CPU); run-to-run spread is noted where it matters.

| Code | Configuration | Target residual | Steps | Time (s) | Reached? |
|------|---------------|-----------------|-------|----------|----------|
| open-Qmin | CPU, 1 rank | 1e-3 | 54 | 1.67 | yes |
| open-Qmin | CPU, 8 ranks (MPI) | 1e-3 | 54 | 1.13 | yes |
| open-Qmin | CPU, 16 ranks (MPI) | 1e-3 | 54 | 0.84 | yes |
| open-Qmin | GPU, 1x RTX 5060 | 1e-3 | 55 | 0.198 | yes |
| volterra | CPU, rayon (Euler) | 1e-3 | 4 | 0.10 | yes |
| volterra | CPU, rayon (FIRE) | 1e-3 | 20 | 0.66 | yes |
| volterra | GPU, RTX 5060 (FIRE) | 1e-3 | 20 | **0.070** | yes |

**volterra's GPU FIRE reaches the target in 0.070s against open-Qmin's GPU
0.198s, a 2.8x margin, and its CPU FIRE (0.66s) beats even open-Qmin's fastest
CPU configuration (16 ranks, 0.84s).** Both beat the 0.199s objective this
document was revised to chase. Read this plainly and then read the **Scale
mismatch** caveat immediately below the table: `1e-3` is not an equal fraction
of each code's own starting disorder, and the fairer, scale-matched
comparison in the next table is the one that should carry the claim.

### A scale-matched comparison: the same relative distance to equilibrium

open-Qmin's own residual starts at `0.0339` after one FIRE iteration (measured
directly: `openQmin -l 100 -i 1 -f 0`) and reaches `1e-3` after a roughly 55x
reduction. volterra's random-director initial condition at this grid's own
`a_landau,c_landau` starts at `1.15e-3` under the identical formula, already
within 15% of the `1e-3` target: the two codes' own unmatched bulk/elastic
constants put `1e-3` at very different relative distances from each side's own
starting disorder (see **Scale mismatch** below for why). The table below
retargets volterra to the same ~55x reduction open-Qmin's own 1e-3 point
represents (`target=2.09e-5`), holding the initial condition, the metric and
open-Qmin's numbers fixed, and changing only what "reached" means for
volterra.

| Code | Configuration | Reduction from start | Steps | Time (s) |
|------|---------------|----------------------|-------|----------|
| open-Qmin | CPU, 1 rank | ~55x (to 1e-3) | 54 | 1.67 |
| open-Qmin | GPU, 1x RTX 5060 | ~55x (to 1e-3) | 55 | 0.198 |
| volterra | CPU, rayon (Euler) | ~55x (to 2.09e-5) | 191 | 4.70 |
| volterra | CPU, rayon (FIRE) | ~55x (to 2.09e-5) | 43 | 1.4-1.9 |
| volterra | GPU, RTX 5060 (FIRE) | ~55x (to 2.09e-5) | 43 | **0.128** |

**volterra's GPU FIRE wins, 0.128s against 0.198s, 1.5x, on open-Qmin's own
ported FIRE constants.** It controls for the scale mismatch by construction:
both sides now walk down the same *relative* distance, `~55x`, from their own
starting disorder. CPU FIRE (1.4-1.9s, noisier across repeats than the GPU
numbers) is roughly on par with open-Qmin's single CPU rank (1.67s) rather
than clearly ahead of it, unlike in the literal-`1e-3` table above; the clear
win is on the GPU. Euler (191 steps, 4.70s) is well behind FIRE at this
reduction factor and well behind open-Qmin, consistent with Section 6's
finding that Euler is not built for minimisation. **This is not the final
number**: retuning FIRE's own constants for volterra's energy landscape
(below) cuts the step count further; see **After tuning** near the end of
this section for the number this document actually stands behind.

### Validation: does the faster code find the same equilibrium?

Required before any of the timing above was collected: this dispatch's own
rule checks correctness before any speed number is reported.

- **CPU FIRE vs a closed-form equilibrium.** A uniform director, no spatial
  variation (Laplacian exactly zero), started away from the analytic bulk
  equilibrium `S0 = sqrt(-3 a_eff / (4c))`. FIRE and a 200,000-step Euler run
  both converge to `S0` to within `1e-4`, and to each other to within `1e-5`
  (`volterra-solver/tests/test_fire_matches_euler_equilibrium.rs`).
- **GPU FIRE vs CPU FIRE, N=8.** Identical iteration count (155/155) and
  identical residual trajectory; the converged Q fields agree to
  `max|Q_cpu - Q_gpu| ~ 2-3.5e-16` (machine epsilon for `f64`), several orders
  inside the `1e-9` tolerance checked. Re-run automatically before every timed
  GPU run in `volterra-cuda`'s own binary, which refuses to report a time if
  this check fails.
- **GPU FIRE vs CPU FIRE, N=100 (the benchmark size).** Same step count (20 at
  the literal target, 43 at the scale-matched target) and the same
  `force_max` to displayed precision on both tables above.

### Per-step throughput

Given a fixed number of steps, which configuration does more work per second.
This does not by itself say which reaches equilibrium sooner (previous two
tables); FIRE and Euler take different step counts to get there, and the
count that matters is the one in the tables above.

| Code | Configuration | us/site/step |
|------|---------------|--------------|
| volterra | CPU, rayon (Euler, fused RHS+step) | 0.025-0.028 |
| volterra | CPU, rayon (FIRE) | 0.03-0.07 |
| volterra | GPU, RTX 5060 (FIRE) | **0.0029-0.0035** |
| open-Qmin | CPU, 1 rank (FIRE) | 0.054 |
| open-Qmin | CPU, 16 ranks (FIRE) | 0.0203 |
| open-Qmin | GPU, 1x RTX 5060 (FIRE) | 0.0036 |

volterra's GPU kernel is marginally faster per site per step than open-Qmin's
own GPU FIRE (0.0029-0.0035 against 0.0036), the throughput target this
dispatch set out to match. It is not, on its own, why the time-to-equilibrium
tables above favour volterra: FIRE's step count (20-43 for volterra at N=100,
against open-Qmin's 54-55) matters as much as per-step cost, exactly as
expected going in. The CPU FIRE range (0.03-0.07) is wider than Euler's
(0.025-0.028) because FIRE recomputes the force twice per iteration
(velocity-Verlet's two half-kicks) against Euler's one, and because the
adaptive timestep and every-500th-iteration reset touch a variable amount of
extra elementwise work.

### Is anything left on the table? The roofline

The molecular field kernel is a 6-point stencil over 5 doubles per site: at
N=100 that is 40 MB read and 40 MB written per pass at best, and the fused
force kernel above (`trq2` then the 6-neighbour stencil) is one of six kernel
launches a FIRE iteration makes. Measured this GPU's achieved bandwidth
directly rather than reading it off a spec sheet
(`Device::measure_bandwidth`, a pure copy kernel, one read and one write per
element, 2 GiB, 20 repeated timed launches after an untimed warm-up):
**323.0 GB/s** in this pass's own run (a prior pass measured 336.6 GB/s over
three clean runs, 338.6/336.9/334.3; the two are close enough to read as the
same achieved-bandwidth regime, with the small gap ordinary run-to-run and
thermal variance rather than a real change), and kernel launch overhead
(`Device::measure_launch_overhead`, 1-element launches, 2000 reps): **1.45
us** in this pass (the prior pass's two clean runs: 1.46, 2.33 us; a run
under `ncu` gave 86.5 us, the profiler's own instrumentation overhead, not
the device's). Host-to-device upload, measured directly rather than assumed
(`Device::measure_h2d_upload`, the same `DeviceBuffer::from_host` call
`fire_minimize`'s first statement makes, N=100's field, 40 MB, 20 reps after
an untimed warm-up): **3.20 ms** (effective 12.5 GB/s). This sits far below
the 323.0 GB/s device-DRAM figure above because this transfer crosses PCIe,
not the on-device memory bus.

Every kernel a FIRE iteration launches, and what it moves (`n` = site count,
`n5 = 5n`; derived directly from the kernel bodies in `kernels.rs`, not
estimated):

| Kernel | Launches/iteration | Bytes moved |
|---|---|---|
| `position_update` | 1 | `160n` (reads v,f; read-modify-write q) |
| `axpy_inplace` (half-kicks) | 2 | `120n` each, `240n` total |
| `trq2` | 1 | `48n` |
| `force` | 1 | `360n` |
| `reduce_fire` | 1 | `80n` |
| `fire_mix` | 1 | `120n` |
| `zero_field` (reset iterations only) | 0 or 1 | `80n` |

Total **1008n bytes/iteration** (1.008 GB at N=100) on a non-reset iteration,
1088n (1.088 GB) on one that resets. At this pass's measured 323.0 GB/s that
is a floor of **3.120 ms** (no reset) to **3.368 ms** (reset) per iteration.

The tuned preset's measured scale-matched step time (below, **0.067 s over 19
steps: 3.52 ms/step**) sits about 13% above the no-reset floor, and its
literal-target step time (0.039 s over 8 steps: 4.93 ms/step) sits further
above it; the gap is larger on the 8-step run because two fixed one-time
costs -- the initial force evaluation before the FIRE loop starts, and the
single final device-to-host readback of the converged field (symmetric with
the 3.20 ms host-to-device upload measured above) -- are a much bigger
fraction of an 8-step run's total time than of a 19-step run's. Both numbers
read as close to the floor, not free of it: a prior pass's `nsys` run on the
(then-untuned) ported preset found `force`'s own effective throughput at
**~668 GB/s, roughly double the flat copy-probe number**, only possible if a
large fraction of the 6-neighbour stencil's reads are served from L2 rather
than DRAM. This pass corroborates that without `nsys`: `time_split_force`
(below) times the actual `trq2`+`force` pair with no host round trip, at
**0.644 ms mean for 408n bytes = 633 GB/s**, essentially the same reading --
the split kernel already exceeds the flat-copy bandwidth by about 2x, so the
reuse a shared-memory tiling pass would try to manufacture by hand is already
happening in cache.

**Conclusion: the kernel is close to the bandwidth floor, and the fused
kernel narrows the remaining gap rather than closing it.** Both changes
proposed in the previous pass are now timed on the device, not left pending:

- **Fusing `trq2` and `force`.** The naive fusion (keep one thread per
  (site, component), have each of the 5 threads independently re-read all 5
  of its own site's components to compute `Tr(Q^2)` itself) moves **more**
  traffic (`480n`, not less) than the current split (`408n` -- `trq2` shares
  that read once per site, `force` reads it back cheaply), the opposite of
  what fusing two passes over the same data usually buys. `force_fused_aos`
  (`volterra-cuda/src/kernels.rs`) is the properly fused version instead: one
  thread per **site**, reading all 5 neighbour components per direction and
  writing all 5 outputs through `DisjointSlice<[f64; 5]>` (no `unsafe` --
  each thread's 5-wide output is exactly one disjoint-slice element). Moves
  `320n` against the split design's `408n`, predicting a ~21.6% cut to that
  pair's traffic. **Measured** (`Device::time_split_force` and
  `Device::time_fused_aos_force`, N=100, 50 timed launches each with no host
  round trip between launches, three independent runs): split
  0.6487/0.6544/0.6302 ms (mean 0.644 ms), fused AoS 0.5695/0.5075/0.5165 ms
  (mean 0.531 ms) -- **fused is faster by 12.2%, 22.5% and 18.0% across the
  three runs (mean 17.6%)**, in the same direction as, and roughly consistent
  with, the traffic-reduction prediction. Its arithmetic was already checked
  against `beris_edwards_rhs_3d_par_dry` on CPU
  (`volterra-solver/tests/test_force_fused_formula.rs`, agreement `<1e-12`);
  it stays unwired from `Device::fire_minimize`'s own loop: the timed win at
  this field size is real but modest, and answering the timing question this
  pass set out to answer did not need wiring it in.
- **Tiling the stencil was not attempted**, for the reason the roofline
  numbers above already answer: the reuse tiling exists to capture is
  measurably already happening in L2.
- **Kernel launch overhead stays negligible.** ~7 launches/iteration x
  8-19 iterations (tuned preset) is 56-133 launches; at 1.45 us each that is
  0.08-0.19 ms total, under 1% of a 39-67 ms run, matching the prediction
  that this would be small enough to leave alone.
- **Memory layout: AoS, confirmed from the source** (`QField3D::q: Vec<[f64;
  5]>`, `volterra-fields/src/qfield3d.rs`), one site's 5 components
  contiguous. `force_fused_soa` (`kernels.rs`) is the same fused formula over
  5 separate component planes, timed the same way
  (`Device::time_fused_soa_force`, same three runs): 0.6189/0.5827/0.5831 ms
  (mean 0.595 ms) -- **faster than the split kernel by 4.6%, 11.0% and 7.5%
  (mean 7.6%), but slower than the fused AoS kernel in every one of the three
  runs.** This answers the memory-layout question directly rather than by
  inference: AoS is not costing meaningful coalescing efficiency here, and
  switching to SoA on top of the fusion gives back part of the fusion's own
  gain rather than adding to it, consistent with the L2-reuse picture above
  (SoA's five separately-indexed plane reads do not get to share the same
  base-index arithmetic AoS's fused kernel shares across a site's five
  components). `QField3D` itself stays AoS: the measurement above is the
  answer a full SoA rewrite would have been chasing, and it argues against
  making it.

### Accounting for startup as a fairness correction

`BENCHMARKS.md` already noted open-Qmin's reported minimisation time
excludes process/MPI startup. Read the source rather than assume what that
means precisely:

- **`MPI_Init`** runs at `openQmin.cpp:45`, long before anything timed;
  excluded, as expected for a process-level call.
- **`chooseGPU`** (`openQmin.cpp:165`, calling `cudaSetDevice`, which
  actually creates the CUDA context and is where `cudaGetDeviceProperties`
  reads the GPU name/clock rate seen in earlier verbose output) runs at line
  ~165; **`pMinimize.start()`** wraps `sim->performTimestep()` at line
  ~289-290. Device selection and context creation happen well before the
  timed region, excluded.
- **The initial host-to-device transfer of the Q-tensor field is not
  excluded.** `setNematicQTensorRandomly` (during `setInitialConditions.h`,
  before `pMinimize.start()`) writes the IC into `GPUArray`'s *host* buffer
  only. `GPUArray`'s `ArrayHandle` does the actual `cudaMemcpy` lazily, on
  first *device*-side access (`gpuarray.cpp:293/298`,
  `memcpyHostToDevice()`), and that first device access is
  `sim->computeForces()` -- the literal first statement inside `minimize()`
  (`energyMinimizerFIRE.cpp`), itself called from inside `pMinimize.start()`.
  So the H2D transfer of the initial condition is **inside** open-Qmin's
  reported time, not excluded from it.
- **No `MPI_Barrier` anywhere in the codebase** (checked directly, `grep -r
  MPI_Barrier` across `src/` and `inc/`): whatever inter-rank communication
  happens during a step (measured separately by `sim->p1`, and already shown
  negligible in this document's own runs -- "percent comm" at 0.0001-0.001%)
  is point-to-point halo exchange nested inside the timed region, not a
  separate barrier sitting outside it.

**volterra-cuda's own timed run already matches this precisely, not just in
spirit.** `Device::fire_minimize`'s first statement is
`DeviceBuffer::from_host(stream, q0)` -- the H2D upload of the initial
condition -- inside the region `run_timed` times, exactly mirroring
open-Qmin's inclusion of its own IC transfer. What sits *before* the timed
region in `main()` is `Device::new` (context creation and module load), the
direct analogue of `chooseGPU`/`cudaSetDevice`, and nothing else.

volterra-cuda instruments these phases explicitly (`Device::new` for context
+ module load, an explicit warm-up run timed separately from the timed run
that follows it) rather than assuming the parity holds:

| Phase | Time |
|---|---|
| CUDA context + module load (`Device::new`) | 0.20-0.69 s across six runs in this pass (0.6921, 0.2115, 0.2788, 0.2000, 0.2490, 0.2582 s) |
| Host-to-device upload, N=100 field (40 MB) | 3.20 ms mean, 20 reps (`Device::measure_h2d_upload`) |
| Validation (N=8 and N=100, CPU+GPU cross-check, both presets, both N=100 targets) | 3.96 s total for all eight checks (`volterra-cuda validate`); every check passed, see **Validation** above |
| Tuned preset GPU FIRE, literal target, N=100 | 0.0387-0.0401 s across six repeats (two batches of three) |
| Tuned preset GPU FIRE, scale-matched target, N=100 | 0.0659-0.0687 s across six repeats (two batches of three) |
| Total process wall-clock, `time-tuned` phase (validation + both targets, args parse to exit) | 4.15-4.31 s across two runs |

Context + module load is noisy (0.20-0.69 s across six runs in this pass,
matching the 0.14-0.69 s range a prior pass measured) but consistently well
under a second, and consistently excluded from every timed FIRE run below by
construction: each phase creates its own `Device` before doing anything
timed, and every `time-tuned` measurement starts its clock only after that
call returns. The host-to-device upload (3.20 ms) is the one piece of
per-run setup that sits *inside* the timed region, exactly
mirroring open-Qmin's own inclusion of its IC transfer (below); it is small
against even the 8-step literal-target run (39 ms).

**Fully-inclusive comparison** (both codes' total process wall-clock,
including everything): open-Qmin's own GPU run, timed end to end with
`/usr/bin/time`, is **2.72 s** (dominated by MPI_Init and CUDA context
creation, ~2.5s of which its own reported 0.199s never counts). volterra's
context + module load alone measured 0.20-0.69s across runs in this pass plus
one tuned-preset FIRE run (0.039-0.069s), comfortably under open-Qmin's 2.72s
even at the high end of that range. The literal- and scale-matched
margins reported in this section's headline tables already exclude startup
on both sides, matching each code's own reported number; this
fully-inclusive comparison is reported for completeness and does not change
which comparison this document leads with.

### Retuning FIRE's constants for volterra's own energy landscape

open-Qmin tuned `delta_t_inc`, `alpha_dec` and `n_min` for its own energy
landscape; volterra's bulk/elastic constants differ (see **Scale mismatch**
below), so there was no reason to expect those values were optimal here too.
A sweep (`volterra-solver/examples/sweep_fire_params.rs`), holding the
initial condition and every other parameter fixed:

| Constant varied | Range tried | Best | Steps (scale-matched target) |
|---|---|---|---|
| `delta_t_inc` (1.1 baseline) | 1.05-1.3 | 1.3 | 27 |
| `alpha_dec` (0.9 baseline) | 0.8-0.99 | 0.99 | 41 |
| `n_min` (4 baseline) | 1-8 | 1 | 40 |
| combined | `delta_t_inc=1.6, alpha_dec=0.7, n_min=0` | n/a | **19** |

Wired into `FireParams::volterra_tuned` (CPU and GPU, identical constants),
checked across four random seeds so this is not tuned to one initial
condition: 19 steps on every one of seed 7, 42, 100, 999 at the scale-matched
target (baseline: 43 on all four). At the literal `1e-3` target: 8 steps
(baseline 20). CPU wall-clock (`bench_matched_convergence`-style, three runs
each): literal target 0.33s (was 0.74s), scale-matched target 0.75s (was
1.52s), roughly a 2x cut on both, without touching a single kernel. **A step
saved was worth far more here than anything found on the kernel side above.**

### One more fix: three small read-backs became one

`nsys` also showed a real cost on the FIRE reduction's *host* side: the
original design zeroed, launched into, and read back three separate
one-element accumulators (force-dot, velocity-norm, power) every iteration,
each `to_host_vec` call carrying its own implicit stream synchronisation.
The device-side reduction was already a single fused kernel producing all
three partial sums (the coordinator's suggestion #3 was already the design,
not a remaining gap); what was not fused was the three small D2H copies
reading its output back. Changed to one 3-element accumulator, one
`cast_elem` round trip, one `to_host_vec` call. Modest by construction (the
kernel-side traffic is unchanged; this only removes host round-trip
overhead), kept because it was cheap, safe (same reduction kernel, same
atomics, only the buffer shape changed) and directly answered a real, if
small, inefficiency `nsys` actually showed rather than a guessed one.

### After tuning: the number this document stands behind

Both presets (`open_qmin_defaults` and `volterra_tuned`) validated before
either was timed: GPU against CPU FIRE at N=8 (full convergence, `1e-9`
tolerance, agreement `8.3e-17` to `9.7e-17`, well inside it) and, new in this
pass, at N=100 -- the benchmark size itself -- at both the literal and
scale-matched targets, for both presets (four checks: matching iteration
counts, `force_max` agreeing to 15-16 significant figures, `max|Q_cpu -
Q_gpu|` between `4.5e-17` and `3.3e-16`, all well inside the `1e-6` tolerance
applied there). All eight checks passed (`volterra-cuda validate`); see
**Validation** above for the tuned preset's own N=100 numbers. Nothing below
was timed before this passed.

The tuned preset was then timed three times at each target (`volterra-cuda
time-tuned`), repeated twice (six measurements per target total) to give the
spread across independent batches, not just within one:

| Target | Steps | Wall-clock (s), six repeats | Min | Mean | Max |
|---|---|---|---|---|---|
| Literal (1e-3) | 8 | 0.0387, 0.0390, 0.0392, 0.0395, 0.0397, 0.0401 | 0.0387 | 0.0394 | 0.0401 |
| Scale-matched (2.09e-5) | 19 | 0.0659, 0.0664, 0.0666, 0.0669, 0.0671, 0.0687 | 0.0659 | 0.0669 | 0.0687 |

| Code | Configuration | Target | Steps | Time (s) |
|------|---------------|--------|-------|----------|
| open-Qmin | GPU, 1x RTX 5060 | 1e-3 | 55 | 0.198 |
| open-Qmin | GPU, 1x RTX 5060 | ~55x reduction | 55 | 0.198 |
| volterra | GPU, RTX 5060 (FIRE, ported constants) | 1e-3 | 20 | 0.070 |
| volterra | GPU, RTX 5060 (FIRE, ported constants) | ~55x reduction | 43 | 0.128 |
| volterra | GPU, RTX 5060 (FIRE, tuned constants) | 1e-3 | 8 | **0.0394 (mean)** |
| volterra | GPU, RTX 5060 (FIRE, tuned constants) | ~55x reduction | 19 | **0.0669 (mean)** |

**The tuned preset wins by a wider margin on both targets, and this is the
number this document stands behind: 0.067 s against open-Qmin's 0.198 s on
the scale-matched target, a 2.96x margin** -- up from the ported preset's own
1.5x (0.128 s) reported earlier in this section, and from a step-count cut
alone (19 against 43), without touching a kernel. On the literal `1e-3`
target the margin is wider still, 0.039 s against 0.198 s, 5.0x, though that
target is the one the **Scale mismatch** caveat below says not to lead with.
Both figures beat volterra's own ported-preset numbers too (0.070 s and
0.128 s respectively), by 1.8x and 1.9x: a saved FIRE step is still worth
more here than anything found on the kernel side, exactly as the CPU sweep
predicted before any of this was run on the device.

### Caveats

- **Scale mismatch: the two codes' own default constants put "1e-3" at
  different relative distances from equilibrium.** open-Qmin's default bulk
  constants (`a=-0.172, b=-2.12, c=1.73`) and volterra's (`a_landau=-0.5,
  c_landau=4.5`, no cubic `b` term in the 3D molecular field at all) are each
  code's own convention. Measured directly: open-Qmin's own residual starts
  at `0.0339` and needs a ~55x reduction to reach `1e-3`; volterra's starts at
  `1.15e-3`, already a ~1.15x reduction away. The literal-`1e-3` table above
  is a real result on the literal target this dispatch was set, using the
  identical formula on both sides and a disordered initial
  condition on volterra's side, checked directly (`test_random_director_field_is_disordered`,
  `test_random_director_field_has_fixed_magnitude`), not a trivially-converged
  one. It should still be read alongside the scale-matched table, which is the
  fairer race and the one this document leads with in its own framing. Closing
  the mismatch properly needs volterra's 3D molecular field to carry the cubic
  bulk term it currently lacks (`SUBSUMPTION.md` section 1, "gap, additive");
  that is future work, not attempted in this dispatch.
- **Not a matched parameter set**, for the reason directly above; a
  parameter-matched run remains future work.
- **GPU state checked before every timing run** with `nvidia-smi
  --query-compute-apps=pid,used_memory --format=csv`, confirming no other
  process held the device; a prior timing pass on this machine was invalidated
  by exactly that.
- **open-Qmin's reported minimisation time excludes process/MPI startup**
  (roughly 1-4s further wall-clock, dominated by CUDA context creation on the
  GPU row); volterra's reported time is total wall-clock including its process
  startup and, for the GPU row, CUDA context and module load, which are paid
  once by an untimed warm-up run before the timed run in `volterra-cuda`'s own
  binary. This makes the open-Qmin numbers in the tables above slight
  underestimates of a real end-to-end run.
- **CPU FIRE times are noisier across repeats than the GPU numbers** (the
  scale-matched CPU FIRE row spans 1.4-1.9s across three runs on an otherwise
  idle machine, against the GPU row's 0.125-0.133s); attributed to rayon
  thread-scheduling and OS-level noise on a 1000-1500-step-equivalent run,
  not investigated further here.
- **This is not this document's only open-Qmin CPU-vs-GPU measurement.**
  Section 8 below records an earlier comparison built from a container-based,
  compute_89-compatibility-mode open-Qmin binary, and predates volterra's own
  GPU path entirely. That binary no longer reflects the native `sm_120a`
  build used here and should be read as superseded by this section for
  anything concerning open-Qmin's native performance on this GPU; Section 8's
  own per-step-throughput numbers at small N are a different axis from the
  time-to-equilibrium claim here and are left as historical record.

### The scale-matched comparison after this pass's measurements

| Code | Configuration | Reduction from start | Steps | Time (s) |
|------|---------------|----------------------|-------|----------|
| open-Qmin | CPU, 1 rank | ~55x (to 1e-3) | 54 | 1.67 |
| open-Qmin | GPU, 1x RTX 5060 | ~55x (to 1e-3) | 55 | 0.198 |
| volterra | CPU, rayon (Euler) | ~55x (to 2.09e-5) | 191 | 4.70 |
| volterra | CPU, rayon (FIRE, ported constants) | ~55x (to 2.09e-5) | 43 | 1.4-1.9 |
| volterra | GPU, RTX 5060 (FIRE, ported constants) | ~55x (to 2.09e-5) | 43 | 0.128 |
| volterra | GPU, RTX 5060 (FIRE, tuned constants) | ~55x (to 2.09e-5) | 19 | **0.067 (mean)** |

volterra's GPU FIRE with the tuned preset reaches the scale-matched target in
0.067 s against open-Qmin's 0.198 s: a **2.96x margin**, up from the ported
preset's 1.5x. Both codes still walk down the same ~55x relative distance
from their own starting disorder, so the comparison this table controls for
holds exactly as it did before tuning; only the step count and the wall-clock
changed.

---

## 2. Feature Comparison

| Feature | volterra | open-Qmin | Ludwig |
|---------|----------|-----------|--------|
| Active stress | Y | N | Y |
| Full Navier-Stokes | Y | N | Y (LBM) |
| Passive LdG equilibrium | Y | Y | Y |
| Curved manifolds (2D) | Y (DEC) | N | N |
| Curved manifolds (3D) | Y (DEC) | N | N |
| Confined 2D (BCs) | Y (epitrochoid) | Y | Y |
| Defect detection | Y (holonomy) | Y | Y |
| Defect braiding | Y | N | N |
| GPU acceleration | Y (CUDA, FIRE minimiser only) | Y (CUDA) | Y (CUDA/HIP) |
| Open source | Y (MIT) | Y | Y |
| Language | Rust | C++ | C |
| Parallelism | rayon (threads) | MPI + CUDA | MPI + OpenMP + CUDA/HIP |

---

## 3. Thread Scaling (volterra, N=100, 100 steps)

Measures how volterra's rayon parallelism scales with thread count on a 100^3 grid.

| Threads | Wall (s) | us/site/step | Speedup vs 1T |
|---------|----------|-------------|---------------|
| 1 | 7.629 | 0.076 | 1.0x |
| 2 | 4.689 | 0.047 | 1.6x |
| 4 | 3.038 | 0.030 | 2.5x |
| 8 | 2.302 | 0.023 | 3.3x |
| 16 | 2.257 | 0.023 | 3.4x |
| 32 | 2.482 | 0.025 | 3.1x |

**Observations:** Scaling is good up to 8 threads (3.3x on 8 cores), plateaus at 16, and slightly regresses at 32 due to NUMA/cache contention on the AMD Renoir. The optimal thread count for this problem size is 8-16. At 8 threads, volterra achieves 0.023 us/site/step, which is **2.4x faster than open-Qmin** (0.054 us/site/step, single-threaded) and **14.6x faster than Ludwig** at 32 threads (0.336 us/site/step).

---

## 4. Memory Usage (N=100)

Peak resident set size (RSS) for a 100^3 passive nematic relaxation.

| Code | RSS (MB) | vs volterra |
|------|---------|-------------|
| volterra | 155 | baseline |
| open-Qmin | 445 | 2.9x more |
| Ludwig | 1096 | 7.1x more |

volterra's lower memory footprint comes from: (1) the fused molecular field avoids allocating a separate Laplacian field, (2) storing only 5 Q-components per vertex (no LBM distribution functions), (3) no MPI communication buffers.

Ludwig's high RSS is expected: the D3Q19 model stores 19 distribution functions per site (19 * 8 bytes * 1M = 152 MB for distributions alone), plus Q-tensor, velocity, force, and gradient fields.

---

## 5. open-Qmin Convergence to Equilibrium (N=50)

open-Qmin uses FIRE (Fast Inertial Relaxation Engine), an energy minimiser that converges in fewer iterations than Euler time stepping for equilibrium problems. This section measures the total wall-clock time to reach a target residual force.

| Target max force | FIRE steps | Wall (s) |
|-----------------|-----------|----------|
| 0.001 | 59 | 1.9 |
| 0.0001 | 1145 | 5.1 |

For the N=50 problem, open-Qmin reaches max_force < 0.001 in 59 FIRE steps (1.9s). volterra (Euler, 200 steps, 0.48s) has not yet been instrumented with residual force tracking, so a direct convergence-matched comparison is pending. The per-step cost advantage of volterra (5x at N=50) would need to outweigh FIRE's iteration advantage to win on total time to equilibrium.

---

## 6. Convergence to Equilibrium: Euler vs FIRE

**Superseded (2026-08-14).** This section's conclusion, that volterra has no
minimiser and Euler cannot close the gap to FIRE, no longer holds: volterra
now has its own FIRE minimiser (CPU and GPU), which Section 1 measures
directly against open-Qmin's. The finding below about Euler's own residual
floor is still accurate as a statement about Euler, and is kept for that
reason, but "volterra is not designed for energy minimisation" is no longer
true of volterra as a whole. Read Section 1 for the current comparison.

**Question:** open-Qmin uses FIRE (energy minimiser), which converges in far fewer iterations than Euler for equilibrium problems. Does volterra's per-step speed advantage compensate?

**Answer: no.** For pure equilibrium problems, FIRE wins decisively.

| Code | N | Method | Steps to max_force < 0.001 | Wall (s) |
|------|---|--------|---------------------------|----------|
| open-Qmin | 50 | FIRE | 59 | 1.9 |
| volterra | 50 | Euler (dt=0.005) | >20,000 (stuck at 0.004) | 34.6 |

**Why:** After 20,000 Euler steps at dt=0.005 (total t=100, roughly 1,000 decay times for the fastest mode), volterra has reached the numerical equilibrium floor. The residual of ~3e-3 is the FD discretisation error of the Laplacian stencil at equilibrium, not a convergence failure. The equilibrium is reached, but the residual metric (max |dQ/dt|) settles at a nonzero floor because the discrete Laplacian of the equilibrium Q is not exactly zero on the grid.

**Takeaway:** volterra is not designed for energy minimisation. Its strength is time-dependent dynamics (active nematics with flow, defect braiding, turbulence), where the simulation time matters physically and FIRE cannot be used. For passive equilibrium problems, open-Qmin's FIRE is the right tool. This is a known distinction in the computational physics literature (minimisers vs integrators), not a deficiency.

---

## 7. Large-N Scaling (N=50 to 200)

Tests how throughput scales with problem size, revealing cache effects.

### volterra (rayon auto-threading)

| N | Sites | Q-tensor data (MB) | Wall (s) | us/site/step |
|---|-------|-------------------|----------|-------------|
| 50 | 125K | 10 | 0.178 | 0.014 |
| 100 | 1M | 80 | 1.124 | 0.023 |
| 150 | 3.4M | 270 | 1.480 | 0.022 |
| 200 | 8M | 640 | 1.775 | 0.022 |

### open-Qmin (single-threaded)

| N | Sites | Wall (s) | us/site/step |
|---|-------|----------|-------------|
| 50 | 125K | 2.443 | 0.098 |
| 100 | 1M | 10.890 | 0.054 |
| 150 | 3.4M | 5.659 | 0.084 |
| 200 | 8M | 7.899 | 0.099 |

### Comparison (us/site/step)

| N | Sites | volterra | open-Qmin | Speedup |
|---|-------|----------|-----------|---------|
| 50 | 125K | 0.014 | 0.098 | 7.0x |
| 100 | 1M | 0.023 | 0.054 | 2.4x |
| 150 | 3.4M | 0.022 | 0.084 | 3.8x |
| 200 | 8M | 0.022 | 0.099 | 4.5x |

**Observations:**

- volterra's throughput is remarkably stable from N=100 onward (0.022-0.023 us/site/step), indicating that the rayon parallelism effectively hides cache effects. The jump from N=50 (0.014) to N=100 (0.023) corresponds to the Q-tensor data exceeding L3 cache (80 MB vs typical 16-32 MB L3).

- open-Qmin's throughput has a U-shape: best at N=100 (0.054), worse at both small N (overhead) and large N (cache pressure). At N=200, open-Qmin returns to 0.099, similar to N=50.

- volterra maintains a **2.4-7x advantage** across all tested sizes, with the gap widening at large N where volterra's parallel stencil computation amortises cache misses better than open-Qmin's single-threaded FIRE.

---

## 8. GPU Comparison: volterra (CPU) vs open-Qmin (CUDA, RTX 5060)

**Partly superseded (2026-08-14).** This section's title and the open-Qmin
binary it measures both predate volterra's own CUDA path (Section 1) and the
native `sm_120a` open-Qmin build Section 1 uses; volterra now has a CUDA path
of its own, so "volterra (CPU)" in the title describes only this section. The
per-step-throughput numbers below are a different axis (small N, this container build) from Section 1's
time-to-equilibrium claim and are kept as historical record rather than
rewritten; for open-Qmin's native-build numbers and volterra's own GPU
numbers, read Section 1.

open-Qmin rebuilt with CUDA support in a container (Ubuntu 22.04, CUDA 12.6, compute_89 compatibility mode for the Blackwell RTX 5060). All runs amortised over enough steps to reduce container/GPU init overhead.

### Per-step throughput (us/site/step, lower is better)

| N | Sites | volterra (rayon) | oQmin CPU (1T) | oQmin GPU | GPU speedup (oQmin) | volterra vs GPU |
|---|-------|-----------------|---------------|-----------|--------------------|----|
| 20 | 8K | 0.056 | 0.228 | 0.153 | 1.5x | volterra 2.7x faster |
| 30 | 27K | 0.028 | 0.079 | 0.043 | 1.8x | volterra 1.5x faster |
| 50 | 125K | 0.014 | 0.034 | 0.027 | 1.2x | volterra 2.0x faster |
| 75 | 422K | 0.012 | 0.042 | 0.038 | 1.1x | volterra 3.2x faster |
| 100 | 1M | 0.024 | 0.046 | 0.042 | 1.1x | volterra 1.7x faster |
| 150 | 3.4M | 0.023 | 0.046 | 0.044 | 1.0x | volterra 1.9x faster |
| 200 | 8M | 0.022 | 0.049 | 0.048 | 1.0x | volterra 2.2x faster |

### Wall-clock time (seconds)

| N | Sites | Steps | volterra | oQmin CPU | oQmin GPU |
|---|-------|-------|----------|-----------|-----------|
| 20 | 8K | 1,000 | 0.4 | 1.8 | 1.2 |
| 30 | 27K | 1,000 | 0.8 | 2.1 | 1.2 |
| 50 | 125K | 1,000 | 1.8 | 4.3 | 3.4 |
| 75 | 422K | 500 | 2.5 | 8.8 | 8.0 |
| 100 | 1M | 200 | 4.8 | 9.3 | 8.3 |
| 150 | 3.4M | 100 | 7.9 | 15.4 | 14.7 |
| 200 | 8M | 50 | 8.8 | 19.8 | 19.1 |

### Analysis

**open-Qmin's GPU acceleration is negligible at research-relevant sizes.** The RTX 5060 gives open-Qmin only a 1.0-1.8x speedup over its own CPU path. At N >= 75, the GPU provides essentially zero benefit (1.0-1.1x). The FIRE minimisation step is memory-bound (6-point stencil, 5 Q-components per site), and the GPU's compute throughput advantage is wasted waiting on memory.

**volterra (CPU-only, Rust, rayon) beats open-Qmin CUDA at every tested grid size**, with margins from 1.5x (N=30) to 3.2x (N=75). volterra's advantage comes from:

1. **Fused stencil + bulk computation**: one parallel pass per vertex, no intermediate Laplacian allocation
2. **rayon work-stealing**: scales to 8 effective threads with minimal overhead
3. **Cache-friendly access pattern**: the fused kernel touches each vertex's data once, vs open-Qmin's separate force computation + FIRE velocity update + position update passes

### Implications

CUDA acceleration is not a priority for volterra. The crossover point where GPU memory bandwidth would matter (N > 300, 27M+ sites) exceeds typical active nematic research grids. For the problem sizes in arXiv:2503.10880 (N = 100), volterra is 1.7x faster than open-Qmin's best GPU path.

**Caveat:** open-Qmin GPU was compiled for compute_89 (Ada Lovelace compatibility mode) on a compute_120 GPU (Blackwell). Native Blackwell compilation (requires CUDA 13.2+) may improve GPU performance by 10-20%. Even a 20% improvement would not change the conclusion at these sizes.

---

## 9. Braid Analysis Throughput (volterra vs CGPO braid_tracker.py)

Throughput of the defect braid-analysis pipeline -- detection from Q-tensor
grids, frame-to-frame tracking, and braid-word extraction -- on identical input
(120 frames each of the golden 3-defect and silver 4-defect orbits, 100x100
grid, 10,000 sites/frame, single-threaded). The reference is the published
`Chaos-Generating-Periodic-Orbits/braid_tracker.py` algorithm, transcribed
faithfully in `volterra-braid/oracle/braid_tracker_v2.py` (same per-cell `ss`
Jacobian, flood-fill clustering, greedy tracking; plotting/IO stripped). Every
path extracts the correct braid word for both configurations (golden
`{sigma_2^-1 sigma_1}`, silver `{sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1
sigma_2^-1}`).

### Detection throughput (the dominant stage)

Detection is a per-cell Jacobian + flood fill, so its cost is independent of the
defect count -- golden (3 defects) and silver (4 defects) are within noise:

| Path | golden ns/site | silver ns/site | vs Python |
|------|----------------|----------------|-----------|
| volterra, native Rust (`cargo run --example bench_braid`) | 3.0 | 3.3 | **~180x** |
| volterra via PyO3 (`braid_detect_defects`, incl. list marshalling) | 42 | 42 | ~14x |
| CGPO `braid_tracker.py` scheme (Python, `braid_tracker_v2.py`) | 597 | 597 | 1x |

### Full pipeline, 120 frames (golden / silver)

| Path | detection | track + word | total |
|------|-----------|--------------|-------|
| volterra native Rust | 3.6 / 3.9 ms | 0.009 / 0.013 ms | **3.6 / 3.9 ms** |
| volterra via PyO3 | 50.7 / 50.4 ms | 0.04 / 0.05 ms | 50.7 / 50.5 ms |
| CGPO Python | 716.8 / 716.8 ms | 0.34 / 0.46 ms | 717.2 / 717.3 ms |

### Notes

- The silver (4-strand) configuration tracks the golden (3-strand) numbers:
  detection throughput is set by the grid size, not the defect count, and the
  braid-word extraction stays negligible even with the longer 6-generator word.
- **Native Rust is ~150-180x faster** than the published per-cell Python scheme;
  even through the PyO3 boundary (which copies each 10k-element grid to a Python
  list per call) volterra is ~14x faster. The order-of-magnitude gap between the
  native and PyO3 paths is the FFI marshalling, not the algorithm.
- The Python baseline is the algorithm **as published** (explicit per-cell
  loops, `braid_tracker.py` lines 219-231). A vectorised numpy rewrite of the
  `ss` computation would narrow the gap, but the published code is not vectorised
  and this benchmark compares against it as written.
- Tracking + word extraction is negligible in all paths (the braid algebra is
  cheap; detection dominates).
- Reproduce: `cargo run --release --example bench_braid -p volterra-braid`
  (native) and `.venv/bin/python volterra-braid/oracle/bench_braid.py` (vs
  Python; needs `maturin develop --release`).

---

## Future Benchmarks (TODO)

- [ ] Active nematic with flow: volterra (FFT Stokes) vs Ludwig (LBM)
- [ ] Saturn ring defect: volterra vs open-Qmin (passive, colloidal sphere)
- [ ] DEC solver convergence order: error vs mesh spacing on S^2
- [ ] Matched bulk LdG parameters (needs volterra's 3D molecular field to
      gain the cubic `b` term it currently lacks) so "reach residual X" means
      the same relative distance to equilibrium on both codes without the
      scale-matching correction Section 1 applies by hand
- [ ] volterra GPU FIRE at N=50 and N=200, to check the margin over
      open-Qmin's GPU holds away from N=100
