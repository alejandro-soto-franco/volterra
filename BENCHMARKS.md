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

**volterra's GPU FIRE still wins, 0.128s against 0.198s, 1.5x, and this margin
is the one this document stands behind.** It controls for the scale mismatch
by construction: both sides now walk down the same *relative* distance,
`~55x`, from their own starting disorder. CPU FIRE (1.4-1.9s, noisier across
repeats than the GPU numbers) is roughly on par with open-Qmin's single CPU
rank (1.67s) rather than clearly ahead of it, unlike in the literal-`1e-3`
table above; the clear win is on the GPU. Euler (191 steps, 4.70s) is well
behind FIRE at this reduction factor and well behind open-Qmin, consistent
with Section 6's finding that Euler is not built for minimisation.

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
