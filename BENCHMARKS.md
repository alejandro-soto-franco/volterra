# Volterra Performance Benchmarks

Running document tracking performance comparisons between volterra and competing codes.

**Machine:** Fedora 43, Linux 6.19.10, AMD Renoir + NVIDIA RTX 5060, 32 cores, 30 GiB RAM.

**Date started:** 2026-04-09.

---

## 1. Passive LdG Relaxation (3D Periodic)

**Physics:** Landau-de Gennes free energy minimisation on a 3D periodic cubic lattice. Zero activity (zeta = 0), no flow. Random initial Q-tensor, evolving toward equilibrium.

**Codes tested:**
- **volterra** v0.2.0 (Rust, rayon auto-threading, Euler integrator with fused parallel molecular field)
- **open-Qmin** v0.11 (C++, single-threaded CPU, FIRE minimiser)
- **Ludwig** latest main (C, MPI + OpenMP, lattice Boltzmann D3Q19)

**Protocol:** 200 time steps (volterra, Ludwig) or 200 FIRE iterations (open-Qmin). All codes start from a random Q-tensor perturbation on a periodic NxNxN grid. Throughput measured as microseconds per lattice site per step.

### Results

| Code | N | Sites | Threads | Wall (s) | us/site/step |
|------|---|-------|---------|----------|-------------|
| volterra | 50 | 125K | rayon | 0.479 | 0.019 |
| open-Qmin | 50 | 125K | 1 | 2.443 | 0.098 |
| Ludwig | 50 | 125K | 1 | 43.58 | 1.743 |
| Ludwig | 50 | 125K | 32 | 28.45 | 1.138 |
| volterra | 100 | 1M | rayon | 5.750 | 0.029 |
| open-Qmin | 100 | 1M | 1 | 10.890 | 0.054 |
| Ludwig | 100 | 1M | 1 | 319.31 | 1.597 |
| Ludwig | 100 | 1M | 32 | 67.14 | 0.336 |

### Summary at N=100 (1M sites)

| Code | Wall-clock | vs volterra |
|------|-----------|-------------|
| volterra (rayon) | 5.8s | baseline |
| open-Qmin (1 thread) | 10.9s | 1.9x slower |
| Ludwig (32 threads) | 67.1s | 11.7x slower |
| Ludwig (1 thread) | 319.3s | 55x slower |

### Notes

- **volterra vs open-Qmin** is an apples-to-apples comparison: both compute the FD molecular field (Laplacian stencil + bulk LdG terms) per step. volterra's advantage comes from: (1) fused Laplacian + bulk computation (one pass, no intermediate allocation), (2) rayon parallel iteration across vertices.

- **Ludwig comparison is not apples-to-apples.** Ludwig runs the full D3Q19 lattice Boltzmann fluid solver at every step (collision + streaming + gradient computation + free energy update), even in passive mode. It cannot disable the fluid. The 12-55x gap reflects this extra work. For active nematics with flow, Ludwig's LBM makes the Stokes solve essentially free (embedded in LBM collision), whereas volterra needs a separate Stokes solve per step. The active-with-flow comparison would narrow this gap.

- **open-Qmin uses FIRE (energy minimiser)** while volterra uses Euler time stepping. FIRE typically converges in fewer iterations than Euler for equilibrium problems, so the total-time-to-equilibrium comparison may favour open-Qmin more than the per-step throughput suggests. A convergence-matched comparison (time to reach a target residual force) is a future benchmark.

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
| GPU acceleration | N | Y (CUDA) | Y (CUDA/HIP) |
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

## Future Benchmarks (TODO)

- [ ] Convergence-matched comparison: instrument volterra with max |dQ/dt| tracking, compare total time to reach equivalent residual as open-Qmin FIRE
- [ ] Active nematic with flow: volterra (FFT Stokes) vs Ludwig (LBM)
- [ ] Saturn ring defect: volterra vs open-Qmin (passive, colloidal sphere)
- [ ] DEC solver convergence order: error vs mesh spacing on S^2
- [ ] Large-N scaling (N=150, 200): measure cache behaviour at 3.4M and 8M sites
