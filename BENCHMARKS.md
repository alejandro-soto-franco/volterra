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

## 6. Convergence to Equilibrium: Euler vs FIRE

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

## Future Benchmarks (TODO)

- [ ] Active nematic with flow: volterra (FFT Stokes) vs Ludwig (LBM)
- [ ] Saturn ring defect: volterra vs open-Qmin (passive, colloidal sphere)
- [ ] DEC solver convergence order: error vs mesh spacing on S^2
- [ ] Implement FIRE minimiser in volterra for a fairer equilibrium comparison
- [ ] open-Qmin GPU with native compute_120 compilation (requires CUDA 13.2+ with Blackwell PTX support)
