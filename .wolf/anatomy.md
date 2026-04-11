# Volterra Anatomy

> Active nematohydrodynamics on Riemannian manifolds. Rust workspace, v0.2.0, MSRV 1.85.

## Workspace Crates

| Crate | Role | Key Types |
|-------|------|-----------|
| `volterra` | Facade re-export | prelude |
| `volterra-core` | Traits, params, error types | `ActiveNematicParams`, `ActiveNematicParams3D`, `NematicParams` |
| `volterra-dec` | DEC on manifolds | `DecDomain`, `QFieldDec`, `ConnectionLaplacian`, `StreamFunctionStokes`, `CurvedStokesSolver` |
| `volterra-fields` | Field containers | `QField3D`, `VelocityField3D`, `ScalarField` |
| `volterra-mars` | MARS experiment presets | dimensionless groups |
| `volterra-solver` | Time integration, Stokes, defects | `ActiveNematicEngine`, `NematicField2D`, `StokesSolver` trait |
| `volterra-py` | PyO3 bindings (PyPI: `volterra-nematic`) | Python wrappers |

## Key Source Files

### volterra-core/src/
- `lib.rs` -- `ActiveNematicParams` (Pe, Er, La, Lc, lambda), error types
- `nematic_params.rs` -- `NematicParams` struct, dimensionless group computation

### volterra-dec/src/
- `connection_laplacian.rs` -- spin-2 parallel-transport Laplacian on meshes
- `curved_stokes.rs` -- `CurvedStokesSolver`: force -> curl -> vorticity -> biharmonic LDL^T -> psi -> velocity
- `domain.rs` -- `DecDomain<M>` struct
- `helfrich.rs` -- Helfrich bending energy on deformable meshes
- `mesh_gen.rs` -- icosphere, torus mesh generators
- `molecular_field_dec.rs` -- `molecular_field_conn()`: Frank elastic + Landau-de Gennes bulk
- `poisson.rs` -- sparse Poisson solver (LDL^T via sprs-ldl)
- `qfield_dec.rs` -- `QFieldDec`: Q-tensor on DEC meshes (q1, q2 per vertex)
- `semi_lagrangian.rs` -- BVH-accelerated RK4 backward trace, deformation gradient pullback
- `stokes_dec.rs` -- `StreamFunctionStokes`: active stress -> vorticity -> stream function -> velocity
- `variational.rs` -- BAOAB stochastic integrator
- `snapshot.rs` -- serialization for checkpointing

### volterra-solver/src/
- `active_nematic_engine.rs` -- `ActiveNematicEngine`: operator-split RK4 (Zhu et al. Algorithm 1)
- `engine.rs` -- legacy engine interface
- `nematic_field_2d.rs` -- `NematicField2D`: wraps QFieldDec + normalisation
- `stokes_trait.rs` -- `StokesSolver` trait (FFT spectral, stream function, Killing backends)
- `beris_3d.rs` -- Cartesian 3D Beris-Edwards
- `runner_3d.rs` -- 3D Cartesian runner
- `runner_dec.rs` -- dry DEC runner (no flow)
- `runner_dec_wet.rs` -- wet DEC runner (with Stokes)

### volterra-solver/examples/
- `sim_sphere.rs` -- S^2 active nematic (icosphere L5, Pe=1)
- `sim_torus.rs` -- torus active nematic
- `sim_wet_2d.rs` -- wet 2D flat nematic with FFT Stokes
- `engine_sphere.rs` -- ActiveNematicEngine on S^2
- `sweep_sphere.rs` -- parameter sweep on S^2
- `bench_*.rs` -- performance benchmarks

### volterra-dec/tests/
- `domain_test.rs` -- DecDomain construction tests
- `test_qfield_dec.rs` -- QFieldDec operations

### volterra-solver/tests/
- `test_active_nematic_engine.rs` -- engine integration tests
- `test_nematic_field_2d.rs` -- field normalization tests
- `test_runner_dec.rs` -- dry DEC runner tests
- `test_runner_dec_wet.rs` -- wet DEC runner tests

## Dependencies (workspace)
- cartan-{core,manifolds,geo,dec,remesh} 0.3
- nalgebra 0.33, rayon 1, thiserror 2, serde 1, rustfft 6, rand 0.9
- pyo3 0.25 + numpy 0.25 (Python bindings)

## Visualization Pipeline
- `tools/viz/pipeline.py` -- master orchestrator
- `tools/viz/render_2d.py`, `render_3d.py`, `render_surface.py`, `render_surface_pv.py`

## Test Commands
```bash
cargo test --workspace          # all tests
cargo test -p volterra-dec      # DEC crate only
cargo test -p volterra-solver   # solver crate only
cargo run --release --example sim_sphere -p volterra-solver  # S^2 sim
```
