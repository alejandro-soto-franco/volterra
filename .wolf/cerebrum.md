# Volterra Cerebrum

## Preferences
- Heavy doc comments on every public function
- thiserror preferred, anyhow secondary
- NO actor model: tokio::spawn + Arc<Mutex/RwLock>
- Integration tests in /tests
- Conventional commits (feat/, fix/, patch/)
- All years are 2026, never 2025
- No Co-Authored-By in commits
- No em dashes anywhere
- `\ell` for math l in all contexts

## Learnings
- cartan v0.3.0 is the current dependency version
- ActiveNematicParams (not MarsParams) is the canonical name post-rename
- S^2 simulation uses icosphere refinement level 5 (10,242 vertices)
- Connection Laplacian automatically handles curvature via holonomy angles
- Semi-Lagrangian advection uses BVH + RK4 backward trace + deformation gradient pullback
- Normalization forces |z|=1 (S=2) at every vertex every timestep

## Do-Not-Repeat
- DEC Laplacian is POSITIVE-semidefinite (positive at maxima). Elastic smoothing in molecular field must use -K*lap, not +K*lap.
- Active force div(Q) on curved surfaces needs covariant tensor divergence (expand Q into 3D via vertex tangent frames). Never use raw (x,y) component projections on non-flat meshes.
- advect_q treats q1,q2 as scalars: only correct on flat meshes. Use advect_q_covariant (with edge_phases from ConnectionLaplacian) on curved surfaces.
