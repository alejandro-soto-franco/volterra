# Volterra Memory

## 2026-04-11: Wolf initialized + branch cleanup + S^2 debug
- Created .wolf/ directory
- Branch cleanup: deleted zhu-framework, membrane-nematic-dec, approach-b (all merged). Archived cgpo-integration (committed LBM code, removed worktree).
- S^2 engine: fixed two bugs. (1) Active force used flat (x,y) projection, now uses covariant 3D tensor divergence. (2) Elastic term had wrong sign (+K*lap should be -K*lap). DEC Laplacian is positive-semidefinite. See buglog.json.
