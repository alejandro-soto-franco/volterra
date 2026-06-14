"""Smoke tests for the 3D bindings (params, field, dry runner).

Grids are 8x8x8 and step counts tiny so this stays fast in a debug build.
"""

import numpy as np
import pytest
import volterra

N = 8


def test_params3d_default_test_valid():
    p = volterra.ActiveNematicParams3D.default_test()
    assert p.validate() is None
    assert p.nx > 0 and p.ny > 0 and p.nz > 0
    assert np.isfinite(p.a_eff())
    assert np.isfinite(p.pi_number())


def test_qfield3d_construction():
    q = volterra.QField3D.random_perturbation(N, N, N, 1.0, 0.05, 5)
    assert q.nx == N and q.ny == N and q.nz == N
    assert len(q) == N * N * N
    assert np.isfinite(q.mean_s())
    assert np.isfinite(q.max_norm())


def test_run_dry_active_nematic_3d(tmp_path):
    p = volterra.ActiveNematicParams3D.default_test()
    p.nx = N
    p.ny = N
    p.nz = N
    q0 = volterra.QField3D.random_perturbation(p.nx, p.ny, p.nz, p.dx, 0.05, 9)

    n_steps, snap_every = 4, 2
    # The 3D runner records snapshots at in-loop multiples of snap_every and
    # writes their frames under out_dir (here a pytest tmp dir).
    q_final, stats = volterra.run_dry_active_nematic_3d(
        q0, p, n_steps, snap_every, str(tmp_path), False
    )
    assert isinstance(q_final, volterra.QField3D)
    assert len(q_final) == p.nx * p.ny * p.nz
    assert np.isfinite(q_final.mean_s())
    assert len(stats) >= 1
    for s in stats:
        assert np.isfinite(s.time)
        assert np.isfinite(s.mean_s)
