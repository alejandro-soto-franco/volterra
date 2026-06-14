"""Smoke tests for the 2D simulation runners and field operators.

Grids are shrunk to 16x16 so the suite stays fast even in a debug build.
"""

import numpy as np
import pytest
import volterra

NX = NY = 16


@pytest.fixture
def params():
    p = volterra.ActiveNematicParams.default_test()
    p.nx = NX
    p.ny = NY
    return p


@pytest.fixture
def q0(params):
    return volterra.QField2D.random_perturbation(params.nx, params.ny, params.dx, 0.05, 11)


def _assert_finite_field(q):
    arr = np.asarray(q.to_numpy())
    assert np.all(np.isfinite(arr)), "field developed non-finite values"


def test_run_dry_active_nematic(params, q0):
    n_steps, snap_every = 10, 5
    q_final, stats = volterra.run_dry_active_nematic(q0, params, n_steps, snap_every)
    assert isinstance(q_final, volterra.QField2D)
    # Legacy runner snapshots at steps 0, snap_every, ... n_steps inclusive.
    assert len(stats) == n_steps // snap_every + 1
    for s in stats:
        assert np.isfinite(s.time)
        assert np.isfinite(s.mean_s)
        assert s.n_defects == s.n_plus + s.n_minus
    _assert_finite_field(q_final)


def test_run_active_nematic_hydro(params, q0):
    n_steps, snap_every = 8, 4
    q_final, stats = volterra.run_active_nematic_hydro(q0, params, n_steps, snap_every)
    assert isinstance(q_final, volterra.QField2D)
    assert len(stats) == n_steps // snap_every + 1
    _assert_finite_field(q_final)


def test_stokes_solve_returns_velocity(params, q0):
    v = volterra.stokes_solve(q0, params)
    assert isinstance(v, volterra.VelocityField2D)
    arr = np.asarray(v.to_numpy())
    assert arr.shape == (params.nx * params.ny, 2)
    assert np.all(np.isfinite(arr))


def test_scan_defects_returns_list(params, q0):
    defects = volterra.scan_defects(q0)
    assert isinstance(defects, list)
    for d in defects:
        assert isinstance(d.plaquette, tuple) and len(d.plaquette) == 2
        assert d.charge_sign in (-1, 1)
        assert np.isfinite(d.angle)


def test_k0_convolution_returns_field(params, q0):
    q_lip = volterra.k0_convolution(q0, params)
    assert isinstance(q_lip, volterra.QField2D)
    assert len(q_lip) == params.nx * params.ny
    _assert_finite_field(q_lip)


def test_run_bech_and_ch_step(params, q0):
    phi0 = volterra.ScalarField2D.uniform(params.nx, params.ny, params.dx, 0.5)
    n_steps, snap_every = 6, 3
    q_final, phi_final, stats = volterra.run_bech(q0, phi0, params, n_steps, snap_every)
    assert isinstance(q_final, volterra.QField2D)
    assert isinstance(phi_final, volterra.ScalarField2D)
    assert len(stats) == n_steps // snap_every + 1
    for s in stats:
        assert np.isfinite(s.mean_phi)
        assert np.isfinite(s.phi_variance)
    _assert_finite_field(q_final)

    # One explicit CH ETD step from Python-level pieces.
    v = volterra.stokes_solve(q0, params)
    q_lip = volterra.k0_convolution(q0, params)
    phi_next = volterra.ch_step_etd(phi0, q_lip, v, params)
    assert isinstance(phi_next, volterra.ScalarField2D)
    assert np.all(np.isfinite(np.asarray(phi_next.to_numpy())))
