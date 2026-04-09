"""Integration tests for volterra 3D Python bindings."""
import tempfile
import os
import numpy as np
import volterra


def test_active_nematic_params_3d():
    p = volterra.ActiveNematicParams3D(
        nx=8, ny=8, nz=8, dx=1.0, dt=0.01,
        k_r=1.0, gamma_r=1.0, zeta_eff=2.0, eta=1.0,
        a_landau=-0.5, c_landau=4.5, b_landau=0.0, lambda_=0.95,
        k_l=0.5, gamma_l=1.0, xi_l=5.0, chi_ms=0.5,
        kappa_ch=1.0, a_ch=1.0, b_ch=1.0, m_l=0.1,
        chi_a=0.0, b0=1.0, omega_b=1.0, noise_amp=0.0,
    )
    assert p.nx == 8
    assert p.nz == 8
    assert p.defect_length() > 0
    # a_eff = a_landau - zeta_eff/2 = -0.5 - 1.0 = -1.5
    assert abs(p.a_eff() - (-1.5)) < 1e-10
    p_def = volterra.ActiveNematicParams3D.default_test()
    assert p_def.nx == 16
    print("test_active_nematic_params_3d PASSED")


def test_qfield3d():
    q = volterra.QField3D.zeros(8, 8, 8, 1.0)
    assert q.nx == 8 and q.ny == 8 and q.nz == 8
    arr = q.q
    assert arr.shape == (8, 8, 8, 5)
    assert np.allclose(arr, 0.0)

    q2 = volterra.QField3D.uniform(4, 4, 4, 1.0, 0.1, 0.0, 0.0, -0.05, 0.0)
    s = q2.scalar_order()
    assert s.shape == (64,)
    assert np.all(s >= 0)

    q3 = volterra.QField3D.random_perturbation(4, 4, 4, 1.0, 0.01, 42)
    assert q3.max_norm() > 0

    # from_numpy round-trip
    arr_in = np.zeros((64, 5), dtype=np.float64)
    arr_in[:, 0] = 0.1
    q4 = volterra.QField3D.from_numpy(arr_in, 4, 4, 4, 1.0)
    assert q4.nx == 4
    print("test_qfield3d PASSED")


def test_velocity_field_3d():
    v = volterra.VelocityField3D.zeros(4, 4, 4, 1.0)
    assert v.nx == 4
    u = v.u
    assert u.shape == (4, 4, 4, 3)
    assert np.allclose(u, 0.0)
    print("test_velocity_field_3d PASSED")


def test_scalar_field_3d():
    phi = volterra.ScalarField3D.zeros(4, 4, 4, 1.0)
    assert phi.nx == 4
    arr = phi.phi
    assert arr.shape == (64,)
    assert np.allclose(arr, 0.0)

    phi2 = volterra.ScalarField3D.uniform(4, 4, 4, 1.0, 0.3)
    assert abs(phi2.mean() - 0.3) < 1e-10

    # from_numpy round-trip
    arr_in = np.ones(64, dtype=np.float64) * 0.5
    phi3 = volterra.ScalarField3D.from_numpy(arr_in, 4, 4, 4, 1.0)
    assert abs(phi3.mean() - 0.5) < 1e-10
    print("test_scalar_field_3d PASSED")


def test_disclination_classes_importable():
    assert hasattr(volterra, 'DisclinationLine')
    assert hasattr(volterra, 'DisclinationEvent')
    print("test_disclination_classes_importable PASSED")


def test_run_dry_active_nematic_3d_dry():
    """Smoke test: run_dry_active_nematic_3d completes and returns correct types."""
    p = volterra.ActiveNematicParams3D(
        nx=8, ny=8, nz=8, dx=1.0, dt=0.01,
        k_r=1.0, gamma_r=1.0, zeta_eff=2.0, eta=1.0,
        a_landau=-0.5, c_landau=4.5, b_landau=0.0, lambda_=0.95,
        k_l=0.5, gamma_l=1.0, xi_l=5.0, chi_ms=0.5,
        kappa_ch=1.0, a_ch=1.0, b_ch=1.0, m_l=0.1,
        chi_a=0.0, b0=1.0, omega_b=1.0, noise_amp=0.0,
    )
    q0 = volterra.QField3D.random_perturbation(8, 8, 8, 1.0, 0.01, 42)
    with tempfile.TemporaryDirectory() as out_dir:
        q_fin, stats = volterra.run_dry_active_nematic_3d(q0, p, 5, 5, out_dir, False)
    assert isinstance(q_fin, volterra.QField3D)
    assert q_fin.nx == 8
    assert len(stats) == 1  # 5 steps, snap_every=5 -> 1 snapshot
    assert isinstance(stats[0], volterra.SnapStats3D)
    assert stats[0].time > 0
    print("test_run_dry_active_nematic_3d_dry PASSED")


def test_run_bech_3d():
    """Smoke test: run_bech_3d completes and returns correct types."""
    p = volterra.ActiveNematicParams3D(
        nx=8, ny=8, nz=8, dx=1.0, dt=0.01,
        k_r=1.0, gamma_r=1.0, zeta_eff=2.0, eta=1.0,
        a_landau=-0.5, c_landau=4.5, b_landau=0.0, lambda_=0.95,
        k_l=0.5, gamma_l=1.0, xi_l=5.0, chi_ms=0.5,
        kappa_ch=1.0, a_ch=1.0, b_ch=1.0, m_l=0.1,
        chi_a=0.0, b0=1.0, omega_b=1.0, noise_amp=0.0,
    )
    q0 = volterra.QField3D.random_perturbation(8, 8, 8, 1.0, 0.01, 42)
    phi0 = volterra.ScalarField3D.uniform(8, 8, 8, 1.0, 0.3)
    with tempfile.TemporaryDirectory() as out_dir:
        q_fin, phi_fin, stats = volterra.run_bech_3d(q0, phi0, p, 5, 5, out_dir, False)
    assert isinstance(q_fin, volterra.QField3D)
    assert isinstance(phi_fin, volterra.ScalarField3D)
    assert len(stats) == 1
    assert isinstance(stats[0], volterra.BechStats3D)
    assert phi_fin.nx == 8
    print("test_run_bech_3d PASSED")


if __name__ == "__main__":
    test_active_nematic_params_3d()
    test_qfield3d()
    test_velocity_field_3d()
    test_scalar_field_3d()
    test_disclination_classes_importable()
    test_run_dry_active_nematic_3d_dry()
    test_run_bech_3d()
    print("\nAll Task 17 integration tests PASSED")
