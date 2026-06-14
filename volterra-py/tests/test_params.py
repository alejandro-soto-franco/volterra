"""Smoke tests for ActiveNematicParams (2D) construction and derived quantities."""

import math

import pytest
import volterra


def test_default_test_is_valid():
    p = volterra.ActiveNematicParams.default_test()
    # default_test must produce a self-consistent, validatable parameter set.
    assert p.validate() is None
    assert p.nx > 0 and p.ny > 0
    assert p.dx > 0.0 and p.dt > 0.0


def test_getters_are_finite():
    p = volterra.ActiveNematicParams.default_test()
    for v in (
        p.dx, p.dt, p.k_r, p.gamma_r, p.zeta_eff, p.eta,
        p.a_landau, p.c_landau, p.lambda_, p.k_l, p.gamma_l, p.xi_l,
        p.noise_amp, p.chi_ms, p.kappa_ch, p.a_ch, p.b_ch, p.m_l,
    ):
        assert math.isfinite(v)


def test_derived_quantities():
    p = volterra.ActiveNematicParams.default_test()
    # a_eff = a_landau - zeta_eff / 2 by definition.
    assert p.a_eff() == pytest.approx(p.a_landau - p.zeta_eff / 2.0)
    # These are physical scales; they must be finite real numbers.
    for v in (p.pi_number(), p.defect_length(), p.ch_coherence_length(), p.phi_eq()):
        assert math.isfinite(v)


def test_setters_roundtrip():
    p = volterra.ActiveNematicParams.default_test()
    p.zeta_eff = 1.25
    assert p.zeta_eff == pytest.approx(1.25)
    p.noise_amp = 0.01
    assert p.noise_amp == pytest.approx(0.01)


def test_repr_mentions_class_name():
    p = volterra.ActiveNematicParams.default_test()
    assert "ActiveNematicParams" in repr(p)


def test_explicit_constructor_validates():
    # Mirror default_test's validated driving values through the explicit
    # constructor (the remaining parameters take their defaults). This keeps the
    # test independent of validate()'s exact admissibility rules.
    d = volterra.ActiveNematicParams.default_test()
    p = volterra.ActiveNematicParams(
        nx=d.nx, ny=d.ny, dx=d.dx, dt=d.dt,
        k_r=d.k_r, gamma_r=d.gamma_r, zeta_eff=d.zeta_eff, eta=d.eta,
        a_landau=d.a_landau, c_landau=d.c_landau, lambda_=d.lambda_,
        k_l=d.k_l, gamma_l=d.gamma_l, xi_l=d.xi_l,
    )
    assert p.validate() is None
    assert p.nx == d.nx and p.ny == d.ny
