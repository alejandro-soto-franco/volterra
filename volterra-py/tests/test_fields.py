"""Smoke tests for QField2D / ScalarField2D and their numpy interop."""

import math

import numpy as np
import pytest
import volterra


def test_qfield_constructors_and_shape():
    nx, ny, dx = 12, 10, 1.0
    z = volterra.QField2D.zeros(nx, ny, dx)
    assert z.nx == nx and z.ny == ny and z.dx == pytest.approx(dx)
    assert len(z) == nx * ny
    # A zero Q-tensor field has zero order parameter everywhere.
    assert z.mean_order_param() == pytest.approx(0.0)

    u = volterra.QField2D.uniform(nx, ny, dx, 0.3, 0.0)
    # S = 2*sqrt(q1^2 + q2^2) = 2*0.3 for a uniform (0.3, 0.0) field.
    assert u.mean_order_param() == pytest.approx(0.6)


def test_numpy_roundtrip_is_lossless():
    nx, ny, dx = 8, 8, 1.0
    q = volterra.QField2D.random_perturbation(nx, ny, dx, 0.05, 7)
    arr = np.asarray(q.to_numpy())
    assert arr.shape == (nx * ny, 2)
    assert np.all(np.isfinite(arr))

    q2 = volterra.QField2D.from_numpy(arr, nx, ny, dx)
    arr2 = np.asarray(q2.to_numpy())
    np.testing.assert_array_equal(arr, arr2)


def test_from_numpy_rejects_wrong_shape():
    bad = np.zeros((10, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        volterra.QField2D.from_numpy(bad, 5, 2, 1.0)


def test_order_param_and_director_shapes():
    nx, ny, dx = 9, 7, 1.0
    q = volterra.QField2D.random_perturbation(nx, ny, dx, 0.05, 3)
    s = np.asarray(q.order_param())
    theta = np.asarray(q.director_angle())
    assert s.shape == (nx * ny,)
    assert theta.shape == (nx * ny,)
    assert np.all(s >= 0.0)
    # Director angle from atan2(.)/2 lies in [-pi/2, pi/2].
    assert np.all(np.abs(theta) <= math.pi / 2 + 1e-12)


def test_scalar_field_stats():
    nx, ny, dx = 6, 6, 1.0
    phi = volterra.ScalarField2D.uniform(nx, ny, dx, 0.4)
    assert phi.mean_value() == pytest.approx(0.4)
    assert phi.variance() == pytest.approx(0.0)
    assert phi.max_value() == pytest.approx(0.4)
    assert phi.min_value() == pytest.approx(0.4)
    arr = np.asarray(phi.to_numpy())
    assert arr.shape == (nx * ny,)
