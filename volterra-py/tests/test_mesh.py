"""Boundary geometry and the conforming mesh, through the Python bindings."""

import math

import numpy as np
import pytest

import volterra as v


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def circle_points(n, r=1.0, cx=0.0, cy=0.0):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def test_epitrochoid_rejects_a_non_half_integer_winding():
    with pytest.raises(ValueError):
        v.PlaneCurve.epitrochoid(q=1.7, d=0.9, r=1.0)


def test_epitrochoid_rejects_a_regularisation_outside_its_range():
    for d in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            v.PlaneCurve.epitrochoid(q=2.0, d=d, r=1.0)


def test_epitrochoid_cusp_count():
    for q, k in [(1.0, 0), (1.5, 1), (2.0, 2), (2.5, 3), (3.0, 4)]:
        c = v.PlaneCurve.epitrochoid(q=q, d=0.9, r=10.0)
        assert len(c.features) == k


def test_from_points_interpolates_its_samples():
    pts = circle_points(64, r=3.0)
    c = v.PlaneCurve.from_points(pts)
    assert c.period == 64.0
    for k in (0, 17, 63):
        assert c.point(float(k)) == pytest.approx(tuple(pts[k]), abs=1e-12)


def test_from_points_takes_any_sequence_of_pairs():
    """A list of pairs and an integer array are both walls a caller reaches for
    before they have thought about dtypes."""
    pts = circle_points(48, r=4.0)
    a = v.PlaneCurve.from_points(pts)
    b = v.PlaneCurve.from_points(pts.tolist())
    assert a.point(7.3) == pytest.approx(b.point(7.3), abs=1e-12)

    ints = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int64)
    c = v.PlaneCurve.from_points(ints)
    assert c.period == 4.0
    assert c.point(1.0) == pytest.approx((10.0, 0.0), abs=1e-12)


def test_from_points_rejects_a_degenerate_table():
    with pytest.raises(ValueError):
        v.PlaneCurve.from_points(np.array([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError):
        v.PlaneCurve.from_points(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_from_points_rejects_a_feature_off_the_parameter_range():
    with pytest.raises(ValueError):
        v.PlaneCurve.from_points(circle_points(32), features=[40.0])


def test_from_callable_matches_the_same_curve_tabulated():
    n = 256
    f = lambda u: (2.0 * math.cos(u), 1.0 * math.sin(u))
    a = v.PlaneCurve.from_callable(f, samples=n)
    b = v.PlaneCurve.from_points(
        np.array([f(2.0 * math.pi * i / n) for i in range(n)])
    )
    for u in (0.0, 13.7, 200.2):
        assert a.point(u) == pytest.approx(b.point(u), abs=1e-12)


def test_from_callable_maps_features_into_sample_indices():
    c = v.PlaneCurve.from_callable(
        lambda u: (math.cos(u), math.sin(u)), samples=360, features=[math.pi]
    )
    assert c.features == pytest.approx([180.0])


def test_from_callable_reports_a_bad_return_value():
    with pytest.raises(ValueError):
        v.PlaneCurve.from_callable(lambda u: math.cos(u), samples=16)


def test_circle_curvature_radius_is_the_radius():
    c = v.PlaneCurve.from_points(circle_points(256, r=5.0))
    for u in (10.5, 100.5, 200.5):
        assert c.curvature_radius(u) == pytest.approx(5.0, rel=2e-3)


def test_normal_points_into_the_domain():
    c = v.PlaneCurve.from_points(circle_points(64, r=2.0, cx=3.0))
    for k in range(0, 64, 7):
        x, y = c.point(float(k))
        nx, ny = c.inward_normal(float(k))
        # The centre lies inside, so the normal has a positive component
        # along the vector from the wall towards it.
        assert (3.0 - x) * nx + (0.0 - y) * ny > 0.0


def test_sample_returns_the_requested_count():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.9, r=4.0)
    s = c.sample(300)
    assert s.shape == (300, 2)


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


def test_disc_mesh_is_non_degenerate():
    c = v.PlaneCurve.epitrochoid(q=1.0, d=0.5, r=20.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.5)
    assert m.n_triangles > 100
    assert m.n_vertices == m.vertices.shape[0]
    assert m.n_triangles == m.triangles.shape[0]
    assert m.min_area > 0.0
    assert 0.0 < m.min_angle_deg < m.max_angle_deg < 180.0
    assert np.isfinite(m.vertices).all()
    assert m.triangles.min() >= 0 and m.triangles.max() < m.n_vertices


def test_boundary_vertices_come_first_and_sit_on_the_wall():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=30.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.25)
    b = m.boundary_vertices
    assert list(b) == list(range(m.n_boundary))
    xy = m.vertices[b]
    for k in range(0, m.n_boundary, 5):
        assert xy[k] == pytest.approx(c.point(m.boundary_params[k]), abs=1e-9)


def test_tangential_anchoring_on_a_smooth_wall_imposes_one_defect():
    for q, d in [(1.0, 0.5), (1.5, 0.7), (2.0, 0.8), (2.5, 0.7)]:
        c = v.PlaneCurve.epitrochoid(q=q, d=d, r=40.0)
        m = v.confined_mesh(c, h_bulk=1.2, h_min=0.3)
        charge, worst, over = m.imposed_charge(1.0)
        assert charge == pytest.approx(1.0, abs=1e-6), f"q={q} d={d}"
        assert worst < 90.0
        assert over == 0


def test_a_winding_anchoring_imposes_its_own_charge():
    c = v.PlaneCurve.epitrochoid(q=1.0, d=0.5, r=40.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.5)
    for q_anchor in (1.0, 2.0, 3.0):
        charge, _, _ = m.imposed_charge(q_anchor)
        assert charge == pytest.approx(q_anchor, abs=1e-6)


def test_a_tabulated_wall_meshes_like_the_analytic_one():
    """The nephroid, once through its formula and once through a table of its
    own points. The tabulated wall is what a caller supplies for a shape with no
    closed form, so it has to reach the same charge and comparable elements."""
    analytic = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=30.0)
    tab = v.PlaneCurve.from_points(analytic.sample(1500))

    ma = v.confined_mesh(analytic, h_bulk=1.0, h_min=0.25, seed=3)
    mt = v.confined_mesh(tab, h_bulk=1.0, h_min=0.25, seed=3)

    assert mt.imposed_charge(1.0)[0] == pytest.approx(ma.imposed_charge(1.0)[0], abs=1e-4)
    assert mt.n_triangles == pytest.approx(ma.n_triangles, rel=0.25)

    # Declaring the cusp is what closes the quality gap. The analytic curve
    # reports its cusp parameters, so the sampler grades towards them and the
    # size field follows; a bare table of the same points is treated as smooth
    # and measures about half the minimum angle, 10.6 degrees against 22.1.
    # Naming the same points as features recovers it.
    marked = v.PlaneCurve.from_points(
        analytic.sample(1500), features=[u * 1500 / (2 * math.pi) for u in analytic.features]
    )
    mm = v.confined_mesh(marked, h_bulk=1.0, h_min=0.25, seed=3)
    assert mm.min_angle_deg > 0.8 * ma.min_angle_deg, (
        f"marked {mm.min_angle_deg:.2f}, analytic {ma.min_angle_deg:.2f}, "
        f"unmarked {mt.min_angle_deg:.2f}"
    )


def test_a_wall_that_is_no_epitrochoid_meshes():
    """A five-lobed flower, which no branch of the reference lattice's
    `set_boundary` covers."""
    c = v.PlaneCurve.from_callable(
        lambda u: ((1.0 + 0.3 * math.cos(5.0 * u)) * 20.0 * math.cos(u),
                   (1.0 + 0.3 * math.cos(5.0 * u)) * 20.0 * math.sin(u)),
        samples=1200,
    )
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.4)
    assert m.min_area > 0.0
    charge, worst, over = m.imposed_charge(1.0)
    assert charge == pytest.approx(1.0, abs=1e-6)
    assert worst < 90.0
    assert over == 0


def sharp_square(m_side=40, length=20.0):
    side = np.linspace(0.0, 1.0, m_side, endpoint=False)
    return np.vstack([
        np.column_stack([side, np.zeros(m_side)]),
        np.column_stack([np.ones(m_side), side]),
        np.column_stack([1.0 - side, np.ones(m_side)]),
        np.column_stack([np.zeros(m_side), 1.0 - side]),
    ]) * length


def filleted_square(n=800, half=45.0, radius=12.0):
    pts = []
    centres = [(half - radius, half - radius), (-half + radius, half - radius),
               (-half + radius, -half + radius), (half - radius, -half + radius)]
    per = n // 8
    for k in range(4):
        cx, cy = centres[k]
        a0 = 0.5 * math.pi * k
        for i in range(per):
            a = a0 + 0.5 * math.pi * i / per
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        nx, ny = centres[(k + 1) % 4]
        a1 = a0 + 0.5 * math.pi
        sx, sy = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
        ex, ey = nx + radius * math.cos(a1), ny + radius * math.sin(a1)
        for i in range(per):
            t = i / per
            pts.append((sx + t * (ex - sx), sy + t * (ey - sy)))
    return np.array(pts)


def test_a_right_angle_corner_is_reported_as_unresolved():
    """A square turns the director by a quarter turn at each corner, which is
    the bound the winding sum needs, so the corner shows up in the diagnostics:
    the worst doubled-angle step is over 90 degrees and the count is non-zero.
    Refining the sampling does not remove it, since a corner is scale free.

    The total charge itself comes out right at 1.0000. It did not before the
    wall sampling was graded, when the square read 0.5 at any sampling density;
    grading the step so consecutive wall edges differ by at most `grade` puts
    enough samples through the corner for the wrapped increments to sum
    correctly, and the reading that remains is the per-step one."""
    c = v.PlaneCurve.from_points(sharp_square(), features=[0.0, 40.0, 80.0, 120.0])
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.5)
    assert m.n_triangles > 200
    charge, worst, over = m.imposed_charge(1.0)
    assert worst > 90.0
    assert over > 0
    assert charge == pytest.approx(1.0, abs=1e-6)

    fine = v.PlaneCurve.from_points(sharp_square(160), features=[0.0, 160.0, 320.0, 480.0])
    _, worst_fine, over_fine = v.confined_mesh(fine, h_bulk=1.0, h_min=0.5).imposed_charge(1.0)
    assert worst_fine > 90.0
    assert over_fine > 0


def test_a_filleted_square_imposes_one_defect():
    """The same domain with its corners rounded over a radius the sampling
    resolves. Every boundary step is then well inside a quarter turn."""
    c = v.PlaneCurve.from_points(filleted_square())
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.5)
    charge, worst, over = m.imposed_charge(1.0)
    assert charge == pytest.approx(1.0, abs=1e-6)
    assert worst < 90.0
    assert over == 0


def test_finer_bulk_size_gives_more_elements():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.8, r=30.0)
    coarse = v.confined_mesh(c, h_bulk=2.0, h_min=0.5)
    fine = v.confined_mesh(c, h_bulk=1.0, h_min=0.5)
    assert fine.n_triangles > coarse.n_triangles


def test_the_same_seed_gives_the_same_mesh():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.8, r=25.0)
    a = v.confined_mesh(c, h_bulk=1.0, h_min=0.4, seed=11)
    b = v.confined_mesh(c, h_bulk=1.0, h_min=0.4, seed=11)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.triangles, b.triangles)


def test_mesh_options_are_validated():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.9, r=10.0)
    with pytest.raises(ValueError):
        v.confined_mesh(c, h_bulk=1.0, h_min=2.0)
    with pytest.raises(ValueError):
        v.confined_mesh(c, h_bulk=-1.0, h_min=0.5)
    with pytest.raises(ValueError):
        v.confined_mesh(c, h_bulk=1.0, h_min=0.5, grade=1.0)
    with pytest.raises(ValueError):
        v.confined_mesh(c, h_bulk=1.0, h_min=0.5, boundary_frac=0.0)
    with pytest.raises(ValueError):
        v.confined_mesh(c, h_bulk=1.0, h_min=1e-9)


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------


def test_planar_anchoring_is_the_wall_tangent():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=30.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.3)
    d = m.anchoring_director(1.0)
    for k in range(0, m.n_boundary, 9):
        t = np.array(c.tangent(m.boundary_params[k]))
        # A director is headless, so the two agree up to sign.
        assert abs(abs(float(np.dot(d[k], t))) - 1.0) < 1e-9


def test_anchoring_amplitude_matches_the_reference_convention():
    """With `Q = s0 (m m - I/2)` the invariant is `Tr(Q^2) = s0^2 / 2`, so the
    reference's own `S = sqrt(Tr(Q^2))` diagnostic reads `s0 / sqrt(2)`."""
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.85, r=30.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.3)
    s0 = 1.4
    q = m.anchoring_q(1.0, s0)
    tr_q2 = 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    assert np.allclose(np.sqrt(tr_q2), s0 / math.sqrt(2.0))


def test_anchoring_arrays_are_shaped_by_the_boundary():
    c = v.PlaneCurve.epitrochoid(q=2.5, d=0.8, r=30.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.3)
    assert m.anchoring_q(1.0, 1.0).shape == (m.n_boundary, 2)
    assert m.anchoring_director(1.0).shape == (m.n_boundary, 2)
    assert m.boundary_normal_angles.shape == (m.n_boundary,)
    assert m.boundary_normals.shape == (m.n_boundary, 2)
    assert m.boundary_params.shape == (m.n_boundary,)


def test_normal_angle_agrees_with_the_stored_normal():
    c = v.PlaneCurve.epitrochoid(q=1.5, d=0.8, r=20.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.3)
    n = m.boundary_normals
    ang = m.boundary_normal_angles
    assert np.allclose(np.cos(ang), -n[:, 0], atol=1e-12)
    assert np.allclose(np.sin(ang), -n[:, 1], atol=1e-12)


def test_repr_reports_the_mesh_size():
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.9, r=20.0)
    m = v.confined_mesh(c, h_bulk=1.0, h_min=0.4)
    r = repr(m)
    assert "ConfinedMesh(" in r and "min_angle" in r
    assert "epitrochoid" in repr(c)
