"""The confined active nematic run, through the Python bindings."""

import math

import numpy as np
import pytest

import volterra as v


def disc(r=24.0, h_bulk=1.6, h_min=1.6):
    """A disc, which is the cheapest domain that still has a wall."""
    c = v.PlaneCurve.epitrochoid(q=1.0, d=0.5, r=r)
    return v.confined_mesh(c, h_bulk=h_bulk, h_min=h_min)


def graded_nephroid(h_bulk=1.6, h_min=0.15):
    """A cusped domain, where the element size varies by an order of magnitude.
    A disc has no feature, so its mesh is uniform and nothing is graded."""
    c = v.PlaneCurve.epitrochoid(q=2.0, d=0.9, r=30.0)
    return v.confined_mesh(c, h_bulk=h_bulk, h_min=h_min)


def run_on(mesh, **kw):
    opts = dict(active_length=6.0, coherence_length=4.0, resolution=48, dt=2e-4)
    opts.update(kw)
    return v.ConfinedRun(mesh, **opts)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_a_run_reports_its_own_configuration():
    m = disc()
    r = run_on(m)
    assert r.n_steps == 0
    assert r.time == 0.0
    assert r.dt == 2e-4
    assert r.wall == "noslip"
    assert r.wall_vertices >= m.n_boundary
    assert r.q.shape == (m.n_vertices, 2)
    assert "ConfinedRun(" in repr(r)


def test_the_wall_argument_is_checked():
    m = disc()
    with pytest.raises(ValueError):
        run_on(m, wall="sticky")
    with pytest.raises(ValueError):
        run_on(m, active_length=0.0)
    with pytest.raises(ValueError):
        run_on(m, coherence_length=-1.0)
    with pytest.raises(ValueError):
        run_on(m, picard=0)
    with pytest.raises(ValueError):
        run_on(m, dt=0.0)


def test_params_are_the_length_scale_constants():
    m = disc()
    r = run_on(m, active_length=6.0, coherence_length=4.0)
    p = r.params()
    # K = 2^14 fixed, zeta = K / als^2, C = K / ncl^2, A = -C.
    assert p["k_frank"] == pytest.approx(16384.0)
    assert p["zeta"] == pytest.approx(16384.0 / 36.0)
    assert p["c_landau"] == pytest.approx(16384.0 / 16.0)
    assert p["a_landau"] == pytest.approx(-16384.0 / 16.0)
    # s0 = sqrt(-2A / C) = sqrt(2) whenever A = -C.
    assert p["s0"] == pytest.approx(math.sqrt(2.0))
    assert p["coherence_length"] == pytest.approx(4.0)
    assert p["q_anchor"] == pytest.approx(1.0)


def test_the_initial_field_is_seeded():
    m = disc()
    a = run_on(m, seed=7).q
    b = run_on(m, seed=7).q
    c = run_on(m, seed=8).q
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_wall_is_anchored_from_the_start():
    """Boundary vertices come first, and their Q is the anchored value the mesh
    reports, before any step is taken."""
    m = disc()
    r = run_on(m)
    nb = m.n_boundary
    assert np.allclose(r.q[:nb], m.anchoring_q(1.0, r.params()["s0"]), atol=1e-12)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


def test_stepping_advances_the_clock_and_moves_the_field():
    m = disc()
    r = run_on(m)
    q0 = r.q.copy()
    r.step(5)
    assert r.n_steps == 5
    assert r.time == pytest.approx(5 * 2e-4)
    assert not np.array_equal(q0, r.q)


def test_the_wall_stays_anchored_through_a_step():
    m = disc()
    r = run_on(m)
    nb = m.n_boundary
    anchored = m.anchoring_q(1.0, r.params()["s0"])
    r.step(10)
    assert np.allclose(r.q[:nb], anchored, atol=1e-10)


def test_the_velocity_vanishes_on_a_no_slip_wall():
    m = disc()
    r = run_on(m, wall="noslip")
    r.step(3)
    u = r.velocity
    nb = m.n_boundary
    assert np.abs(u[:nb]).max() < 1e-8


def test_stats_carry_the_step_diagnostics():
    m = disc()
    r = run_on(m)
    r.step(4)
    s = r.stats()
    for key in ("step", "time", "n_plus", "n_minus", "charge", "s_median",
                "speed_max", "worst_dq", "courant", "cg_iterations",
                "stokes_iterations"):
        assert key in s
    assert s["step"] == 4
    assert s["time"] == pytest.approx(4 * 2e-4)
    assert s["s_median"] > 0.0


def test_the_run_is_deterministic():
    m = disc()
    a = run_on(m, seed=3)
    b = run_on(m, seed=3)
    a.step(6)
    b.step(6)
    assert np.allclose(a.q, b.q, atol=0.0)


# ---------------------------------------------------------------------------
# The wall toggle
# ---------------------------------------------------------------------------


def test_the_two_wall_conditions_give_different_flows():
    """The clamped wall adds dpsi/dn = 0 to the simply supported one, so the same
    field drives a slower flow through it. The mean speed is the statistic that
    orders: measured over five seeds the clamped wall is slower every time, while
    the maximum, which sits on one vertex, changes places once in five."""
    m = disc()
    for seed in (1, 2, 3, 4, 5):
        slow = run_on(m, wall="noslip", seed=seed)
        fast = run_on(m, wall="freeslip", seed=seed)
        assert np.array_equal(slow.q, fast.q), "the two start from one field"

        u_slow = np.linalg.norm(slow.velocity, axis=1)
        u_fast = np.linalg.norm(fast.velocity, axis=1)
        assert u_fast.mean() > u_slow.mean(), f"seed {seed}"
        # The difference is a large fraction of the flow, rather than a nudge.
        assert np.abs(u_fast - u_slow).max() > 0.1 * u_slow.max()
    assert run_on(m, wall="noslip").wall == "noslip"
    assert run_on(m, wall="freeslip").wall == "freeslip"


def test_a_free_slip_wall_lets_the_fluid_move_along_it():
    m = disc()
    r = run_on(m, wall="freeslip")
    r.step(3)
    nb = m.n_boundary
    # psi = 0 pins the normal component, so the wall vertices are still pinned.
    # The two conditions differ in the layer of interior vertices next to the
    # wall, where the clamped form also sets the normal derivative.
    interior = np.linalg.norm(r.velocity[nb:], axis=1)
    assert interior.max() > 0.0


def test_the_wall_layer_grows_with_its_threshold():
    m = graded_nephroid()
    thin = run_on(m, wall_h=0.0)
    mid = run_on(m, wall_h=0.8)
    thick = run_on(m, wall_h=1.5)
    assert thin.wall_vertices == m.n_boundary
    assert thick.wall_vertices > mid.wall_vertices > thin.wall_vertices


# ---------------------------------------------------------------------------
# Stress and mask toggles
# ---------------------------------------------------------------------------


def test_the_stress_toggle_changes_the_flow():
    """The active term alone omits the elastic backflow that opposes it."""
    m = disc()
    full = run_on(m, full_stress=True, seed=2)
    active = run_on(m, full_stress=False, seed=2)
    assert not np.allclose(full.velocity, active.velocity)


def test_the_elastic_mask_covers_the_wall_by_default():
    m = disc(h_bulk=1.6, h_min=0.4)
    on = run_on(m, elastic_mask=True, elastic_h=1.0)
    off = run_on(m, elastic_mask=False)
    assert on.elastic_mask_vertices >= m.n_boundary
    assert off.elastic_mask_vertices == 0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_a_field_can_be_set_and_read_back():
    m = disc()
    r = run_on(m)
    q = r.q.copy()
    q[m.n_boundary:] *= 0.5
    r.set_q(q)
    assert np.allclose(r.q[m.n_boundary:], q[m.n_boundary:], atol=1e-12)
    # The wall is re-imposed rather than taken from the array.
    assert np.allclose(r.q[:m.n_boundary],
                       m.anchoring_q(1.0, r.params()["s0"]), atol=1e-12)


def test_set_q_checks_its_shape():
    m = disc()
    r = run_on(m)
    with pytest.raises(ValueError):
        r.set_q(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        r.set_q(np.zeros((m.n_vertices, 3)))


def test_reset_returns_the_run_to_a_fresh_field():
    m = disc()
    r = run_on(m, seed=1)
    q0 = r.q.copy()
    r.step(4)
    r.reset(seed=1)
    assert r.n_steps == 0
    assert np.array_equal(r.q, q0)


def test_a_passive_relaxation_lowers_the_free_energy():
    m = disc()
    r = run_on(m)
    before = r.free_energy()
    steps, last = r.relax(40)
    assert steps > 0
    assert last >= 0.0
    assert r.free_energy() < before


# ---------------------------------------------------------------------------
# Fields and diagnostics
# ---------------------------------------------------------------------------


def test_the_field_accessors_are_shaped_by_the_mesh():
    m = disc()
    r = run_on(m)
    r.step(2)
    assert r.q.shape == (m.n_vertices, 2)
    assert r.velocity.shape == (m.n_vertices, 2)
    assert r.order_parameter.shape == (m.n_vertices,)
    assert r.director_angle.shape == (m.n_vertices,)
    assert r.local_element_size.shape == (m.n_vertices,)


def test_the_order_parameter_matches_its_definition():
    m = disc()
    r = run_on(m)
    r.step(2)
    q = r.q
    assert np.allclose(r.order_parameter, np.sqrt(2.0 * (q ** 2).sum(axis=1)))


def test_the_director_angle_is_half_the_q_angle():
    m = disc()
    r = run_on(m)
    q = r.q
    assert np.allclose(r.director_angle, 0.5 * np.arctan2(q[:, 1], q[:, 0]))


def test_defects_come_back_as_positions_and_charges():
    m = disc()
    r = run_on(m)
    r.step(2)
    for x, y, charge in r.defects():
        assert charge in (-1, 1)
        assert math.isfinite(x) and math.isfinite(y)


def test_the_diffusive_limit_is_reported():
    m = disc()
    r = run_on(m)
    lim = r.diffusive_dt_limit
    p = r.params()
    h = r.local_element_size
    h_min = h[h > 0].min()
    assert lim == pytest.approx(p["gamma"] * h_min ** 2 / (4.0 * p["k_frank"]))


def test_the_run_hands_back_its_own_mesh():
    m = disc()
    r = run_on(m)
    got = r.mesh
    assert got.n_vertices == m.n_vertices
    assert got.n_triangles == m.n_triangles
    assert np.allclose(got.vertices, m.vertices)


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_a_runaway_field_raises_rather_than_being_handed_back():
    """A saturated field is a fixed point of the semi-implicit solve, so the
    change per step reads zero once it has run away and finiteness alone catches
    nothing. The bound is on the order parameter against the equilibrium the
    Landau potential sets, and the error names the step and the worst vertex."""
    m = disc()
    r = run_on(m)
    q = r.q.copy()
    q[m.n_boundary:] *= 5000.0
    r.set_q(q)
    with pytest.raises(RuntimeError, match="unstable"):
        r.step(1)


def test_the_scheme_is_stable_across_a_wide_range_of_timestep():
    """The Frank term is implicit, so the step is bounded by accuracy rather than
    by the smallest element. On a graded cusped mesh whose smallest element is a
    fortieth of the bulk, the order parameter stays at its equilibrium scale from
    1e-3 through 0.2, where an explicit scheme would be limited by the smallest
    element. At 1e-3 to 5e-2 it has settled on the equilibrium value of 1 by 150
    steps; at 0.2 it is still approaching it, so the test bounds it rather than
    pinning it."""
    m = graded_nephroid()
    for dt in (1e-3, 1e-2, 5e-2, 0.2):
        r = v.ConfinedRun(m, active_length=3.0, coherence_length=4.0,
                          resolution=48, dt=dt, wall_h=0.0, elastic_mask=False)
        r.step(150)
        s_max = r.order_parameter.max()
        assert 0.5 < s_max < 2.0, f"dt {dt}: S max {s_max}"
        if dt <= 5e-2:
            assert s_max == pytest.approx(1.0, abs=1e-6), f"dt {dt}"
