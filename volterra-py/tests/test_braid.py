"""Smoke tests for the braid-group defect-trajectory analysis bindings."""

import math

import numpy as np

import pytest
import volterra


def test_braidword_construction_and_codes():
    bw = volterra.BraidWord(3, [1, 2, -1])
    assert bw.n_strands == 3
    assert bw.codes == [1, 2, -1]
    assert len(bw) == 3
    # exponent sum = +1 +1 -1 = 1.
    assert bw.exponent_sum() == 1
    assert "BraidWord" in repr(bw)


def test_braidword_rejects_out_of_range_generator():
    # |code| must be in [1, n_strands - 1]; code 3 is invalid for 3 strands.
    with pytest.raises(ValueError):
        volterra.BraidWord(3, [3])


def test_permutation_length():
    bw = volterra.BraidWord(4, [1, 2, 3])
    perm = bw.permutation()
    assert sorted(perm) == [0, 1, 2, 3]


def test_entropy_is_nonnegative():
    bw = volterra.BraidWord(3, [1, -2, 1, -2])
    h = bw.entropy()
    assert math.isfinite(h)
    assert h >= -1e-12


def test_topological_entropy_free_function_matches_method():
    n, codes = 3, [1, -2, 1, -2]
    h_fn = volterra.braid_topological_entropy(n, codes)
    h_method = volterra.BraidWord(n, codes).entropy()
    assert h_fn == pytest.approx(h_method)


def test_braid_word_from_frames():
    # Two strands that swap positions over three frames (a sigma_1-like crossing).
    frames = [
        [(0.0, 0.0, 1), (1.0, 0.0, -1)],
        [(0.5, 0.2, 1), (0.5, -0.2, -1)],
        [(1.0, 0.0, 1), (0.0, 0.0, -1)],
    ]
    n_strands, codes = volterra.braid_word_from_frames(frames)
    assert n_strands == 2
    assert isinstance(codes, list)
    # Round-trip through the class form.
    bw = volterra.BraidWord.from_frames(frames)
    assert bw.n_strands == 2


# ---------------------------------------------------------------------------
# Defect detection
# ---------------------------------------------------------------------------


def _core(n, m, cx, cy, s0=math.sqrt(2.0)):
    """A single disclination of charge `m`, row-major with x as the outer index,
    which is the layout both detectors read."""
    qxx = np.zeros(n * n)
    qxy = np.zeros(n * n)
    for x in range(n):
        for y in range(n):
            phi = m * math.atan2(y - cy, x - cx)
            qxx[x * n + y] = s0 * (math.cos(phi) ** 2 - 0.5)
            qxy[x * n + y] = s0 * math.cos(phi) * math.sin(phi)
    return qxx, qxy


@pytest.mark.parametrize("m,charge", [(0.5, 1), (-0.5, -1), (1.0, 1)])
def test_the_winding_detector_reads_a_core_at_a_plaquette_centre(m, charge):
    """The holonomy is a topological quantity, so it takes no threshold."""
    n = 64
    qxx, qxy = _core(n, m, 32.5, 32.5)
    found = volterra.braid_detect_defects_winding(qxx, qxy, n, n, [True] * (n * n))
    assert len(found) == 1, found
    x, y, c = found[0]
    assert c == charge
    assert abs(x - 32.5) < 1.5 and abs(y - 32.5) < 1.5


def test_a_negative_core_on_a_lattice_site_is_missed():
    """The one placement the winding detector does not read.

    With the singularity exactly on a sampled point, `Q` vanishes there and the
    director comes from `atan2(0, 0)`, which is zero rather than undefined. The
    four plaquettes meeting at that point then round inconsistently, and a
    negative core is lost where a positive one survives. A field from a solver
    never sits exactly on a site, so this is recorded rather than repaired."""
    n = 64
    on_site = volterra.braid_detect_defects_winding(
        *_core(n, -0.5, 32.0, 32.0), n, n, [True] * (n * n))
    assert on_site == []
    off_site = volterra.braid_detect_defects_winding(
        *_core(n, -0.5, 32.5, 32.5), n, n, [True] * (n * n))
    assert len(off_site) == 1 and off_site[0][2] == -1


def test_the_saddle_splay_detector_is_a_marker_rather_than_a_census():
    """`braid_detect_defects` thresholds the saddle-splay density, not an angle.

    At the scale the reference plots with, about `0.05 * S0`, a single `+1/2`
    core fragments into several components of both signs, so the count means
    nothing. The threshold is also on the field's own gradients, so the same
    number behaves differently on a sharp synthetic core and on a relaxed solver
    field: `pi / 2` finds this core and finds nothing at all on a settled run,
    which reads as a field with no defect rather than as a misused parameter.
    Use the winding detector for a census."""
    n = 64
    qxx, qxy = _core(n, 0.5, 32.5, 32.5)
    mask = [True] * (n * n)

    at_plot_scale = volterra.braid_detect_defects(qxx, qxy, n, n, 0.05, mask)
    assert len(at_plot_scale) > 1
    assert {c for _, _, c in at_plot_scale} == {1, -1}

    # The winding detector reads the one core that is actually there.
    assert len(volterra.braid_detect_defects_winding(qxx, qxy, n, n, mask)) == 1

    # And the saddle-splay count moves with the threshold, which a topological
    # count would not.
    counts = [len(volterra.braid_detect_defects(qxx, qxy, n, n, t, mask))
              for t in (0.01, 0.05, 0.5)]
    assert len(set(counts)) > 1, counts
