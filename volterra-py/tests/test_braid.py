"""Smoke tests for the braid-group defect-trajectory analysis bindings."""

import math

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
