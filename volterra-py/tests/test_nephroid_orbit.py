"""The nephroid's periodic orbit, end to end and reproducible.

The confined nephroid at the published phase point settles into a four-defect
periodic orbit whose braid is the silver braid,

    sigma_3 sigma_1 sigma_2 sigma_3^-1 sigma_1^-1 sigma_2^-1

on four strands, with topological entropy log(3 + 2 sqrt 2) = 1.762747. This
runs it from the Python bindings and reads the invariant back.

The nephroid is the test case because its braid stabilises fastest: measured
over five seeds the defect count reaches four and stays there between t = 0.22
and t = 0.94, so a run of two time units gives four to five clean periods.

The chain each test covers, from the wall to the topological invariant:
the epicycloid's true cusp imposes a total charge of +2, which the interior
carries as four `+1/2` cores, and those four cores are the four strands.
"""

import math

import numpy as np
import pytest

import volterra as v

# The lattice-matched nephroid at Lx = 100: the epicycloid of the published
# study, at the scale that gives it the same effective system length.
RADIUS = 49.778694002
# The per-shape divisor that turns the published dimensionless lengths into
# lattice units. For the nephroid it is 0.764031 Lx.
DIVISOR = 0.764031 * 100.0
# The published phase point, (l_a, l_c) = (0.0128, 0.0766).
ACTIVE_LENGTH = 0.0128 * DIVISOR
COHERENCE_LENGTH = 0.0766 * DIVISOR

# The canonical silver word and its entropy.
SILVER = [3, 1, 2, -3, -1, -2]
SILVER_ENTROPY = math.log(3 + 2 * math.sqrt(2))

STEPS = 20_000
EVERY = 200
DT = 1e-4


def nephroid_mesh(h=2.0):
    """The epicycloid with the sharp cusp treatment.

    At `d = 1` the tip has no radius to refine towards, so the cusp is excised
    at the element size: one vertex on the exact cusp with its two boundary
    edges an element long. That is the wall whose turning number is `k/2 + 1`.

    `h = 2.0` is a measured choice. The core has to sit above about twice the
    bulk element for a defect count to mean anything, which caps `h` at
    `ncl / 2 = 2.93`. Inside that cap the answer is not monotone in `h`: at
    `h = 2.5` two seeds of three reached a different orbit, of entropy 1.9248,
    and the runs settled only in the last quarter of the window. At `h = 2.0`
    five seeds of five gave the silver braid, and settling was complete by
    `t = 0.94` in the worst of them.
    """
    curve = v.PlaneCurve.epitrochoid(q=2.0, d=1.0, r=RADIUS)
    return v.confined_mesh(curve, h_bulk=h, h_min=h, cusp_edge=h)


def orbit(mesh, seed, steps=STEPS, every=EVERY):
    """Run to the periodic state and return (frames, first stable frame)."""
    run = v.ConfinedRun(
        mesh,
        active_length=ACTIVE_LENGTH,
        coherence_length=COHERENCE_LENGTH,
        resolution=100,
        q_anchor=1.0,
        wall="noslip",
        dt=DT,
        seed=seed,
    )
    frames = []
    for _ in range(steps // every):
        run.step(every)
        frames.append([tuple(d) for d in run.defects()])
    counts = [
        (sum(1 for t in f if t[2] > 0), sum(1 for t in f if t[2] < 0)) for f in frames
    ]
    settled = next(
        (i for i in range(len(counts)) if all(c == (4, 0) for c in counts[i:])), None
    )
    return frames, settled, counts


def rotations(word):
    """Every cyclic rotation of a word, since a period has no preferred start."""
    return [tuple(word[i:] + word[:i]) for i in range(len(word))]


def silver_variants():
    """The silver word up to the one relation that reorders it.

    `sigma_1` and `sigma_3` act on disjoint strand pairs and therefore commute,
    so the canonical word and the one with that pair exchanged are the same
    braid. A run may also start anywhere in the period.
    """
    swapped = [1, 3, 2, -1, -3, -2]
    return set(rotations(SILVER)) | set(rotations(swapped))


# ---------------------------------------------------------------------------
# The wall
# ---------------------------------------------------------------------------


def test_the_cusped_wall_imposes_two_units_of_charge():
    """A re-entrant cusp has interior angle 2 pi and contributes -pi to the
    boundary's turning, so the tangent's turning number is k/2 + 1 = 2 rather
    than 1, and the interior takes that whole number."""
    m = nephroid_mesh()
    charge, worst_step, over = m.imposed_charge(q_anchor=1.0)
    assert charge == pytest.approx(2.0, abs=1e-9)
    assert worst_step < 90.0
    assert over == 0


def test_the_mesh_is_well_shaped():
    m = nephroid_mesh()
    assert m.min_angle_deg > 25.0
    assert m.min_area > 0.0


# ---------------------------------------------------------------------------
# The orbit
# ---------------------------------------------------------------------------


def test_the_orbit_settles_to_four_positive_cores():
    m = nephroid_mesh()
    _, settled, counts = orbit(m, seed=0)
    assert settled is not None, f"never settled; last counts {counts[-5:]}"
    # Measured over five seeds the latest settling was frame 47 of 100.
    assert settled < 60, f"settled only at frame {settled}"
    assert counts[-1] == (4, 0)


def test_the_braid_is_the_silver_braid():
    m = nephroid_mesh()
    frames, settled, _ = orbit(m, seed=0)
    assert settled is not None
    word = v.BraidWord.from_frames(frames[settled:])

    assert word.n_strands == 4
    period = word.fundamental_period()
    assert len(period) >= 6, f"too few generators to see a period: {period}"

    block = tuple(period[-6:])
    assert block in silver_variants(), f"period block {block} is no silver word"

    one = v.BraidWord(4, list(block))
    assert one.entropy() == pytest.approx(SILVER_ENTROPY, abs=1e-9)
    assert one.permutation() == [3, 2, 1, 0]
    assert one.exponent_sum() == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_every_seed_reaches_the_same_invariant(seed):
    """The orbit is reproducible in the topological invariant, which is the
    claim that matters: the field is chaotic, so two seeds agree on the braid
    rather than on the trajectory."""
    m = nephroid_mesh()
    frames, settled, counts = orbit(m, seed=seed)
    assert settled is not None, f"seed {seed} never settled; last {counts[-5:]}"
    word = v.BraidWord.from_frames(frames[settled:])
    block = tuple(word.fundamental_period()[-6:])
    assert block in silver_variants(), f"seed {seed}: block {block}"
    assert v.BraidWord(4, list(block)).entropy() == pytest.approx(
        SILVER_ENTROPY, abs=1e-9
    )


def test_the_same_seed_gives_the_same_worldlines():
    """Reproducible in the trajectory too, not only in the invariant."""
    m = nephroid_mesh()
    a, sa, _ = orbit(m, seed=0, steps=4000)
    b, sb, _ = orbit(m, seed=0, steps=4000)
    assert sa == sb
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa == fb


def test_the_orbit_returns_to_its_configuration():
    """The braid repeating is a statement about crossings. The positions return
    as well, which is what makes it an orbit rather than a sequence of
    exchanges.

    The cores travel round a loop, so no individual radius is constant: over one
    period each varies by about a quarter of its mean. The statement that holds
    is a return. Take the sorted distances from the four cores to their
    centroid, one vector per frame, and compare frames a fixed lag apart against
    the distance between two unrelated frames. On the run of this test the best
    lag is 17 frames at 0.375 of that baseline, while the worst lag reaches
    1.38; over a run five times longer the configuration comes back to 0.064 of
    baseline at lag 83, and the elementary period is 8.3 frames, two of which
    make one period of the braid."""
    m = nephroid_mesh()
    frames, settled, _ = orbit(m, seed=0)
    conf = []
    for f in frames[settled:]:
        p = np.array([[x, y] for x, y, _ in f])
        conf.append(np.sort(np.linalg.norm(p - p.mean(axis=0), axis=1)))
    conf = np.array(conf)
    n = len(conf)
    assert conf.shape[1] == 4
    assert n >= 40, f"only {n} stable frames"

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    baseline = np.linalg.norm(conf[i] - conf[j], axis=1).mean()
    lags = range(4, max(5, n // 3))
    best = min(
        np.linalg.norm(conf[lag:] - conf[:-lag], axis=1).mean() for lag in lags
    )
    assert best < 0.6 * baseline, f"best return {best:.3f} against baseline {baseline:.3f}"
