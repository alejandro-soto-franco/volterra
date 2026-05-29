"""Demo of the volterra.BraidWord Python class.

Requires the compiled extension (it is not on PyPI with these symbols yet):

    cd <workspace root>            # ~/volterra
    maturin develop --release
    .venv/bin/python volterra-braid/examples/braid_demo.py

(`uv run` will not work here: it builds a fresh env without the `volterra`
extension. Use the venv that `maturin develop` installed into.)
"""

from __future__ import annotations

import math

import volterra


def sigma1_crossing(steps: int = 12) -> list[list[tuple[float, float, int]]]:
    """A well-sampled sigma_1: two strands swap, the right one passing above.

    Greedy nearest-neighbour tracking needs the crossing sampled finely (with a
    y-separation that keeps the strands apart), or it grabs the wrong endpoints
    and sees no crossing at all. realize_braid does this internally; here we do
    it by hand for the demo.
    """
    frames = []
    for k in range(steps + 1):
        t = k / steps
        pert = 2.0 * math.sin(math.pi * t)
        a = (t, -pert, 1)            # left strand drifts right, passing low
        b = (1.0 - t, 1e-3 + pert, 1)  # right strand drifts left, passing high
        frames.append([a, b])
    return frames


def show(label: str, word: "volterra.BraidWord") -> None:
    print(f"{label}: {word}")
    print(f"    n_strands   {word.n_strands}")
    print(f"    codes       {word.codes}")
    print(f"    permutation {word.permutation()}")
    print(f"    entropy     {word.entropy():.6f}")


def main() -> None:
    show("golden (cardioid)", volterra.BraidWord(3, [-2, 1]))
    show("silver (nephroid)", volterra.BraidWord(4, [3, 1, 2, -3, -1, -2]))

    # from_frames: a single sigma_1 crossing, sampled finely enough to track.
    show("from_frames (one crossing)", volterra.BraidWord.from_frames(sigma1_crossing()))


if __name__ == "__main__":
    main()
