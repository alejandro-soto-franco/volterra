#!/usr/bin/env python3
"""Analysis of the periodic active-nematic runs, against the two papers.

Reads a run directory written by `examples/periodic_active_nematic.rs` and
reports the quantities the papers state:

Mitchell, Sabbir, Geumhan, Smith, Klein and Beller, Phys. Rev. E 109, 014606
(2024): whether the late state is periodic, its period T, the steady defect
count, the dimensionless topological entropy h_tilde = h t_a, and the braid
prediction h_tilde_max = log(phi + sqrt(phi)) / (T_tilde / 4).

Mitchell, Sabbir, Klein and Beller, arXiv:2506.20996: RMS and median of the two
director rotation rates over the last N frames.

Usage:  uv run --with numpy analyse_periodic.py <run-dir> [<run-dir> ...]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0
# Maximal topological entropy per operation for the braid of Fig. 2a.
H_TEPO = math.log(PHI + math.sqrt(PHI))


def read_stats(run):
    rows = [l.strip().split(",") for l in (run / "stats.csv").read_text().splitlines()]
    head, body = rows[0], rows[1:]
    cols = {name: i for i, name in enumerate(head)}
    out = {}
    for name, i in cols.items():
        out[name] = np.array([float(r[i]) if r[i] not in ("", "NaN") else np.nan
                              for r in body])
    return out


def dominant_period(t, y):
    """Period of `y(t)` from the first interior maximum of its autocorrelation.

    Returns `(T, peak)`, with `peak` the normalised autocorrelation at `T`: a
    value near one is a periodic signal, a small one is not.
    """
    y = y - y.mean()
    if np.allclose(y, 0.0):
        return float("nan"), 0.0
    n = len(y)
    ac = np.correlate(y, y, mode="full")[n - 1:]
    ac /= ac[0]
    dt = t[1] - t[0]
    # First index past the initial descent, so lag 0 is not the answer.
    i = 1
    while i < len(ac) - 1 and ac[i] < ac[i - 1]:
        i += 1
    best, best_v = None, -2.0
    for j in range(i, len(ac) - 1):
        if ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1] and ac[j] > best_v:
            best, best_v = j, ac[j]
            break
    if best is None:
        return float("nan"), float(ac[i:].max() if len(ac) > i else 0.0)
    return float(best * dt), float(best_v)


def report(run: Path):
    cfg = json.loads((run / "config.json").read_text())
    p = cfg["params"]
    s = read_stats(run)
    t_a = cfg["active_time"]
    ell_a = math.sqrt(p["k_elastic"] / p["zeta"])
    lock = p.get("locking")

    print(f"\n=== {run.name}")
    print(f"    {p['lx']}x{p['ly']} periodic, ell_a = {ell_a:.3f}, ell_n = "
          f"{cfg['ell_n']:.3f}, S_eq = {p['s0']:.4f}, "
          f"locking = {'sigma=' + str(lock['sigma']) if lock else 'off'}")
    print(f"    t_a = {t_a:.6f}, run to t = {s['t'][-1]:.3f} "
          f"({s['t'][-1] / t_a:.0f} t_a)")

    # Late window: the last third of the run.
    t = s["t"]
    late = t >= t[-1] - (t[-1] - t[0]) / 3.0
    npl, nmi = s["n_plus"][late], s["n_minus"][late]
    print(f"    late defects: +1/2 {npl.mean():.2f} +- {npl.std():.2f} "
          f"(mode {int(np.bincount(npl.astype(int)).argmax())}), "
          f"-1/2 {nmi.mean():.2f} +- {nmi.std():.2f}")
    print(f"    late rms u  : {s['rms_u'][late].mean():.4f} "
          f"+- {s['rms_u'][late].std():.4f} "
          f"(relative spread {s['rms_u'][late].std() / s['rms_u'][late].mean():.4f})")
    print(f"    max |div u| : {np.nanmax(s['max_div_u']):.3e}")

    T, peak = dominant_period(t[late], s["rms_u"][late])
    if peak > 0.5 and np.isfinite(T):
        t_tilde = T / t_a
        print(f"    PERIODIC    : T = {T:.4f} (autocorrelation {peak:.4f}), "
              f"T_tilde = T / t_a = {t_tilde:.1f}")
        print(f"    braid prediction h_tilde_max = log(phi + sqrt phi) / (T_tilde / 4)"
              f" = {H_TEPO / (t_tilde / 4.0):.4e}")
    else:
        print(f"    APERIODIC   : best autocorrelation {peak:.4f} "
              f"at lag {T:.4f}")

    ent = run / "entropy.json"
    if ent.exists():
        e = json.loads(ent.read_text())
        print(f"    entropy     : h = {e['h']:.4f} +- {e['h_sem']:.4f} "
              f"per unit integration time")
        print(f"                  h_tilde = {e['h_tilde']:.4e} "
              f"+- {e['h_tilde_sem']:.1e}   "
              f"(fit over t in [{e['fit_window'][0]:.2f}, {e['fit_window'][1]:.2f}], "
              f"{len(e['per_line'])} lines)")

    if not np.all(np.isnan(s["omega_a_rms"])):
        # The reference's own statistics are over 100 frames of a developed
        # state; take the last 100 recorded frames.
        k = min(100, len(t))
        sl = slice(len(t) - k, len(t))
        print(f"    rotation rates over the last {k} frames "
              f"(t in [{t[sl][0]:.3f}, {t[sl][-1]:.3f}]):")
        print(f"      RMS    omega_A = {s['omega_a_rms'][sl].mean():.4f}   "
              f"omega_F = {s['omega_f_rms'][sl].mean():.4f}")
        print(f"      median|omega_A|= {s['omega_a_median'][sl].mean():.4f}   "
              f"|omega_F|= {s['omega_f_median'][sl].mean():.4e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for d in sys.argv[1:]:
        report(Path(d))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
