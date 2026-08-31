#!/usr/bin/env python3
"""The register of rendered films, so a finished video can be found again.

Films are written under `output/` beside the run that produced them, and a name
alone does not say what a film shows. Two days later `braid.mp4` is one of six
with that name. This keeps a manifest that answers the question.

The manifest is a TSV in the repository, one row per film, tab separated with a
header. Small, greppable, versioned with the code that made it. The films
themselves stay under `output/`, which is not tracked: a manifest of gigabytes
of MP4 belongs in git, the MP4s do not.

    video_library.py list [pattern]        newest first, pattern matches any field
    video_library.py path <pattern>        the best single match, for opening
    video_library.py publish <mp4> --run=<dir> --shows="..." [--title=...]
    video_library.py scan                  films on disk that no row accounts for

`publish` reads the run's own `meta.json`, `stats.csv`, `etec.json` and
`braid.json`, so the activity, the ensemble and the period are recorded from the
run rather than typed in and mistyped. Paths are stored relative to the
repository, which keeps one machine's directory layout out of a public file.

`sphere_braid_video.py` calls `publish` itself once ffmpeg returns, so a film
that renders is a film that is registered.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "videos" / "LIBRARY.tsv"
ROOTS = [REPO / "output"]
COLS = ["date", "title", "path", "run", "pe", "tracers", "period", "tu",
        "frames", "res", "mb", "script", "shows"]


def rel(p: Path) -> str:
    """Relative to the repository where possible, so rows travel between machines."""
    p = p.resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def rows() -> list[dict]:
    if not MANIFEST.exists():
        return []
    out = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        out.append(dict(zip(COLS, f + [""] * (len(COLS) - len(f)))))
    return out


def write(rs: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rs = sorted(rs, key=lambda r: (r["date"], r["title"]), reverse=True)
    body = "\n".join("\t".join(r.get(c, "") for c in COLS) for r in rs)
    MANIFEST.write_text("# " + "\t".join(COLS) + "\n" + body + "\n")


def probe(mp4: Path) -> tuple[str, str]:
    """Frames and resolution from the container, so a truncated film shows as one."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,width,height",
             "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, timeout=600)
        parts = out.stdout.strip().split(",")
        if len(parts) >= 3:
            w, h, n = parts[0], parts[1], parts[2]
            return n, f"{w}x{h}"
    except Exception:
        pass
    return "?", "?"


def from_run(run: Path | None) -> dict:
    """What the run itself records, rather than what a caller remembers."""
    if run is None or not run.exists():
        return {}
    out: dict[str, str] = {}
    stats = run / "stats.csv"
    if stats.exists():
        rs = list(csv.DictReader(open(stats)))
        tail = [float(r["pe_measured"]) for r in rs[len(rs) // 2:]
                if r.get("pe_measured")]
        if tail:
            out["pe"] = f"{sum(tail) / len(tail):.4f}"
        if rs and rs[-1].get("t"):
            out["tu"] = f"{float(rs[-1]['t']):.0f}"
    for name, key, fmt in (("etec.json", "tracers", "{}"),
                           ("braid.json", "period", "{}")):
        f = run / name
        if f.exists():
            try:
                v = json.loads(f.read_text()).get(key)
                if v is not None:
                    out[key] = fmt.format(v)
            except Exception:
                pass
    return out


def cmd_publish(argv: list[str]) -> None:
    mp4 = Path(argv[0]).resolve()
    opt = {a.split("=")[0].lstrip("-"): a.split("=", 1)[1] for a in argv[1:] if "=" in a}
    if not mp4.exists():
        sys.exit(f"{mp4} does not exist")
    run = Path(opt["run"]).resolve() if "run" in opt else None
    frames, res = probe(mp4)
    r = {
        # From the FILE, not from today. A row written a week after the render
        # would otherwise claim to be the newest film there is, and "the most
        # recent video of X" is the question this manifest exists to answer.
        "date": opt.get("date",
                        datetime.fromtimestamp(mp4.stat().st_mtime)
                        .strftime("%Y-%m-%d %H:%M")),
        "title": opt.get("title", f"{run.name}_{mp4.stem}" if run else mp4.stem),
        "path": rel(mp4),
        "run": rel(run) if run else "",
        "frames": frames,
        "res": res,
        "mb": f"{mp4.stat().st_size / 1e6:.1f}",
        "script": opt.get("script", ""),
        "shows": opt.get("shows", ""),
    }
    r.update({k: v for k, v in from_run(run).items() if k not in ("shows",)})
    rs = [x for x in rows() if x["path"] != r["path"]]
    rs.append(r)
    write(rs)
    print("\t".join(f"{k}={v}" for k, v in r.items() if v))


def match(rs: list[dict], pattern: str) -> list[dict]:
    p = pattern.lower()
    return [r for r in rs if any(p in str(v).lower() for v in r.values())]


def cmd_list(argv: list[str]) -> None:
    rs = rows()
    if argv:
        rs = match(rs, argv[0])
    if not rs:
        print("no films registered" if not argv else f"nothing matches {argv[0]!r}")
        return
    w = max(len(r["title"]) for r in rs)
    for r in rs:
        pe = f"Pe={r['pe']}" if r["pe"] else ""
        print(f"{r['date']}  {r['title']:<{w}}  {pe:<11} {r['res']:>9}  "
              f"{r['frames']:>4} fr  {r['mb']:>5} MB")
        if r["shows"]:
            print(f"{'':12}{r['shows']}")
        print(f"{'':12}{r['path']}")


def cmd_path(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: video_library.py path <pattern>")
    m = match(rows(), argv[0])
    if not m:
        sys.exit(f"nothing matches {argv[0]!r}")
    # Newest first is already the file order, so the first match is the most
    # recent one, which is what "the latest video of X" asks for.
    print(str(REPO / m[0]["path"]))


def cmd_scan(_argv: list[str]) -> None:
    known = {r["path"] for r in rows()}
    loose = []
    for root in ROOTS:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.mp4")):
            if rel(f) not in known:
                loose.append(f)
    if not loose:
        print("every film on disk is registered")
        return
    print(f"{len(loose)} film(s) with no row:")
    for f in loose:
        st = f.stat()
        print(f"  {date.fromtimestamp(st.st_mtime)}  {st.st_size / 1e6:8.1f} MB  {rel(f)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    {"publish": cmd_publish, "list": cmd_list, "path": cmd_path,
     "scan": cmd_scan}[sys.argv[1]](sys.argv[2:])
