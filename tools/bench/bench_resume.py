"""An annual update of the bench cell, timed and checked.

The cell is bench_cell.py's (h024v013 chunk 00030008, 500 x 500, 1984-2023; `--size` takes a
smaller corner of it). Per strip of `strip` rows:

  1. the stored run: detect_cube(record=True) on the dates before `cut`;
  2. the update: detect_cube(resume=, freeze_statistics=True) on every date, from 1's output:
     the preset has no statistics cutoff, and the update keeps its statistics at 1's last date;
  3. the full run: detect_cube() on every date, statistics_end capped at 1's last date.

So the update equals the full run with statistics capped at the stored end, not a plain full
run on every date (whose statistics would read the new dates too).

Each runs in its own process over the whole cell (`--only stored|update|full` runs one; the
update's process runs 1 untimed for its input) and prints its detect time (conversion outside)
and its peak memory, also above the peak once the cell is read. 2 must equal 3 column for
column (every segment array, bit for bit): the update and full processes save their segment
columns per strip (`--save <dir>`, a temporary directory), and the script counts the pixels
where they differ and exits non-zero if any do. It also reports the resume's status counts and
the fingerprint of the update's output.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

PHASES = ("stored", "update", "full")
SEGMENTS = ("row", "col", "start_day", "end_day", "break_day", "curve_qa", "observation_count",
            "change_probability", "intercept", "coefficients", "rmse", "magnitude")


def differing_pixels(a: dict[str, NDArray[Any]], b: dict[str, NDArray[Any]]) -> int:
    """The number of pixels whose segments differ between the segment columns `a` and `b`."""
    bad = set()
    for pixel in set(zip(a["row"], a["col"])) | set(zip(b["row"], b["col"])):
        ia = (a["row"] == pixel[0]) & (a["col"] == pixel[1])
        ib = (b["row"] == pixel[0]) & (b["col"] == pixel[1])
        if ia.sum() != ib.sum() or not all(np.array_equal(a[k][ia], b[k][ib]) for k in SEGMENTS):
            bad.add(pixel)
    return len(bad)


def run_phase(args: argparse.Namespace) -> int:
    """One phase over the whole cell, in this process."""
    _common.import_build(args.package_dir)
    import ccdcpp
    import zarr  # pyright: ignore[reportMissingImports]
    from ccdcpp import presets

    N = args.size
    cut = dt.date.fromisoformat(args.cut).toordinal()
    print("ccdcpp", _common.build_of(), "numpy", np.__version__,
          "zarr", zarr.__version__)
    cell = _common.read_cell(args.cube, N)
    dates = cell.dates
    n = int(np.searchsorted(dates, cut))
    print(f"T={dates.size}; stored run on {n} dates up to {dt.date.fromordinal(int(dates[n - 1]))}")
    base = _common.peak_mb()

    profile = presets.landsat_c2()
    # The full run reads statistics up to the stored run's last date, or the preset's cutoff if
    # that comes first.
    end, last = ccdcpp.config("ccdc").shared()["statistics_end"], int(dates[n - 1])
    if end != "none":
        last = min(last, end if isinstance(end, int) else dt.date.fromisoformat(end).toordinal())
    capped = ccdcpp.config("ccdc").replace(shared__statistics_end=last)

    seconds = 0.0
    counts = {"resumed": 0, "rerun_input_changed": 0, "failed": 0}
    for r0 in range(0, N, args.strip):
        r1 = min(N, r0 + args.strip)
        sp, qa = _common.c2_strip(cell, r0, r1)
        tic = time.perf_counter()
        if args.only == "full":
            out = ccdcpp.detect_cube(dates, sp, qa, spec=profile, config=capped,
                                     threads=args.threads)
        else:
            out = ccdcpp.detect_cube(dates[:n], sp[..., :n], qa[..., :n], spec=profile,
                                     threads=args.threads, record=True)
            if args.only == "update":
                tic = time.perf_counter()
                out = ccdcpp.detect_cube(dates, sp, qa, spec=profile, threads=args.threads,
                                         resume=out, freeze_statistics=True)
                resumed = out.get("counts")
                assert resumed is not None   # a resume's output holds its counts
                for k in counts:
                    counts[k] += resumed[k]
        toc = time.perf_counter()
        seconds += toc - tic
        print(f"  {args.only} rows {r0}-{r1 - 1}: {toc - tic:.1f} s", flush=True)
        if args.save and args.only != "stored":
            np.savez(os.path.join(args.save, f"{args.only}_{r0}.npz"),
                     failures=np.array(json.dumps(out["failures"])),
                     **{k: np.asarray(out[k]) for k in SEGMENTS})
        del sp, qa, out

    print(_common.report(f"{args.only} t{args.threads} strip{args.strip}",
                         f"{seconds:.1f} s, {seconds / (N * N) * 1e3:.3f} ms/px",
                         _common.peak_mb(), base, "the cell read"))
    _common.emit({"seconds": seconds, "counts": counts})
    return 0


def check(args: argparse.Namespace, results: dict[str, dict[str, Any]], saved: str) -> int:
    """Compares the update with the full run strip by strip, and reports the times, the status
    counts and the update's fingerprint."""
    N = args.size
    digest = hashlib.sha256()
    differing = 0
    for r0 in range(0, N, args.strip):
        with np.load(os.path.join(saved, f"update_{r0}.npz")) as fa, \
                np.load(os.path.join(saved, f"full_{r0}.npz")) as fb:
            a = {k: fa[k] for k in SEGMENTS}
            b = {k: fb[k] for k in SEGMENTS}
            same_failures = str(fa["failures"]) == str(fb["failures"])
        same = all(np.array_equal(a[k], b[k]) for k in SEGMENTS) and same_failures
        if not same:
            differing += differing_pixels(a, b)
        print(f"  rows {r0}-{min(N, r0 + args.strip) - 1}: {'equal' if same else 'DIFFERENT'}")
        for k in SEGMENTS:
            digest.update(k.encode())
            digest.update(np.ascontiguousarray(a[k]).tobytes())

    t_first, t_resume, t_full = (results[p]["seconds"] for p in PHASES)
    counts = results["update"]["counts"]
    print(f"stored run {t_first:.1f} s; update {t_resume:.1f} s; full run (statistics capped "
          f"at the stored end) {t_full:.1f} s; update / full = {t_resume / t_full:.3f}")
    print(f"status {counts}; pixels where the update differs from the full run with statistics "
          f"capped at the stored end: {differing}")
    print("fingerprint", digest.hexdigest())
    if sum(counts.values()) != N * N or counts["failed"]:
        sys.exit(f"{sys.argv[0]}: expected {N * N} pixels and none failed, got {counts}")
    return 1 if differing else 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package_dir")
    p.add_argument("cube", help="cube.zarr")
    p.add_argument("threads", type=int, nargs="?", default=8)
    p.add_argument("strip", type=int, nargs="?", default=50)
    p.add_argument("cut", nargs="?", default="2023-01-01", help="an ISO date")
    p.add_argument("--size", type=int, default=_common.CELL_SIZE,
                   help="the corner of the cell to run, size x size pixels (default: all of it)")
    p.add_argument("--only", choices=PHASES, help="run one phase")
    p.add_argument("--save", help="with --only update or full: save its segment columns here")
    args = p.parse_args(argv)
    if args.only is not None:
        return run_phase(args)
    with tempfile.TemporaryDirectory(prefix="bench_resume.") as saved:
        results = _common.run_sides(__file__, [*argv, "--save", saved], PHASES)
        return check(args, results, saved)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
