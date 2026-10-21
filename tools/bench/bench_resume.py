"""An annual update of the bench cell, timed and checked.

The cell is bench_cell.py's (h024v013 chunk 00030008, 500 x 500, 1984-2023; `--size` takes a
smaller corner of it). Per strip of `strip` rows:

  1. the stored run: detect_cube(record=True) on the dates before `cut`;
  2. the update: detect_cube(resume=) on every date, from 1's output;
  3. the full run: detect_cube() on every date, statistics_end capped at 1's last date.

2 must equal 3 column for column (every segment array, bit for bit); the script counts the
pixels where they differ and exits non-zero if any do. It reports each step's detect time
(conversion outside), the resume's status counts and the fingerprint of the update's output.
"""
import argparse
import datetime as dt
import hashlib
import os
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

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
    args = p.parse_args(argv)
    _common.import_build(args.package_dir)
    import ccdcpp
    import zarr  # pyright: ignore[reportMissingImports]
    from ccdcpp import presets

    N = args.size
    cut = dt.date.fromisoformat(args.cut).toordinal()

    print("ccdcpp", ccdcpp.__version__, ccdcpp.__file__, "numpy", np.__version__,
          "zarr", zarr.__version__)
    cell = _common.read_cell(args.cube, N)
    dates = cell.dates
    n = int(np.searchsorted(dates, cut))
    print(f"T={dates.size}; stored run on {n} dates up to {dt.date.fromordinal(int(dates[n - 1]))}")

    profile = presets.landsat_c2()
    # The full run reads statistics up to the stored run's last date, or the preset's cutoff if
    # that comes first.
    end, last = ccdcpp.config("ccdc").shared()["statistics_end"], int(dates[n - 1])
    if end != "none":
        last = min(last, end if isinstance(end, int) else dt.date.fromisoformat(end).toordinal())
    capped = ccdcpp.config("ccdc").replace(shared__statistics_end=last)

    digest = hashlib.sha256()
    t_first = t_resume = t_full = 0.0
    counts = {"resumed": 0, "rerun_input_changed": 0, "failed": 0}
    differing = 0
    for r0 in range(0, N, args.strip):
        r1 = min(N, r0 + args.strip)
        sp, qa = _common.c2_strip(cell, r0, r1)

        tic = time.perf_counter()
        stored = ccdcpp.detect_cube(dates[:n], sp[..., :n], qa[..., :n], input=profile,
                                    threads=args.threads, record=True)
        t1 = time.perf_counter()
        update = ccdcpp.detect_cube(dates, sp, qa, input=profile, threads=args.threads,
                                    resume=stored)
        t2 = time.perf_counter()
        full = ccdcpp.detect_cube(dates, sp, qa, input=profile, config=capped,
                                  threads=args.threads)
        t3 = time.perf_counter()
        t_first += t1 - tic
        t_resume += t2 - t1
        t_full += t3 - t2

        resumed = update.get("counts")
        assert resumed is not None   # a resume's output holds its counts
        counts["resumed"] += resumed["resumed"]
        counts["rerun_input_changed"] += resumed["rerun_input_changed"]
        counts["failed"] += resumed["failed"]
        a = {k: update[k] for k in SEGMENTS}
        b = {k: full[k] for k in SEGMENTS}
        same = (all(np.array_equal(a[k], b[k]) for k in SEGMENTS)
                and update["failures"] == full["failures"])
        if not same:
            differing += differing_pixels(a, b)
        print(f"  rows {r0}-{r1 - 1}: stored {t1 - tic:.1f} s, update {t2 - t1:.1f} s, "
              f"full {t3 - t2:.1f} s, {'equal' if same else 'DIFFERENT'}", flush=True)
        for k in SEGMENTS:
            digest.update(k.encode())
            digest.update(np.ascontiguousarray(a[k]).tobytes())
        del sp, qa, stored, update, full, a, b

    print(f"stored run {t_first:.1f} s; update {t_resume:.1f} s; full run {t_full:.1f} s; "
          f"update / full = {t_resume / t_full:.3f}")
    print(f"status {counts}; pixels where the update differs from the full run: {differing}; "
          f"maxrss {_common.peak_mb() / 1e3:.1f} GB")
    print("fingerprint", digest.hexdigest())
    if sum(counts.values()) != N * N or counts["failed"]:
        sys.exit(f"{sys.argv[0]}: expected {N * N} pixels and none failed, got {counts}")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
