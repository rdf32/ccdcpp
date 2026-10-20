"""bench_resume.py <ccdcpp_package_parent> <cube.zarr> [threads=8] [strip=50] [cut=2023-01-01]

An annual update of the bench cell (bench_cell.py's: h024v013 chunk 00030008, 500 x 500,
1984-2023), timed and checked. Per strip of `strip` rows:

  1. the stored run: detect_cube(record=True) on the dates before `cut`;
  2. the update: detect_cube(resume=) on every date, from 1's output;
  3. the full run: detect_cube() on every date, stat_ord capped at 1's last date.

2 must equal 3 column for column (every segment array, bit for bit); the script counts the
pixels where they differ and exits non-zero if any do. It reports each step's detect time
(conversion outside), the resume's status counts and the fingerprint of the update's output.
"""
import datetime as dt
import hashlib
import os
import resource
import sys
import time

if len(sys.argv) < 3:
    sys.exit(__doc__)
sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import zarr  # noqa: E402

import ccdcpp  # noqa: E402

if not os.path.realpath(ccdcpp.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep):
    sys.exit(f"{sys.argv[0]}: ccdcpp loaded from {ccdcpp.__file__}, not from {sys.argv[1]}")

from ccdcpp import presets  # noqa: E402

CUBE = sys.argv[2]
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STRIP = int(sys.argv[4]) if len(sys.argv) > 4 else 50
CUT = dt.date.fromisoformat(sys.argv[5] if len(sys.argv) > 5 else "2023-01-01").toordinal()
R0, C0, N = 4000, 1500, 500
SEGMENTS = ("row", "col", "start_day", "end_day", "break_day", "curve_qa", "observation_count",
            "change_probability", "intercept", "coefficients", "rmse", "magnitude")

print("ccdcpp", ccdcpp.__version__, ccdcpp.__file__, "numpy", np.__version__, "zarr", zarr.__version__)
g = zarr.open_group(CUBE, mode="r")
sr, bt = g["landsat-c2ard-sr"], g["landsat-c2ard-bt"]
dates = np.array([dt.date.fromisoformat(t).toordinal() for t in sr.attrs["time"]], np.int64)
s = sr[R0:R0 + N, C0:C0 + N, :, :]
th = bt[R0:R0 + N, C0:C0 + N, 0, :]
n = int(np.searchsorted(dates, CUT))
print(f"T={dates.size}; stored run on {n} dates up to {dt.date.fromordinal(int(dates[n - 1]))}")

profile = presets.landsat_c2()
capped = ccdcpp.config("ccdc").replace(
    shared__stat_ord=min(ccdcpp.config("ccdc").shared()["stat_ord"], int(dates[n - 1])))

digest = hashlib.sha256()
t_first = t_resume = t_full = 0.0
counts = {"resumed": 0, "rerun_input_changed": 0, "failed": 0}
differing = 0
for r0 in range(0, N, STRIP):
    r1 = min(N, r0 + STRIP)
    sp = np.empty((r1 - r0, N, 7, dates.size), np.float64)
    opt = s[r0:r1, :, :6]
    sp[:, :, :6] = opt * 2.75e-5 - 0.2
    sp[:, :, :6][opt == 0] = -1.0
    t = th[r0:r1]
    sp[:, :, 6] = (t * 0.00341802 + 149.0 - 273.15) / 100.0
    sp[:, :, 6][t == 0] = -9.0
    qa = np.ascontiguousarray(s[r0:r1, :, 6])
    del opt, t

    tic = time.perf_counter()
    stored = ccdcpp.detect_cube(dates[:n], sp[..., :n], qa[..., :n], input=profile,
                                threads=THREADS, record=True)
    t1 = time.perf_counter()
    update = ccdcpp.detect_cube(dates, sp, qa, input=profile, threads=THREADS, resume=stored)
    t2 = time.perf_counter()
    full = ccdcpp.detect_cube(dates, sp, qa, input=profile, config=capped, threads=THREADS)
    t3 = time.perf_counter()
    t_first += t1 - tic
    t_resume += t2 - t1
    t_full += t3 - t2

    for k, v in update["counts"].items():
        counts[k] += v
    same = (all(np.array_equal(update[k], full[k]) for k in SEGMENTS)
            and update["failures"] == full["failures"])
    if not same:
        bad = set()
        a = {k: update[k] for k in SEGMENTS}
        b = {k: full[k] for k in SEGMENTS}
        for pixel in set(zip(a["row"], a["col"])) | set(zip(b["row"], b["col"])):
            ia = (a["row"] == pixel[0]) & (a["col"] == pixel[1])
            ib = (b["row"] == pixel[0]) & (b["col"] == pixel[1])
            if ia.sum() != ib.sum() or not all(np.array_equal(a[k][ia], b[k][ib]) for k in SEGMENTS):
                bad.add(pixel)
        differing += len(bad)
    print(f"  rows {r0}-{r1 - 1}: stored {t1 - tic:.1f} s, update {t2 - t1:.1f} s, "
          f"full {t3 - t2:.1f} s, {'equal' if same else 'DIFFERENT'}", flush=True)
    for k in SEGMENTS:
        digest.update(k.encode()); digest.update(np.ascontiguousarray(update[k]).tobytes())
    del sp, qa, stored, update, full

print(f"stored run {t_first:.1f} s; update {t_resume:.1f} s; full run {t_full:.1f} s; "
      f"update / full = {t_resume / t_full:.3f}")
print(f"status {counts}; pixels where the update differs from the full run: {differing}; "
      f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB")
print("fingerprint", digest.hexdigest())
if sum(counts.values()) != N * N or counts["failed"]:
    sys.exit(f"{sys.argv[0]}: expected {N * N} pixels and none failed, got {counts}")
sys.exit(1 if differing else 0)
