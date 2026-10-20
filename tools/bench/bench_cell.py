"""bench_cell.py <ccdcpp_package_parent> <cube.zarr> [threads=8] [strip=50] [statistics_end]

The real-cell timing of the ccdcpp state-machine refactor: a 500 x 500 cell of a Landsat C2
ARD cube, through detect_cube with encoding=presets.landsat_c2() (raw QA_PIXEL decoded by the
Landsat C2 table) and the "ccdc" configuration (with its shared statistics_end set to the
argument `statistics_end`, an ordinal day, when given: 736694 is pyccd's 2017-12-31), `threads` threads, in strips of `strip` rows as the nlcd plugin runs
it. The refactor's cell is h024v013 chunk 00030008 (rows 4000-4499, cols 1500-1999 of the
tile), 1984-2023, all 2473 dates.

`cube.zarr` is a zarr 3 group holding `landsat-c2ard-sr`, (rows, cols, 7, T) uint16 SR_B1..B6
and QA_PIXEL, and `landsat-c2ard-bt`, (rows, cols, 1, T) uint16 ST, with the ISO dates in the
`time` attribute of `landsat-c2ard-sr`; the cell is rows 4000-4499, cols 1500-1999 of it.

The cell is read once (uint16) before any timing. Each strip is converted to landsat_c2 units
(float64) outside the timed region; only the detect_cube calls are timed, and their sum
is the number reported. The outputs of every strip are hashed in key order, so two builds that
print the same fingerprint gave byte-identical results on all 250,000 pixels.

Run under /usr/bin/time -l for the process's peak RSS, with a Python that has zarr 3 and
numpy (the refactor used zarr 3, numpy 1.26, Python 3.11). `ccdcpp_package_parent` is a
directory holding a `ccdcpp` package: python/ccdcpp/*.py of the tree under test plus the
extension built from it.
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

# A wrong module_dir would silently time whatever ccdcpp the interpreter has installed.
if not os.path.realpath(ccdcpp.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep):
    sys.exit(f"{sys.argv[0]}: ccdcpp loaded from {ccdcpp.__file__}, not from {sys.argv[1]}")

from ccdcpp import presets  # noqa: E402

CUBE = sys.argv[2]
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STRIP = int(sys.argv[4]) if len(sys.argv) > 4 else 50
CONFIG = (ccdcpp.config("ccdc").replace(shared__statistics_end=int(sys.argv[5]))
          if len(sys.argv) > 5 else ccdcpp.config("ccdc"))
R0, C0, N = 4000, 1500, 500

print("ccdcpp", ccdcpp.__version__, ccdcpp.__file__, "numpy", np.__version__, "zarr", zarr.__version__)
t0 = time.perf_counter()
g = zarr.open_group(CUBE, mode="r")
sr, bt = g["landsat-c2ard-sr"], g["landsat-c2ard-bt"]
dates = np.array([dt.date.fromisoformat(t).toordinal() for t in sr.attrs["time"]], np.int64)
s = np.asarray(sr[R0:R0 + N, C0:C0 + N, :, :])  # (500, 500, 7, T) uint16: SR_B1..B6 + QA_PIXEL
th = np.asarray(bt[R0:R0 + N, C0:C0 + N, 0, :])  # (500, 500, T) uint16: ST
print(f"read {s.nbytes / 1e9 + th.nbytes / 1e9:.2f} GB in {time.perf_counter() - t0:.1f} s; T={dates.size}; "
      f"maxrss after reading {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB")

profile = presets.landsat_c2()
digest = hashlib.sha256()
t_det, segments, failures = 0.0, 0, 0
t_loop = time.perf_counter()
for r0 in range(0, N, STRIP):
    r1 = min(N, r0 + STRIP)
    t_conv = time.perf_counter()
    sp = np.empty((r1 - r0, N, 7, dates.size), np.float64)
    opt = s[r0:r1, :, :6]
    sp[:, :, :6] = opt * 2.75e-5 - 0.2
    sp[:, :, :6][opt == 0] = -1.0                        # nodata -> out of range
    t = th[r0:r1]
    sp[:, :, 6] = (t * 0.00341802 + 149.0 - 273.15) / 100.0
    sp[:, :, 6][t == 0] = -9.0
    qa = np.ascontiguousarray(s[r0:r1, :, 6])
    del opt, t
    tic = time.perf_counter()
    out = ccdcpp.detect_cube(dates, sp, qa, encoding=profile, config=CONFIG, threads=THREADS)
    toc = time.perf_counter()
    t_det += toc - tic
    print(f"  rows {r0}-{r1 - 1}: convert {tic - t_conv:.1f} s, detect {toc - tic:.1f} s", flush=True)
    failures += len(out["failures"])
    segments += len(out["row"])
    for k in sorted(out):
        v = out[k]
        if isinstance(v, np.ndarray):
            digest.update(k.encode()); digest.update(str(v.dtype).encode()); digest.update(np.ascontiguousarray(v).tobytes())
    del sp, qa, out

px = N * N
print(f"strip loop {time.perf_counter() - t_loop:.1f} s")
print(f"cell detect t{THREADS} strip{STRIP}: {t_det:.1f} s, {t_det / px * 1e3:.3f} ms/px, "
      f"segments {segments}, failures {failures}, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB")
print("fingerprint", digest.hexdigest())
