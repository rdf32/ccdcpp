"""bench_synthetic.py <module_dir> <corpus.bin> <dtype: f64|u16> <entry: cols|cube> <threads> <reps>

The synthetic timing of one build: the corpus of make_corpus.cpp tiled 2x2 (8000 pixels),
through ccdcpp.detect_cube_columns ("cols") or ccdcpp.detect_cube ("cube") with default
options, `threads` threads, best of `reps` calls. Prints the best wall time of the call and its
time per pixel, and the file `import ccdcpp` loaded.

`module_dir` is a directory from which `import ccdcpp` loads the build under test: the build
directory (it holds the bare extension) or a package directory. Input conversion and copying
happen outside the timed call.
"""
import os
import resource
import sys
import time

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402

# A wrong module_dir would silently time whatever ccdcpp the interpreter has installed.
if not os.path.realpath(ccdcpp.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep):
    sys.exit(f"{sys.argv[0]}: ccdcpp loaded from {ccdcpp.__file__}, not from {sys.argv[1]}")

corpus, dtype, entry = sys.argv[2], sys.argv[3], sys.argv[4]
threads, reps = int(sys.argv[5]), int(sys.argv[6])

with open(corpus, "rb") as f:
    H, W, T = np.frombuffer(f.read(24), dtype=np.uint64).astype(int)
    dates = np.frombuffer(f.read(8 * T), dtype=np.int64).copy()
    spec = np.frombuffer(f.read(8 * H * W * 7 * T), dtype=np.float64).reshape(H, W, 7, T)
    qas = np.frombuffer(f.read(H * W * T), dtype=np.uint8).reshape(H, W, T)

qas = np.ascontiguousarray(np.tile(qas, (2, 2, 1)))
src = np.empty((2 * H, 2 * W, 7, T), dtype=np.uint16 if dtype == "u16" else np.float64)
for i in range(2):
    for j in range(2):
        np.clip(spec, 0, 65535, out=src[i * H:(i + 1) * H, j * W:(j + 1) * W], casting="unsafe")
del spec

h = ccdcpp.HarmonicOptions()
fn = ccdcpp.detect_cube_columns if entry == "cols" else ccdcpp.detect_cube
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
best = float("inf")
for _ in range(reps):
    # A fresh float64 input per call, as a caller's buffer would be.
    s = src.copy() if (reps > 1 and dtype == "f64") else src
    t0 = time.perf_counter()
    r = fn(dates, s, qas, h, threads)
    best = min(best, time.perf_counter() - t0)
    del r, s

npx = qas.shape[0] * qas.shape[1]
print(f"{entry} {dtype} t{threads}: {best * 1e3:.0f} ms, {best / npx * 1e3:.4f} ms/px, "
      f"input {src.nbytes / 1e6:.0f} MB, maxrss_before_call {rss0 / 1e6:.0f} MB, "
      f"ccdcpp {ccdcpp.__file__}")
