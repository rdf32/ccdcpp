"""bench_pixel.py <package_dir> <corpus.bin> <pixels> <reps>

The per-pixel timing of one build: the first `pixels` pixels of the corpus of make_corpus.cpp,
one ccdcpp.detect call each, single-threaded, best of `reps` passes, in three forms:

  hoptions   detect(d, s, q, h): the call's options, the "ccdc" configuration by default
  profile    detect(d, s, q, profile=landsat_c1_dn()): the profile's options and configuration
  config     detect(d, s, q, h, config=cfg), cfg = ccdcpp.config("ccdc") built once

The first two are the default paths, what a caller writes without thinking about the
configuration; the third is the floor they should reach. Prints, per form, the best pass's
time per pixel and the process's peak RSS, and the file `import ccdcpp` loaded.

`package_dir` is a directory holding the ccdcpp package (profiles need it, not the bare
extension). Input conversion happens outside the timed calls.
"""
import os
import resource
import sys
import time

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402
from ccdcpp import presets  # noqa: E402

if not os.path.realpath(ccdcpp.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep):
    sys.exit(f"{sys.argv[0]}: ccdcpp loaded from {ccdcpp.__file__}, not from {sys.argv[1]}")

corpus, pixels, reps = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

with open(corpus, "rb") as f:
    H, W, T = np.frombuffer(f.read(24), dtype=np.uint64).astype(int)
    dates = np.frombuffer(f.read(8 * T), dtype=np.int64).copy()
    spec = np.frombuffer(f.read(8 * H * W * 7 * T), dtype=np.float64).reshape(H * W, 7, T)
    qas = np.frombuffer(f.read(H * W * T), dtype=np.uint8).reshape(H * W, T)

px = [(np.ascontiguousarray(spec[i]), np.ascontiguousarray(qas[i]))
      for i in range(min(pixels, H * W))]
del spec

h = ccdcpp.HarmonicOptions()
profile = presets.landsat_c1_dn()
cfg = ccdcpp.config("ccdc")
forms = {
    "hoptions": lambda s, q: ccdcpp.detect(dates, s, q, h),
    "profile": lambda s, q: ccdcpp.detect(dates, s, q, profile=profile),
    "config": lambda s, q: ccdcpp.detect(dates, s, q, h, config=cfg),
}

for name, call in forms.items():
    call(*px[0])
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for s, q in px:
            call(s, q)
        best = min(best, time.perf_counter() - t0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"pixel {name}: {best / len(px) * 1e3:.4f} ms/px over {len(px)} px, "
          f"maxrss {rss / 1e6:.0f} MB")
print(f"ccdcpp {ccdcpp.__file__}")
