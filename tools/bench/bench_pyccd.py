"""bench_pyccd.py <module_dir> <pyccd_dir> <corpus.bin> <pixels> [<pixel.csv> ...] [--only pyccd|ccdcpp]

ccdcpp against PyCCD on the same pixels, one thread each. Takes `pixels` pixels spread evenly
over the corpus of make_corpus.cpp, plus any PyCCD-format CSV pixels given (day, the seven
bands, the QA code per line, e.g. tests/ccd/machine/test_3657_3610_observations.csv). Each
pixel runs once through PyCCD's ccd.detect and best of 5 through ccdcpp.detect with the default
input and the "ccdc" configuration at PyCCD's statistics cutoff. Prints per-pixel times, the median and total of each and their ratio, and how many
pixels gave the same segments (start, end and break days) in both: not all, since the 0.8.0
fixes move some pixels on purpose (see CHANGELOG.md).

With --only, one implementation runs and the process's peak resident set size is printed, in
MB of 10^6 bytes as /usr/bin/time reports it: run once per implementation for each one's memory
(--only ccdcpp also skips PyCCD's import; ccdcpp's module, small, is always imported). The peak
is also given above the peak after the input and the first pixel, i.e. what the other pixels add.

`module_dir` is a directory from which `import ccdcpp` loads the build under test (as for
bench_synthetic.py); `pyccd_dir` is one from which `import ccd` loads PyCCD (lcmap-pyccd
2021.7.19, its dependencies installed in the interpreter: numpy, scipy, scikit-learn).
"""
import os
import resource
import statistics
import sys
import time
import warnings

args, only, i = [], None, 1
while i < len(sys.argv):
    a = sys.argv[i]
    if a == "--only":
        if i + 1 >= len(sys.argv):
            sys.exit("--only needs a value: pyccd or ccdcpp")
        only = sys.argv[i + 1]
        i += 2
        continue
    if a.startswith("--only="):
        only = a[len("--only="):]
    elif a.startswith("--"):
        sys.exit(f"unknown option {a}")
    else:
        args.append(a)
    i += 1
if only not in (None, "pyccd", "ccdcpp"):
    sys.exit("--only pyccd or --only ccdcpp")
sys.argv = [sys.argv[0]] + args
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
import numpy as np  # noqa: E402

import ccdcpp  # noqa: E402

if not os.path.realpath(ccdcpp.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep):
    sys.exit(f"{sys.argv[0]}: ccdcpp loaded from {ccdcpp.__file__}, not from {sys.argv[1]}")

# PyCCD 2021 uses the numpy aliases that numpy 1.24 removed.
np.bool, np.int, np.float = bool, int, float
warnings.filterwarnings("ignore")
if only != "ccdcpp":
    import ccd  # noqa: E402

PARAMS = {"QA_BITPACKED": False, "QA_FILL": 255, "QA_CLEAR": 0, "QA_WATER": 1,
          "QA_SHADOW": 2, "QA_SNOW": 3, "QA_CLOUD": 4}

corpus, count = sys.argv[3], int(sys.argv[4])
with open(corpus, "rb") as f:
    H, W, T = np.frombuffer(f.read(24), dtype=np.uint64).astype(int)
    dates = np.frombuffer(f.read(8 * T), dtype=np.int64).copy()
    spec = np.frombuffer(f.read(8 * H * W * 7 * T), dtype=np.float64).reshape(H, W, 7, T)
    qas = np.frombuffer(f.read(H * W * T), dtype=np.uint8).reshape(H, W, T)

pixels = []
for i in np.linspace(0, H * W - 1, count).astype(int):
    r, c = divmod(int(i), W)
    pixels.append((f"corpus {r},{c}", dates, spec[r, c], qas[r, c]))
for path in sys.argv[5:]:
    d = np.genfromtxt(path, delimiter=",")
    pixels.append((os.path.basename(path), d[:, 0].astype(np.int64), d[:, 1:8].T, d[:, 8]))

hopt = ccdcpp.Input()
# The "ccdc" configuration with PyCCD's statistics cutoff, 2017-12-31.
PYCCD = ccdcpp.config("ccdc").replace(shared__statistics_end=736694)


def peak_mb():
    # ru_maxrss is bytes on macOS, kilobytes on Linux; MB of 10^6 bytes.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r * 1024 / 1e6


def run_pyccd(d, s64, q8):
    # PyCCD gets integer DN where the values are whole, as its own tests pass them.
    py_s = s64.astype(np.int64) if np.all(s64 == np.round(s64)) else s64
    return ccd.detect(d, *[np.array(py_s[b]) for b in range(7)], q8.astype(np.int64), params=PARAMS)


py_t, cpp_t, same = [], [], 0
before = None
# The first pixel runs twice, untimed the first time: PyCCD's first call pays for lazy imports.
for k, (name, d, s, q) in enumerate(pixels[:1] + pixels):
    d = np.ascontiguousarray(d, dtype=np.int64)
    s64 = np.ascontiguousarray(s, dtype=np.float64)
    q8 = np.ascontiguousarray(q, dtype=np.uint8)
    if k == 0:
        if only != "ccdcpp":
            run_pyccd(d, s64, q8)
        if only != "pyccd":
            ccdcpp.detect(d, s64, q8, input=hopt, config=PYCCD)
        before = peak_mb()
        continue
    py = cpp = None
    if only != "ccdcpp":
        t0 = time.perf_counter()
        py = run_pyccd(d, s64, q8)
        py_t.append(time.perf_counter() - t0)
    if only != "pyccd":
        best = None
        for _ in range(5):
            t0 = time.perf_counter()
            cpp = ccdcpp.detect(d, s64, q8, input=hopt, config=PYCCD)
            el = time.perf_counter() - t0
            best = el if best is None else min(best, el)
        cpp_t.append(best)
    if only is None:
        a = [(m["start_day"], m["end_day"], m["break_day"]) for m in cpp["models"]]
        b = [(m["start_day"], m["end_day"], m["break_day"]) for m in py["change_models"]]
        same += a == b
        print(f"{name}: PyCCD {1e3 * py_t[-1]:.1f} ms, ccdcpp {1e3 * best:.2f} ms, "
              f"{py_t[-1] / best:.0f}x, segments {len(a)}/{len(b)}"
              f"{'' if a == b else ' DIFFER'}")

n = len(pixels)
if only is None:
    pm, cm = statistics.median(py_t), statistics.median(cpp_t)
    print(f"{n} pixels; median PyCCD {1e3 * pm:.1f} ms, ccdcpp {1e3 * cm:.3f} ms "
          f"({pm / cm:.0f}x); total PyCCD {sum(py_t):.1f} s, ccdcpp {sum(cpp_t):.3f} s "
          f"({sum(py_t) / sum(cpp_t):.0f}x); same segments {same}/{n}")
else:
    times = py_t or cpp_t
    print(f"{n} pixels, {only} only; median {1e3 * statistics.median(times):.3f} ms, total "
          f"{sum(times):.3f} s; peak RSS {peak_mb():.0f} MB, {peak_mb() - before:.1f} MB above "
          f"the peak after loading the input and the first pixel ({before:.0f} MB)")
print(f"ccdcpp {ccdcpp.__file__}; PyCCD {ccd.__file__ if only != 'ccdcpp' else '-'}; "
      f"numpy {np.__version__}")
