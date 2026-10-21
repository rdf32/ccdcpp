"""ccdcpp against PyCCD on the same pixels, one thread each.

Takes `pixels` pixels spread evenly over the corpus of `golden_driver gen` (c16.bin), plus any
PyCCD-format CSV pixels given (day, the seven bands, the QA code per line, e.g.
tests/ccd/data/test_3657_3610_observations.csv). Each pixel runs once through PyCCD's
ccd.detect and best of 5 through ccdcpp.detect with the default input and the "ccdc"
configuration at PyCCD's statistics cutoff. Prints per-pixel times, the median and total of
each and their ratio, and how many pixels gave the same segments (start, end and break days)
in both: not all, since the 0.8.0 fixes move some pixels on purpose (see CHANGELOG.md).

Each implementation runs in its own process (`--only pyccd` or `--only ccdcpp` runs one), which
prints its median and total time and its peak memory, also above the peak after the input and
the first pixel, i.e. what the other pixels add; the pyccd process does not import ccdcpp, nor
the ccdcpp process PyCCD.

`module_dir` is a directory from which `import ccdcpp` loads the build under test (as for
bench_synthetic.py); `pyccd_dir` is one from which `import ccd` loads PyCCD (lcmap-pyccd
2021.7.19, its dependencies installed in the interpreter: numpy, scipy, scikit-learn).
"""
import argparse
import os
import statistics
import sys
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

SIDES = ("pyccd", "ccdcpp")
#: One implementation's detect on one pixel (dates, spectral, QA), giving its segments' (start,
#: end, break) days.
Detect = Callable[[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8]],
                  list[tuple[int, int, int]]]
PARAMS = {"QA_BITPACKED": False, "QA_FILL": 255, "QA_CLEAR": 0, "QA_WATER": 1,
          "QA_SHADOW": 2, "QA_SNOW": 3, "QA_CLOUD": 4}


def pixel_inputs(corpus_path: str, count: int,
                 csvs: list[str]) -> list[tuple[str, NDArray[np.int64], NDArray[np.float64],
                                                NDArray[np.uint8]]]:
    """The pixels by name: `count` spread evenly over the corpus, then the CSV pixels."""
    corpus = _common.read_corpus(corpus_path)
    H, W = corpus.qas.shape[:2]
    pixels = []
    for i in np.linspace(0, H * W - 1, count).astype(int):
        r, c = divmod(int(i), W)
        pixels.append((f"corpus {r},{c}", corpus.dates, corpus.spectral[r, c], corpus.qas[r, c]))
    for path in csvs:
        d = np.genfromtxt(path, delimiter=",")
        pixels.append((os.path.basename(path), d[:, 0].astype(np.int64), d[:, 1:8].T, d[:, 8]))
    return [(name, np.ascontiguousarray(d, dtype=np.int64), np.ascontiguousarray(s, dtype=np.float64),
             np.ascontiguousarray(q, dtype=np.uint8)) for name, d, s, q in pixels]


def pyccd_side(pyccd_dir: str) -> tuple[Detect, str]:
    """PyCCD's Detect and the file it loaded from."""
    sys.path.insert(0, pyccd_dir)
    # PyCCD 2021 uses the numpy aliases that numpy 1.24 removed.
    for alias, builtin in (("bool", bool), ("int", int), ("float", float)):
        setattr(np, alias, builtin)
    warnings.filterwarnings("ignore")
    import ccd  # pyright: ignore[reportMissingImports]  # pyccd, optional

    def detect(d: NDArray[np.int64], s64: NDArray[np.float64],
               q8: NDArray[np.uint8]) -> list[tuple[int, int, int]]:
        # PyCCD gets integer DN where the values are whole, as its own tests pass them.
        py_s = s64.astype(np.int64) if np.all(s64 == np.round(s64)) else s64
        r = ccd.detect(d, *[np.array(py_s[b]) for b in range(7)], q8.astype(np.int64),
                       params=PARAMS)
        return [(int(m["start_day"]), int(m["end_day"]), int(m["break_day"]))
                for m in r["change_models"]]

    return detect, str(ccd.__file__)


def ccdcpp_side(module_dir: str) -> tuple[Detect, str]:
    """ccdcpp's Detect and the file it loaded from."""
    _common.import_build(module_dir)
    import ccdcpp
    h = ccdcpp.Input()
    config = ccdcpp.config("ccdc").replace(shared__statistics_end=_common.PYCCD_STATISTICS_END)

    def detect(d: NDArray[np.int64], s64: NDArray[np.float64],
               q8: NDArray[np.uint8]) -> list[tuple[int, int, int]]:
        r = ccdcpp.detect(d, s64, q8, input=h, config=config)
        return [(m["start_day"], m["end_day"], m["break_day"]) for m in r["models"]]

    return detect, str(ccdcpp.__file__)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module_dir")
    p.add_argument("pyccd_dir")
    p.add_argument("corpus", help="c16.bin")
    p.add_argument("pixels", type=int)
    p.add_argument("csv", nargs="*", help="PyCCD-format pixel CSVs")
    p.add_argument("--only", choices=SIDES, help="run one implementation")
    args = p.parse_args(argv)
    if args.only is None:
        return compare(_common.run_sides(__file__, argv, SIDES))

    if args.only == "pyccd":
        detect, where = pyccd_side(args.pyccd_dir)
        reps = 1
    else:
        detect, where = ccdcpp_side(args.module_dir)
        reps = 5
    pixels = pixel_inputs(args.corpus, args.pixels, args.csv)
    # The first pixel runs once untimed first: PyCCD's first call pays for lazy imports.
    detect(*pixels[0][1:])
    base = _common.peak_mb()
    times: list[float] = []
    segments: list[list[tuple[int, int, int]]] = []
    for _, d, s, q in pixels:
        best, segs = _common.best_of(reps, lambda: detect(d, s, q))
        times.append(best)
        segments.append(segs)
    print(_common.report(f"{args.only:>6}",
                         f"median {1e3 * statistics.median(times):.3f} ms, total "
                         f"{sum(times):.3f} s over {len(pixels)} pixels",
                         _common.peak_mb(), base, "the input and the first pixel"))
    print(f"{args.only} {where}; numpy {np.__version__}")
    _common.emit({"names": [name for name, *_ in pixels], "times": times, "segments": segments})
    return 0


def compare(results: dict[str, dict[str, Any]]) -> int:
    """Prints the two sides' times and segments pixel by pixel, then the medians, the totals and
    their ratios, and how many pixels have the same segments."""
    py, cpp = results["pyccd"], results["ccdcpp"]
    same = 0
    for name, pt, ct, a, b in zip(py["names"], py["times"], cpp["times"], cpp["segments"],
                                  py["segments"]):
        same += a == b
        print(f"{name}: PyCCD {1e3 * pt:.1f} ms, ccdcpp {1e3 * ct:.2f} ms, {pt / ct:.0f}x, "
              f"segments {len(a)}/{len(b)}{'' if a == b else ' DIFFER'}")
    n = len(py["names"])
    pm, cm = statistics.median(py["times"]), statistics.median(cpp["times"])
    pt, ct = sum(py["times"]), sum(cpp["times"])
    print(f"{n} pixels; median PyCCD {1e3 * pm:.1f} ms, ccdcpp {1e3 * cm:.3f} ms ({pm / cm:.0f}x); "
          f"total PyCCD {pt:.1f} s, ccdcpp {ct:.3f} s ({pt / ct:.0f}x); same segments {same}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
