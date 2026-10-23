"""The synthetic timing of one build.

The corpus of `golden_driver gen` (c16.bin) tiled 2x2 (8000 pixels), through ccdcpp.detect_cube
with the default DataSpec, `threads` threads, best of `reps` calls. Prints the best wall time of
the call and its time per pixel, the process's peak memory, also above the peak with the input
built, and the file `import ccdcpp` loaded.

`module_dir` is a directory from which `import ccdcpp` loads the build under test: the build
directory (it holds the bare extension) or a package directory. Input conversion and copying
happen outside the timed call.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module_dir")
    p.add_argument("corpus", help="c16.bin")
    p.add_argument("dtype", choices=("f64", "u16"))
    p.add_argument("threads", type=int)
    p.add_argument("reps", type=int)
    args = p.parse_args(argv)
    _common.import_build(args.module_dir)
    import ccdcpp

    corpus = _common.read_corpus(args.corpus)
    H, W, _, T = corpus.spectral.shape
    dates = corpus.dates
    qas = np.ascontiguousarray(np.tile(corpus.qas, (2, 2, 1)))
    src = np.empty((2 * H, 2 * W, 7, T), dtype=np.uint16 if args.dtype == "u16" else np.float64)
    for i in range(2):
        for j in range(2):
            np.clip(corpus.spectral, 0, 65535, out=src[i * H:(i + 1) * H, j * W:(j + 1) * W],
                    casting="unsafe")
    del corpus

    h = ccdcpp.DataSpec()
    base = _common.peak_mb()
    best = float("inf")
    for _ in range(args.reps):
        # A fresh float64 input per call, as a caller's buffer would be.
        s = src.copy() if (args.reps > 1 and args.dtype == "f64") else src
        t0 = time.perf_counter()
        r = ccdcpp.detect_cube(dates, s, qas, spec=h, threads=args.threads)
        best = min(best, time.perf_counter() - t0)
        del r, s

    npx = qas.shape[0] * qas.shape[1]
    print(_common.report(f"cube {args.dtype} t{args.threads}",
                         f"{best * 1e3:.0f} ms, {best / npx * 1e3:.4f} ms/px, "
                         f"input {src.nbytes / 1e6:.0f} MB", _common.peak_mb(), base))
    print(f"ccdcpp {ccdcpp.__file__}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
