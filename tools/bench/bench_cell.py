"""The real-cell timing of one build.

A 500 x 500 cell of a Landsat C2 ARD cube, through detect_cube with spec=presets.landsat_c2()
(raw QA_PIXEL decoded by the Landsat C2 table) and the "ccdc" configuration (with its shared
statistics_end set to the argument `statistics_end`, an ordinal day, when given: 736694 is
pyccd's 2017-12-31), `threads` threads, in strips of `strip` rows as the nlcd plugin runs it.
The bench cell is h024v013 chunk 00030008 (rows 4000-4499, cols 1500-1999 of the tile),
1984-2023, all 2473 dates; `--size` takes a smaller corner of it.

`cube.zarr` is a zarr 3 group holding `landsat-c2ard-sr`, (rows, cols, 7, T) uint16 SR_B1..B6
and QA_PIXEL, and `landsat-c2ard-bt`, (rows, cols, 1, T) uint16 ST, with the ISO dates in the
`time` attribute of `landsat-c2ard-sr`; the cell is rows 4000-4499, cols 1500-1999 of it.

The cell is read once (uint16) before any timing. Each strip is converted to landsat_c2 units
(float64) outside the timed region; only the detect_cube calls are timed, and their sum
is the number reported. The outputs of every strip are hashed in key order, so two builds that
print the same fingerprint gave byte-identical results on all 250,000 pixels.

Prints the process's peak memory at the end, also above the peak once the cell is read (the
input's share). Run with a Python that has zarr 3 and numpy. `package_dir` is a directory holding a `ccdcpp`
package: python/ccdcpp/*.py of the tree under test plus the extension built from it.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package_dir")
    p.add_argument("cube", help="cube.zarr")
    p.add_argument("threads", type=int, nargs="?", default=8)
    p.add_argument("strip", type=int, nargs="?", default=50)
    p.add_argument("statistics_end", type=int, nargs="?")
    p.add_argument("--size", type=int, default=_common.CELL_SIZE,
                   help="the corner of the cell to run, size x size pixels (default: all of it)")
    args = p.parse_args(argv)
    _common.import_build(args.package_dir)
    import ccdcpp
    import zarr  # pyright: ignore[reportMissingImports]
    from ccdcpp import presets

    config = ccdcpp.config("ccdc")
    if args.statistics_end is not None:
        config = config.replace(shared__statistics_end=args.statistics_end)
    N = args.size

    print("ccdcpp", _common.build_of(), "numpy", np.__version__,
          "zarr", zarr.__version__)
    t0 = time.perf_counter()
    cell = _common.read_cell(args.cube, N)
    print(f"read {cell.sr.nbytes / 1e9 + cell.st.nbytes / 1e9:.2f} GB in "
          f"{time.perf_counter() - t0:.1f} s; T={cell.dates.size}")
    base = _common.peak_mb()

    profile = presets.landsat_c2()
    digest = hashlib.sha256()
    t_det, segments, failures = 0.0, 0, 0
    t_loop = time.perf_counter()
    for r0 in range(0, N, args.strip):
        r1 = min(N, r0 + args.strip)
        t_conv = time.perf_counter()
        sp, qa = _common.c2_strip(cell, r0, r1)
        tic = time.perf_counter()
        out = ccdcpp.detect_cube(cell.dates, sp, qa, spec=profile, config=config,
                                 threads=args.threads)
        toc = time.perf_counter()
        t_det += toc - tic
        print(f"  rows {r0}-{r1 - 1}: convert {tic - t_conv:.1f} s, detect {toc - tic:.1f} s",
              flush=True)
        failures += len(out["failures"])
        segments += len(out["row"])
        for k in sorted(out):
            v = out[k]
            if isinstance(v, np.ndarray):
                digest.update(k.encode())
                digest.update(str(v.dtype).encode())
                digest.update(np.ascontiguousarray(v).tobytes())
        del sp, qa, out

    px = N * N
    print(f"strip loop {time.perf_counter() - t_loop:.1f} s")
    print(_common.report(f"cell detect t{args.threads} strip{args.strip}",
                         f"{t_det:.1f} s, {t_det / px * 1e3:.3f} ms/px, segments {segments}, "
                         f"failures {failures}", _common.peak_mb(), base, "the cell read"))
    print("fingerprint", digest.hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
