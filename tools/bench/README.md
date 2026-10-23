# Timing harness

How a build of ccdcpp is timed against another. A change reports the synthetic bench and, if it
moves the hot loop, the real cell and the per-pixel bench; its cost is the ratio against the
build before it, measured in the same session. [docs/benchmarks.md](../../docs/benchmarks.md)
has the results.

| File | What it does |
|---|---|
| `bench_synthetic.py` | Times `detect_cube` on the synthetic corpus (`c16.bin`, [below](#setup)) tiled 2x2 (8000 pixels), best of N calls. |
| `bench.sh` | Runs `bench_synthetic.py` for the two reported configs, each in its own process. |
| `bench_cell.py` | Times `detect_cube` with `spec=presets.landsat_c2()` on a 500 x 500 Landsat C2 ARD cell (`--size n`: its n x n corner) and prints a fingerprint of every output. |
| `bench_pixel.py` | Times `detect` per pixel, single-threaded, with a DataSpec, a profile and a named configuration, each in its own process (`--only <form>` runs one). |
| `bench_resume.py` | An annual update of the same cell: a recorded run on the dates before a cut, `detect_cube(resume=, freeze_statistics=True)` on all of them, and the full run with statistics capped at the stored end; checks the update equals that full run on every pixel and times the three, each in its own process (`--only <phase>` runs one). |
| `bench_pyccd.py` | Times PyCCD's `ccd.detect` and `ccdcpp.detect` on the same pixels, one thread each (corpus pixels and PyCCD-format CSVs), and counts the pixels whose segments agree; each runs in its own process (`--only pyccd` or `--only ccdcpp` runs one). |
| `bench_preprocess.cpp` | Times `apply()` of each `ccdc` preprocessing chain against the chain written out by hand (`tests/ccd/support/preprocess_reference.hpp`) on PyCCD's reference pixel, and prints its peak memory; exits 1 if a mask differs. Built with the tests as the target `bench_preprocess`; ctest does not run it. |
| `_common.py` | What the scripts share: loading the build under test (each checks `import ccdcpp` loaded from the directory given), reading the corpus and the cell, the C2 conversion, best-of-N timing, the peak memory, and running each side of a comparison in its own process. |
| `patch_pyccd.py` | Copies PyCCD and applies ccdcpp's corrections to it (README, "Differences from pyccd"), so the two can be timed on the same algorithm. |

## Setup

Build the tree under test Release with the Python extension, with the same configure line for
every build you compare. A build directory holds the bare extension (`ccdcpp.cpython-*.so`),
which is enough for the synthetic bench. The cell bench needs the package: copy
`python/ccdcpp/*.py` of the tree and the built extension into `<dir>/ccdcpp/`.

The synthetic corpus is the golden corpus's `c16` cube: 2000 pixels at a 16-day cadence, from
`tests/ccd/support/synthetic.hpp`. Make it once with the golden driver
(`tools/linux-compare/golden_driver.cpp`), which writes `c16.bin` with the other cubes:

```
cmake -S <repo> -B <build dir> -DCMAKE_BUILD_TYPE=Release -DCCD_BUILD_GOLDEN_DRIVER=ON
cmake --build <build dir> --target golden_driver
<build dir>/golden_driver gen <corpus dir>
```

## Method

Every bench prints its peak memory (the process's peak resident set size, in MB of 10^6 bytes)
next to its time, and how far it is above the peak once the input is held; a comparison runs each
side (implementation, call form or phase) in its own process, so each peak is its own.

Two builds compared run in the same session, in rounds whose order alternates (A B, B A, ...);
a ratio is the median of the per-round ratios. Start a block of runs on a quiet machine (the
1-minute load below 1.5, or other processes under 100% CPU for 60 s), note `sysctl
vm.loadavg` before each run, and run the block again once if other processes pass 200% CPU
during it.

1. Synthetic. Run `bash tools/bench/bench.sh <python> <build dir> c16.bin` three times, with
   nothing else of yours running. The default configs are `f64 t1` (best of 2) and `f64 t8`
   (best of 3); a config is `"dtype threads reps"`, e.g. `"u16 8 3"` for uint16 input.
   Report the per-run numbers and the median. Rerun if the spread is above 3%.
2. Cell. Run three times:

   ```
   <python with zarr 3> -I -W ignore tools/bench/bench_cell.py <package dir> <cube.zarr> 8 50
   ```

   Report the median detect time, the peak memory and the fingerprint; the peak above the one
   once the cell is read is what detection adds to the input. A build that prints the
   baseline's fingerprint gave byte-identical results on all 250,000 pixels. A third number
   after the strip, `8 50 736694`, sets `ccdc`'s `statistics_end` (pyccd's 2017-12-31), the
   cutoff 0.10.0 ran at. With `8 5 --size 20` (a 20 x 20 corner in strips of 5) it only checks
   that the script runs. For thread scaling, run it once per thread count.
3. Measure the baseline build again in the same session whenever the load differs much from
   its last run.
4. PyCCD. With an interpreter that has lcmap-pyccd 2021.7.19's dependencies (numpy, scipy,
   scikit-learn) and PyCCD's `ccd` package in `<pyccd dir>`:

   ```
   <python> -I -W ignore tools/bench/bench_pyccd.py <package dir> <pyccd dir> c16.bin 402 \
       tests/ccd/data/test_3657_3610_observations.csv \
       tests/ccd/data/test_atlanta_024013_00030008_481_393.csv
   ```

   Run it twice and report the medians, totals and ratios of the last lines. For the same
   algorithm on both sides, run it again with `<pyccd dir>` a copy made by
   `python3 -I tools/bench/patch_pyccd.py <pyccd dir>/ccd <out dir>` (394 of the 404 pixels then
   have the same segments; the other 10 are InsufficientClear pixels that ccdcpp's
   degrees-Celsius thermal check rejects, which the copy does not patch). Each implementation's
   line gives its peak memory.
5. Per pixel. `<python> -I tools/bench/bench_pixel.py <package dir> c16.bin 500 3` times the
   first 500 corpus pixels, best of 3, for each call form in its own process; run it five
   times, alternating with the baseline.
6. Annual update. `<python with zarr 3> -I -W ignore tools/bench/bench_resume.py <package dir>
   <cube.zarr> 8 50` runs the stored run, the update and the full run (statistics capped at
   the stored end) each in its own process, and exits non-zero if the update differs from the
   full run on any pixel. Run it twice and
   report each phase's time and peak and the update / full ratio.
