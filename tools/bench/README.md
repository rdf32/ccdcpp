# Timing harness

How the state-machine refactor of the Standard procedure is timed. Every phase reports a
synthetic bench and, at the phases that move the hot loop, a real cell. The budget is a ratio
against v0.6.1 measured in the same session.

| File | What it does |
|---|---|
| `make_corpus.cpp` | Writes the synthetic corpus: 2000 pixels at a 16-day cadence, from `tests/ccd/golden/synthetic.hpp` (the golden corpus's `c16` cube). |
| `bench_synthetic.py` | Times `detect_cube` on that corpus tiled 2x2 (8000 pixels), best of N calls. |
| `bench.sh` | Runs `bench_synthetic.py` for the two reported configs, each in its own process. |
| `bench_cell.py` | Times `detect_cube` with `input=presets.landsat_c2()` on a 500 x 500 Landsat C2 ARD cell (`--size n`: its n x n corner) and prints a fingerprint of every output. |
| `bench_pixel.py` | Times `detect` per pixel, single-threaded, with an Input, a profile and a named configuration, each in its own process (`--only <form>` runs one). |
| `bench_resume.py` | An annual update of the same cell: a recorded run on the dates before a cut, `detect_cube(resume=)` on all of them, and the full run; checks the update equals the full run on every pixel and times the three, each in its own process (`--only <phase>` runs one). |
| `bench_pyccd.py` | Times PyCCD's `ccd.detect` and `ccdcpp.detect` on the same pixels, one thread each (corpus pixels and PyCCD-format CSVs), and counts the pixels whose segments agree; each runs in its own process (`--only pyccd` or `--only ccdcpp` runs one). |
| `_common.py` | What the scripts share: loading the build under test (each checks `import ccdcpp` loaded from the directory given), reading the corpus and the cell, the C2 conversion, best-of-N timing, the peak memory, and running each side of a comparison in its own process. |
| `patch_pyccd.py` | Copies PyCCD and applies ccdcpp's corrections to it (README, "Differences from pyccd"), so the two can be timed on the same algorithm. |

## Setup

Build the tree under test Release with the Python extension, with the same configure line for
every build you compare. A build directory holds the bare extension (`ccdcpp.cpython-*.so`),
which is enough for the synthetic bench. The cell bench needs the package: copy
`python/ccdcpp/*.py` of the tree and the built extension into `<dir>/ccdcpp/`.

Make the corpus once:

```
c++ -std=c++17 -O2 -I <repo>/tests/ccd/golden <repo>/tools/bench/make_corpus.cpp -o make_corpus
./make_corpus c16.bin
```

## Method

Every bench prints its peak memory (the process's peak resident set size, in MB of 10^6 bytes)
next to its time, and how far it is above the peak once the input is held; a comparison runs each
side (implementation, call form or phase) in its own process, so each peak is its own.

1. Synthetic. Run `bash tools/bench/bench.sh <python> <build dir> c16.bin` three times back to
   back, with nothing else of yours running. The default configs are `f64 t1` (best of 2) and
   `f64 t8` (best of 3). Note `sysctl vm.loadavg` before each run.
   Report the per-run numbers and the median. Rerun if the spread is above 3%, or if the
   1-minute load is far above about 10.
2. Cell. Run three times:

   ```
   <python with zarr 3> -I -W ignore tools/bench/bench_cell.py <package dir> <cube.zarr> 8 50
   ```

   Report the median detect time, the peak memory and the fingerprint; the peak above the one
   once the cell is read is what detection adds to the input.
   With `8 5 --size 20` (a 20 x 20 corner in strips of 5) it checks that the script runs. A build that prints the baseline's fingerprint gave byte-identical results on
   all 250,000 pixels.
3. Measure v0.6.1 again in the same session whenever the load differs much from the last
   baseline. The budget (at most 2% slower) is checked on the finished machine against v0.6.1.
4. PyCCD. With an interpreter that has lcmap-pyccd 2021.7.19's dependencies (numpy, scipy,
   scikit-learn) and PyCCD's `ccd` package in `<pyccd dir>`:

   ```
   <python> -I -W ignore tools/bench/bench_pyccd.py <package dir> <pyccd dir> c16.bin 402 \
       tests/ccd/machine/test_3657_3610_observations.csv \
       tests/ccd/machine/test_atlanta_024013_00030008_481_393.csv
   ```

   Run it twice and report the medians, totals and ratios of the last lines. For the same
   algorithm on both sides, run it again with `<pyccd dir>` a copy made by
   `python3 -I tools/bench/patch_pyccd.py <pyccd dir>/ccd <out dir>` (every pixel then has the
   same segments). Each implementation's line gives its peak memory.
