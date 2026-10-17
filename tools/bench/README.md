# Timing harness

How the state-machine refactor of the Standard procedure is timed. Every phase reports a
synthetic bench and, at the phases that move the hot loop, a real cell. The budget is a ratio
against v0.6.1 measured in the same session.

| File | What it does |
|---|---|
| `make_corpus.cpp` | Writes the synthetic corpus: 2000 pixels at a 16-day cadence, from `tests/ccd/golden/synthetic.hpp` (the golden corpus's `c16` cube). |
| `bench_synthetic.py` | Times one entry point on that corpus tiled 2x2 (8000 pixels), best of N calls. |
| `bench.sh` | Runs `bench_synthetic.py` for the three reported configs, each under `/usr/bin/time -l`. |
| `bench_cell.py` | Times `detect_cube_columns` with `presets.landsat_c2()` on a 500 x 500 Landsat C2 ARD cell and prints a fingerprint of every output. |
| `bench_resume.py` | An annual update of the same cell: a recorded run on the dates before a cut, `resume_cube_columns` on all of them, and the full run; checks the update equals the full run on every pixel and times the three. |

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

1. Synthetic. Run `bash tools/bench/bench.sh <python> <build dir> c16.bin` three times back to
   back, with nothing else of yours running. The default configs are `cols f64 t1` (best of 2),
   `cols f64 t8` and `cube f64 t8` (best of 3 each). Note `sysctl vm.loadavg` before each run.
   Report the per-run numbers and the median. Rerun if the spread is above 3%, or if the
   1-minute load is far above about 10.
2. Cell. Run three times:

   ```
   /usr/bin/time -l <python with zarr 3> -I -W ignore tools/bench/bench_cell.py <package dir> <cube.zarr> 8 50
   ```

   Report the median detect time, the peak RSS (`maximum resident set size`) and the
   fingerprint. A build that prints the baseline's fingerprint gave byte-identical results on
   all 250,000 pixels.
3. Measure v0.6.1 again in the same session whenever the load differs much from the last
   baseline. The budget (at most 2% slower) is checked on the finished machine against v0.6.1.
