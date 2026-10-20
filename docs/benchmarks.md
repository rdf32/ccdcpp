# Benchmarks

How fast ccdcpp is, measured on what, and how to measure it again. The tools and their exact
method are in [`tools/bench/README.md`](../tools/bench/README.md); this page is the results.

All numbers are for ccdcpp 0.10.0 on one machine:

- Apple M5 Max, 18 cores (6 performance, 12 efficiency), 64 GB, macOS 26.7.
- Release build (`-O3`, Apple clang 21, libomp), Python 3.11, numpy 1.26.4.
- The default input and configuration (`ccdcpp.Input()` and the `ccdc` preset, at pyccd's
  statistics cutoff as 0.10.0 ran it), or `encoding=presets.landsat_c2()` for the real cell.
- The machine was not otherwise idle: the 1-minute load is given with each table. A
  comparison is only made between runs of the same session.
- Memory is the peak resident set size of the process (`/usr/bin/time -l`), which includes the
  interpreter and the input it holds; each section says how much of it the input is.

## The workloads

| Name | What it is |
|---|---|
| Synthetic | `make_corpus.cpp`'s 2000 pixels at a 16-day cadence, tiled 2 x 2 (8000 pixels, float64 input), through `detect_cube` as columns ("cols") or as a dict per pixel ("cube"). Small and repeatable: the per-change bench. |
| Atlanta cell | A real 500 x 500 Landsat Collection 2 ARD cell (h024v013, 1984-2023, 2473 dates, 7 bands, uint16), through `detect_cube` with `encoding=presets.landsat_c2()` in strips of 50 rows, as a production plugin runs it. 250,000 pixels, 455,231 segments. Read whole before timing; only detection is timed. |
| Annual update | The same cell recorded up to 2022-12-31, then resumed to 2023 with `detect_cube(resume=...)`, against a full run to 2023. |
| PyCCD pixels | Pixels run one at a time through PyCCD's `ccd.detect` and `ccdcpp.detect`. |

## Against PyCCD

One pixel at a time, one thread each: 402 pixels spread over the synthetic corpus and the two
PyCCD-format pixels in `tests/ccd/machine/` (PyCCD's reference pixel `test_3657_3610` and one
Atlanta pixel), 404 in all (`bench_pyccd.py`; PyCCD is lcmap-pyccd 2021.7.19 on numpy 1.26.4,
scipy 1.10.1, scikit-learn 1.9.1). Two rounds, alternating, load 8-12 (other software running):

| | PyCCD with ccdcpp's corrections | PyCCD as released | ccdcpp |
|---|---|---|---|
| median per pixel | 54.6, 52.4 ms | 51.8, 52.5 ms | 0.27 ms |
| total, 404 pixels | 27.4, 27.1 s | 26.7, 26.9 s | 0.15 s |
| same segments as ccdcpp | 404 of 404 | 155 of 404 | |
| peak memory (own process) | 243 MB | | 131 MB |

The two pixels from PyCCD's own test data, over the four rounds: 68-73x on its 443-observation
reference pixel (192-205 ms against 2.8 ms), 250-256x on the Atlanta pixel (313-322 ms against
1.25 ms). Each PyCCD time is one call, each ccdcpp time the best of five.

- **Speed:** ccdcpp is about 200 times faster per pixel (median), 174-178 times on the total.
- **The same answers:** ccdcpp corrects PyCCD's code where it evidently does not do what it
  intends (README, [Differences from pyccd](../README.md#differences-from-pyccd)), and the
  synthetic corpus is built to reach those cases, so against PyCCD as released only 155 of the
  404 pixels have the same segments. `patch_pyccd.py` makes a copy of PyCCD with exactly those
  corrections; against it every pixel has the same segments (404 of 404), and its speed is
  PyCCD's.
- **Memory:** each in its own process (`--only`), peak resident memory is 131 MB for ccdcpp and
  243 MB for PyCCD, almost all of it the interpreter, the libraries (PyCCD's SciPy and
  scikit-learn) and the loaded corpus: after the first pixel, the other 403 raise the peak by
  under 3 MB in either.


## The Atlanta cell: threads

`detect_cube` on the whole Atlanta cell, strips of 50 rows, one run per thread count
(load 6-18 before the runs, from the run before):

| Threads | Detect | ms per pixel | Speed-up | Peak memory |
|---|---|---|---|---|
| 1 | 292.4 s | 1.170 | 1.00 | 17.6 GB |
| 2 | 149.7 s | 0.599 | 1.95 | 19.9 GB |
| 4 | 80.0 s | 0.320 | 3.66 | 19.9 GB |
| 6 | 55.2 s | 0.221 | 5.30 | 19.9 GB |
| 8 | 42.4 s | 0.169 | 6.90 | 19.8 GB |
| 12 | 30.6 s | 0.122 | 9.56 | 19.7 GB |
| 18 | 22.8 s | 0.091 | 12.8 | 19.9 GB |

Every run gave the same 455,231 segments and the same output fingerprint. The first six threads
are the M5 Max's performance cores; past them the efficiency cores add less each.

The memory is mostly the input as this bench holds it: the cell read whole (9.9 GB of uint16,
13.0 GB peak while reading), then each 50-row strip converted to float64 (3.5 GB) on top, about
7 GB above the cell at the peak with the strip's outputs; it barely moves with the thread count.
A caller holding less of the cell at once (the nlcd plugin reads per strip) needs less.


## Release 0.10.0 against 0.9.0

The 0.10.0 changes (see [`floating-point-sums.md`](floating-point-sums.md) and the CHANGELOG)
halve the time. Three rounds each, alternating builds, load 7-18:

| Workload | 0.9.0 | 0.10.0 | Ratio | Peak memory, 0.9.0 / 0.10.0 |
|---|---|---|---|---|
| Synthetic cols, 1 thread | 5750 ms (0.72 ms/px) | 3059 ms (0.38 ms/px) | 0.53 | 837 / 836 MB |
| Synthetic cols, 8 threads | 784 ms | 434 ms | 0.55 | 842 / 842 MB |
| Synthetic cube, 8 threads | 828 ms | 467 ms | 0.56 | 918 / 919 MB |
| Atlanta cell, 8 threads | 79.6 s (0.32 ms/px) | 39.5 s (0.16 ms/px) | 0.50 | 19.8 / 19.9 GB |

Medians of three. Same 455,231 segments on the cell; coefficients, RMSEs and magnitudes differ
in their last bits only (results version 2). The memory did not move: in the synthetic bench
478 MB of it is the process with the corpus loaded before the call, and on the cell almost all
of it is the cell held in memory.

Releases 0.6.1 to 0.9.0 (the redesign into a state machine, the 0.8.0 fixes and resume) kept
the real cell within 2% of 0.6.1's speed; 0.8.0's refit at 24 observations cost about 6% on the
single-thread synthetic bench (CHANGELOG, 0.8.0). Against 0.6.1 as measured in those sessions,
0.10.0 is about twice as fast on the cell (39.5 s against 78.7-83.9 s) and 1.8-1.9 times on the
single-thread synthetic bench.

## Annual update

The Atlanta cell, 2,384 dates recorded to 2022-12-31, 89 dates of 2023 added, 8 threads, two
rounds, load 9-12:

| | Time | Peak memory |
|---|---|---|
| Full run to 2023 | 40.1 s, 38.7 s | |
| Update (`detect_cube(resume=...)`) | 14.6 s, 13.9 s | |
| Update / full | 0.36 | |
| The whole bench (stored run, update and full run in one process) | | 19.9 GB |

Every pixel resumed, and the update equals the full run on all 250,000 pixels. The update
still reads each pixel's whole series and refits its open segment, so its cost is mostly that
refit, not the new year's tests.

## Measuring again

The commands, the rounds and the tolerances are in
[`tools/bench/README.md`](../tools/bench/README.md#method). In short:

- Build Release with the same configure line for every build compared, and run them in the
  same session, alternating.
- Note the load (`sysctl vm.loadavg`) before each run; rerun if the spread passes 3%.
- The cell bench prints a fingerprint of every output: equal fingerprints mean byte-identical
  results.
