# Benchmarks

How fast ccdcpp is, measured on what, and how to measure it again. The tools and their exact
method are in [`tools/bench/README.md`](../tools/bench/README.md); this page is the results.

The numbers are for ccdcpp 1.0, against PyCCD and against 0.10.0, on one machine:

- Apple M5 Max, 18 cores (6 performance, 12 efficiency), 64 GB, macOS 26.7, on AC power.
- Release build (`-O3 -DNDEBUG`, Apple clang 21, Homebrew libomp 23.1.1), Python 3.11.16,
  numpy 1.26.4 (zarr 3.1.5 for the cell).
- The default data spec and configuration (`ccdcpp.DataSpec()` and the `ccdc` preset, which has no
  statistics cutoff), or `spec=presets.landsat_c2()` for the real cell; where a run is at
  pyccd's statistics cutoff, `statistics_end=736694` (2017-12-31), the table says so.
- Memory is the peak resident set size of the process (MB of 10^6 bytes), as each bench prints
  it (the number `/usr/bin/time -l` reports), which includes the interpreter and the input it
  holds; each section says how much of it the input is. Every time has its side's peak beside
  it.

## Method

- **One process per side.** Each side of a comparison (build, implementation, call form or
  phase) runs in its own process, so each peak is its own.
- **Alternating rounds.** The two builds compared run in the same session, in rounds whose
  order alternates, except where a section says otherwise; a ratio is the median of the
  per-round ratios (paired medians).
- **The load rule.** A block of runs starts when the 1-minute load is below 1.5, or the
  processes outside the block have used under 100% CPU for 60 s; if those other processes
  pass 200% (60 s rolling mean) during the block, it runs again once and the rerun counts.
  The bench's own threads raise the load during a block; the table gives the load at its start
  where it matters.

## The workloads

| Name | What it is |
|---|---|
| Synthetic | The golden corpus's `c16` cube (`golden_driver gen`), 2000 pixels at a 16-day cadence, tiled 2 x 2 (8000 pixels), through `detect_cube`, float64 or uint16 input. Small and repeatable: the per-change bench. |
| Atlanta cell | A real 500 x 500 Landsat Collection 2 ARD cell (h024v013, 1984-2023, 2473 dates, 7 bands, uint16), through `detect_cube` with `spec=presets.landsat_c2()` in strips of 50 rows, as a production plugin runs it. 250,000 pixels: 458,018 segments under `ccdc`, 455,231 at `statistics_end=736694`. Read whole before timing (9.89 GB in 3 s); only detection is timed. |
| Annual update | The same cell recorded up to 2022-12-31, then resumed to 2023 with `detect_cube(resume=..., freeze_statistics=True)`, against a full run to 2023 with statistics capped at 2022-12-31. |
| PyCCD pixels | Pixels run one at a time through PyCCD's `ccd.detect` and `ccdcpp.detect`. |
| Corpus pixels | The first 500 pixels of the synthetic corpus, one `ccdcpp.detect` call each, single-threaded (`bench_pixel.py`). |

## Against PyCCD

One pixel at a time, one thread each: 402 pixels spread over the synthetic corpus and the two
PyCCD-format pixels in `tests/ccd/data/` (PyCCD's reference pixel `test_3657_3610` and one
Atlanta pixel), 404 in all (`bench_pyccd.py`, ccdcpp at pyccd's statistics cutoff; PyCCD is
lcmap-pyccd 2021.7.19 on numpy 1.26.4, scipy 1.10.1, scikit-learn 1.9.1). Each side in its own
process, two rounds, load 1.4-4.0 at the start of each (37 for the last, from the round before):

| | PyCCD with ccdcpp's corrections | PyCCD as released | ccdcpp |
|---|---|---|---|
| median per pixel | 53.0, 56.0 ms | 51.3, 53.3 ms | 0.261-0.266 ms |
| total, 404 pixels | 27.1, 28.4 s | 26.4, 27.0 s | 0.151-0.152 s |
| ratio of the medians | 203x, 211x | 195x, 204x | |
| ratio of the totals | 178x, 188x | 175x, 178x | |
| same segments as ccdcpp | 394 of 404 | 145 of 404 | |
| peak memory (own process) | 241 MB | 240-241 MB | 134 MB |

The two pixels from PyCCD's own test data, over the four rounds: 69-73x on its 443-observation
reference pixel (192-196 ms against 2.65-2.77 ms), 236-256x on the Atlanta pixel (311-314 ms
against 1.23-1.32 ms). Each PyCCD time is one call, each ccdcpp time the best of five.

- **Speed:** ccdcpp is about 200 times faster per pixel (median), 175-188 times on the total.
- **The same answers:** ccdcpp corrects PyCCD's code where it evidently does not do what it
  intends (README, [Differences from pyccd](../README.md#differences-from-pyccd)), and the
  synthetic corpus is built to reach those cases, so against PyCCD as released only 145 of the
  404 pixels have the same segments. `patch_pyccd.py` makes a copy of PyCCD with those
  corrections; against it 394 of the 404 pixels have the same segments, and its speed is
  PyCCD's. The other 10 are synthetic pixels where PyCCD fits one InsufficientClear segment and
  ccdcpp none, because ccdcpp's InsufficientClear chain tests the thermal range in degrees
  Celsius (README, "The thermal range outside Standard"), which the patched copy does not.
- **Memory:** each in its own process (`--only`), peak resident memory is 134 MB for ccdcpp and
  241 MB for PyCCD, almost all of it the interpreter, the libraries (PyCCD's SciPy and
  scikit-learn) and the loaded corpus: after the first pixel, the other 403 raise the peak by
  under 3 MB in either.

## Per pixel: 1.0 and 0.10.0

`bench_pixel.py` on the first 500 corpus pixels (0.10.0's forms by the same loop), one thread,
best of 3 passes, five alternating rounds, each form in its own process, load 1.5-6.7. 1.0 runs
`ccdc` without a statistics cutoff, 0.10.0 its default at 2017-12-31. The time is the median of
the rounds (their range in brackets), the ratio the median of the paired rounds:

| Form | 0.10.0 | 1.0 | Ratio | Peak memory, 0.10.0 / 1.0 |
|---|---|---|---|---|
| `spec=ccdcpp.DataSpec()` (0.10.0: `HarmonicOptions()`, `LassoOptions()`) | 0.3910 ms (0.3850-0.3935) | 0.3908 ms (0.3880-0.3923) | 0.998 | 128 / 131 MB |
| `spec=presets.landsat_c1_dn()` (0.10.0: `profile=`) | 0.4033 ms (0.3913-0.4093) | 0.3931 ms (0.3918-0.4071) | 0.973 | 128 / 131 MB |
| `config=` a `Config` built once (0.10.0: none) | | 0.3979 ms (0.3902-0.4149) | | 131 MB |

The fixed cost of a call, on one all-fill pixel (the corpus's dates, every QA code fill, no
segment), 20,000 calls, best of 5, three rounds:

| Form | 0.10.0 | 1.0 | Peak memory, 0.10.0 / 1.0 |
|---|---|---|---|
| `DataSpec()` (0.10.0: `HarmonicOptions()`) | 7.2 us | 6.3 us | 126 / 127 MB |
| a profile | 11.5 us | 6.5 us | 126 / 127 MB |
| `config=` | | 6.3 us | 127 MB |

A call resolves and builds nothing it has before: a `Config` compiles its `ccd::Algorithm` once,
when it is built, and the `Algorithm` resolves the configuration once per data spec it meets. The
memory is almost all the interpreter, numpy and the loaded corpus.

## Synthetic cube: 1.0 and 0.10.0

`bench.sh` (`detect_cube`, returning columns), three alternating rounds, load 1.2-4.0 at the
start. 1.0 runs `ccdc` without a statistics cutoff, 0.10.0 its default at 2017-12-31. Medians,
with the range of the rounds:

| Config | 0.10.0 | 1.0 | Ratio | Peak memory, 0.10.0 / 1.0 |
|---|---|---|---|---|
| float64, 1 thread | 2920 ms (2908-3060) | 3013 ms (2948-3029) | 1.010 | 838 / 838 MB |
| float64, 8 threads | 409 ms (407-414) | 419 ms (419-421) | 1.029 | 842 / 846 MB |
| uint16, 1 thread | 2919 ms (2896-3014) | 3023 ms (3022-3047) | 1.036 | 238 / 239 MB |
| uint16, 8 threads | 409 ms (408-419) | 424 ms (422-426) | 1.032 | 242 / 246 MB |

The input is 343 MB as float64 and 86 MB as uint16; with the corpus loaded the process holds
about 480 MB (float64) or 225 MB (uint16) before the call, and detection adds 360 MB or 15-22 MB.

## The Atlanta cell: 8 threads

`bench_cell.py` on the whole cell, 8 threads, three rounds, load 1.5-9.6 at the start of each
run (from the run before):

| Run | Detect (median; the rounds) | Ratio | Segments | Fingerprint | Peak memory |
|---|---|---|---|---|---|
| 0.10.0 | 39.7 s (38.9, 39.7, 40.2) | | 455,231 | `c7739aca...` | 19.9-20.0 GB |
| 1.0, `statistics_end=736694` | 40.8 s (40.5, 41.2, 40.8) | 1.038 | 455,231 | `c7739aca...` | 19.9-20.0 GB |
| 1.0, `ccdc` | 41.9 s (41.6, 42.3, 41.9) | 1.066 | 458,018 | `30a6bd19...` | 19.9 GB |

At pyccd's statistics cutoff 1.0 prints 0.10.0's fingerprint: its output is byte-identical on
all 250,000 pixels. Under `ccdc`, which reads the whole series for its statistics, 2,787 more
segments are found.

On a cube 1.0 measures 1-4% slower than 0.10.0 in wall time (1.010-1.036 on the synthetic cube,
1.038 on the cell at the same cutoff), but instruction counts put the real difference at about
1%: at the same configuration and identical outputs, 1.0's detect runs 0.7-1.05% more
instructions and 0.9-1.4% more cycles than 0.10.0 (cell, 8 threads: 1.0105 and 1.0116;
synthetic cube: 1.0071-1.0083 and 1.0088-1.0136). 0.10.0 rebuilt with today's compiler matches
the released build within 0.05%, and `CCD_OBSERVE=OFF` saves 0.03-0.12%. So 1.0 is about 1%
slower on cubes, and at or below 0.10.0 per pixel; the rest of the wall-time gap was the order
of the rounds (the cell's ran 0.10.0 first each time, and one round with 1.0 first gave 39.6 s
against 0.10.0's 40.2 s) and the machine heating through the block. The counts are the
instructions retired that `/usr/bin/time -l` reports per process, three rounds in rotated
order, with `OPENBLAS_NUM_THREADS=1 KMP_BLOCKTIME=0` (otherwise idle threads' spinning is
counted).

## The Atlanta cell: threads

`bench_cell.py` on the whole cell under `ccdc`, one run per thread count, load 1.5-6.9 at the
start of each (from the run before):

| Threads | Detect | ms per pixel | Speed-up | Peak memory |
|---|---|---|---|---|
| 1 | 284.4 s | 1.137 | 1.00 | 19.9 GB |
| 2 | 145.8 s | 0.583 | 1.95 | 19.8 GB |
| 4 | 75.0 s | 0.300 | 3.79 | 19.9 GB |
| 8 | 42.8 s | 0.171 | 6.64 | 19.9 GB |
| 12 | 29.3 s | 0.117 | 9.71 | 19.9 GB |
| 18 | 20.6 s | 0.082 | 13.8 | 20.0 GB |

Every run gave the same 458,018 segments and the same output fingerprint. The first six threads
are the M5 Max's performance cores; past them the efficiency cores add less each.

The memory is mostly the input as this bench holds it: the cell read whole (9.9 GB of uint16,
about 12.9 GB peak once read), then each 50-row strip converted to float64 on top, about 7 GB
above the cell at the peak with the strip's outputs; it barely moves with the thread count. A
caller holding less of the cell at once (the nlcd plugin reads per strip) needs less.

## Annual update

`bench_resume.py` on the Atlanta cell, 2,384 dates recorded to 2022-12-31, 89 dates of 2023
added, 8 threads, each phase in its own process, two rounds, load 1.5-8.6 at the start:

| Phase | Round 1 | Round 2 | Peak memory |
|---|---|---|---|
| Stored run (`record=True`, to 2022-12-31) | 41.5 s | 41.6 s | 19.9 GB |
| Update (`detect_cube(resume=..., freeze_statistics=True)`) | 14.2 s | 14.2 s | 19.9-20.0 GB |
| Full run to 2023, statistics capped at 2022-12-31 | 40.7 s | 40.9 s | 19.9 GB |
| Update / full | 0.348 | 0.347 | |

All 250,000 pixels resumed, and the update is identical to the full run with statistics capped
at the stored run's end on every one of them (the same fingerprint in both rounds). `ccdc` has
no statistics cutoff, so the update passes `freeze_statistics=True`: its statistics stay those
of the stored run until a full rerun reads the new year too. Each phase's peak is the cell held in memory (about 12.9 GB) and
about 7 GB of strip and outputs on top. The update still reads each pixel's whole series and
refits its open segment, so its cost is mostly that refit, not the new year's tests.

## Measuring again

The commands, the rounds and the tolerances are in
[`tools/bench/README.md`](../tools/bench/README.md#method). In short:

- Build Release with the same configure line for every build compared, and run them in the
  same session, alternating the order of the rounds. `ccdcpp.build_info()` says what a build
  is: its `build_type` is `Release` and `openmp` is `True` for a threaded timing.
- Run each side in its own process; report its time and its peak memory.
- Start a block on a quiet machine and note the load (`sysctl vm.loadavg`) before each run;
  rerun if other processes load the machine during it, or if the spread passes 3%.
- The cell bench prints a fingerprint of every output: equal fingerprints mean byte-identical
  results.
