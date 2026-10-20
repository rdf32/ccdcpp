# ccdcpp

ccdcpp is a C++17 implementation of the Continuous Change Detection and Classification (CCDC) algorithm, with Python bindings through pybind11. It reproduces the USGS LCMAP PyCCD algorithm and its outputs, doing what PyCCD's code intends where the code does otherwise (every such difference is listed under [Differences from pyccd](#differences-from-pyccd)), and runs one pixel about 200 times faster than PyCCD in about half the process memory. A run takes one pixel or a whole image cube, in parallel with OpenMP, and can be continued when new observations arrive instead of rerun.

> **Step 5 state.** The description of the data is `ccdcpp.Input` (C++ `ccd::Input`): the QA
> codes, a role per row (`roles`), `saturation_bands`, `reflectance_unit` (Collection-1 DN 10000,
> reflectance 1) and the bounds and thermal transform, in snake_case. `HarmonicOptions` is gone,
> and the entry points take `input=` where they took `hoptions=`; `profile.input()` replaces
> `profile.options()`. A configuration is `{"shared": ..., "stages": ...}` in physical terms
> (`ccdcpp.Config`, C++ `config::Config`): the values several stages read are written once in
> `shared` (`detection_bands` by role, `change_probability`, `outlier_probability`, the Lasso's
> `alpha`, `coef_floor`, `max_iter`, `tolerance`, the coefficient rule, `stat_ord`,
> `green_filter` and `green_filter_scope`), reflectance values are multiplied by the input's
> `reflectance_unit`, and the change and outlier thresholds are the chi-squared quantiles of the
> probabilities for the detection-band count, all when a run resolves the configuration against
> its input (C++ `ccd::Algorithm`, which caches the resolution per input). Some examples below
> still use the earlier names; the full documentation comes with the two-call API.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Configuring the algorithm](#configuring-the-algorithm)
- [Diagnostics](#diagnostics)
- [Resuming a run](#resuming-a-run)
- [How it works](#how-it-works)
- [Performance](#performance)
- [Differences from pyccd](#differences-from-pyccd)
- [Reference](#reference): [input](#input-arrays), [band layout](#band-layout), [input units](#input-unit-convention), [QA](#qa-input), [cadence](#cadence-parameters), [sensor profiles](#sensor-profiles), [preprocessing](#preprocessing)
- [Build and test](#build-and-test)
- [License and Attribution](#license-and-attribution)

# Features

- **PyCCD's algorithm as it intends**, every difference from its code listed and checked against Zhu's MATLAB reference ([Differences from pyccd](#differences-from-pyccd)).
- **Fast**: about 200 times PyCCD's speed per pixel in about half its process memory; a 500 x 500 Landsat cell over 40 years in 23 s on 18 threads ([Performance](#performance)).
- **One pixel or a cube**: `detect` for a pixel, `detect_cube` and `detect_cube_columns` for an image cube in parallel (OpenMP, the GIL released), reading float64, float32 or uint16 cubes in any layout without copying them.
- **Sensor profiles**: band roles, units, QA decoding, options and the configuration for the input in one object, with presets for Landsat Collection 1 and 2, HLS v2.0 and Sentinel-2 L2A (`ccdcpp.presets`).
- **Raw QA input**: Landsat `QA_PIXEL` and `pixel_qa`, HLS `Fmask` and Sentinel-2 `SCL` decoded through lookup tables per pixel.
- **A configurable algorithm**: one fixed state machine whose decisions are stages, each decided by a named strategy with parameters; configurations are immutable values, TOML presets and part of a run's record ([Configuring the algorithm](#configuring-the-algorithm)).
- **Diagnostics**: every transition, change test, dropped observation and segment of the pixels you name, as rows for pandas, without changing the output ([Diagnostics](#diagnostics)).
- **Resume**: a stored run continued on new observations, equal bit for bit to a full rerun, at about a third of its time ([Resuming a run](#resuming-a-run)).
- **Custom preprocessing**: the filters that build the processing mask are a plan you can edit ([Preprocessing](#preprocessing)).
- **A C++ API** of two calls, `ccd::run_pixel` and `ccd::run_cube`, and a CMake build.

# Quick start

## Install

ccdcpp builds with CMake through scikit-build-core. It needs Python 3.11 or later, numpy, a C++17 compiler, Boost, Eigen 3 and pybind11; OpenMP is optional (without it a cube runs on one thread). `environment.yml` is the conda environment it is developed in (a Windows export; elsewhere install the same packages from conda-forge).

```bash
conda env create -f environment.yml
conda activate geo
pip install .
```

## One pixel

`detect` runs one pixel. The arrays are `dates` (T,), proleptic Gregorian ordinals in non-decreasing order, `spectral` (bands, T) and `qas` (T,). A sensor profile says what the rows are and in which units; here `presets.landsat_c2()`, Landsat Collection 2 surface reflectance with its thermal band in degrees Celsius / 100 and the raw `QA_PIXEL` band. The pixel is synthetic: twenty years at a 16-day cadence with an annual cycle, noise, 15% of observations cloudy and a land-cover change in June 2012.

```python
from datetime import date

import numpy as np
import ccdcpp
from ccdcpp import presets

rng = np.random.default_rng(42)
dates = np.arange(date(2000, 1, 1).toordinal(), date(2020, 1, 1).toordinal(), 16, dtype=np.int64)
T = dates.size
season = np.sin(2 * np.pi * dates / 365.25)

# Rows: blue, green, red, nir, swir1, swir2 (reflectance 0..1), thermal (degrees C / 100).
level = np.array([0.04, 0.07, 0.06, 0.30, 0.18, 0.09])
spectral = np.empty((7, T))
spectral[:6] = level[:, None] * (1 + 0.3 * season) + rng.normal(0, 0.005, (6, T))
spectral[:6, dates >= date(2012, 6, 1).toordinal()] *= [[1.3], [1.4], [1.6], [0.6], [1.5], [1.7]]
spectral[6] = 0.20 + 0.10 * season

qas = np.full(T, 21824, dtype=np.uint16)   # QA_PIXEL clear
qas[rng.random(T) < 0.15] = 22280          # QA_PIXEL cloud

out = ccdcpp.detect(dates, spectral, qas, profile=presets.landsat_c2())

for m in out["models"]:
    print(date.fromordinal(m["start_day"]), date.fromordinal(m["end_day"]),
          "break", date.fromordinal(m["break_day"]),
          "change probability", m["change_probability"], "observations", m["observation_count"])
print(out["band_names"])
print(sorted(out["models"][0]["bands"][0]))
```

```text
2000-01-17 2012-05-25 break 2012-06-10 change probability 1.0 observations 239
2012-06-10 2019-07-16 break 2019-08-01 change probability 0.0 observations 131
('blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal')
['coefficients', 'intercept', 'magnitude', 'rmse']
```

Two segments: the change on the first clear observation after 2012-06-01 (change probability 1), and the last curve, which ends at the series end (change probability 0). Each segment has one fit per row (`bands`): intercept, seven harmonic coefficients, RMSE and change magnitude. The result also holds the `processing_mask`, the observations the fit used. Without a profile, pass `hoptions` (a `ccdcpp.HarmonicOptions`, the input's description) and pre-classified QA codes; the run then uses the `ccdc` configuration ([Configuring the algorithm](#configuring-the-algorithm)). The defaults of both are PyCCD's, for Collection 1 values in DN ([Input unit convention](#input-unit-convention)).

The examples below continue from this one.

## A cube

`detect_cube_columns` runs every pixel of an (H, W, bands, T) cube in parallel and returns the segments as columns, one entry per segment; `detect_cube` returns one dict per pixel, as `detect` does.

```python
H, W = 3, 4
cube = np.ascontiguousarray(np.broadcast_to(spectral, (H, W, 7, T)) + rng.normal(0, 0.002, (H, W, 7, T)))
cube_qa = np.ascontiguousarray(np.broadcast_to(qas, (H, W, T)))

cols = ccdcpp.detect_cube_columns(dates, cube, cube_qa, threads=4, profile=presets.landsat_c2())

print(cols["pixels_total"], cols["pixels_with_segments"], len(cols["row"]), cols["failures"])
print(cols["coefficients"].shape)
print(sorted(k for k in cols if not k.startswith("pixels")))
```

```text
12 12 24 []
(24, 7, 7)
['band_names', 'break_day', 'change_probability', 'coefficients', 'col', 'curve_qa', 'end_day', 'failures', 'intercept', 'magnitude', 'num_bands', 'observation_count', 'rmse', 'row', 'start_day', 'threads']
```

`threads=0` (the default) uses the OpenMP runtime's thread count. A pixel that raises does not stop the cube: it is listed in `failures` with its row, column and message.

## From C++

The C++ API is two calls in `include/ccd/run/run.hpp`, `ccd::run_pixel` and `ccd::run_cube`. What a call runs is a `ccd::RunSpec`: the input's options (`ccd::HarmonicOptions`), the configuration (the `ccdc` preset unless you give one), and optionally a QA table, a preprocessing plan, an observer and whether to return resume records. The arrays are `ccd::ArrayView`s over your memory (any strides; float64, float32 or uint16 values; uint8 or uint16 QA). Link the CMake target `ccd_core`.

```cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include "ccd/ccd.hpp"

// One pixel: T dates, 7 rows of T values (row after row), T QA codes (0 clear).
void one_pixel(const std::vector<std::int64_t>& dates,
               const std::vector<double>& spectral,
               const std::vector<std::uint8_t>& qas)
{
    const std::size_t T = dates.size();

    ccd::RunSpec spec{ccd::HarmonicOptions{}};
    ccd::PixelRun run = ccd::run_pixel(spec,
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}));

    for (const ccd::ChangeModel& m : run.result.models)
        std::cout << m.start_day << " " << m.end_day << " " << m.break_day << " "
                  << m.bands[3].rmse << "\n";
}

// A cube: (H, W, 7, T) values and (H, W, T) QA, on 8 threads.
void a_cube(const std::vector<std::int64_t>& dates, const std::vector<double>& cube,
            const std::vector<std::uint8_t>& qas, std::size_t H, std::size_t W)
{
    const std::size_t T = dates.size();

    ccd::CubeResult r = ccd::run_cube({ccd::HarmonicOptions{}},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(cube.data(), {H, W, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {H, W, T}),
        /*threads=*/8);

    for (const ccd::PixelResult& p : r.pixels)
        std::cout << p.row << "," << p.col << ": " << p.result.models.size() << " segments\n";
    for (const ccd::CubeFailure& f : r.failures)
        std::cerr << f.row << "," << f.col << ": " << f.message << "\n";
}
```

`run_pixel` returns a `ccd::PixelRun` (`result`, and with `RunSpec::record` or a resume, `record`, `status` and `header`); `run_cube` a `ccd::CubeResult` (`pixels` in raster order, `failures`, `threads`, `header`). Both throw `std::invalid_argument` before any pixel runs on options, dates, a plan or a configuration they refuse.

# Configuring the algorithm

ccdcpp runs one fixed machine, the CCDC phases and routes, whose decisions are made by **stages**: the procedure a pixel takes, its noise floor, where the search for a stable start begins, the start window's outlier screen, fit and stability rule, Lookback, the monitoring model and when it is refitted, the change test, the outlier rule, and the fits a segment reports. Each stage is decided by a **strategy**, chosen by name, with parameters. A **configuration** names, for every stage, its strategy and that strategy's parameters.

The options (`HarmonicOptions`) describe the input; the configuration is the algorithm. A run without one uses the preset `ccdc`, each strategy at its defaults: PyCCD's algorithm with ccdcpp's corrections, for Collection 1 DN values. A run with `profile=` uses `profile.config()`, that preset written for the profile's input.

```python
cfg = ccdcpp.config("ccdc")
print(cfg)
```

```text
Config(
  change_test: chi2(change_probability=0.99, change_threshold=15.086272469388987, model_rmse_up_to=24, peek_reference_days=16.0, peek_size=6, seasonal_count=24, seasonal_divisor=4.0)
  edge_fit: lasso(alpha=1.0, coef_floor=1.0, coefficients=4, max_iter=1000, min_observations=12, tolerance=0.0001)
  lookback: pyccd(change_probability=0.99, change_threshold=15.086272469388987, outlier_threshold=35.88818687961042, peek=0)
  mode: pyccd(clear_pct_threshold=0.25, snow_pct_threshold=0.75)
  model: lasso(alpha=1.0, coef_floor=1.0, coefficient_max=8, coefficient_mid=6, coefficient_min=4, max_iter=1000, num_obs_factor=3, tolerance=0.0001)
  noise_floor: variogram(min_lag_days=30)
  outlier: floored(threshold=35.88818687961042)
  refit: span-factor(factor=1.33, min_obs=24)
  segment_fit: model-fit()
  stability_fit: lasso(alpha=1.0, coef_floor=1.0, coefficient_max=8, coefficient_mid=6, coefficient_min=4, max_iter=1000, num_obs_factor=3, tolerance=0.0001)
  stability_rule: pyccd(change_probability=0.99, change_threshold=15.086272469388987, peek=0)
  start_search: pyccd(day_delta=365, meow_size=12)
  window_screen: tmask(bands=[1, 4], t_const=4.89)
)
```

A configuration's parameters are in the input's units, as the options' are: the Lasso's `alpha` and `coef_floor` are 1.0 for DN and 1e-4 for reflectance. So for a profile, start from `profile.config()`: the `ccdc` preset with the profile's Lasso constants, Tmask bands, cadence and thresholds. A `Config` is immutable; `replace(stage__parameter=value)` returns a changed copy, and `replace(stage="strategy")` swaps a stage's strategy, its parameters starting from the new strategy's defaults. Every entry point takes it as `config=`:

```python
c2 = presets.landsat_c2()
base = c2.config()
print(base.params("model"))

stricter = base.replace(change_test__change_probability=0.999, refit__factor=2.0)
out2 = ccdcpp.detect(dates, spectral, qas, profile=c2, config=stricter)
print(len(out2["models"]), base.hash() != stricter.hash())

assert ccdcpp.detect(dates, spectral, qas, profile=c2, config=base)["models"] == out["models"]
```

```text
{'max_iter': 1000, 'alpha': 0.0001, 'tolerance': 0.0001, 'coef_floor': 0.0001, 'coefficient_min': 4, 'coefficient_mid': 6, 'coefficient_max': 8, 'num_obs_factor': 3}
2 True
```

Mistakes are refused by name, before any pixel runs:

```python
for bad in (lambda: base.replace(refit__fctor=2.0),
            lambda: base.replace(refit__factor=0.0).check(),
            lambda: base.replace(window_screen__bands=[1, 9]).check(bands=7)):
    try:
        bad()
    except ValueError as e:
        print(e)
```

```text
refit 'span-factor' has no parameter 'fctor'; it has min_obs, factor
refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
window_screen 'tmask' parameter 'bands' holds band 9; the input has 7 bands
```

`ccdcpp.strategies()` lists every stage's strategies and their parameters, with unit, bounds, default and meaning:

```python
for p in ccdcpp.strategies()["refit"]["span-factor"]:
    print(p["name"], "|", p["unit"], "|", p["bounds"], "|", p["default"], "|", p["doc"])
```

```text
min_obs | observations | at least 1 | 24 | observations up to which every test refits
factor | none | greater than 0 | 1.33 | refit once the window spans this many times the last fit's span
```

`cfg.hash()` is a digest equal for equal configurations on any platform, `cfg.to_toml()` writes the configuration as a preset file, and a `Config` pickles, so it can go to worker processes. A preset is a TOML file, one table per stage; `ccdcpp.config(path)` reads one, and a file may `extends = "ccdc"` and list only what differs. In C++ a configuration is a `config::ConfigTree` in `RunSpec::config` (by default `strategies::library_registry().preset("ccdc")`), and a new decision rule is a strategy class registered by name through a build extension (CMake's `CCD_EXTENSIONS`). The public library has one strategy per stage.

[docs/config.md](docs/config.md) covers configurations from Python and C++, presets, experiments and writing a strategy; [docs/ccdc-state-machine.md](docs/ccdc-state-machine.md#stages-and-strategies) lists the stages, their strategies and their parameters.

# Diagnostics

`trace=True` on `detect`, or `trace_pixels=[(row, col), ...]` on `detect_cube_columns`, also returns the run step by step: every phase transition with the machine's cursor, every Extend change test with its outcome and statistics, every observation a test drops, every segment written and how the pixel ended, one dict per event, ready for `pandas.DataFrame`. The results are the untraced run's, bit for bit.

```python
import pandas as pd

traced = ccdcpp.detect(dates, spectral, qas, profile=presets.landsat_c2(), trace=True)
trace = pd.DataFrame(traced["trace"])
print(trace.kind.value_counts().to_dict())
print(trace[trace.kind == "segment"][["day", "day_to", "outcome", "observations"]].to_string())
print(trace[trace.kind == "exit"]["outcome"].item())

cube_traced = ccdcpp.detect_cube_columns(dates, cube, cube_qa, threads=4, profile=presets.landsat_c2(),
                                         trace_pixels=[(1, 2)])
print(len(cube_traced["traces"][(1, 2)]))
```

```text
{'transition': 356, 'test': 327, 'segment': 2, 'exit': 1}
        day  day_to                                  outcome  observations
449  730136  734648  curve_qa 8 break 734664 change 1.000000           239
682  734664  737256  curve_qa 8 break 737272 change 0.000000           131
SeriesEnd
686
```

In C++ an observer is `RunSpec::observer` (`ccd::TextObserver` prints the events), and `RunSpec::observe` names a cube's pixels. The hooks compile out with CMake's `-DCCD_OBSERVE=OFF`. See [docs/diagnostics.md](docs/diagnostics.md) for the event fields and the C++ side.

# Resuming a run

When new observations arrive, a stored run can be continued instead of rerun. The first run asks for a resume record (`resume_record=True`); the update passes the whole series again (the stored observations and the new ones), the record and the stored segments, and gets the complete result for the whole series:

```python
c2 = presets.landsat_c2()
n = int(np.searchsorted(dates, date(2019, 1, 1).toordinal()))   # the first run: up to 2018

first = ccdcpp.detect_cube_columns(dates[:n], cube[..., :n], cube_qa[..., :n],
                                   threads=4, profile=c2, resume_record=True)
# Store first's segment columns, first["resume"] (one entry per pixel) and first["resume_header"].

SEGMENT_COLUMNS = ("row", "col", "start_day", "end_day", "break_day", "curve_qa",
                   "observation_count", "change_probability", "intercept", "coefficients",
                   "rmse", "magnitude")
update = ccdcpp.resume_cube_columns(
    dates, cube, cube_qa, threads=4, profile=c2,
    resume=first["resume"], resume_header=first["resume_header"],
    segments={k: first[k] for k in SEGMENT_COLUMNS})

print(update["counts"])
print(all(np.array_equal(update[k], cols[k]) for k in SEGMENT_COLUMNS))
```

```text
{'resumed': 12, 'rerun_input_changed': 0, 'failed': 0}
True
```

- **The result is the full run's, bit for bit**: `resume_cube_columns` gives what `detect_cube_columns` gives on the whole series with every statistics cutoff (`STAT_ORD`, and a `GreenMedian` step's date) capped at the first run's series end. With the Landsat presets' `STAT_ORD` (2017-12-31) and a first run past it, as here, that is the plain full rerun. The HLS and Sentinel-2 presets have no cutoff, so their statistics (the procedure choice, the cadence, the variogram) stay those of the first run until a full rerun re-baselines them.
- **What it saves is compute**: each pixel the Standard machine was running continues from where its stored run stopped, its earlier segments kept as stored and its open segment refitted once. The pixels with nothing to continue (PermanentSnow, InsufficientClear and the early exits) are rerun whole and count as resumed, since their result is the rerun's. On a 500 x 500 Landsat cell the annual update takes 0.36 of a full run's time ([Performance](#performance)).
- **A pixel whose stored observations changed** (a reprocessed value, a back-filled scene) is rerun in full, and its status says so (`update["resume"]["status"]`: 0 resumed, 1 rerun because its input changed, 2 failed). A scene the pixel's QA marks fill does not count as a change.
- **A record that does not match the call is refused** (`ValueError`): another record format or results version (`ccdcpp.RESULTS_VERSION` changes whenever a release changes outputs), other options, preset, plan, QA table or configuration, a series that ends before the stored one, columns that do not fit. Nothing falls back to a full rerun silently.
- The record is plain columns (`ccdcpp.resume_schema()` gives their dtypes); the caller stores them beside its segments.
- One pixel: `detect(..., resume_record=True)` and `resume(..., resume=, resume_header=, models=)`. C++: `ccd::run_pixel` and `ccd::run_cube` with `RunSpec::record`, and the stored run as their `resume` argument (a `ccd::PixelResume`, a `ccd::CubeResumeFrom`; `include/ccd/results/record.hpp`).

```python
n = int(np.searchsorted(dates, date(2019, 1, 1).toordinal()))
stored = ccdcpp.detect(dates[:n], spectral[:, :n], qas[:n], profile=c2, resume_record=True)
more = ccdcpp.resume(dates, spectral, qas, profile=c2, resume=stored["resume"],
                     resume_header=stored["resume_header"], models=stored["models"])
print(more["status"], more["models"] == out["models"])
```

```text
resumed True
```

# How it works

Per pixel, a run decodes the QA, computes the QA statistics, chooses the pixel's procedure, builds the processing mask (the observations it fits) and fits them. A pixel with enough clear observations takes the **Standard** procedure, CCDC's change detection; a mostly snowy pixel (**PermanentSnow**) and one with too few clear observations (**InsufficientClear**) get one curve over the whole series.

The Standard procedure is a small state machine: a per-pixel record, one transition function per phase, each making one decision and returning the next phase, and one loop that steps it. Bracketed names are why the machine stopped:

```
                              ┌─ size <= meow_size ───────────────────────► Done  [TooFewObservations]
                              ├─ a NaN in the variogram ──────────────────► Done  [VariogramNaN]
   Prepare ───────────────────┼─ no room for a start window ──────────────► EndCatch [NoRoomToInitialize]
                              ▼
       ┌───────────────► Initialize ◄──┐  ShortSpan, AllOutliers, RejectedOutliers: grow
       │                  │    │   └───┘  Unstable, DroppedThenUnstable: shift
       │                  │    └── Exhausted ─────────────────────────────► EndCatch [InitializeExhausted]
       │                  │ Stable, DroppedThenStable
       │                  ▼
       │   window.start > prev_break ?──── no ─────────────┐
       │                  │ yes                            │
       │                  ▼                                │
       │      ┌──► Lookback  Change / Dropped / Included   │
       │      └───────┘  (while window.start > prev_break) │
       │                  │                                │
       │                  ▼                                ▼
       │   no segment yet and window.start - prev_break > peek ?
       │                  │ yes                            │ no
       │                  ▼                                │
       │              StartCatch (the Start segment)       │
       │                  ▼                                ▼
       │   a full peek after the window ? ── no ──────────────────────────► EndCatch [PeekBeyondEnd]
       │                  │ yes
       │                  ▼
       │      ┌──► Extend  Change / Dropped / SeriesEnd / Grown
       │      └───────┘  (while a full peek fits)
       │                  │ series end: segment, change probability 0 ───────────────► EndCatch [SeriesEnd]
       │                  │ Change: segment, change probability 1
       │                  ▼
       └─── room ──── Restart: prev_break = window.stop, window = (stop, stop + meow_size)
                          │ no room
                          ▼
                      EndCatch [NoRoomToInitialize] ── the End segment, if prev_break + peek < size ──► Done
```

The phases and routes are fixed; what a phase decides is a stage, decided by the strategy the configuration names for it ([Configuring the algorithm](#configuring-the-algorithm)). Because the record holds everything that survives from one step to the next, a run can stop at a size guard and continue on a longer series ([Resuming a run](#resuming-a-run)), and an observer can report each step ([Diagnostics](#diagnostics)).

- [CCDC as a state machine](docs/ccdc-state-machine.md): the phases, guards and routes, the record, the stages and strategies, resuming, and why the code is shaped this way.
- [Floating-point sums](docs/floating-point-sums.md): why the order of the Lasso solver's sums is part of the output, and how they are made fast.

# Performance

Measured for ccdcpp 0.10.0 on an Apple M5 Max (18 cores: 6 performance, 12 efficiency; 64 GB; macOS 26.7), Release build, Python 3.11, numpy 1.26.4, the default options and `ccdc` configuration (`presets.landsat_c2()` for the real cell), on a machine that was not otherwise idle. Memory is the peak resident set size of the process, including the interpreter and the input it holds. [docs/benchmarks.md](docs/benchmarks.md) has every table, the load during each run, and how to measure again.

**Against PyCCD**, one pixel at a time on one thread, 404 pixels (the synthetic corpus and PyCCD's own test pixels), two rounds:

| | PyCCD with ccdcpp's corrections | PyCCD as released | ccdcpp |
|---|---|---|---|
| median per pixel | 54.6, 52.4 ms | 51.8, 52.5 ms | 0.27 ms |
| total, 404 pixels | 27.4, 27.1 s | 26.7, 26.9 s | 0.15 s |
| same segments as ccdcpp | 404 of 404 | 155 of 404 | |
| peak memory (own process) | 243 MB | | 131 MB |

About 200 times faster per pixel (median), 174-178 times on the total, and 68-73 times on PyCCD's 443-observation reference pixel (192-205 ms against 2.8 ms). Both memory figures are almost all the interpreter, the libraries and the loaded corpus: after the first pixel, the other 403 raise the peak by under 3 MB in either.

**A real cell**: a 500 x 500 Landsat Collection 2 ARD cell (h024v013, 1984-2023, 2473 dates, 250,000 pixels, 455,231 segments) through `detect_cube_columns` in strips of 50 rows:

| Threads | Detect | ms per pixel | Speed-up | Peak memory |
|---|---|---|---|---|
| 1 | 292.4 s | 1.170 | 1.00 | 17.6 GB |
| 8 | 42.4 s | 0.169 | 6.90 | 19.8 GB |
| 18 | 22.8 s | 0.091 | 12.8 | 19.9 GB |

The memory is mostly the input as the bench holds it: the cell read whole (9.9 GB of uint16), plus each strip converted to float64 (3.5 GB); it barely moves with the thread count, and a caller that reads per strip needs less.

**The annual update** of that cell (recorded to 2022-12-31, 89 dates of 2023 added, 8 threads, two rounds): `resume_cube_columns` 14.6 s and 13.9 s against a full run's 40.1 s and 38.7 s, 0.36 of the time, equal to the full run on all 250,000 pixels; 19.9 GB peak for the whole bench (stored run, update and full run in one process).

# Differences from pyccd

ccdcpp recreates pyccd. Where pyccd's code does not do what it evidently intends, ccdcpp does what it intends, so the two differ on the inputs that reach it:

- **RLM convergence.** pyccd's robust fit stops when `not numpy.any(numpy.fabs(x0 - x > tol))` (`ccd/models/robust_fit.py`, `_check_converge`): the parenthesis takes the absolute value of the comparison, not of the difference, so a coefficient that fell by more than `tol` counts as converged. ccdcpp stops when every coefficient moved by less than `tol` in either direction. A fit that pyccd stops early gets different weights and coefficients.
- **Date order.** pyccd orders observations with `np.argsort(dates)`, which is not a stable sort, so which of two observations on the same date survives the duplicate-date filter depends on the sort. ccdcpp takes dates in non-decreasing order (unsorted dates raise `ValueError`) and keeps the first observation of each date.
- **Green median filter.** pyccd's `insufficient_clear_filter` computes the green mask for the observations up to `STAT_ORD` and applies it with `standard_mask[standard_mask][:n] &= green_mask` (`ccd/qa.py`), which writes into a copy, so the filter drops nothing. ccdcpp applies it, with the `StatWindow` scope by default.
- **Seasonal RMSE ties.** pyccd's `find_closest_doy` (`ccd/change.py`) takes the 24 fit-window observations closest in day of year to the last peek date with `np.argsort(d_yr)[:num]`, which is not a stable sort. Observations a multiple of 1461 days apart are exactly equally distant, so which of the tied observations are kept depends on the sort. ccdcpp breaks ties in window order, so the seasonal RMSE, and with it the change magnitudes, can differ on dense series.

On the synthetic corpus, with pyccd patched for these four (the convergence test as `|x0 - x| > tol`, a stable date sort, the green mask written through `standard_mask[np.flatnonzero(standard_mask)[:n]] &= green_mask`, and `np.argsort(d_yr, kind="stable")` in `find_closest_doy`), ccdcpp's defaults, with the Standard procedure as pyccd's, match pyccd to 1e-6 on every InsufficientClear pixel (175 in DN) and on every pixel where a saturation-band value on a bound reaches the saturation filter (106, and the golden test's bounds pixel); with the green filter left unpatched, `Off` matches on the same 175. One more difference is kept on purpose: an undefined raw QA value decodes to fill where pyccd raises (see [QA Input](#qa-input)). One pyccd quirk is kept for parity: the PermanentSnow and InsufficientClear procedures test the untransformed thermal band against the Celsius bounds (see `harmonic.hpp`).

The Standard procedure (`ccd/procedures.py`) is pyccd's with five such corrections, each checked against Zhu's MATLAB reference, TrendSeasonalFit, and each in one function of the state machine (see [CCDC as a state machine](docs/ccdc-state-machine.md)):

- **Lookback's peek far from the break.** `slice(start - 1, start - peek, -1)` (line 753) reads one observation fewer than the peek; ccdcpp reads the peek, as Extend does after the window and TrendSeasonalFit does both ways.
- **Lookback's peek near the break.** The guard against reading past the start of the series (`start - peek <= 0`, line 755) also fires after a break, where it reads the previous segment's observations; ccdcpp reads down to the previous break and never below, as TrendSeasonalFit does. It takes an adjusted peek of 13 or more, a median gap of about 7.7 days or less (HLS, Sentinel-2), never Landsat's 16 days.
- **Coefficient counts.** Initialize fits a literal 4 coefficients and the peeks predict with a literal 8-term basis; ccdcpp reads the Lasso fits' `coefficient_min` and `coefficient_max` (`stability_fit`, `model`; TrendSeasonalFit's `min_num_c`, `max_num_c`). The same at the defaults.
- **The series end.** The check before Extend's grow (line 691) tests `>` where its comment, and pyccd's own `>=` before it changed its loop guard, mean the window to stop before the grow, so the check never holds and the last curve grows once more: its end day is its break day. ccdcpp stops at the last full peek, so a series-end segment ends one observation earlier, before its break day (as after an outlier drop). That curve is the last: pyccd restarts after it, and at peeks of 26 or more runs Initialize on the last observations and drops observations no segment uses; ccdcpp goes to the End segment's test, as TrendSeasonalFit ends.
- **Refit at 24 observations.** Extend refits while the window holds fewer than 24 observations (line 632) but compares against the model RMSE while it holds at most 24 (line 646), so at exactly 24 it compares an older fit's RMSE; ccdcpp refits at most 24, as TrendSeasonalFit does with one `<=` for both.

The measured agreement with PyCCD, as released and with exactly these corrections applied, is in [docs/benchmarks.md](docs/benchmarks.md#against-pyccd).

# Reference

## Input arrays

| Entry point | `dates` | `spectral` | `qas` |
|---|---|---|---|
| `detect`, `resume` | `(T,)` int64 | `(B, T)` float64 | `(T,)` |
| `detect_cube`, `detect_cube_columns`, `resume_cube_columns` | `(T,)` int64, shared | `(H, W, B, T)` float64, float32 or uint16 | `(H, W, T)` |

`qas` is uint8 QA codes, or raw uint8 or uint16 QA with a QA table ([QA Input](#qa-input)). `dates` are proleptic Gregorian ordinals (`datetime.date.toordinal()`) in non-decreasing order: every entry point raises `ValueError` (`std::invalid_argument` in C++, `ccd::validate_dates`) on a date earlier than the one before it, once per call and before any pixel runs. Equal dates are allowed, and the filters keep the first observation of each date.

Input arrays are never modified. The cube entry points read the spectral cube in any layout with non-negative strides, converting one pixel at a time inside each worker thread, so a uint16 cube is never widened to float64 as a whole; the results are those of the same values passed as float64. Other dtypes are converted to a C-contiguous float64 array first, as long as numpy can do so safely. C-contiguous arrays, each pixel's series contiguous, are the fastest.

## Band Layout

`HarmonicOptions` says which row of `spectral` plays which role. The defaults are the Landsat layout, seven bands in the order blue, green, red, nir, swir1, swir2, thermal:

| Field | Default | Read by |
|-------|---------|---------|
| `NUM_BANDS` | `7` | every entry point: the band count `spectral` must have |
| `DETECTION_BANDS` | `[1, 2, 3, 4, 5]` | change magnitude and the stability check, in list order |
| `SATURATION_BANDS` | `[0, 1, 2, 3, 4, 5]` | the saturation filter (`REFL_MIN`, `REFL_MAX`) |
| `GREEN_IDX` | `1` | the InsufficientClear green median filter |
| `THERMAL_IDX` | `6` | the thermal transform and filter; `ccdcpp.NO_BAND` for none |

Every row is fitted and reported, so results have `NUM_BANDS` bands in row order. The Tmask outlier screen's bands are the configuration's (`window_screen__bands`, `[1, 4]`, green and swir1). The configuration's change and outlier thresholds (`change_test__change_threshold`, `outlier__threshold` and the stages that hold the same values, [Cadence Parameters](#cadence-parameters)) are chi-squared quantiles with 5 degrees of freedom, the default detection band count; a layout with a different count sets them to match (a sensor profile's `config()` does this itself). `BLUE_IDX`, `RED_IDX`, `NIR_IDX`, `SWIR1_IDX`, `SWIR2_IDX` and `QA_IDX` are informational and not read. A layout that does not fit the input (wrong band count, an index off the band axis, a repeated index, an empty detection list, no green band, thermal in the saturation list, more than `ccdcpp.CCD_MAX_BANDS` bands) raises `ValueError` before any pixel runs, as does a configuration band list that does not fit it (an empty, repeated or out-of-range `window_screen__bands`).

## Input Unit Convention

The observation filters compare `spectral` against absolute thresholds, so each threshold is a statement about the unit of the input, as are the configuration's Lasso `alpha` and `coef_floor` (in `model`, `stability_fit` and `edge_fit`). The defaults are Collection 1 DN, as pyccd consumes it; the physical-unit preset (`presets.landsat_c1_physical()`) is every one of them divided by 10000 (the measured equivalence is in the input unit convention in `include/ccd/input/harmonic.hpp`). The defaults are pyccd's as pyccd intends them: exclusive saturation and thermal bounds, and the green median filter scope its InsufficientClear filter computes.

| Field | Default | Read by |
|-------|---------|---------|
| `THERMAL_SCALE`, `THERMAL_OFFSET` | `10.0`, `-27315.0` | the Standard procedure's thermal transform, `thermal * scale + offset` |
| `REFL_MIN`, `REFL_MAX` | `0.0`, `10000.0` | the saturation filter, on `SATURATION_BANDS` |
| `REFL_MIN_INCLUSIVE`, `REFL_MAX_INCLUSIVE` | `False`, `False` | whether a value equal to `REFL_MIN` / `REFL_MAX` is kept |
| `THERM_MIN`, `THERM_MAX` | `-9320.0`, `7070.0` | the thermal filter, on `THERMAL_IDX`, after the transform |
| `THERM_MIN_INCLUSIVE`, `THERM_MAX_INCLUSIVE` | `False`, `False` | whether a value equal to `THERM_MIN` / `THERM_MAX` is kept |
| `MEDIAN_GREEN_FILTER` | `400.0` | the InsufficientClear green median filter's offset |
| `GREEN_FILTER_SCOPE` | `StatWindow` | where the green median filter drops observations |

- **Bounds.** The defaults are exclusive at both ends of both ranges, pyccd's `0 < v < 10000` and `-9320 < v < 7070`: an observation with a saturation band at exactly `0` or `10000`, or NaN, is dropped. With both ends of a range inclusive the test is `!(v < lo || v > hi)`, under which the bounds are kept and NaN passes, reaching the fit (a pixel with a NaN kept this way returns no segments and its processing mask). The `*_INCLUSIVE` flags are not unit constants, so the physical preset leaves them as they are.
- **Green median filter.** The InsufficientClear procedure drops cloud-like observations by their green value: the threshold is the median green of the masked observations dated up to `STAT_ORD`, plus `MEDIAN_GREEN_FILTER`, and an observation at or above it is dropped. `GREEN_FILTER_SCOPE` (a `ccdcpp.GreenFilterScope`) says where: `StatWindow` (the default, the scope pyccd's filter computes its mask for) only up to `STAT_ORD`, `All` at any date, `Off` never (what pyccd's code does; see [Differences from pyccd](#differences-from-pyccd)). With no masked observation up to `STAT_ORD` nothing is dropped.

## QA Input

`qas` reaches the procedures as one class code per observation. It can be passed two ways.

**Pre-classified codes (uint8).** `qas` already holds the codes, the `HarmonicOptions` `QA_*` fields:

| Field | Default | Read by |
|-------|---------|---------|
| `QA_FILL` | `255` | the quality statistics: not counted as a valid observation |
| `QA_CLEAR` | `0` | the fit (kept), the clear percentage |
| `QA_WATER` | `1` | the fit (kept), the clear percentage |
| `QA_SNOW` | `3` | the snow percentage and the PermanentSnow procedure |
| `QA_CLOUD` | `4` | the quality statistics' cloud count |
| `QA_SHADOW` | `2` | not read by the procedures; written by the QA tables |
| `QA_CIRRUS1`, `QA_CIRRUS2`, `QA_OCCLUSION` | `8`, `9`, `10` | informational; not read by ccdcpp |

Any code other than `QA_CLEAR` and `QA_WATER` keeps the observation out of the fit; a code that is none of the above counts only toward the total.

**Raw QA plus a table (uint8 or uint16).** Pass the archive's own QA band and a `ccdcpp.QaTable` as `qa_table=` (a sensor profile supplies its own). Each pixel's values are decoded through the table, per thread, before the procedures run, so the run is exactly the run of the decoded codes passed as uint8, without a decoded copy of the cube. A uint16 `qas` without a table raises `TypeError`.

```python
h = c2.options()
table = ccdcpp.qa_tables.landsat_c2_qa_pixel(h)
explicit = ccdcpp.detect(dates, spectral, qas, h, qa_table=table, config=c2.config())
print(explicit["models"] == out["models"])
```

```text
True
```

| Builder | Product | Validated for production |
|---------|---------|--------------------------|
| `landsat_c2_qa_pixel(options)` | Landsat Collection 2 `QA_PIXEL` (uint16) | yes |
| `landsat_c1_pixel_qa(options)` | Landsat Collection 1 `pixel_qa` (uint16) | yes |
| `hls_fmask(options, exclude_high_aerosol=False)` | HLS v2.0 `Fmask` (uint8) | decode checked on real data: all 43 values present (38 in L30, 43 in S30) match the HLS v2.0 User Guide (T16SGC, 2020–2023) |
| `sentinel2_scl(options)` | Sentinel-2 L2A `SCL` (uint8) | decode checked on real data: every class present (0, 2–11) matches SentiWiki (T16SGC, 2020–2023) |

Each builder writes the `QA_*` codes of the `HarmonicOptions` it is given, so build the table from the options the run uses. Each decodes most-excluding first: fill, then cloud, then shadow, then snow, then water, then clear.

- **Landsat C2 `QA_PIXEL`:** bit 0 fill; bits 3, 1, 2 (cloud, dilated cloud, cirrus) cloud; bit 4 shadow; bit 5 snow; bit 7 water; otherwise clear. Bit 6 (clear) and the confidence bits 8-15 are not read.
- **Landsat C1 `pixel_qa`:** pyccd's `qabitval`: bit 0 fill; bit 5 cloud; bit 3 shadow; bit 4 snow; bit 2 water; bit 1 clear; bits 8 and 9 together, or bit 10, clear. A value pyccd rejects decodes to fill.
- **HLS `Fmask`:** 255 fill; bits 1, 2, 0 (cloud, adjacent to cloud or shadow, reserved) cloud; bit 3 shadow; bit 4 snow; bit 5 water; otherwise clear. The aerosol level (bits 6-7) is ignored unless `exclude_high_aerosol=True`, which makes high aerosol cloud. Values above 255 are fill. Bit 0 is reserved and unused in v2.0, so it decodes to cloud only should it ever be set.
- **Sentinel-2 `SCL`:** 0, 1 fill; 2, 3 (cast shadows, cloud shadows) shadow; 4, 5 clear; 6 water; 7-10 (unclassified, cloud medium and high probability, thin cirrus) cloud; 11 snow; anything else fill.

For any other product, `ccdcpp.QaTable(codes)` takes a `(65536,)` uint8 array whose entry `v` is the code of raw value `v`.

## Cadence Parameters

The Standard procedure has parameters that assume something about the revisit interval of the input or the false-alarm rate of the change test. They are parameters of the configuration's strategies, set as `stage__parameter` (`cfg.replace(refit__factor=2.0)`, or a profile's `cadence`, below), each defaulting to pyccd's value, tuned for 16-day Landsat. Where pyccd has one constant, several stages each hold their own value of it ("Also in"): the `ccdc` preset sets them alike, and a caller changing one sets the others to match, or apart on purpose.

| Parameter | Default | Unit | Also in | Read by |
|---|---|---|---|---|
| `change_test__peek_size` | `6` | observations | | the peek at the reference cadence; the stability rule and Lookback use the test's peek (their `peek` is 0) |
| `change_test__peek_reference_days` | `16.0` | days | | the peek size: `round(peek_size * peek_reference_days / median gap)`, never below `peek_size` |
| `change_test__change_probability` | `0.99` | probability | `stability_rule__change_probability`, `lookback__change_probability` | the change threshold of a peek longer than `peek_size`: the chi-squared quantile at `1 - (1 - change_probability) ** (peek_size / peek)` |
| `change_test__change_threshold` | `15.0863` | none | `stability_rule__change_threshold`, `lookback__change_threshold` | the change threshold at a peek of `peek_size` |
| `outlier__threshold` | `35.8882` | none | `lookback__outlier_threshold` | Extend's outlier rule (and Lookback's) |
| `noise_floor__min_lag_days` | `30` | days | | the variogram: the smallest lag whose most common separation exceeds this, over the pairs further apart than this |
| `refit__min_obs` | `24` | observations | `change_test__model_rmse_up_to` | Extend: refit while the window holds at most this many; such a window compares against the model RMSE, a longer one against the seasonal RMSE |
| `refit__factor` | `1.33` | ratio | | Extend: refit once the window spans at least this times the span of the last fit |
| `change_test__seasonal_count` | `24` | observations | | the seasonal RMSE: residuals of this many fit observations closest in day of year to the last peek observation |
| `change_test__seasonal_divisor` | `4.0` | none | | the seasonal RMSE: their norm divided by this. pyccd's 4 is `sqrt(24 - 8)`; it is not derived from `seasonal_count` |
| `start_search__meow_size` | `12` | observations | `edge_fit__min_observations` | the start window, and the observations a start needs after it; the observations a whole-series curve needs |
| `start_search__day_delta` | `365` | days | | the least span of a start window |

`ccdcpp.chi2_quantile(p, df)` is the chi-squared quantile the procedure computes. The change threshold is `chi2_quantile(change_probability, 5)` to the last bit, so a caller changing the probability or the detection band count sets the threshold to match, in each stage that holds it:

```python
cfg = ccdcpp.config("ccdc")
print(cfg.params("change_test")["change_threshold"] == ccdcpp.chi2_quantile(0.99, 5))

p = 0.995
t = ccdcpp.chi2_quantile(p, 5)
stricter = cfg.replace(**{f"{stage}__{name}": value
                          for stage in ("change_test", "stability_rule", "lookback")
                          for name, value in (("change_probability", p), ("change_threshold", t))})
print(round(t, 4), stricter.params("stability_rule"))
```

```text
True
16.7496 {'change_threshold': 16.749602343639044, 'change_probability': 0.995, 'peek': 0}
```

A sensor profile's `cadence` holds the parameters that follow its sensor, by the same `stage__parameter` names, and `profile.config()` applies them with `Config.replace()`. Start windows of 24 observations and refits up to 48, for a dense series:

```python
import dataclasses

dense = dataclasses.replace(presets.hls(), name="hls_dense", cadence={
    "start_search__meow_size": 24, "edge_fit__min_observations": 24,
    "refit__min_obs": 48, "change_test__model_rmse_up_to": 48})
cfg = dense.config()
print(cfg.params("start_search"), cfg.params("refit"), cfg.params("change_test")["model_rmse_up_to"])
```

```text
{'meow_size': 24, 'day_delta': 365} {'min_obs': 48, 'factor': 1.33} 48
```

A value outside its parameter's bounds (`ccdcpp.strategies()` lists them) raises `ValueError` naming the stage, strategy and parameter, before any pixel runs: `peek_reference_days` or `min_lag_days` of zero or less, a count below 1, a `peek_size` below 2, a probability outside (0, 1), a factor or divisor of zero or less, a NaN threshold, a negative or NaN Lasso constant, and coefficient counts that decrease from `coefficient_min` to `coefficient_max` or exceed 8 (the harmonic basis's width).

**Denser cadences.** The peek size follows the cadence: it is `round(peek_size * peek_reference_days / (median gap + 0.001))`, never below `peek_size`, where the gap is between consecutive observations the processing mask keeps, up to `STAT_ORD`; at 8 days it is 12, at 5 days 19, at 4 days 24. The change threshold falls as the window grows, so that a longer window keeps the default window's false-alarm rate: with the defaults it is 9.24 at a peek of 12, 6.83 at 19 and 5.90 at 24. Both are computed per pixel, as pyccd's `adjustpeek` and `adjustchgthresh` do; a series whose median gap is about 14.8 days or more keeps `peek_size` and `change_threshold` as given. `meow_size` and the refit's `min_obs` are observation counts and are deliberately not scaled with the cadence, as pyccd does not scale them: at a 4–5 day cadence 12 observations are about 50–60 days, where at 16 days they are about half a year. A caller who wants them in time sets them from the cadence, as above.

## Sensor Profiles

A `ccdcpp.SensorProfile` describes one sensor's input in one immutable object: the band names, the role each band plays, the detection, Tmask and saturation lists by role name, the unit-coupled thresholds, the Lasso constants, its cadence parameters, `STAT_ORD`, the QA table and optionally a preprocessing plan. Every entry point takes one as `profile=`.

- The profile supplies `hoptions`, `qa_table`, `preprocess` and the configuration (`profile.config()`). Any of them passed explicitly is used instead, whole: `hoptions=` replaces the profile's `HarmonicOptions` rather than being merged into it, `qa_table=None` reads `qas` as pre-classified codes, `preprocess=None` runs the default plan, and `config=` replaces `profile.config()` (start from `profile.config().replace(...)` to keep the profile's units and thresholds).
- `spectral` is checked against the profile's band count (`ValueError`), and only the count. `band_names=` (keyword only, only with `profile=`) names the rows as the caller holds them; a stack in another order then raises `ValueError` naming the first row that differs, before anything runs. A Sentinel-2 store that keeps B08 with the 10 m bands is refused by `sentinel2_l2a()` this way rather than having B08 read as red edge 1.
- The outputs of `detect`, `resume`, `detect_cube_columns` and `resume_cube_columns` gain `band_names`; `detect_cube`'s per-pixel dicts do not.
- `profile.options()` returns a new `HarmonicOptions`, `profile.config(base="ccdc")` the configuration `base` written for the profile's input (its `cadence`, its `alpha` and `coef_floor` in `model`, `stability_fit` and `edge_fit`, its Tmask roles as `window_screen__bands`, the change and outlier thresholds of its detection band count), `profile.qa_table()` the table (or `None`), `profile.check(spectral, band_names=None)` the band check. Profiles are hashable and picklable, so a profile can be sent to worker processes. Without `profile=`, `hoptions` is required, `config=` defaults to `"ccdc"`, and a call is the call of the extension function of the same name (`ccdcpp.ccdcpp.detect`, ...).

### Presets

`ccdcpp.presets` holds the inputs ccdcpp is validated on, Landsat, and presets for HLS v2.0 and Sentinel-2 L2A, which follow the products' documentation, are tested on synthetic series and were validated on a real sample: one cell of MGRS tile T16SGC near Atlanta, 2020–2023. The three Landsat presets take seven bands in the order blue, green, red, nir, swir1, swir2, thermal (TM/ETM+ bands 1, 2, 3, 4, 5, 7, 6; OLI/TIRS bands 2, 3, 4, 5, 6, 7, 10).

| Preset | Optical input | Thermal input | QA input | Validated for production |
|--------|---------------|---------------|----------|--------------------------|
| `landsat_c1_dn()` | reflectance x 10000 | Kelvin x 10 | pre-classified codes | yes |
| `landsat_c1_physical()` | reflectance 0..1 (DN / 10000) | degrees Celsius / 100 | pre-classified codes | yes |
| `landsat_c2()` | reflectance 0..1 (`SR_B* * 2.75e-5 - 0.2`) | degrees Celsius / 100 | raw `QA_PIXEL` (uint16) | yes |
| `hls_l30()` | reflectance 0..1 (`B* * 0.0001`) | B10 degrees Celsius / 100 (`B10 * 0.0001`) | raw `Fmask` (uint8) | yes (T16SGC, 2020–2023) |
| `hls_s30()` | reflectance 0..1 (`B* * 0.0001`) | none | raw `Fmask` (uint8) | yes, with a caveat: Fmask over bright surfaces (T16SGC, 2020–2023) |
| `hls()` | reflectance 0..1 (`B* * 0.0001`), L30 and S30 | none | raw `Fmask` (uint8) | yes, with a caveat: Fmask over bright surfaces (T16SGC, 2020–2023) |
| `sentinel2_l2a()` | reflectance 0..1 (`(DN + BOA_ADD_OFFSET) / 10000`) | none | raw `SCL` (uint8) | validated at median level on Planetary Computer's Esri/MSFT/S2RP L2A, 2020–23; ESA's own processing not sampled |

`landsat_c1_dn().options()` is `HarmonicOptions()`, field for field, and `landsat_c1_dn().config()` is the `ccdc` preset. `landsat_c1_physical()` divides every absolute threshold and both Lasso constants of its configuration (`alpha`, `coef_floor`) by 10000 and uses the identity thermal transform: the same fit, with coefficients in the input's units. `landsat_c2()` has the same thresholds (Collection 2's saturation range in DN, 7273..43636, is reflectance 0..1) and decodes `QA_PIXEL`.

The HLS and Sentinel-2 presets have the thresholds and Lasso constants of `landsat_c1_physical()`, detect change on green, red, nir, swir1 and swir2 with the default chi-squared(5) thresholds, use green and swir1 for Tmask, saturation-screen every optical row, have no statistics cutoff (`stat_ord=None`) and keep pyccd's cadence values.

| Preset | Rows (role) | Left out |
|--------|-------------|----------|
| `hls_l30()` | B01 (coastal), B02 (blue), B03 (green), B04 (red), B05 (nir), B06 (swir1), B07 (swir2), B10 (thermal) | B09 cirrus, B11 TIRS 2 |
| `hls_s30()` | B01 (coastal), B02 (blue), B03 (green), B04 (red), B05, B06, B07 (red_edge_1..3), B08 (nir_broad), B8A (nir), B11 (swir1), B12 (swir2) | B09 water vapour, B10 cirrus |
| `hls()` | coastal, blue, green, red, nir (L30 B05, S30 B8A), swir1, swir2 | thermal, which S30 lacks |
| `sentinel2_l2a()` | B02 (blue), B03 (green), B04 (red), B05, B06, B07 (red_edge_1..3), B08 (nir_broad), B8A (nir), B11 (swir1), B12 (swir2) | B01, B09 (60 m, atmospheric); L2A has no B10 |

- **HLS scaling.** Every HLS layer is `int16` times 0.0001; for L30 B10 (degrees Celsius, scale 0.01) that is degrees Celsius / 100, so one divisor serves every row. The fill, -9999, becomes -0.9999 and is dropped by the bounds.
- **HLS QA.** `Fmask` decodes with the aerosol level ignored; `dataclasses.replace(preset, qa_table_options={"exclude_high_aerosol": True})` treats high aerosol as cloud.
- **S30 and Sentinel-2 bands.** The red edges and B08 are fitted, reported and saturation-screened but not detected on.
- **Sentinel-2 offset.** From processing baseline 04.00 (25 January 2022) L2A stores `DN = reflectance * 10000 - BOA_ADD_OFFSET`, with a band-dependent offset in each product's metadata. ccdcpp never sees the metadata, so the caller applies it; a series that mixes corrected and uncorrected scenes jumps at the baseline change and detects a false break. Some archives distribute corrected bands while their metadata still declares the offset (earth-search's `sentinel-2-l2a` COGs); there it must not be applied twice.
- **Sentinel-2 QA.** `SCL` class 7 (unclassified) is cloud, which costs water pixels (below). A caller who wants them passes `qa_table=ccdcpp.QaTable(codes)` with entry 7 changed.
- **L30 and S30 in one series.** An L30 and an S30 observation on one date are deduplicated like any other pair, the first in array order kept: put L30 before S30 on a shared date.

**Validation on real data.** The HLS and Sentinel-2 presets were run on Landsat ARD tile h024v013 chunk 00030008 near Atlanta, cut to MGRS tile T16SGC, 2020-01-01 to 2023-12-31, from Microsoft Planetary Computer (`hls2-l30`, `hls2-s30`, `sentinel-2-l2a` at 20 m), against Landsat Collection 2 ARD of the same cell run with `landsat_c2()` without its statistics cutoff. Four years of one cell is the whole record.

| Preset | Pixels, dates | What was measured |
|--------|---------------|-------------------|
| `hls_l30()` | 300,288 at 30 m, 238 | Break agreement per pixel with Landsat 8+9 ARD kappa 0.44 (0.43 with Landsat 7 added); matched breaks a median 8 days apart, 85% within 96 days, no date bias; model levels within 0.004 reflectance. No break before the second half of 2021 (a thin 2020). |
| `hls_s30()` | 300,288, 461 | 41% more break pixels than L30, a median 16 days earlier. Fmask on Sentinel-2 calls bright barren land cloud: over the 1,862 quarry pixels it flags 63% of observations as cloud and sends 1,145 of them to InsufficientClear. |
| `hls()` | 300,288, 699 | Mostly follows S30 (kappa 0.71 with S30, 0.45 with L30); L30 first on a shared date worked on all 85 same-date pairs. 1,741 pixels go to InsufficientClear (S30: 2,101; L30: 2). |
| `sentinel2_l2a()` | 675,648 at 20 m, 565 | With the per-scene offset the 2022 baseline change looks like any other window (0.34% of pixels break within 60 days of it; 92.3% without), apart from 24 water pixels. Against S30, levels about 0.01 higher and about twice the RMSE, kappa 0.48, fewer break pixels (2.35% against 3.62%). SCL 7 (2.13% of observations), with SCL 2 on small water bodies, sends 1,355 of the cell's 3,983 water pixels to InsufficientClear (2,545 decoded as fill, 587 as clear). |

High aerosol is flagged on about half the HLS observations; treating it as cloud would remove another 16.5% (L30) and 16.8% (S30) of what QA keeps, which is why the presets ignore it. Five of the sample's pixels (four in all three products, a fifth in the two HLS products) are pinned with every preset's output in `tests/python/fixtures/t16sgc_2020_2023` (`tests/python/test_real_fixtures.py`).

### Adding a sensor

A sensor is one `SensorProfile` call; `python/ccdcpp/presets.py` writes its profiles that way. **Not shipped:** this Landsat 8/9 Collection 2 profile with the coastal band shows the shape; its values are not validated.

```python
from ccdcpp import CELSIUS_HUNDREDTHS, REFLECTANCE, SensorProfile

def landsat_oli_c2():
    return SensorProfile(
        name="landsat_oli_c2",
        band_names=("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"),
        roles=("coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"),
        detection=("green", "red", "nir", "swir1", "swir2"),
        tmask=("green", "swir1"),
        saturation=("coastal", "blue", "green", "red", "nir", "swir1", "swir2"),
        units=REFLECTANCE,
        alpha=1e-4,
        coef_floor=1e-4,
        thermal=CELSIUS_HUNDREDTHS,
        stat_ord=736694,  # 2017-12-31, the Landsat statistics cutoff
        qa_table_builder="landsat_c2_qa_pixel",
    )

oli = landsat_oli_c2()
h, cfg = oli.options(), oli.config()
print(h.NUM_BANDS, list(h.DETECTION_BANDS), cfg.params("window_screen")["bands"], h.THERMAL_IDX,
      cfg.params("model")["alpha"])
```

```text
8 [2, 3, 4, 5, 6] [2, 5] 7 0.0001
```

- **Roles**: `blue`, `green`, `red`, `nir`, `swir1`, `swir2`, `thermal`, `coastal`, `red_edge_1`, `red_edge_2`, `red_edge_3`, `nir_broad` and `other` (`ccdcpp.ROLES`). `nir` is the 865 nm band (TM/ETM+ band 4, OLI band 5, Sentinel-2 B8A); the broad 842 nm B08 is `nir_broad`. Each role but `other` names at most one band; `other` bands are fitted and reported but no list can name them. Each list keeps the order the profile gives, which is the order the procedures accumulate in.
- **Units and thermal**: `units` is a `ccdcpp.Units` (`REFL_MIN`, `REFL_MAX`, `MEDIAN_GREEN_FILTER`, `refl_inclusive`; `ccdcpp.DN` and `ccdcpp.REFLECTANCE`), `thermal` a `ccdcpp.Thermal` (transform, bounds, `inclusive`; `ccdcpp.KELVIN_TENTHS`, `ccdcpp.CELSIUS_HUNDREDTHS`) or `None`. `alpha` and `coef_floor` have no default, because both are statements about the input unit and a wrong value does not raise.
- **Thresholds**: `change_threshold` and `outlier_threshold` default to `None`, which computes the chi-squared quantiles for the profile's detection band count (for five bands the `ccdc` preset's to the last bit); `config()` writes them into every stage that holds them ([Cadence Parameters](#cadence-parameters)).
- **Other fields**: `cadence`, `stage__parameter` -> value for the configuration's parameters that follow the sensor ([Cadence Parameters](#cadence-parameters)), which `config()` applies with `Config.replace()`; `green_filter_scope`, `"stat_window"` (the default), `"all"` or `"off"`; `qa_table_options`, keyword arguments for the `qa_table_builder` of exactly the types `ccdcpp.QA_TABLE_OPTIONS` lists; `preprocess`, a `ccdcpp.Preprocess` ([Preprocessing](#preprocessing)).
- A profile that breaks a layout rule (an unknown or repeated role, a list naming a role the profile lacks, no green band, thermal in the saturation list, a thermal band without `thermal=`) raises `ValueError` when it is constructed.

## Preprocessing

Before a procedure fits a pixel it builds the processing mask, the observations it fits: an optional transform of one band of the pixel's private copy, then a chain of filter steps run in order. A `PreprocessPlan` holds one such `ProcedurePlan` (transform and chain) per procedure, `standard`, `snow` (PermanentSnow) and `insufficient_clear`, and each pixel runs the one of the procedure it takes. The steps:

| Step | Effect |
|------|--------|
| `KeepQa(codes)` | keep only observations whose QA code is in `codes` |
| `AddQa(codes)` | also keep every observation whose code is in `codes`; must be followed by `Dedupe` |
| `ThermalRange(band, range)` | keep only observations whose thermal value is in the range |
| `BandRange(bands, range)` | keep only observations on which every band is in the range, tested in list order |
| `GreenMedian(band, offset, max_date, scope)` | drop observations at or above the median of `band` over the kept observations dated up to `max_date`, plus `offset`, at the dates `scope` names |
| `Dedupe` | keep the first kept observation of each date |

A range keeps `min < x < max`; an inclusive end also keeps a value equal to it, and NaN passes only a range inclusive at both ends. A chain holds at most `CHAIN_CAPACITY` (8) steps.

### The default plan

Without a plan, a run uses `default_plan(hoptions)`, pyccd's preprocessing as pyccd intends it. For `HarmonicOptions()` it is (`print(ccdcpp.default_plan(ccdcpp.HarmonicOptions()))`):

```text
standard
  transform  band 6 <- x * 10 - 27315
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  Dedupe
snow
  transform  none
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  Dedupe
  4  AddQa codes {3}
  5  Dedupe
insufficient_clear
  transform  none
  0  KeepQa codes {0, 1}
  1  ThermalRange band 6, -9320 < x < 7070
  2  BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000
  3  Dedupe
  4  GreenMedian band 1, drop x >= median(dates <= 736694) + 400, scope StatWindow
  5  Dedupe
```

Only `standard` has a transform (`THERMAL_SCALE`/`THERMAL_OFFSET`, none when that is the identity), so the other two test the thermal band in the caller's unit, as pyccd does. Passing the default plan explicitly gives the same output, to the bit, as passing none.

### A custom plan

A plan is a value: edit the default, or build chains from the step constructors (`ccdcpp.Step.keep_qa`, `add_qa`, `thermal_range`, `band_range`, `green_median`, `dedupe`). Every entry point takes one as `preprocess=` and checks it against the band count (`ValueError`); `ccdcpp.validate_plan(plan, bands)` checks one directly. Here the Standard procedure keeps snow observations too, on a DN pixel with pre-classified codes:

```python
h = ccdcpp.HarmonicOptions()
plan = ccdcpp.default_plan(h)
plan.standard.chain[0] = ccdcpp.Step.keep_qa([h.QA_CLEAR, h.QA_WATER, h.QA_SNOW])
print(plan.standard.chain[0])

dn = spectral * 10000                      # Collection 1 DN, as the default options expect
dn[6] = (spectral[6] * 100 + 273.15) * 10  # Kelvin x 10
codes = np.where(qas == 21824, h.QA_CLEAR, h.QA_CLOUD).astype(np.uint8)
snowy = ccdcpp.detect(dates, dn, codes, h, preprocess=plan)
print(len(snowy["models"]))
```

```text
<Step KeepQa codes {0, 1, 3}>
2
```

With a sensor profile a plan is written by role and QA code name instead, as frozen dataclasses: `KeepQa(codes=("clear", "water"))`, `AddQa`, `ThermalRange`, `BandRange(roles=...)`, `GreenMedian`, `Dedupe()` and `Transform(role, scale, offset)`, gathered per procedure as `ProcedureChain(steps=, transform=)` and per plan as `Preprocess(standard=, snow=, insufficient_clear=)`. A field or procedure left `None` takes what the default plan would, so `Preprocess()` is the default plan. A profile holds one as `SensorProfile(preprocess=...)`. Landsat Collection 2 with its thermal band as the raw `ST_B10` DN, converted on every procedure to the degrees Celsius / 100 of `landsat_c2()`'s bounds:

```python
import dataclasses
from ccdcpp import Preprocess, ProcedureChain, Transform

# ((DN * 0.00341802 + 149.0) - 273.15) / 100
st_b10 = Transform("thermal", scale=3.41802e-5, offset=-1.2415)
c2_raw_thermal = dataclasses.replace(
    presets.landsat_c2(),
    name="landsat_c2_raw_thermal",
    preprocess=Preprocess(
        standard=ProcedureChain(transform=st_b10),
        snow=ProcedureChain(transform=st_b10),
        insufficient_clear=ProcedureChain(transform=st_b10),
    ),
)

raw = spectral.copy()
raw[6] = (spectral[6] * 100 + 273.15 - 149.0) / 0.00341802   # ST_B10 DN
from_raw = ccdcpp.detect(dates, raw, qas, profile=c2_raw_thermal)
print([m["break_day"] for m in from_raw["models"]] == [m["break_day"] for m in out["models"]])
```

```text
True
```

In C++ the same plan is a `ccd::preprocess::PreprocessPlan` in `RunSpec::plan`, shared by every pixel and thread of the call:

```cpp
#include "ccd/ccd.hpp"

namespace pp = ccd::preprocess;

ccd::RunSpec spec_with_snow(const ccd::HarmonicOptions& h)
{
    pp::PreprocessPlan plan = pp::default_plan(h);

    // The thermal transform on every procedure, not only Standard.
    plan.snow.transform = plan.standard.transform;
    plan.insufficient_clear.transform = plan.standard.transform;

    // Standard keeps snow observations too.
    plan.standard.chain = pp::Chain{
        pp::Step::keep_qa(pp::qa_set({h.QA_CLEAR, h.QA_WATER, h.QA_SNOW})),
        pp::Step::thermal_range(h.THERMAL_IDX, ccd::filter::thermal_range(h)),
        pp::Step::band_range(h.SATURATION_BANDS, ccd::filter::reflectance_range(h)),
        pp::Step::dedupe(),
    };

    ccd::RunSpec spec{h};
    spec.plan = std::make_shared<const pp::PreprocessPlan>(plan);
    return spec;
}
```

`pp::validate(plan, bands)` runs once per call, before any pixel, and throws `std::invalid_argument`, naming the procedure and the step, when a band is not below the band count, a `GreenMedian` has no band, a transformed band is also tested by a `BandRange` of its procedure, an `AddQa` is not followed by a `Dedupe`, or a chain is over capacity. A `Preprocess` that does not fit its profile raises `ValueError`: an unknown QA code name or scope when it is constructed, a role the profile lacks when the profile is, and these checks when the plan is built.

# Build and test

```bash
cmake -S . -B build                    # C++ library, tests and the Python module
cmake --build build --config Release -j8
pip install .                          # the Python package
```

CMake options: `CCD_ENABLE_OPENMP` (ON), `CCD_BUILD_PYTHON` (ON), `CCD_BUILD_TESTS` (ON), `CCD_OBSERVE` (ON; the diagnostics hooks), `CCD_EXTENSIONS` (strategy extension directories; none by default) and `CCD_INSTALL_CPP` (OFF; installs `ccd_core` and the headers). On macOS with Apple clang, CMake finds Homebrew's libomp only when told where it is: add `-DOpenMP_CXX_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_CXX_INCLUDE_DIR=$(brew --prefix libomp)/include -DOpenMP_libomp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib` to the configure line.

The tests run in three tiers, from the cheapest:

| Tier | Command | What it runs | When |
|---|---|---|---|
| Quick | `ctest --test-dir build -j8 -LE corpus` | every test, the sweeps over cuts (`PrefixStability.*`, `ResumeEqualsTheFullRun.*`) on their sample shard: every eighth cut, each series' last cut and the cuts through the peek after each break | while changing code |
| Full | `ctest --test-dir build -j8` | every test, every cut | before a change is merged, and after any change to the machine, the fits, the record or resume |
| Release | the full tier on a sanitizer build, below, and on Linux (`tools/linux-compare`) | memory errors, undefined behaviour and asserts; another compiler's arithmetic | before a release, or a change that can behave differently by platform |

The sanitizer build is Debug at `-O2` (asserts on; at `-O0` the sweeps run several hundred times slower than Release):

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O2 -g" \
  -DCCD_BUILD_PYTHON=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j8
ctest --test-dir build-asan -j8
```

`Golden.BitIdenticalToFixture`, `Bands.BitIdenticalToFixture` and `Transitions.BitIdenticalToFixture` compare against fixtures written by a Release build on one platform (clang, arm64, macOS), so they fail on the sanitizer build and skip on Linux; every other test passes on both. The labels need CMake 3.22 or later; before it every test runs unlabelled.

---

# License and Attribution

`ccdcpp` is released into the public domain under the **Unlicense**.

This project is a C++17 translation of the USGS LCMAP PyCCD software.
The original PyCCD implementation was also released into the public domain.

The goals of this project are to:

- preserve PyCCD's algorithm, doing what its code intends where the two differ
- reproduce PyCCD's outputs except where [Differences from pyccd](#differences-from-pyccd) says otherwise
- provide a high-performance native C++ implementation
- enable Python interoperability through pybind11

Minor implementation changes were necessary to accommodate modern NumPy versions and to adapt portions of the Python implementation into efficient C++.

---

# Independent Development

`ccdcpp` is an independent software project developed separately from the U.S. Geological Survey (USGS).

This project was created independently for research, software engineering, and performance optimization of the publicly available CCD algorithm.

The author is not affiliated with, employed by, contracted by, or acting on behalf of the USGS in the development of this project. The USGS did not commission, fund, review, endorse, or provide technical support for this implementation.

The relationship to PyCCD is limited to algorithmic translation and validation against the publicly available implementation. All software architecture, memory management strategies, parallelization, optimizations, and implementation decisions in `ccdcpp` are original work developed independently for this project.
