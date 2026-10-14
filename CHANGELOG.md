# Changelog

## 0.5.0 - 2026-10-13

Preprocessing plans, in C++ and Python: per procedure, the transform and the
filter chain that build the processing mask, with the default plan the
options describe. API only; outputs are those of 0.4.0, bit for bit.

### Added (Python API)

- Preprocessing plans in Python. The bound classes `StepKind`, `Step` (named
  constructors `keep_qa`, `add_qa`, `thermal_range`, `band_range`,
  `green_median`, `dedupe`), `Chain`, `Affine`, `ProcedurePlan` and
  `PreprocessPlan`, compared with `==` and printed by `describe()` and
  `repr()`; `default_plan(hoptions)`, `validate_plan(plan, bands)` and
  `CHAIN_CAPACITY`.
- `detect`, `detect_cube` and `detect_cube_columns` take `preprocess=`,
  keyword only: a `PreprocessPlan`, or `None` (the default) for
  `default_plan(hoptions)`. The plan is checked against the band count
  (`ValueError`) and copied before the GIL is released. A call without it is
  unchanged, and so is its output, bit for bit.
- Plans by role for sensor profiles, as frozen dataclasses: `KeepQa`,
  `AddQa`, `ThermalRange`, `BandRange`, `GreenMedian`, `Dedupe`, `Transform`,
  `ProcedureChain` and `Preprocess`. Roles resolve to the profile's rows,
  QA code names (`QA_CODE_FIELDS`) to the run's `QA_*` codes, and a field
  left `None` to the value the default plan takes from the run's options;
  `Preprocess()` is the default plan.
- `SensorProfile.preprocess`, default `None` (the default plan of the
  profile's options), and `SensorProfile.preprocess_plan(options=None)`,
  which builds the `PreprocessPlan`. A profile with a plan stays hashable and
  picklable. The entry points run the profile's plan unless `preprocess=` is
  passed, as with `qa_table=`: a `PreprocessPlan` or `Preprocess` replaces
  it, `None` runs the default plan. Every preset has `preprocess=None`.
- README: a Preprocessing section, with the default chains of each
  procedure, custom plans in C++ and Python, and the remaining differences
  from pyccd.

### Added (C++ API)

- `ccd/preprocess/preprocess.hpp`: preprocessing plans. A `PreprocessPlan`
  holds one `ProcedurePlan` per procedure (`standard`, `snow`,
  `insufficient_clear`): an optional `Affine` transform of one band, applied
  to the per-pixel copy, and a `Chain` of up to 8 `Step`s (`KeepQa`, `AddQa`,
  `ThermalRange`, `BandRange`, `GreenMedian`, `Dedupe`) that builds the
  processing mask. `Bound` and `Range` are `filter::Bound`/`filter::Range`.
- `preprocess::default_plan(hoptions)` is the plan the options describe, the
  chains `filter::standard`, `filter::snow` and `filter::insufficientclear`
  run and the Standard-only thermal transform; those three functions are now
  wrappers over `preprocess::apply(default_plan(o).<procedure>.chain, ...)`.
- `preprocess::validate(plan, bands)` throws `std::invalid_argument` for a
  band index not below `bands`, a `GreenMedian` step without a band, a
  transform band that a `BandRange` step of the same procedure tests, an
  `AddQa` step not immediately followed by `Dedupe`, and a chain over
  capacity.
- `detect` and `detect_cube` take an optional trailing
  `const PreprocessPlanPtr& preprocess = nullptr`
  (`std::shared_ptr<const PreprocessPlan>`), after `qa_table`. Null runs
  `default_plan(hoptions)`, built once per call; either plan is validated
  once per call, before any pixel runs. Existing calls are unchanged, and
  their outputs are unchanged, bit for bit.

### Changed (C++ API)

- `FitProcedure::run` and `StandardProcedure::run` take the processing mask:
  `run(hworkspace, mask, lworkspace, solver)`. `ccd::detect` builds it per
  pixel: decode the QA, compute the QA statistics, select the procedure,
  apply that procedure's thermal transform (Standard only, as before), then
  run the procedure's filter chain. `run(hworkspace, lworkspace, solver)` on
  each procedure builds the mask with its filter (`filter::standard`,
  `filter::snow`, `filter::insufficientclear`) and calls it.
- `select_observations` is no longer a virtual of `FitProcedure`.
- Outputs are unchanged, bit for bit.

## 0.4.0 - 2026-10-13

### Changed (outputs)

The observation filters' defaults are now pyccd's, as pyccd intends them.
`HarmonicOptions()` and every preset change together (`landsat_c1_dn()` is
still `HarmonicOptions()` field for field), so a caller that builds its
options from either sees both changes below. The millworks nlcd plugin builds
its options from `HarmonicOptions()` and will see both when it upgrades to
0.4.0.

**Reflectance bounds are exclusive.** `REFL_MIN_INCLUSIVE` and
`REFL_MAX_INCLUSIVE` default to `False` (`Units.refl_inclusive=(False,
False)`), so the saturation filter keeps `REFL_MIN < v < REFL_MAX`, which is
pyccd's `0 < v < 10000` (`ccd/qa.py`, `filter_saturated`). Before, both ends
were inclusive.

- Who is affected: a pixel with an observation the filters kept (clear or
  water, thermal in range) on which a saturation band is exactly `REFL_MIN` or
  `REFL_MAX` (`0` or `10000` in DN, `0.0` or `1.0` in reflectance), or NaN.
  That observation is now dropped, and the pixel's segments, breaks and
  coefficients can change. Integer input clipped to the bounds is the common
  case: a `uint16` array where negative reflectance was clipped to `0` now
  gives the result of the unclipped values. A NaN in a saturation band no
  longer reaches the fit, where it left the pixel with no segments; the pixel
  now fits without that observation.
- To keep observations on the bounds: set `REFL_MIN_INCLUSIVE` and
  `REFL_MAX_INCLUSIVE` to `True` (`refl_inclusive=(True, True)` in a
  profile). That reproduces 0.3.0 output.

**The green median filter drops only observations up to `STAT_ORD`.**
`GREEN_FILTER_SCOPE` defaults to `StatWindow` (`green_filter_scope=
"stat_window"`). The threshold is unchanged, the median green of the masked
observations dated up to `STAT_ORD` plus `MEDIAN_GREEN_FILTER`; an observation
at or above it is now dropped only when it is itself dated up to `STAT_ORD`,
which is the set pyccd's `insufficient_clear_filter` computes its green mask
for. Before, it was dropped at any date.

- Who is affected: InsufficientClear pixels (`curve_qa` 44) with masked
  observations dated after `STAT_ORD` whose green is at or above the
  threshold. Those observations are now kept, so the pixel's single segment
  has more observations and its coefficients change; a pixel that had too few
  observations to fit can now fit. Standard and PermanentSnow pixels are
  unaffected.
- To drop them at every date: set `GREEN_FILTER_SCOPE` to
  `ccdcpp.GreenFilterScope.All` (`green_filter_scope="all"`). That reproduces
  0.3.0 output.

With both escape hatches set, 0.4.0 gives 0.3.0's output bit for bit.

### Added

- `HarmonicOptions.REFL_MIN_INCLUSIVE` and `REFL_MAX_INCLUSIVE` (default
  `False`) and `THERM_MIN_INCLUSIVE` and `THERM_MAX_INCLUSIVE` (default
  `False`): whether a value equal to a saturation or thermal bound is kept.
  With both ends of a range inclusive NaN passes; with either end exclusive
  it is rejected. In profiles: `Units.refl_inclusive=(lo, hi)` and
  `Thermal.inclusive=(lo, hi)`.
- `HarmonicOptions.GREEN_FILTER_SCOPE`, a `ccdcpp.GreenFilterScope`: which
  observations the InsufficientClear green median filter drops. `StatWindow`
  (the default) drops those dated up to `STAT_ORD`, as pyccd intends, `All`
  drops them at any date, and `Off` disables the filter. In profiles:
  `SensorProfile.green_filter_scope`, `"stat_window"`, `"all"` or `"off"`.
- `ccd::validate_dates(dates)`: throws `std::invalid_argument` unless `dates`
  is non-decreasing.

### Changed

- `detect`, `detect_cube` and `detect_cube_columns`, in C++ and Python, raise
  `ValueError` (`std::invalid_argument`) when `dates` decreases anywhere,
  before any pixel runs. Equal dates are still accepted and deduplicated.
  Unsorted dates were checked only by a debug-build assertion, and a Release
  build returned results computed on the wrong order.

## 0.3.0 - 2026-10-13

### Sensor profiles

- `ccdcpp.SensorProfile`: band names and roles, the detection, Tmask and
  saturation lists by role name, the unit-coupled thresholds, the Lasso
  constants, cadence options, `STAT_ORD` and a QA table builder, in one
  immutable, picklable object. `options()` resolves it to
  `(HarmonicOptions, LassoOptions)`, `qa_table()` builds its table and
  `check(spectral)` checks the band count. The change and outlier thresholds
  are computed for the detection band count, with a warning when a profile
  of other than five detection bands is given the chi2(5) defaults.
- `ccdcpp.presets`: `landsat_c1_dn()` (exactly `HarmonicOptions()` and
  `LassoOptions()`), `landsat_c1_physical()` and `landsat_c2()` (physical
  units, raw `QA_PIXEL`).
- `profile=` on `detect`, `detect_cube` and `detect_cube_columns`. Explicit
  `hoptions`, `loptions` and `qa_table` override the profile's. With a
  profile, `detect` and `detect_cube_columns` return `band_names`. Calls
  without `profile=` take the same arguments and return the same output as
  before; the three functions are now Python functions in `ccdcpp` that call
  the extension's (`ccdcpp.ccdcpp.detect`, ...).

## 0.2.0

### Changed output: the change threshold follows the adjusted peek window

The Standard procedure now lowers the change threshold when it lengthens the
peek window, as pyccd does. The peek window was already lengthened for a dense
series (`PEEK_SIZE * PEEK_REFERENCE_DAYS / median gap`), but the threshold was
computed from the adjusted peek size against itself, so it stayed at
`CHANGE_THRESHOLD` (15.09) for every window. It is now the chi-squared
quantile, one degree of freedom per detection band, at
`1 - (1 - CHANGE_PROBABILITY) ** (PEEK_SIZE / peek)`, from the `PEEK_SIZE` and
`CHANGE_THRESHOLD` the caller passes: 9.24 for a peek window of 12 (an 8-day
series) and 12.24 for a window of 8.

Who is affected: any pixel whose median gap between consecutive observations
the processing mask keeps (up to `STAT_ORD`) is under about 14.8 days, which is
where the peek window grows past `PEEK_SIZE`. Those pixels can gain or lose
segments and breaks, and their coefficients can change. A pixel at a 16-day
cadence or sparser gives the same output as before, bit for bit. There is no
option to restore the old threshold; outputs of 0.1.x are reproducible from the
`v0.1.x` tags.

`CHANGE_PROBABILITY` therefore now changes detection results on a dense series.

### Sensor-agnostic inputs

- Band count validated against the options at every entry point (`ValueError`
  from Python), and `ccdcpp.__version__`.
- Generic band layout: `NUM_BANDS`, `DETECTION_BANDS`, `TMASK_BANDS`,
  `SATURATION_BANDS`, `GREEN_IDX` and `THERMAL_IDX` (or no thermal band) in
  `HarmonicOptions`; outputs sized by the band count and reported with
  `num_bands`. The default layout is Landsat's seven bands.
- QA lookup tables: raw QA decoded through an optional 65,536-entry table
  (`qa_table=`, `ccdcpp.qa_tables`: Landsat Collection 1 and 2, HLS Fmask,
  Sentinel-2 SCL); pre-classified QA codes work as before.
- Cadence options: `PEEK_REFERENCE_DAYS`, `VARIOGRAM_MIN_LAG_DAYS`,
  `RETRAIN_MIN_OBS`, `RETRAIN_SPAN_FACTOR`, `SEASONAL_RMSE_COUNT`,
  `SEASONAL_RMSE_DIVISOR` and `CHANGE_PROBABILITY`, defaulting to pyccd's
  values, and `ccdcpp.chi2_quantile`.

### Since 0.1.2, with output unchanged

- Performance work that keeps every output bit for bit (fewer fits,
  allocations and copies; uint16/float32/strided cubes read in place), plus
  unit-agnostic thresholds, `detect_cube_columns`, `threads=`, raster-ordered
  `detect_cube` results, and inputs no longer modified in place.
