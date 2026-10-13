# Changelog

## Unreleased

### Added

- `HarmonicOptions.REFL_MIN_INCLUSIVE` and `REFL_MAX_INCLUSIVE` (default
  `True`) and `THERM_MIN_INCLUSIVE` and `THERM_MAX_INCLUSIVE` (default
  `False`): whether a value equal to a saturation or thermal bound is kept.
  With both ends of a range inclusive NaN passes; with either end exclusive
  it is rejected. The defaults are the bounds as they were, so output is
  unchanged. In profiles: `Units.refl_inclusive=(lo, hi)` and
  `Thermal.inclusive=(lo, hi)`.
- `HarmonicOptions.GREEN_FILTER_SCOPE`, a `ccdcpp.GreenFilterScope`: which
  observations the InsufficientClear green median filter drops. `All` (the
  default, and the filter's behaviour so far) drops them at any date,
  `StatWindow` only those dated up to `STAT_ORD`, as pyccd intends, and `Off`
  disables the filter. In profiles: `SensorProfile.green_filter_scope`,
  `"all"`, `"stat_window"` or `"off"`.

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
