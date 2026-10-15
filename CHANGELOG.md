# Changelog

## Unreleased

### Fixed

- `PEEK_SIZE` below 2 is refused by `ccd::validate()` and every entry point
  (`ValueError` in Python: "PEEK_SIZE is 1; it must be at least 2"). It
  crashed `detect()` before: Lookback peeks `PEEK_SIZE - 1` observations far
  from the previous break, and both peeks read their first magnitude. Valid
  inputs are unchanged.

### Changed (C++ internals; outputs bit for bit)

- The Standard procedure keeps its per-pixel state in records instead of
  hand-kept pairs and the options. `ccd::Series`
  (`include/ccd/procedure/series.hpp`) owns the processing mask and the
  observations it keeps, and is the only way to drop one: `drop(position,
  window)` and Initialise's tentative Tmask removal `drop_outliers_if`,
  each moving the caller's window by one rule. `ccd::standard::Cadence`
  holds the adjusted peek and change threshold (`measure_cadence`), and
  `ccd::Variogram` the per-band variogram, so the caller's `PEEK_SIZE` and
  `CHANGE_THRESHOLD` are no longer overwritten in a per-pixel copy.
- `HarmonicWorkspace` refers to the caller's `HarmonicOptions` instead of
  copying them, and `options()` is const only; it cannot be built from a
  temporary. `FitProcedure::run` and `StandardProcedure::run` take a const
  workspace. `tmask()` and `stable()` take a `const Variogram&`.
- Removed, unused: `update_processing_mask` (both overloads), and
  `remove_outliers` / `remove_observation` (now inside `Series`), the
  vector-returning `StandardProcedure::calc_residuals`, `Window::rewind`
  and `Window::extend`.

### Tests

- `tests/ccd/golden/test_transitions.cpp` (ctest `Transitions.*`): 17
  characterisation pixels (`characterisation.hpp`) built for the Standard
  procedure's transitions, guards and pyccd quirks that the golden corpus
  misses or only Python reaches: Lookback reading below the previous break
  and Lookback's `PEEK_SIZE - 1` peek each deciding an output, the re-entry
  after a series-end segment at peek 32 with its unused Tmask and Lookback
  exclusions, the peek guard, the variogram NaN exit, `COEFFICIENT_MIN`
  against Initialise's literal 4, and the procedure choice at both
  thresholds. Their output is pinned byte for byte in
  `transitions_fixture.txt`, written by v0.6.1. The test's header lists
  which pixel takes which transition and which pinned pixels each decided
  0.8.0 fix moves. No library change.
- `Transitions.PathsAsDocumented` also pins, to a tolerance, the values the
  quirks decide on the pixels whose shapes they leave alone (the mask count
  of `lookback_below_break_decides`; the per-band RMSE of
  `lookback_peek_minus_one_decides` and `coefficient_min_6`), so fixes 2, 4
  and 7 move a pin off the fixture platform too.
- `tests/ccd/procedure/test_series.cpp` (ctest `Series.*`, `Variogram.*`,
  `StandardCadence.*`, `HarmonicWorkspace.*`): the records' contracts.

## 0.6.1 - 2026-10-15

The HLS and Sentinel-2 presets promoted on a real sample, a band-order check
on the entry points, and real-data regression fixtures. Outputs are those of
0.6.0, bit for bit.

### Changed (validation status)

- The presets were run on one cell, Landsat ARD h024v013 chunk 00030008
  near Atlanta cut to MGRS tile T16SGC, 2020-01-01 to 2023-12-31, from
  Microsoft Planetary Computer (HLS v2.0 L30 and S30; Sentinel-2 L2A at
  20 m), against Landsat Collection 2 ARD of the same cell and years. Four
  years of one cell is the whole record, as the README and the module
  docstring say; each preset docstring names the cell and 2020-2023.
- `hls_l30()`: validated. Break agreement with Landsat 8+9 ARD kappa 0.44
  (Landsat 7 added to the ARD: 0.43), matched breaks a median 8 days apart,
  model levels within 0.004 reflectance.
- `hls_s30()` and `hls()`: validated, with a caveat: Fmask over bright
  surfaces. Fmask on Sentinel-2 calls bright barren land cloud: over all
  1,862 quarry (land cover 31) pixels on the HLS grid it flags 63% of S30
  observations as cloud, and 1,145 of them go to InsufficientClear in S30
  (68% cloud over those); Landsat goes InsufficientClear there far less
  (170 of ARD's 173 such pixels, both of L30's 2, are quarries). `hls()`
  sends 1,741 of 300,288 pixels there.
- `sentinel2_l2a()`: validated at median level on Planetary Computer's
  Esri/MSFT/S2RP L2A, 2020–23; ESA's own processing not sampled. Levels
  about 0.01 above S30 with about twice the RMSE; the per-scene offset holds
  across the 2022 baseline change.
- SCL 7 (unclassified) stays cloud. Documented: with SCL 2 on small water
  bodies it sends 1,355 of 3,983 water pixels, 3,426 pixels in all, to
  InsufficientClear (2,545 as fill, 587 as clear).

### Added (Python API)

- `band_names=` on `detect`, `detect_cube` and `detect_cube_columns`,
  keyword only, default `None`: the names of `spectral`'s rows as the caller
  holds them. Given, they must be the profile's `band_names` in the
  profile's order; a misordered stack (a Sentinel-2 store with B08 among the
  10 m bands) raises `ValueError` naming the first row that differs, other
  names raise `ValueError` listing what is missing, extra or repeated, and a
  str or a non-str name raises `TypeError`. Only with `profile=`
  (`TypeError` otherwise). Without it a call is unchanged.
  `SensorProfile.check(spectral, band_names=None)` takes the same argument.

### Tests

- `tests/python/fixtures/t16sgc_2020_2023`: four pixels (evergreen forest
  `481_393`, named 1-based (column, row); open water; high-intensity
  developed; a pixel with a 2022 break) in HLS L30, HLS S30 and Sentinel-2
  L2A, and a fifth in L30 and S30 whose result depends on the coastal B01
  low bound (7 L30 and 2 S30 observations dropped by it alone), as raw
  values and raw QA with each scene's Sentinel-2 offset, the STAC item ids, a provenance and license note, and the output of every
  HLS and Sentinel-2 preset on them. `test_real_fixtures.py`, ctest
  `python_real_fixtures`, compares against that output.

### Documentation

- README: Presets and QA-table rows carry the validation status; a
  Validation on real data section; the measured cadence (`hls()` 4–5 days,
  peek 19–24; S30 and Sentinel-2 5 days, peek 19) in place of the 2–4 days
  and 32–48 documented before. The Fmask bit 0 is named reserved, in the
  README and in `qa_table.hpp`, where SCL 2 is named cast shadows (comments
  only), and in the `qa_tables` docstrings, which no longer call it cirrus.
  Those docstrings now say the HLS and Sentinel-2 tables decode as
  documented on every value present in the T16SGC 2020-2023 sample and that
  no production run has used them yet.

## 0.6.0 - 2026-10-15

Presets for HLS v2.0 and Sentinel-2 L2A, not validated in production, and
QA-table builder arguments on sensor profiles. Python and documentation
only; outputs are those of 0.5.2, bit for bit.

### Added (Python API)

- `ccdcpp.presets.hls_l30()`, `hls_s30()`, `hls()` (L30 and S30 in one
  series) and `sentinel2_l2a()`, each labelled "not validated in
  production": the values follow the HLS v2.0 User Guide and ESA's
  Sentinel-2 L2A documentation and are tested on synthetic series only. All
  four take physical units (HLS: every `int16` layer times 0.0001, which
  puts L30 B10 in degrees Celsius / 100; Sentinel-2: offset-corrected BOA
  reflectance, `(DN + BOA_ADD_OFFSET) / 10000`), have the thresholds and
  Lasso constants of `landsat_c1_physical()`, detect on green, red, nir
  (B05 on L30, B8A on S30 and Sentinel-2), swir1 and swir2 with the default
  thresholds, saturation-screen every optical row (coastal B01 included),
  have no statistics cutoff and keep pyccd's cadence values. The red edges
  and B08 are fitted and reported, not detected on. QA is raw `Fmask`
  (`qa_tables.hls_fmask`, aerosol ignored) or raw `SCL`
  (`qa_tables.sentinel2_scl`). The single-product presets name their rows
  by product band ID, `hls()` by role.
- `SensorProfile.qa_table_options`: keyword arguments for the
  `qa_table_builder`, a mapping or `(name, value)` pairs; none given by
  default. `dataclasses.replace(presets.hls_l30(),
  qa_table_options={"exclude_high_aerosol": True})` decodes high aerosol as
  cloud. Each value is of exactly the type the builder declares, listed in
  `ccdcpp.QA_TABLE_OPTIONS` (`exclude_high_aerosol` takes `True` or `False`;
  `None`, `0`, `1`, `"True"` or `numpy.bool_(True)` raise). A profile stores
  every argument its builder declares, sorted by name, with the default
  written out where none is given: the HLS presets store
  `(("exclude_high_aerosol", False),)`, and `()`,
  `{"exclude_high_aerosol": False}` and the preset are one profile (equal,
  same hash, one cached table). A profile without a builder stores none.
  Tables are cached per builder, codes and arguments. Options without a
  builder, a repeated name, a name the builder does not take, a value of
  another type, input that is not a mapping or `(name, value)` pairs, or a
  `qa_table_builder` that is not a str raise `ValueError` at construction.
- `ccdcpp.QA_TABLE_OPTIONS`: per `ccdcpp.qa_tables` builder, the keyword
  arguments it takes beyond `options`, with their types and defaults.
- The role `nir_broad`, for the broad 842 nm Sentinel-2 B08.

### Changed (Python API)

- The role `nir_narrow` is removed; `nir` is the 865 nm Landsat-equivalent
  near infrared on every sensor, B8A on Sentinel-2. A profile naming
  `nir_narrow` raises `ValueError` (unknown role). No preset used it.

### Documentation

- README: the Presets section lists the HLS and Sentinel-2 presets with
  their rows, units, QA and validation status, the Sentinel-2 offset the
  caller applies, and the L30-before-S30 order on a shared date; the HLS
  `Fmask` notes that the User Guide reserves bit 0; the SCL shadow classes
  are named. The `ccdcpp.profiles` and README "Adding a sensor" example is
  a Landsat 8/9 Collection 2 profile with the coastal band, not shipped.

## 0.5.2 - 2026-10-14

### Fixed (build)

- The static core library `ccd_core` is built position-independent
  (`POSITION_INDEPENDENT_CODE ON`). It is linked into the Python extension, a
  shared object, and Linux linkers refuse non-PIC objects there
  (`relocation R_X86_64_PC32 ... recompile with -fPIC`), so the Python package
  did not build on Linux with conda-forge's gcc. macOS builds were unaffected.
  Outputs are unchanged, bit for bit.

## 0.5.1 - 2026-10-14

### Fixed (outputs)

**The peek window's median gap is no longer truncated.** The Standard
procedure sizes its peek window from the median gap between consecutive
observations up to `STAT_ORD`, and took that median in integer days: for an
even number of gaps whose two middle values differ by one (8 and 9, say) it
got the lower one, where pyccd's `np.median` gets the half day (8.5). The
window then came out at `round(96 / 8.001) = 12` where pyccd's is
`round(96 / 8.501) = 11`, with the change threshold that goes with it (9.24
against 9.80). The median is now taken in `scalar_t`, and `median()` refuses
an integral type at compile time.

- Who is affected: a pixel whose kept observations up to `STAT_ORD` have an
  even number of gaps with differing middle values, where the half day moves
  `round(PEEK_SIZE * PEEK_REFERENCE_DAYS / (median + 0.001))` across an
  integer. Its segments, breaks and coefficients can change. On 1257 sampled
  Atlanta pixels (tile 024013) one changed, chunk 00030008 pixel (481, 393),
  which now matches pyccd (with the controls the README lists under
  Differences from pyccd); the synthetic corpus and the golden fixtures are
  unchanged, bit for bit.

### Documentation

- README, Differences from pyccd: pyccd's unstable sort in `find_closest_doy`
  picks different tied seasonal-RMSE observations on dense series; ccdcpp breaks
  ties in window order.

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
