# Changelog

## 1.0.0 - Unreleased

The input says only how to read the data; the configuration says what to do with it, in
physical units; the run translates once (0–1 reflectance × 10000 = the 0–10000 DN check, as in
0.10.0). Two calls, `detect` and `detect_cube`, take the input (`input=`, how to read the data) and a
configuration (`config=`, the `ccdc` preset by default) in Python and C++. One state machine runs
the CCDC phases, and each of its decisions is a stage decided by a named strategy. Outputs are
0.10.0's bit for bit for the same input, configuration and statistics cutoff, except for two
deliberate changes of defaults: `ccdc` has no statistics cutoff, and the PermanentSnow and
InsufficientClear chains test their thermal range in degrees Celsius on every input. 0.10.0
resume records are refused: rerun once
with `record=True`. [docs/migration-1.0.md](docs/migration-1.0.md) maps every 0.10.0 name to
its replacement.

### Changed

- **No statistics cutoff by default.** `ccdc`'s `shared.statistics_end` is `NO_STATISTICS_END`:
  the QA statistics, the cadence, the variogram and the green median read the whole series.
  0.10.0 and pyccd stopped them at 2017-12-31 (ordinal 736694), LCMAP's 2018 production
  baseline; `ccdcpp.config("ccdc").replace(shared__statistics_end=736694)` is that run, bit for
  bit. On the Atlanta cell (h024v013, 250,000 pixels) 57.1% of the pixels differ, 6.5% in their
  segment count.
- **The snow and insufficient-clear thermal check compares in degrees Celsius on every input.**
  pyccd's PermanentSnow and InsufficientClear filters test the untransformed thermal band
  against the Celsius bounds. `ccdc`'s `snow` and `insufficient_clear` chains test the stored
  band against the bounds taken back through the input's thermal transform
  (`ccd::preprocess::stored_range()`: 1799.5 to 3438.5 on Collection-1 DN, 179.95 to 343.85 K),
  so they keep exactly the values whose transform Standard's range keeps. On Collection-1 DN
  they drop, besides saturated (20000) and NaN values as before, the implausible temperatures
  0.10.0 kept: no change on the 1,500 reference Collection-1 pixels, 218 of the synthetic
  corpus's 2,900 pixels, none of the Atlanta cell's 250,000 (Collection 2). Physical-unit
  outputs are 0.10.0's. An input's `thermal_scale` must be a positive number and its
  `thermal_offset` a finite one.
- **`ccd::Input` (`ccdcpp.Input`) replaces `HarmonicOptions`** and says only how to read the data: QA
  codes and `qa_table`, `roles`, `reflectance_scale`/`reflectance_offset`,
  `thermal_scale`/`thermal_offset` with `thermal_at_100c`, `saturation_code` and
  `thermal_saturation_code`. Immutable in Python (`Input(**fields)`, `replace()`).
- **The algorithm is a configuration** (`config::Config`, `ccdcpp.Config`): `shared` values,
  13 stages each with a strategy and its parameters, and a preprocessing chain per procedure, in
  physical units (reflectance, degrees Celsius, probabilities, bands by role, QA codes by name).
  Every 0.10.0 algorithm field and `LassoOptions`' constants are values of it; thresholds are
  derived from probabilities. It is checked and compiled when built (`ccd::Algorithm`) and
  resolved once per input.
- **Python: two calls.** `detect(dates, spectral, qas, *, input, config="ccdc", record=False,
  resume=None, band_names=None, trace=False)` and `detect_cube(..., *, input, config="ccdc",
  threads=0, record=False, resume=None, band_names=None, trace_pixels=None, masks=False)`, which
  returns 0.10.0's `detect_cube_columns` columns, names and dtypes unchanged; `resume=` is an
  earlier call's output.
- **C++: two calls.** `ccd::detect` and `ccd::detect_cube` run a `ccd::RunSpec{input,
  algorithm, record, masks, observer, observe}` on views of float64, float32 or uint16 values,
  replacing the 20 `detect`, `detect_recorded`, `resume`, `detect_cube` and `resume_cube`
  overloads. `RunSpec::masks` is off by default (0.10.0's `keep_masks` was on).
- **Profiles describe the data only**: `SensorProfile(name, band_names, roles, units, thermal,
  saturation_code, thermal_saturation_code, qa_table_builder, qa_table_options)`, with
  `Units(scale, offset)` and `Thermal(scale, offset, at_100c)`; `profile.input()` replaces
  `options()` and `qa_table()`. `CELSIUS_HUNDREDTHS` is `CELSIUS_OVER_100`.
- **Error messages** name the stage, strategy and parameter (`change_test 'chi2' parameter
  'peek_size' is 1; it must be at least 2`), or `shared` for a shared value.
- **Resume**: the header's digest (`digest`, was `options_digest`) covers the input and the
  resolved configuration: `RESUME_FORMAT` is 2, and a 0.10.0 record is refused by its format
  (`the resume record's format is 1; this ccdcpp reads 2`). `RESULTS_VERSION` is 3: the
  default outputs changed.
- **The source layout**: `include/ccd/` and `src/ccd/` in `core/`, `input/`, `regression/`,
  `config/`, `strategies/`, `machine/`, `results/` and `run/`; `ccd/ccd.hpp` includes every
  header. `HarmonicWorkspace` is `PixelSeries`.

### Added

- Stages and strategies (`include/ccd/strategies/`, [docs/ccdc-state-machine.md](docs/ccdc-state-machine.md#stages-and-strategies)):
  `mode`, `noise_floor`, `start_search`, `window_screen`, `stability_fit`, `stability_rule`,
  `lookback`, `model`, `refit`, `change_test`, `outlier`, `segment_fit`, `edge_fit`, each with
  its pyccd strategy; a registry of strategies, steps and presets with parameter schemas;
  strategy extensions through CMake's `CCD_EXTENSIONS`.
- Configurations from Python ([docs/config.md](docs/config.md)): `ccdcpp.config()` (a preset,
  a TOML file with `extends`, a dict), `Config.replace(shared__name=, stage__param=,
  stage="strategy", preprocess__...=)`, `resolve()`, `hash()`, `to_toml()`;
  `ccdcpp.strategies()`, `shared_values()`, `steps()`, `plan_of()`; `python/ccdcpp/configs/ccdc.toml`.
- `detect_cube(masks=True)`: every pixel's processing mask in one `(H, W, T)` uint8 array;
  `ccdcpp.to_pixels()` reads the columns per pixel.
- Diagnostics ([docs/diagnostics.md](docs/diagnostics.md)): `ccd::Observer`,
  `ccd::TextObserver`, `detect(trace=True)` and `detect_cube(trace_pixels=)`, compiled out with
  `CCD_OBSERVE=OFF`.
- `Input.saturation_code` and `thermal_saturation_code`, declared by `landsat_c1_dn()` (20000);
  [docs/units.md](docs/units.md) lists which products store one.
- Types: `py.typed`, `ccdcpp.pyi`, `qa_tables.pyi`, and the results as `TypedDict`s in
  `ccdcpp.results`.
- [docs/benchmarks.md](docs/benchmarks.md) and the bench tools (`tools/bench/`), with peak
  memory beside every time; per pixel 1.0 runs as fast as 0.10.0.

### Removed

- Python: `HarmonicOptions`, `LassoOptions`, the `hoptions`/`loptions` arguments, `profile=`,
  `qa_table=`, `preprocess=`, `resume_record=`, `detect_cube_columns`, `resume`,
  `resume_cube_columns`, the per-pixel-dict `detect_cube`, `default_plan`, `validate_plan`,
  `Transform`, `DEFAULT_TRANSFORM`, `CADENCE_FIELDS`, `GREEN_FILTER_SCOPES`,
  `OUTLIER_PROBABILITY`, `QA_CODE_FIELDS`, `NO_STAT_CUTOFF` (`NO_STATISTICS_END`), and the
  profiles' algorithm fields (`detection`, `tmask`, `saturation`, `alpha`, `coef_floor`,
  `cadence`, `stat_ord`, `change_threshold`, `outlier_threshold`, `green_filter_scope`,
  `preprocess`).
- C++: `HarmonicOptions`, `LassoOptions` as a call argument, `CLEAR_OBSERVATION_THRESHOLD`,
  `QA_IDX`, `Recorded`, `Resumed`, `CubeResult` (`CubeOutput`), `DetectScratch` (`Worker`),
  `options_digest()`, `default_plan()`, `PreprocessPlanPtr`, `select_procedure()`,
  `standard::measure_cadence()`, `standard::coefficient_count()`, `Clock`, `elapsed_ms` and
  `print_change_model(s)`.

### Fixed

- Values 0.10.0 ran are refused when the configuration is built: `NUM_OBS_FACTOR` 0,
  `MEOW_SIZE` 0 (`min_observations`), a negative or NaN Lasso constant (`alpha`, `coef_floor`,
  `tolerance`), a NaN change or outlier probability.
- A build without OpenMP compiles and runs a cube on one thread.
- A Start or End segment is written only on more observations than the edge fit's
  coefficients (`coefficient_min`), as well as more than the peek: a `peek_size` below
  `coefficient_min` (2 or 3, or `coefficient_min` 7 or 8 with the default peek) wrote a
  segment of too few observations, with a NaN or infinite RMSE (and an assertion in a debug
  build). The defaults' outputs are unchanged.

## 0.10.0 - 2026-10-17

Speed, and outputs moved in their last bits only. The Atlanta h024v013 cell
(500 x 500, 1984-2023, 8 threads) runs in 39.7 s against 0.9.0's 79.3 s;
the synthetic bench at one thread in 3.06 s against 5.76 s. Every segment
is unchanged on that cell and on the golden corpus (the same days, counts,
change probabilities and curve QA); coefficients, RMSEs and magnitudes
move by about 1e-10 relative at most, from the Lasso solver's sums in a
new order (`docs/floating-point-sums.md`). The other changes give the same
bits. **Results version 2**: a resume record stored by 0.9.0 is refused, so
the first run after upgrading is a full one. The annual update of that
cell (2023 added to 1984-2022) takes 14 s, 0.36x the full run (the update
is mostly the reading and the open segments' refit, which the faster
state machine leaves as they were). Tests run in a quick tier (`ctest -LE
corpus`) and a full one.

### Changed (outputs in their last bits; results version 2)

- The Lasso solver's two dot products (rho = X_j^T r in the coordinate
  descent, X^T r in the duality-gap test) sum in four interleaved partial
  sums: the same sum in another order, rounded at other places, about 20%
  faster over the whole run (the synthetic bench at one thread 3.87 s ->
  3.06 s). Measured on the Atlanta h024v013 cell and the golden corpus,
  every segment is unchanged (the same days, counts and change
  probabilities; 455,231 and 26,021 segments), no fit stops at another
  iteration (182,147 band fits), and the coefficients, RMSEs and
  magnitudes move by about 1e-10 relative at most (1.4e-10 there; 5.6e-10
  on small coefficients of an independent synthetic set; typically
  1e-16). A fit
  whose convergence test fell within rounding of its threshold could
  differ by up to the solver tolerance; none did. `RESULTS_VERSION` is 2:
  a record stored by 0.9.0 is refused, so the first run after the upgrade
  is a full one. The Golden, Bands and Transitions fixtures are rewritten
  (their diff moves band values only, at most 1.1e-11 relative).
  `docs/floating-point-sums.md` explains the solver's sums, why their order
  is part of the output and why four.

### Changed (speed; outputs bit for bit)

- Extend refits the detection bands only, the bands its tests read; the
  others are fitted when the segment is written, on the last refit's
  window with its coefficient count, with their residuals over the last
  test's peek (before the drop when a drop ends the curve): the same models,
  magnitudes and segments, without fitting the other bands at every refit.
  At the defaults (5 detection bands of 7) that is two fits in seven.
- `find_closest_doy()` (the seasonal RMSE's observations) computes the
  seasonal distance in integer quarter days and selects by a counting sort:
  the same observations in the same order, without a division per
  observation and a partial sort. Extend selects through a `SeasonalOrder`:
  the fit window's observations ordered by phase in the year once per refit,
  each test walking out from its target's phase, so a selection costs its
  count rather than the window's size.
- Together: the synthetic bench at one thread 5.90 s -> 3.92 s (-34%; the
  seasonal order about 4% of it).

### Tests

- Test tiers. The sweeps over cuts (`PrefixStability.*` and
  `ResumeEqualsTheFullRun.*` but the chain) run as 8 shards each
  (`Shards/<suite>.<test>/sample` and `/rest1` to `/rest7`), so ctest runs
  them in parallel; together they test the same 14,695 cuts.
  `ctest -LE corpus`, the quick tier, runs every test but the `rest` shards:
  the sample shard takes every eighth cut, each series' last cut and the cuts
  through the peek after each break (15-26% of a corpus's cuts). Plain `ctest` is
  the full tier. The README's "Run the Tests" lists the tiers, the sanitizer
  build among them (Debug at `-O2 -g`, asserts on). The labels need CMake
  3.22 (discovery filters); before it every test runs unlabelled.

## 0.9.0 - 2026-10-17

Resume from previous results: a run can keep a record of where each pixel's
machine stopped, and a later run on the longer series (the annual update:
1985-2023 stored, 1985-2024 given) continues each pixel from it and returns
the complete result, equal bit for bit to a full run with the statistics
cutoffs capped at the stored series end. A pixel whose earlier observations
changed is rerun and says so. **Outputs unchanged** from 0.8.0 (the results
version stays 1). Timing: detection level with 0.8.0 (synthetic 1.00, the
Atlanta cell 1.00); the annual update of that cell (1984-2022 stored, 2023
added) takes 0.21x the full run's detect time, all 250,000 pixels equal to
the full run (`tools/bench/bench_resume.py`).

### Added (C++)

- The resume record (`include/ccd/results/record.hpp`): `ccd::detect_recorded()`
  returns `detect()`'s result, bit for bit, with a `ResumeHeader` (the record
  format, `RESULTS_VERSION`, `options_digest()` of the options, plan and QA
  table, the series end and the statistics end) and a `PixelRecord`: the
  pixel's `input_fingerprint()` (every non-fill observation up to the series
  end: date, QA code, value bits), its procedure, where the Standard machine
  stopped (`standard::Suspended`: the exit, the cursor, the fit window's
  stop, a pending grow, the final segment count) and the days the machine
  dropped. What a later resume reads to continue a run on a longer series.
- `ccd::resume()` continues a recorded run on the whole of a longer series
  and returns the complete result, equal bit for bit to `detect()` with every
  statistics cutoff capped at the first run's series end (STAT_ORD and each
  GreenMedian step's date), with the record of this run for the next resume.
  The Standard machine continues from where it stopped (`standard::resume()`:
  Prepare as usual, the stored drops, the cursor, the guard that failed
  tested again; at a series end the open segment's models refitted on its fit
  window); the stored final segments are kept as given. A pixel whose
  observations up to the stored series end changed is rerun the same way,
  with `ResumeStatus::RerunInputChanged`; the pixels with nothing to resume
  (PermanentSnow, InsufficientClear, the early exits) are rerun too. Refuses
  another record format or results version, other options, a shorter series
  and too few stored segments. `standard::Suspended` also keeps the last
  peek test's coefficient count and magnitudes, which write a series end's
  provisional segment again when a longer series still ends there.
- Cubes: `detect_cube(..., record = true)` fills each `PixelResult::record`
  and the call's `CubeResult::header`; `resume_cube()` (`CubeResumeFrom`: the
  stored header and, in raster order, each pixel's record and stored
  segments) resumes every pixel in parallel, each as `resume()` would, with
  its `PixelResult::status`; a pixel with too few stored segments is a
  `CubeFailure`, the rest run.

### Added (Python)

- `detect(..., resume_record=True)` and `detect_cube_columns(...,
  resume_record=True)` also return `resume`, the record (one pixel's as a
  dict; a cube's as columns over its H x W pixels in raster order, the
  excluded days as `excluded_offsets` [N+1] and `excluded_days`), and
  `resume_header`. `resume(..., resume=, resume_header=, models=)` and
  `resume_cube_columns(..., resume=, resume_header=, segments=)` take them
  back with the stored segments (as `detect()` and `detect_cube_columns()`
  return them) and the whole series, and return the complete result with
  the next record, `status` and, for cubes, `counts`. `resume_schema()`
  names the record columns and their dtypes; `RESUME_FORMAT` and
  `RESULTS_VERSION` are exported. Every record and segment column is checked
  for its length and shape, the records for raster order and their offsets
  against the excluded days (`ValueError` otherwise); a record that does not
  fit its pixel's series is that pixel's failure (status 2), refused before
  anything reads at its positions. Stored segments with placeholders where
  the caller blanks a field (nlcd's break day and magnitude on no-break
  rows, no observation count) come back as given.

### Added (tools)

- `tools/bench/bench_resume.py`: the annual update of the bench cell, timed
  against the full run and checked against it pixel by pixel.

### Tests

- `ResumeEqualsTheFullRun.*` (`tests/ccd/golden/test_resume.cpp`): at every
  cut of the characterization and CSV pixels and twelve of each golden cube
  pixel (14,695 cuts, 11,014 continued from the record), the resumed result
  and record equal the full run's; chains of two resumes; a reprocessed
  observation rerun, a fill-only scene not a change, nothing new giving the
  stored output again, a plan without Dedupe (a pixel whose stored drop is on
  a date kept twice is rerun, since the day does not say which), and the
  refusals.
- `PrefixStability.*` also checks that the record's final segment count is
  the segments a longer series cannot rewrite, at every split.

## 0.8.0 - 2026-10-16

The Standard procedure keeps pyccd's intent where pyccd's code says
otherwise: seven decided fixes, each checked against Zhu's MATLAB reference
(TrendSeasonalFit), each moving only the pins listed with it, and each
holding the new prefix-stability test (new observations never rewrite a
final segment). **Outputs change**: on Landsat at the defaults through the
full Lookback peek, the series end and the refit at 24 observations (the
Atlanta h024v013 cell: 455,231 segments against 0.7.0's 456,157), at dense
cadences also through Lookback near the break, and every early exit now
returns its processing mask. The C++ segment record is a neutral
`ccd::BandFit`; the Python API is unchanged. Timing: level with 0.7.0 on the
real cell; the refit at 24 observations costs about 6% on the single-thread
synthetic bench (an algorithm cost, accepted; the optimisation stage
targets it). The tests that failed on Linux pass there.

### Changed (pyccd's intent, not its literals)

- Extend refits while the window holds at most `RETRAIN_MIN_OBS` (24)
  observations, the same windows that compare against the model RMSE, as Zhu's
  TrendSeasonalFit does with one `<=` for both; pyccd refits while fewer, so
  at exactly 24 it compared a 23-observation fit's plain RMSE. Default output
  changes on segments that grow through 24 observations. Pins moved: the
  golden fixture 177 of 621 pixels (13 with a segment boundary or count
  moved), the bands fixture 137 of 400, the transitions fixture
  `lookback_far_peek_decides` (its first segment's fit), `expected_hls_l30.json`
  3 of 5 pixels, all 144 check.sh dumps; `CCD.DetectStandard` compares
  segments 0 and 1, which grow through 24 observations, with pyccd's shapes
  only, and segments 2-4 still with pyccd's fits.
- Near the previous break Lookback peeks down to the break and never below
  it, into the previous segment, as Zhu's TrendSeasonalFit does. pyccd's
  guard against reading past the series start (`start <= peek`) also fired
  after a break, which needs an adjusted peek of 13 or more (a median gap
  of about 7.7 days or less: HLS, not Landsat). Pins moved: in the transitions
  fixture, `lookback_near_break_decides` keeps 493 observations in its mask
  (was 483), every segment unchanged; it and `lookback_near_break` are
  renamed from `lookback_below_break*`; no golden dump.
- Far from the previous break, Lookback peeks the adjusted peek of
  observations before the window, as Extend peeks after it. pyccd's
  `slice(start - 1, start - peek, -1)` reads one fewer, a porting slip
  (Zhu's TrendSeasonalFit tests `conse` observations both ways). Pins moved:
  in the transitions fixture, `lookback_far_peek_decides` (renamed from
  `lookback_peek_minus_one_decides`): its second segment's Lookback reaches
  back to 127 observations (was 109), its mask keeps 294 (was 299); no golden
  dump. Default output changes on real data (the Atlanta cell, fix 1
  alone on 0.7.0: 455,363 segments against 456,157). `PEEK_SIZE` below 2 is
  still refused, now because a change needs consecutive observations.
- Initialize fits `COEFFICIENT_MIN` coefficients and both peek bases (Lookback's
  and Extend's) are `COEFFICIENT_MAX` wide, as in Zhu's TrendSeasonalFit
  (`min_num_c`, `max_num_c`), in place of pyccd's literals 4 and 8. The
  defaults are 4 and 8, so default output is unchanged; a run with other
  counts changes. Pins moved: in the transitions fixture,
  `coefficient_min_6` (`COEFFICIENT_MIN` = 6), the fits of its first two
  segments (RMSE 0.4% to 10%), shapes unchanged; no golden dump.
- Lookback compares against the stable start window's RMSE as a stated rule
  (TrendSeasonalFit refits nothing while growing backward either); no output
  change.

### Fixed

- A curve that reaches the series end ends at its last full peek, and is the
  last curve. pyccd's check before Extend's grow had become dead (its `>=`
  turned into `>` when the loop guard changed), so the window grew once more
  and a series-end segment's end day was its break day; the check works
  again, so the segment ends before the observation its last peek starts at
  (end day < break day, as after an outlier drop), and EndCatch follows
  instead of Restart, which at a peek of 25 re-entered Initialize, and at
  26 and more ran it on the last observations and dropped observations no
  segment used (Zhu's
  TrendSeasonalFit ends the same way). A new internal `Exit::SeriesEnd`.
  Pins moved, each a series-end segment that ended after a grow, its end day
  one observation earlier and its count one lower, its start, break day,
  change probability, curve_qa, fits and magnitudes unchanged (checked pixel
  by pixel): the golden fixture 328 of 621 pixels, the bands fixture 225 of
  400, the transitions fixture 7 of 17 (one of them `dense_series_end`,
  renamed from `reentry_after_series_end`, whose mask also keeps the 4
  observations the re-entry dropped), the four real-data `expected_*.json` (11 of 19 pixels), all 144
  check.sh dumps; `CCD.DetectStandardHalfDayMedianGap` now ends one
  observation before pyccd's reference (738727, 959 against pyccd's 738751,
  960), every fit still pyccd's.
- The early exits return the processing mask, as pyccd does: a Standard pixel
  with at most `MEOW_SIZE` kept observations or a NaN variogram, and a
  PermanentSnow or InsufficientClear pixel with fewer than `MEOW_SIZE`, give
  no segments and their mask, where they gave an empty mask. Nothing else
  changes, and `detect_cube_columns` returns no mask. Pins moved, every one
  such a pixel's mask line only: the golden fixture 146 of 621 pixels (128
  Standard, 18 PermanentSnow), the bands fixture 37 of 400, the transitions
  fixture `snow_too_few`, `standard_too_few` and `variogram_nan`; of the
  check.sh dumps, 94 of 144 (the detect and detect_cube dumps; no column
  dump).
- `ccd::validate()` and every entry point refuse a `COEFFICIENT_MIN` below 1,
  coefficient counts that decrease from `COEFFICIENT_MIN` to `COEFFICIENT_MAX`,
  and a `COEFFICIENT_MAX` above 8, the harmonic basis's width: a model wider
  than the peek basis, or a basis wider than its storage, had no defined
  result.

### Changed (C++ API; outputs bit for bit)

- A segment's per-band fit is a `ccd::BandFit` (`include/ccd/machine/fit.hpp`):
  `intercept`, `coefficients`, `rmse`, `magnitude` and the solver's
  `iterations`, in place of a `LassoResult`. `ChangeModel::bands` is a
  `std::vector<BandFit>`; `ccd::band_fit()` converts a Lasso fit. Inside the
  Standard machine the fits stay `LassoResult`s. The Python outputs are
  unchanged.

### Tests

- `PrefixStability.*` (`tests/ccd/golden/test_prefix.cpp`): a run on the
  observations up to a date T keeps every final segment of the run on the
  whole series, bit for bit, with the same processing mask below its break
  (both runs with `STAT_ORD` at min(`STAT_ORD`, T)). Final segments are the
  Start segment and every segment up to the last that ended in a change.
  Swept over the characterization and CSV pixels at every split date and
  over the golden synthetic cubes at twelve.
- `Transitions.InputsAreTheSameOnEveryPlatform` pins a fingerprint of each
  characterization pixel's input. `sparse_random` drew two values in one
  call's arguments, whose evaluation order is the compiler's (clang went left
  to right, GCC right to left), so its pixels differed on Linux and
  `Transitions.PathsAsDocumented` failed there. The draws are sequenced in
  clang's order: the Mac's pixels, and every fixture, are unchanged.
- `Bands.ChangeThresholdTakesItsDegreesOfFreedom` compares the chi-squared
  thresholds within 4 ulp: glibc and Apple's libm differ in the last bit.
  Both tests now pass on Linux.
- `test_golden.cpp` states which tests are platform-bound (the three
  fixtures, as same-machine before-and-after checks) and that every other
  test is machine-agnostic.

## 0.7.0 - 2026-10-16

The Standard procedure as an explicit per-pixel state machine, with the
parts a second change-detection procedure needs shared, and a document of
the design (`docs/ccdc-state-machine.md`, linked from the README). C++ API
changes only; the Python API is unchanged, except that a `PEEK_SIZE` below
2 is refused. Outputs of valid inputs are those of 0.6.1, bit for bit.

### Fixed

- `PEEK_SIZE` below 2 is refused by `ccd::validate()` and every entry point
  (`ValueError` in Python: "PEEK_SIZE is 1; it must be at least 2"). It
  crashed `detect()` before: far from the previous break Lookback peeks the
  adjusted peek minus one observations, at least `PEEK_SIZE - 1`, and both
  peeks read their first magnitude. Valid inputs are unchanged.

### Changed (C++ API and internals; outputs bit for bit)

- The Standard procedure keeps its per-pixel state in records instead of
  hand-kept pairs and the options. `ccd::MaskedSeries`
  (`include/ccd/input/masked_series.hpp`) owns the processing mask and the
  observations it keeps, and is the only way to drop one: `drop(position,
  window)` and Initialise's tentative Tmask removal `drop_outliers_if`,
  each moving the caller's window by one rule. `ccd::standard::Cadence`
  holds the adjusted peek and change threshold (`measure_cadence`), and
  `ccd::Variogram` the per-band variogram, so the caller's `PEEK_SIZE` and
  `CHANGE_THRESHOLD` are no longer overwritten in a per-pixel copy.
- `HarmonicWorkspace` refers to the caller's `HarmonicOptions` instead of
  copying them, and `options()` is const only; it cannot be built from a
  temporary. `standard::Procedure::run` takes a const workspace. `tmask()`
  and `stable()` take a `const Variogram&`.
- The procedures share kernels instead of near-copies of each loop.
  `ccd::HarmonicBasis` (`include/ccd/regression/harmonic_basis.hpp`) builds
  the harmonic design with its column order documented; `LassoWorkspace`
  builds its basis through it, and the peek predictions use one directly
  (without the centring only a fit reads). `ccd::fit_bands`
  (`include/ccd/machine/kernels.hpp`) is the one per-band fit loop of
  Initialise, Extend's refit, both catches and the whole-series fits, the
  latter without its redundant workspace reset; `ccd::catch_segment` writes
  the start and end catches; `ccd::initialize_step`, one candidate window
  of the start-window search, takes its `MaskedSeries` and an
  `InitializeCriteria` (Standard passes its 4 stability coefficients there)
  instead of reading a procedure's members, and returns what it did
  (`InitializeOutcome`).
  The Standard procedure's own steps are free functions in
  `ccd::standard`: `lookback`, `extend`, `coefficient_count`,
  `peek_residuals` (both peeks' residual loop) and `change_magnitude`,
  which replaces `ccd::change_magnitude` and the internal
  `detection_magnitude` (the same sum in the same order) and reads the
  `Variogram` directly. The 8-coefficient peek bases stay at Standard's
  call sites.
- The Standard procedure is a per-pixel state machine,
  `ccd::standard::Machine` (`include/ccd/machine/standard.hpp`): a
  `State` record (phase, exit reason, `MaskedSeries`, `Cursor`, segments,
  `Cadence`, `Variogram`, the stable fits, Extend's `Extension`), a
  read-only `Context`, one transition per phase (`prepare`, `initialize`,
  `lookback`, `start_catch`, `extend`, `restart`, `end_catch`) and named
  routing between them (`after_stable_start`, `after_lookback`,
  `peek_guard`). `ccd::drive` (`include/ccd/machine/machine.hpp`) is the
  one loop, for this machine and any later one, and logs one DEBUG line per
  transition. Each transition makes one decision: `initialize` tests one
  candidate window, `lookback` one observation before the window (its peek
  window is `lookback_peek()`), `extend` one peek after it, each repeating
  itself until a guard sends it on, so the trace shows every decision.
  `Extension` carries Extend's state from one peek to the next (training,
  fit window and span, the last peek and coefficient count, the models, the
  peek residuals), its buffers kept across segments and pixels. pyccd's
  dead series-end check before Extend's grow is an assert. pyccd's `start`
  flag is `segments.empty()`. `standard::lookback` and `standard::extend`
  are now these transitions.
- Each repeating step says what it did. `ccd::initialize_step` returns an
  `InitializeOutcome`, which names the step that drops Tmask's outliers and
  then fits the shortened window (`DroppedThenUnstable`,
  `DroppedThenStable`, beside `Unstable` and `Stable` for a step with
  nothing flagged). `standard::lookback_step` and `standard::extend_step`
  make Lookback's and Extend's decision and return a `LookbackOutcome`
  (`Change`, `Dropped`, `Included`) or an `ExtendOutcome` (`Change`,
  `Dropped`, `Grown`); the transitions `lookback` and `extend` route on it,
  `extend` writing the segment. The trace names every outcome.
- Initialize's criteria are built once per pixel, in Prepare
  (`standard::initialize_criteria`, `State::initialize_criteria`), not on
  every candidate window. `InitializeCriteria` holds its band lists by
  value.
- The per-pixel procedure is shared, for any change detector:
  `include/ccd/machine/pixel_procedure.hpp` declares `PixelProcedure`
  (moved from `procedure/fit.hpp`), `ccd::select_procedure` (pyccd's
  clear-then-snow choice; `detect_pixel` calls it) and
  `ccd::fit_whole_series` (the PermanentSnow / InsufficientClear fit, one
  segment of `COEFFICIENT_MIN` coefficients, curve_qa 54 / 44).
  `ccd::standard::Detector` has one call, `fit`: the Standard machine for
  Standard pixels and `fit_whole_series` for the others;
  `DetectScratch::standard` is a `Detector`. `span()` is in
  `procedure/fit.hpp`.
- Renamed (C++): `FitProcedureType` is `PixelProcedure`. `StandardProcedure`
  is `ccd::standard::Procedure`, a plain class owning the machine's record
  and kept apart from `Detector`, so that a second machine has its own
  `Procedure` beside it; its `run(workspace, mask, lasso, solver)` is
  unchanged.
- CMake sets policies CMP0167 (`find_package(Boost)` reads Boost's
  `BoostConfig.cmake`) and CMP0135 where the CMake knows them, which
  silences the FindBoost warning on CMake 3.30 and later.
- Removed (C++): the `FitProcedure` base class with its virtual
  `coefficient_count()` and `curve_qa()` hooks, the `PermanentSnow` and
  `InsufficientClear` classes and their headers
  (`procedure/permanent_snow.hpp`, `procedure/insufficient_clear.hpp`; call
  `fit_whole_series` with the procedure and its filter's mask),
  `StandardProcedure`'s hook overrides and its overload that built the
  standard filter mask itself (pass `filter::standard(workspace)`, as the
  whole-series fit takes its filter's mask), and `standard::Detector::select`
  (now `ccd::select_procedure`).
- Removed, unused: `update_processing_mask` (both overloads), and
  `remove_outliers` / `remove_observation` (now inside `MaskedSeries`), the
  vector-returning `StandardProcedure::calc_residuals`, `Window::rewind`
  and `Window::extend`.

### Tests

- `tests/ccd/golden/test_transitions.cpp` (ctest `Transitions.*`): 17
  characterization pixels (`characterisation.hpp`) built for the Standard
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
- `tests/ccd/machine/test_kernels.cpp` (ctest `HarmonicBasis.*`,
  `FitBands.*`, `PeekResiduals.*`, `ChangeMagnitude.*`, `CatchSegment.*`):
  each shared kernel against the per-site loop it stands for, bit for bit,
  on the masked series of the pyccd reference pixel and the Collection 2
  pixel.
- `tests/ccd/machine/test_machine.cpp` (ctest `StandardRouting.*`,
  `StandardPrepare.*`, `StandardInitialize.*`, `StandardLookback.*`,
  `StandardExtend.*`, `StandardExit.*`, `StandardMachine.*`,
  `StandardDetector.*`): the machine's guards and routing from hand-built
  records; one transition call per row of Initialize's, Lookback's and
  Extend's guard tables, on a short series with the deciding observation
  edited in; its exit reason on the characterization pixels that stop it
  each way (each run also equal to `detect()`'s result); every pixel
  stepped by hand, each self-transition one decision; a record reused
  across pixels. The guard-row tests of Initialize, Lookback and Extend also
  call the step function and check the outcome it reports.
  `SelectProcedure.*`: the procedure choice at both thresholds;
  `StandardDetector.*`: the detector's fit of each procedure.
- `tests/ccd/machine/test_masked_series.cpp` (ctest `MaskedSeries.*`,
  `Variogram.*`, `StandardCadence.*`, `HarmonicWorkspace.*`): the records'
  contracts.
- `tools/bench/`: the refactor's timing harness under version control: the
  synthetic corpus generator, the synthetic bench (`bench.sh`) and the
  real-cell bench (`bench_cell.py`), data paths as arguments, and the method
  in its README. Both benches refuse a `module_dir` that does not supply the
  `ccdcpp` they import (a wrong one timed the interpreter's installed
  package), and `bench.sh` writes no temporary file. `bench.sh` passes the bench's own
  messages (its import check, a traceback) through its filter, and the
  synthetic bench prints the `ccdcpp` file it timed.
- `tools/linux-compare/`: the Linux dump comparison under version control.
  `compare.sh` builds several git refs with conda-forge GCC in one
  throwaway linux/amd64 container (`in_container.sh`) and compares every
  golden-driver dump and the Golden, Bands and Transitions fixtures byte for
  byte; each fixture write must leave the container's platform header.
  `golden_driver.cpp` is the golden check's one driver;
  `-DCCD_BUILD_GOLDEN_DRIVER=ON` builds it as the target `golden_driver`.
- The tests that read CSVs open them under the build directory by absolute
  path (`CCD_TEST_DATA_DIR`), so every test binary passes from any working
  directory, not only under ctest.

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

- `ccd/input/mask/preprocess.hpp`: preprocessing plans. A `PreprocessPlan`
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
