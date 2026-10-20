# Migrating to 1.0

In 1.0 the input says only how to read the data. The configuration says what to do with it, in
physical units. The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check, exactly
as today.

- **`ccdcpp.Input` replaces `HarmonicOptions`** and holds the data's encoding only: the QA
  codes and the table that decodes raw QA (`qa_table`), a role per row (`roles`),
  `reflectance_unit` and `reflectance_offset` (Collection-1 DN 10000 and 0, reflectance 1 and
  0), the thermal transform (`thermal_scale`, `thermal_offset`, with `thermal_unit`, the value
  of 100 °C after it) and `saturated`. It is immutable: `ccdcpp.Input(**fields)` by keyword,
  `h.replace(**fields)` for a changed copy, where `HarmonicOptions` set attributes. Every
  algorithm value of 0.10.0, `LassoOptions`' four, the bounds, the green filter and the statistics cutoff are the
  configuration's ([config.md](config.md)): `shared` values, `stage__parameter` parameters and
  the `preprocess` steps, in physical units.
- **Two calls.** `ccdcpp.detect(dates, spectral, qas, *, input, config="ccdc", ...)` and
  `ccdcpp.detect_cube(..., *, input, config="ccdc", threads=0, ...)`, which returns columns;
  `input=` is an `Input` or a `SensorProfile`, `config=` a `Config` or a preset's name.
- **Results are bit-identical** to 0.10.0 for the same input, configuration and statistics
  cutoff. The `ccdc` configuration has **no statistics cutoff**; 0.10.0's Landsat default was
  2017-12-31: `ccdcpp.config("ccdc").replace(shared__stat_ord=736694)` runs it, bit for bit.
- **0.10.0 resume records do not continue.** The resume digest covers the input and the
  resolved configuration, so a 0.10.0 record is refused with `ValueError` ("the resume record
  was made with another input or configuration") before any pixel runs. Run once on 1.0 with
  `record=True` and store its records; they continue from then on.

## Python

### The entry points

| 0.10.0 | 1.0 |
|---|---|
| `detect(d, s, q, h, lo)` | `detect(d, s, q, input=h, config=cfg)` |
| `detect(..., profile=p)` | `detect(..., input=p)` |
| `detect_cube_columns(d, s, q, h, threads)` | `detect_cube(d, s, q, input=h, threads=threads)` (the same columns) |
| `detect_cube(...)` (a dict per pixel) | `ccdcpp.to_pixels(detect_cube(...))` (the models; the processing mask is `detect()`'s) |
| `resume_record=True` | `record=True` |
| `resume(d, s, q, h, resume=r, resume_header=hd, models=m)` | `detect(d, s, q, input=h, resume=out)`, `out` the earlier `detect(..., record=True)` |
| `resume_cube_columns(..., resume=r, resume_header=hd, segments=cols)` | `detect_cube(..., resume=out)`, `out` the earlier `detect_cube(..., record=True)` |
| `qa_table=t` | the input's `qa_table`: `h.replace(qa_table=t)` (a profile's `input()` carries its table) |
| `preprocess=plan` | the configuration's `preprocess` section (`Config.replace(preprocess__...=)`) |
| `default_plan(h)` | `ccdcpp.plan_of(config, input)` |
| `ccdcpp.LassoOptions`, the `loptions` argument | the shared `alpha`, `coef_floor`, `max_iter`, `tolerance` |

### Profiles

A `SensorProfile` describes the data: `name`, `band_names`, `roles`, `units`
(`Units(reflectance_unit, offset)`), `thermal` (`Thermal(scale, offset, unit)`), `saturated`,
`qa_table_builder` and `qa_table_options`. `profile.input()` is its `ccdcpp.Input`, QA table
included.

| 0.10.0 | 1.0 |
|---|---|
| `profile.options()` | `profile.input()` |
| `profile.qa_table(h)` | `profile.input().qa_table` |
| `detection`, `tmask` | `shared__detection_bands`, `window_screen__bands` (roles; "ccdc"'s are pyccd's) |
| `alpha`, `coef_floor` | `shared__alpha`, `shared__coef_floor` in reflectance (DN / 10000) |
| `cadence` | `stage__parameter` changes of the configuration |
| `stat_ord` | `shared__stat_ord` (`None`: no cutoff, the default) |
| `green_filter_scope`, `Units.median_green_filter` | `green_median`'s `scope` and `offset` (reflectance) |
| `Units.refl_min/refl_max/refl_inclusive`, `Thermal.min/max/inclusive` | `band_range`'s and `thermal_range`'s bounds and flags |
| `preprocess=` | the configuration's `preprocess` section |

### Examples

```python
# 0.10.0
out = ccdcpp.detect(dates, dn, codes, ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions())
# 1.0
out = ccdcpp.detect(dates, dn, codes, input=ccdcpp.Input())
```

```python
# 0.10.0: physical units
lo = ccdcpp.LassoOptions()
lo.alpha, lo.coef_floor = 1e-4, 1e-4
out = ccdcpp.detect_cube_columns(dates, cube, codes, h_physical, lo, threads=4)
# 1.0: the configuration is the same for every input; the input says the unit
h = ccdcpp.Input(reflectance_unit=1.0, thermal_scale=1.0, thermal_offset=0.0, thermal_unit=1.0)
out = ccdcpp.detect_cube(dates, cube, codes, input=h, threads=4)   # or input=presets.landsat_c1_physical()
```

```python
# 0.10.0
h, lo = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
h.CHANGE_PROBABILITY = 0.995
h.CHANGE_THRESHOLD = ccdcpp.chi2_quantile(0.995, 5)
h.RETRAIN_SPAN_FACTOR = 2.0
h.MEOW_SIZE = 16
h.TMASK_BANDS = [1, 2, 4]
lo.max_iter = 500
out = ccdcpp.detect(dates, dn, codes, h, lo)
# 1.0
cfg = ccdcpp.config("ccdc").replace(
    shared__stat_ord=736694,                 # 0.10.0's cutoff
    shared__change_probability=0.995,        # every change test, its thresholds derived
    refit__factor=2.0,
    shared__min_observations=16,             # the start window and a whole-series curve's least
    window_screen__bands=["green", "red", "swir1"],
    shared__max_iter=500)
out = ccdcpp.detect(dates, dn, codes, input=ccdcpp.Input(), config=cfg)
```

## The options' fields and what they became

Where one 0.10.0 field set several values, set each of them to keep the 0.10.0 run. A threshold
is now the chi-squared quantile of a probability for the detection-band count: set the
probability.

| 0.10.0 field | Default | 1.0 |
|---|---|---|
| `QA_FILL` ... `QA_OCCLUSION` | | `Input.qa_fill` ... `Input.qa_occlusion` |
| `THERMAL_SCALE`, `THERMAL_OFFSET` | 10, -27315 | `Input.thermal_scale`, `Input.thermal_offset` |
| `NUM_BANDS`, `*_IDX` | | `Input.roles` (one role per row) |
| `SATURATION_BANDS` | [0..5] | `band_range`'s `bands`: `"reflective"` (every row but thermal and other) or roles |
| `CLEAR_OBSERVATION_THRESHOLD` | | none (nothing read it) |
| `REFL_MIN`, `REFL_MAX` and their flags | 0, 10000 | `band_range`'s `min`, `max` (reflectance: 0, 1) and flags |
| `THERM_MIN`, `THERM_MAX` and their flags | -9320, 7070 | `thermal_range`'s `min`, `max` (degrees Celsius: -93.2, 70.7) and flags |
| `MEDIAN_GREEN_FILTER` | 400 | `green_median`'s `offset` (reflectance: 0.04) |
| `GREEN_FILTER_SCOPE` | StatWindow | `green_median`'s `scope` ("stat_window") |
| `STAT_ORD` | 736694 | `shared__stat_ord` (default: no cutoff) |
| `DETECTION_BANDS` | [1..5] | `shared__detection_bands` (roles) |
| `MEOW_SIZE` | 12 | `shared__min_observations` (read as `start_search`'s `meow_size` and `edge_fit`'s `min_observations`) |
| `PEEK_SIZE` | 6 | `change_test__peek_size` (`stability_rule__peek` and `lookback__peek` are 0, the change test's peek) |
| `DAY_DELTA` | 365 | `start_search__day_delta` |
| `CLEAR_PCT_THRESHOLD` | 0.25 | `mode__clear_pct_threshold` |
| `SNOW_PCT_THRESHOLD` | 0.75 | `mode__snow_pct_threshold` |
| `CHANGE_PROBABILITY`, `CHANGE_THRESHOLD` | 0.99, 15.0863 | `shared__change_probability` |
| `OUTLIER_THRESHOLD` | 35.8882 | `shared__outlier_probability` (0.999999) |
| `T_CONST` | 4.89 | `window_screen__t_const` |
| `TMASK_BANDS` | [1, 4] | `window_screen__bands` (roles) |
| `PEEK_REFERENCE_DAYS` | 16.0 | `change_test__peek_reference_days` |
| `VARIOGRAM_MIN_LAG_DAYS` | 30 | `noise_floor__min_lag_days` |
| `RETRAIN_MIN_OBS` | 24 | `refit__min_obs`, `change_test__model_rmse_up_to` |
| `RETRAIN_SPAN_FACTOR` | 1.33 | `refit__factor` |
| `SEASONAL_RMSE_COUNT` | 24 | `change_test__seasonal_count` |
| `SEASONAL_RMSE_DIVISOR` | 4.0 | `change_test__seasonal_divisor` |
| `COEFFICIENT_MIN`, `_MID`, `_MAX`, `NUM_OBS_FACTOR` | 4, 6, 8, 3 | `shared__coefficient_min`, `_mid`, `_max`, `shared__num_obs_factor` |
| `LassoOptions.alpha`, `coef_floor` | 1.0, 1.0 (DN) | `shared__alpha`, `shared__coef_floor` (reflectance: 1e-4) |
| `LassoOptions.max_iter`, `tolerance` | 1000, 1e-4 | `shared__max_iter`, `shared__tolerance` |

## C++

| 0.10.0 | 1.0 |
|---|---|
| `HarmonicOptions` | `ccd::Input` (the encoding) and `config::Config` (the algorithm) |
| `detect(d, s, q, h, l, table, plan)` | `run_pixel(spec, d, s, q).result`, `spec.input` holding its `qa_table` and the configuration its preprocessing |
| `detect_recorded(...)` | `run_pixel` with `RunSpec::record = true` |
| `resume(d, s, q, h, l, header, record, segments)` | `run_pixel(spec, d, s, q, PixelResume{header, record, segments})` |
| `detect_cube(d, s, q, h, l, threads, keep_masks, table, plan, record)` | `run_cube(spec, d, s, q, threads)` |
| `resume_cube(..., from, ...)` | `run_cube(..., threads, &from)` |
| `Recorded`, `Resumed`; `DetectScratch` | `PixelRun`; `Worker` |
| `LassoOptions` in a run | the shared Lasso values (`LassoOptions` remains the solver's own options type) |
| `default_plan(h)` | `plan_of(config, input)` (`run/algorithm.hpp`) |
| `standard::coefficient_count()` | the model strategy's `coefficients()`, a `CoefficientRule` (`include/ccd/strategies/coefficients.hpp`) |
| `ccd/types.hpp`, `ccd/procedure/standard.hpp`, `ccd/record.hpp`, ... | `ccd/core/types.hpp`, `ccd/machine/standard.hpp`, `ccd/results/record.hpp`, ... (`ccd/ccd.hpp` includes every header) |

`RunSpec` is `{input, algorithm, record, keep_masks, observer, observe}`; `algorithm` is a
compiled configuration (`ccd::Algorithm`, `run/algorithm.hpp`), the `ccdc` preset's
(`Algorithm::preset("ccdc")`) unless set.

```cpp
// 0.10.0
ccd::HarmonicOptions h;
ccd::LassoOptions l;
h.CHANGE_PROBABILITY = 0.995;
h.CHANGE_THRESHOLD = 16.7496;
ccd::FitResult r = ccd::detect(dates, spectral, qas, h, l);

// 1.0
ccd::config::Config config = ccd::strategies::library_registry().preset("ccdc");
config.shared["stat_ord"] = ccd::config::integer(736694);          // 0.10.0's cutoff
config.shared["change_probability"] = ccd::config::real(0.995);
ccd::RunSpec spec;                                                   // spec.input: Collection-1 DN
spec.algorithm = std::make_shared<const ccd::Algorithm>(config);
ccd::FitResult r = ccd::run_pixel(spec, dates, spectral, qas).result;
```

### Messages

`validate()` checks the input. The configuration is checked by its schemas when it is compiled
(`ccd::Algorithm`, `ccdcpp.Config`), by stage, strategy, step and parameter, with
`std::invalid_argument` (`ValueError` in Python) before any pixel runs:

| 0.10.0 | 1.0 |
|---|---|
| `PEEK_SIZE is 1; it must be at least 2` | `change_test 'chi2' parameter 'peek_size' is 1; it must be at least 2` |
| `COEFFICIENT_MAX is 9; the harmonic basis has at most 8` | the strategy reading the shared value refuses it: `model 'lasso' parameter 'coefficient_max' is 9; the harmonic basis has at most 8` |
| `TMASK_BANDS is empty` | `window_screen 'tmask' parameter 'bands' is empty` |

A `MEOW_SIZE` of 0, a NaN change or outlier threshold and negative or NaN Lasso constants ran
in 0.10.0; 1.0 refuses them (`shared parameter 'min_observations' is 0; it must be at least
1`).
