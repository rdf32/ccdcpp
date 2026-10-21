# Migrating from 0.10.0 to 1.0

1.0 separates what the data is from what to do with it:

| | Example |
|---|---|
| The input says only how to read the data. | `HarmonicOptions` is gone; `ccdcpp.Input()` says only how to read the data: `reflectance_scale` 10000, the thermal transform (10, -27315), the QA codes. |
| The configuration says what to do with it, in physical units. | `HarmonicOptions.REFL_MAX = 10000` is `band_range`'s `max = 1.0` (reflectance) in `ccdcpp.config("ccdc")`. |
| The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check, exactly as today. | `ccdcpp.plan_of("ccdc", ccdcpp.Input())` tests `0 < x < 10000`, 0.10.0's `REFL_MIN < x < REFL_MAX`. |

What to do:

1. Pass how to read the data as `input=` (a `ccdcpp.Input` or a sensor profile) and the
   algorithm as `config=` (a `ccdcpp.Config` or a preset's name); `hoptions`, `loptions`,
   `profile=`, `qa_table=` and `preprocess=` are gone.
2. Move every algorithm field you set on `HarmonicOptions` or `LassoOptions` to the
   configuration ([the table](#the-options-fields-and-what-they-became)).
3. Decide on the statistics cutoff: 1.0's `ccdc` has none; 0.10.0's and pyccd's is
   `ccdcpp.config("ccdc").replace(shared__statistics_end=736694)`
   ([outputs](#what-changes-in-the-outputs)).
4. Rerun once with `record=True`: 0.10.0 resume records are refused.

Everything else gives 0.10.0's outputs bit for bit. Measured on 1,500 real Landsat Collection-1
DN pixels (2,473 dates each) with `statistics_end=736694`: every segment, coefficient, intercept,
RMSE, magnitude and processing mask is bit-identical between 0.10.0 and 1.0 through the default
input, `presets.landsat_c1_dn()`, `presets.landsat_c1_physical()` and an explicit physical-unit
input, and with changed options (the [example below](#harmonicoptions-and-lassooptions)); a
3 x 4 cube's columns are equal to 0.10.0's `detect_cube_columns`, and `to_pixels` of them to
0.10.0's `detect_cube`.

The examples use `dates`, `dn` (7 rows of Collection-1 DN) and `codes` (uint8 QA codes) of one
such pixel, and `cube`, `cube_codes` of twelve: (3, 4, 7, T) and (3, 4, T). A block marked
`# 0.10.0` runs on 0.10.0, the block after it on 1.0.

## What changes in the outputs

Two changes of default outputs are deliberate.

**(a) No statistics cutoff by default.** 0.10.0, like pyccd, computed the QA statistics (the
procedure choice), the cadence (the peek size and its threshold), the variogram and the green
median from the observations up to 2017-12-31 (`STAT_ORD` 736694). That date reproduced LCMAP's
2018 production baseline; it is not part of the algorithm, and for a series that runs past 2017
it ignores the years since. 1.0's `ccdc` uses the whole series (`shared.statistics_end` is
`NO_STATISTICS_END`). pyccd and LCMAP parity is
`ccdcpp.config("ccdc").replace(shared__statistics_end=736694)`, bit for bit as 0.10.0. On the
Atlanta cell (Landsat Collection 2 ARD h024v013, 1984-2023, 250,000 pixels) the change makes
57.1% of the pixels differ and changes the segment count of 6.5% of them (32,751 breaks new,
30,370 gone); on the 1,500 Collection-1 DN pixels above, 55.6% and 6.1%. A resumed run still
freezes its statistics at the first run's series end ([README](../README.md#resuming-a-run)).

```python
# 0.10.0
out = ccdcpp.detect(dates, dn, codes, ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions())
print([m["break_day"] for m in out["models"]], out["models"][0]["bands"][3]["rmse"])
```

```text
[738751] 435.1414437544068
```

```python
pyccd = ccdcpp.config("ccdc").replace(shared__statistics_end=736694)
same = ccdcpp.detect(dates, dn, codes, input=ccdcpp.Input(), config=pyccd)
print([m["break_day"] for m in same["models"]], same["models"][0]["bands"][3]["rmse"])

whole = ccdcpp.detect(dates, dn, codes, input=ccdcpp.Input())    # "ccdc": the whole series
print([m["break_day"] for m in whole["models"]])
```

```text
[738751] 435.1414437544068
[738095, 738751]
```

**(b) Every procedure's thermal check compares in degrees Celsius and covers every observation
its fit uses.** pyccd's PermanentSnow and InsufficientClear filters, and 0.10.0's default plan with
them, test the thermal band against the Celsius bounds without converting it. On Collection-1
DN that compares Kelvin x 10 with -9320 to 7070: it drops saturated (20000) and NaN values and
keeps every temperature below 707 K; on any other input it is a test of the wrong unit. `ccdc`
tests those two chains' stored thermal band against the bounds taken back through the input's
thermal transform, 1799.5 to 3438.5 on Collection-1 DN (179.95 to 343.85 K), so every
procedure keeps -93.2 to 70.7 °C, the values whose transform Standard's range keeps. The snow
chain tests the range on its clear and water observations, as pyccd does, and again after it
adds the snow observations back (pyccd tests only the clear and water ones), so a saturated or
NaN thermal value no longer reaches a snow fit. On
Collection-1 DN it drops, besides 20000 and NaN as before, the implausible temperatures 0.10.0
kept in those fits: none on the 1,500 reference Collection-1 pixels; 218 of the synthetic
corpus's 2,900 pixels change (snow and insufficient-clear pixels, and 8 without a segment),
whose thermal values range far beyond the bounds. Physical-unit outputs are 0.10.0's, and the
DN run segments as the physical-unit run does. On the Atlanta cell (Collection 2) no pixel changes.

## Resume records

A 0.10.0 record does not continue: its header has `options_digest` where 1.0's has `digest`, and
1.0's digest covers the input (its QA table included) and the resolved configuration (its
preprocessing included), so `RESUME_FORMAT` is 2 (0.10.0's was 1); 1.0's default outputs
differ, so `RESULTS_VERSION` is 3 (0.10.0's was 2). A 0.10.0 record is refused before any pixel
runs, with `ValueError: the resume record's format is 1; this ccdcpp reads 2`. Rerun the
detection once with `record=True` and store its records; they continue from then on.

```python
# 0.10.0
first = ccdcpp.detect(dates[:2400], dn[:, :2400], codes[:2400],
                      ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions(), resume_record=True)
more = ccdcpp.resume(dates, dn, codes, ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions(),
                     resume=first["resume"], resume_header=first["resume_header"],
                     models=first["models"])
print(more["status"], sorted(first["resume_header"]))
```

```text
resumed ['format', 'options_digest', 'results_version', 'series_end', 'statistics_end']
```

```python
first = ccdcpp.detect(dates[:2400], dn[:, :2400], codes[:2400], input=ccdcpp.Input(),
                      config=pyccd, record=True)
more = ccdcpp.detect(dates, dn, codes, input=ccdcpp.Input(), config=pyccd, resume=first)
print(more["status"], sorted(first["resume_header"]))
```

```text
resumed ['digest', 'format', 'results_version', 'series_end', 'statistics_end']
```

## Python

### The entry points

| 0.10.0 | 1.0 |
|---|---|
| `detect(d, s, q, h, lo)` | `detect(d, s, q, input=h, config=cfg)` |
| `detect(..., profile=p)` | `detect(..., input=p)` |
| `detect_cube_columns(d, s, q, h, lo, threads)` | `detect_cube(d, s, q, input=h, threads=threads)`: the same columns, names and dtypes |
| `detect_cube(d, s, q, h, lo, threads)` (a dict per pixel) | `ccdcpp.to_pixels(detect_cube(..., masks=True))` |
| `keep_masks` (the dicts held each pixel's mask) | `detect_cube(..., masks=True)`: `"processing_mask"`, `(H, W, T)` uint8; `to_pixels` gives each pixel its row |
| `resume_record=True` | `record=True` |
| `resume(d, s, q, h, lo, resume=r, resume_header=hd, models=m)` | `detect(d, s, q, input=h, resume=out)`, `out` the earlier `detect(..., record=True)` |
| `resume_cube_columns(..., resume=r, resume_header=hd, segments=cols)` | `detect_cube(..., resume=out)`, `out` the earlier `detect_cube(..., record=True)` |
| `qa_table=t` | the input's `qa_table`: `h.replace(qa_table=t)` (a profile's `input()` carries its table) |
| `preprocess=plan` | the configuration's `preprocess` section ([below](#preprocessing)) |
| `ccdcpp.default_plan(h)` | `ccdcpp.plan_of(config, input)` |
| `ccdcpp.validate_plan(plan, bands)` | the configuration is checked when it is built and resolved |
| `resume_header["options_digest"]` | `resume_header["digest"]` |

The calls take everything but the arrays by keyword. A positional `HarmonicOptions` is a
`TypeError`.

```python
# 0.10.0
cols = ccdcpp.detect_cube_columns(dates, cube, cube_codes, ccdcpp.HarmonicOptions(),
                                  ccdcpp.LassoOptions(), threads=4)
rows = ccdcpp.detect_cube(dates, cube, cube_codes, ccdcpp.HarmonicOptions(),
                          ccdcpp.LassoOptions(), threads=4)
print(len(cols["row"]), cols["coefficients"].dtype, rows[5]["models"][0]["bands"][3]["rmse"])
```

```text
19 float64 316.68166070440054
```

```python
cols = ccdcpp.detect_cube(dates, cube, cube_codes, input=ccdcpp.Input(), config=pyccd,
                          threads=4, masks=True)
rows = ccdcpp.to_pixels(cols)
print(len(cols["row"]), cols["coefficients"].dtype, rows[5]["models"][0]["bands"][3]["rmse"])
print(cols["processing_mask"].shape, rows[5]["processing_mask"].sum())
```

```text
19 float64 316.68166070440054
(3, 4, 2473) 798
```

### `HarmonicOptions` and `LassoOptions`

`ccdcpp.HarmonicOptions` is `ccdcpp.Input`, which keeps only how to read the data, its fields
in snake case; every algorithm field moved to the configuration, and so did `LassoOptions`'
`alpha`, `coef_floor`, `max_iter` and `tolerance`. `LassoOptions`' `fit_intercept`,
`unbiased_rmse` and `warm_start` are fixed at their defaults (true, true, false). The `Input`
is immutable: `ccdcpp.Input(**fields)` by keyword, `replace(**fields)` for a changed copy, where
`HarmonicOptions` set attributes.

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
print(len(out["models"]), out["models"][-1]["bands"][3]["rmse"])
```

```text
2 319.50556559382443
```

```python
cfg = ccdcpp.config("ccdc").replace(
    shared__statistics_end=736694,           # 0.10.0's cutoff
    shared__change_probability=0.995,        # every test; the thresholds are derived from it
    refit__factor=2.0,
    shared__min_observations=16,             # the start window and a whole-series curve's least
    window_screen__bands=["green", "red", "swir1"],
    shared__max_iter=500)
out = ccdcpp.detect(dates, dn, codes, input=ccdcpp.Input(), config=cfg)
print(len(out["models"]), out["models"][-1]["bands"][3]["rmse"])
```

```text
2 319.50556559382443
```

A physical-unit input set the bounds, the green filter and the Lasso constants itself; in 1.0
the configuration is the same for every input, and the input says the unit:

```python
# 0.10.0
h = ccdcpp.HarmonicOptions()
h.THERMAL_SCALE, h.THERMAL_OFFSET = 1.0, 0.0
h.REFL_MIN, h.REFL_MAX, h.MEDIAN_GREEN_FILTER = 0.0, 1.0, 0.04
h.THERM_MIN, h.THERM_MAX = -0.932, 0.707
lo = ccdcpp.LassoOptions()
lo.alpha, lo.coef_floor = 1e-4, 1e-4
physical = dn / 10000.0
physical[6] = (dn[6] * 10 - 27315) / 10000.0
out = ccdcpp.detect(dates, physical, codes, h, lo)
print(out["models"][0]["bands"][3]["rmse"])
```

```text
0.04351414437544069
```

```python
h = ccdcpp.Input(reflectance_scale=1.0, thermal_scale=1.0, thermal_offset=0.0, thermal_at_100c=1.0)
physical = dn / 10000.0
physical[6] = (dn[6] * 10 - 27315) / 10000.0
out = ccdcpp.detect(dates, physical, codes, input=h, config=pyccd)
print(out["models"][0]["bands"][3]["rmse"], h == presets.landsat_c1_physical().input())
```

```text
0.04351414437544069 True
```

### The options' fields and what they became

Where one 0.10.0 field set several values, 1.0 holds it once. A threshold is now the chi-squared
quantile of a probability for the detection-band count: set the probability. A `stage__param`
is a parameter of that stage's strategy, `shared__name` a shared value (read by every stage
that needs it, set nowhere else), and `preprocess` a step of a procedure's chain
([config.md](config.md)).

| 0.10.0 field | 0.10.0 default | 1.0 |
|---|---|---|
| `QA_FILL`, `QA_CLEAR`, `QA_WATER`, `QA_SHADOW`, `QA_SNOW`, `QA_CLOUD`, `QA_CIRRUS1`, `QA_CIRRUS2`, `QA_OCCLUSION` | 255, 0, 1, 2, 3, 4, 8, 9, 10 | `Input.qa_fill` ... `Input.qa_occlusion` |
| `THERMAL_SCALE`, `THERMAL_OFFSET` | 10, -27315 | `Input.thermal_scale`, `Input.thermal_offset`, with `Input.thermal_at_100c` (10000 for this transform) |
| `NUM_BANDS`, `BLUE_IDX` ... `THERMAL_IDX` | 7, 0 ... 6 | `Input.roles`, one role per row (`Input.bands()`, `Input.index(role)`) |
| `QA_IDX` | | none (nothing read it) |
| `CLEAR_OBSERVATION_THRESHOLD` | | none (nothing read it) |
| `SATURATION_BANDS` | [0, 1, 2, 3, 4, 5] | `preprocess` `band_range`'s `bands`: `"reflective"` (every row but thermal and other) or roles |
| `REFL_MIN`, `REFL_MAX`, `REFL_MIN_INCLUSIVE`, `REFL_MAX_INCLUSIVE` | 0, 10000, false, false | `preprocess` `band_range`'s `min`, `max` (reflectance: 0, 1) and `min_inclusive`, `max_inclusive` |
| `THERM_MIN`, `THERM_MAX`, `THERM_MIN_INCLUSIVE`, `THERM_MAX_INCLUSIVE` | -9320, 7070, false, false | `preprocess` `thermal_range`'s `min`, `max` (degrees Celsius: -93.2, 70.7) and flags |
| `MEDIAN_GREEN_FILTER` | 400 | `preprocess` `green_median`'s `offset` (reflectance: 0.04) |
| `GREEN_FILTER_SCOPE` | `StatWindow` | `preprocess` `green_median`'s `scope` (`"stat_window"`) |
| `STAT_ORD` | 736694 | `shared__statistics_end` (default: none, `NO_STATISTICS_END`) |
| `DETECTION_BANDS` | [1, 2, 3, 4, 5] | `shared__detection_bands` (roles: green, red, nir, swir1, swir2) |
| `MEOW_SIZE` | 12 | `shared__min_observations` (read as `start_search`'s `meow_size` and `edge_fit`'s `min_observations`) |
| `PEEK_SIZE` | 6 | `change_test__peek_size` (`stability_rule__peek` and `lookback__peek` are 0: the change test's peek) |
| `DAY_DELTA` | 365 | `start_search__day_delta` |
| `CLEAR_PCT_THRESHOLD` | 0.25 | `mode__clear_pct_threshold` |
| `SNOW_PCT_THRESHOLD` | 0.75 | `mode__snow_pct_threshold` |
| `CHANGE_PROBABILITY`, `CHANGE_THRESHOLD` | 0.99, 15.0863 | `shared__change_probability`: `change_test`, `stability_rule` and `lookback` derive their thresholds from it |
| `OUTLIER_THRESHOLD` | 35.8882 | `shared__outlier_probability` (0.999999): `outlier` and `lookback` derive theirs |
| `T_CONST` | 4.89 | `window_screen__t_const` |
| `TMASK_BANDS` | [1, 4] | `window_screen__bands` (roles: green, swir1) |
| `PEEK_REFERENCE_DAYS` | 16.0 | `change_test__peek_reference_days` |
| `VARIOGRAM_MIN_LAG_DAYS` | 30 | `noise_floor__min_lag_days` |
| `RETRAIN_MIN_OBS` | 24 | `refit__min_obs` and `change_test__model_rmse_up_to` (two parameters, set alike) |
| `RETRAIN_SPAN_FACTOR` | 1.33 | `refit__factor` |
| `SEASONAL_RMSE_COUNT` | 24 | `change_test__seasonal_count` |
| `SEASONAL_RMSE_DIVISOR` | 4.0 | `change_test__seasonal_divisor` |
| `COEFFICIENT_MIN`, `COEFFICIENT_MID`, `COEFFICIENT_MAX`, `NUM_OBS_FACTOR` | 4, 6, 8, 3 | `shared__coefficient_min`, `shared__coefficient_mid`, `shared__coefficient_max`, `shared__num_obs_factor` |
| `LassoOptions.alpha`, `LassoOptions.coef_floor` | 1.0, 1.0 (DN) | `shared__alpha`, `shared__coef_floor` (reflectance: 1e-4) |
| `LassoOptions.max_iter`, `LassoOptions.tolerance` | 1000, 1e-4 | `shared__max_iter`, `shared__tolerance` |
| `LassoOptions.fit_intercept`, `unbiased_rmse`, `warm_start` | true, true, false | fixed |

A stage cannot set a value it reads from `shared`: `replace(start_search__meow_size=16)` is
refused, naming `shared__min_observations`.

### Profiles

A `SensorProfile` describes the data only: `name`, `band_names`, `roles`, `units`
(`Units(scale, offset)`), `thermal` (`Thermal(scale, offset, at_100c)`), `saturation_code`,
`thermal_saturation_code`, `qa_table_builder` and `qa_table_options`. The algorithm fields are
the configuration's, the same for every profile.

| 0.10.0 | 1.0 |
|---|---|
| `profile.options()` (`(HarmonicOptions, LassoOptions)`) | `profile.input()` (the `ccdcpp.Input`, its QA table included) |
| `profile.qa_table(h)` | `profile.input().qa_table` |
| `detection`, `tmask` | `shared__detection_bands`, `window_screen__bands` |
| `saturation` | `band_range`'s `bands` (`"reflective"` in `ccdc`, the rows the profiles listed) |
| `alpha`, `coef_floor` | `shared__alpha`, `shared__coef_floor`, in reflectance for every profile |
| `cadence` | `stage__parameter` changes of the configuration |
| `stat_ord` | `shared__statistics_end` (none by default; 0.10.0's Landsat profiles had 736694) |
| `change_threshold`, `outlier_threshold` | `shared__change_probability`, `shared__outlier_probability` |
| `green_filter_scope` | `green_median`'s `scope` |
| `Units(refl_min, refl_max, median_green_filter, refl_inclusive)` | `Units(scale, offset)`; the bounds and offset are `band_range`'s and `green_median`'s, in reflectance |
| `Thermal(scale, offset, min, max, inclusive)` | `Thermal(scale, offset, at_100c)`; the bounds are `thermal_range`'s, in degrees Celsius |
| `preprocess` | the configuration's `preprocess` section |
| `CELSIUS_HUNDREDTHS` | `CELSIUS_OVER_100` (thermal in degrees Celsius / 100) |

```python
# 0.10.0
out = ccdcpp.detect(dates, dn, codes, profile=presets.landsat_c1_dn())
h, lo = presets.landsat_c1_dn().options()
print(out["band_names"], h.STAT_ORD, lo.alpha)
```

```text
('blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal') 736694 1.0
```

```python
out = ccdcpp.detect(dates, dn, codes, input=presets.landsat_c1_dn(), config=pyccd)
h = presets.landsat_c1_dn().input()
print(out["band_names"], h.saturation_code, pyccd.resolve(h)["shared"]["alpha"])
```

```text
('blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal') 20000.0 1.0
```

### Preprocessing

The chain is part of the configuration. `preprocess=`, the profile's `preprocess`, `Transform`,
`DEFAULT_TRANSFORM`, `default_plan()` and `validate_plan()` are gone; `Preprocess`,
`ProcedureChain` and the step classes write chains for `Config.replace()` and TOML:

| 0.10.0 | 1.0 |
|---|---|
| `detect(..., preprocess=plan)` | `config=cfg.replace(preprocess__<procedure>=ProcedureChain(...))`, or `preprocess__<procedure>__<step>__<param>=` |
| `ProcedureChain(steps, transform="default")` / `transform=None` | `ProcedureChain(steps, thermal_transform=True)` / `False` (the input's transform) |
| `Transform(role, scale, offset)` | the input's `thermal_scale`, `thermal_offset` |
| `BandRange(roles=...)` | `BandRange(bands=...)` (roles or `"reflective"`) |
| `GreenMedian(offset, scope, max_date, role)` | `GreenMedian(offset, scope)`; its date is the shared `statistics_end`, its band the green role |
| the bounds in the input's units | the bounds in physical units: reflectance, degrees Celsius |
| `default_plan(h)` | `plan_of("ccdc", h)` |

```python
# 0.10.0
plan = ccdcpp.default_plan(ccdcpp.HarmonicOptions())
print(plan.standard.chain[2], plan.snow.chain[1])
```

```text
<Step BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000> <Step ThermalRange band 6, -9320 < x < 7070>
```

```python
plan = ccdcpp.plan_of(pyccd, ccdcpp.Input())
print(plan.standard.chain[2], plan.snow.chain[1])
```

```text
<Step BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000> <Step ThermalRange band 6, 1799.5 < x < 3438.5>
```

The snow chain tests the stored thermal band against the same degrees Celsius as Standard's
transformed one, and tests it again as its last step, after the snow observations are added
back ([change (b)](#what-changes-in-the-outputs)).

### Other names

| 0.10.0 | 1.0 |
|---|---|
| `ccdcpp.NO_STAT_CUTOFF` | `ccdcpp.NO_STATISTICS_END` |
| `ccdcpp.CADENCE_FIELDS` | none: the cadence is the configuration's parameters |
| `ccdcpp.OUTLIER_PROBABILITY` | `ccdcpp.config("ccdc").shared()["outlier_probability"]` |
| `ccdcpp.QA_CODE_FIELDS` | `ccdcpp.QA_CODE_NAMES` (the names a configuration's `codes` take) |
| `ccdcpp.GREEN_FILTER_SCOPES` | the names `green_median`'s `scope` takes: `"all"`, `"stat_window"`, `"off"` |
| `qa_tables.<builder>(options)` | `qa_tables.<builder>(input)`, an `Input` |

## Newly refused values

0.10.0 ran these; 1.0 refuses them with `ValueError` (`std::invalid_argument` in C++) when the
configuration is built, before any pixel runs:

| 0.10.0 | 1.0 |
|---|---|
| `NUM_OBS_FACTOR = 0` | `shared parameter 'num_obs_factor' is 0; it must be at least 1` |
| `MEOW_SIZE = 0` | `shared parameter 'min_observations' is 0; it must be at least 1` |
| `LassoOptions.tolerance = -1e-4` | `shared parameter 'tolerance' is -0.0001; it must be at least 0` |
| `LassoOptions.alpha = nan` | `shared parameter 'alpha' is nan; it must be at least 0` |
| `LassoOptions.alpha = -1`, `coef_floor = -1` | `shared parameter 'alpha' is -1; it must be at least 0` (and `'coef_floor'`) |
| `CHANGE_THRESHOLD = nan`, `OUTLIER_THRESHOLD = nan` | thresholds are derived; a NaN probability: `shared parameter 'change_probability' is nan; it must be in (0, 1)` |

## Error messages

0.10.0 named a field; 1.0 names the stage, the strategy and the parameter, or `shared` for a
shared value outside its own bounds. Which stage is named depends on which parameters are set:
a shared value's own bounds are checked as `shared`, a rule over several values by the first
stage that reads them, and a role against the input when the configuration is resolved.

| 0.10.0 | 1.0 |
|---|---|
| `PEEK_SIZE is 1; it must be at least 2` | `change_test 'chi2' parameter 'peek_size' is 1; it must be at least 2` |
| `COEFFICIENT_MAX is 9; the harmonic basis has at most 8` | `model 'lasso' parameter 'coefficient_max' is 9; the harmonic basis has at most 8` |
| `COEFFICIENT_MIN` above `COEFFICIENT_MID` | `model 'lasso' parameters 'coefficient_min', 'coefficient_mid', 'coefficient_max' are 7, 6, 8; they must not decrease` |
| `TMASK_BANDS is empty` | `window_screen 'tmask' parameter 'bands' is empty` |
| a band past the input in `TMASK_BANDS` | `window_screen 'tmask' parameter 'bands' holds band 12; the input has 7 bands` |
| a detection band the input lacks | `shared value 'detection_bands' holds 'coastal', which the input does not have; its rows are blue, green, red, nir, swir1, swir2, thermal` |
| (none: one field held it) | `start_search 'pyccd' parameter 'meow_size' is the shared value 'min_observations'; set shared__min_observations instead` |

```python
for bad in (dict(change_test__peek_size=1), dict(shared__coefficient_max=9),
            dict(window_screen__bands=[])):
    try:
        ccdcpp.config("ccdc").replace(**bad)
    except ValueError as e:
        print(e)
```

```text
change_test 'chi2' parameter 'peek_size' is 1; it must be at least 2
model 'lasso' parameter 'coefficient_max' is 9; the harmonic basis has at most 8
window_screen 'tmask' parameter 'bands' is empty
```

## C++

| 0.10.0 | 1.0 |
|---|---|
| `HarmonicOptions` (`ccd/harmonic/harmonic.hpp`) | `ccd::Input` (`ccd/input/input.hpp`, how to read the data) and `config::Config` (`ccd/config/registry.hpp`, the algorithm) |
| `LassoOptions` in a call | the shared `alpha`, `coef_floor`, `max_iter`, `tolerance` (`LassoOptions` remains the solver's own options type, inside the `lasso` strategies) |
| `detect(d, s, q, h, l, table, plan)` (8 overloads) | `detect(spec, d, s, q).result`; `spec.input` holds its `qa_table`, the configuration its preprocessing |
| `detect_recorded(...)` | `detect` with `RunSpec::record = true` |
| `resume(d, s, q, h, l, header, record, segments)` | `detect(spec, d, s, q, Resume{header, {record}, {segments}})` |
| `detect_cube(d, s, q, h, l, threads, keep_masks, table, plan, record)` | `detect_cube(spec, d, s, q, threads)` |
| `keep_masks = true` (the default) | `RunSpec::masks = false` (the default): ask for the masks; they are `CubeOutput::masks`, one (H, W, T) array, `mask(row, col)` per pixel, and `detect`'s `result.mask` |
| `resume_cube(..., from, ...)`, `CubeResumeFrom{header, pixels, segments}` | `detect_cube(spec, d, s, q, threads, &from)`, `Resume{header, records, segments}` |
| `Recorded`, `Resumed`; `CubeResult`; `PixelResult`; `DetectScratch` | `PixelOutput`; `CubeOutput`; `CubePixel` (one pixel of a `CubeOutput`); none (the run's per-thread buffers are internal) |
| `options_digest()`, `ResumeHeader::options_digest` | `ResumeHeader::digest`, the resolution's |
| `default_plan(h)`, `PreprocessPlanPtr`, the `preprocess` argument | `plan_of(config, input)` (`ccd/run/algorithm.hpp`), the plan the configuration resolves to |
| `select_procedure()`, `standard::measure_cadence()` | the `mode` stage's `choose()`, the change test's `prepare()` |
| `HarmonicWorkspace` | `PixelSeries` (`ccd/input/pixel_series.hpp`): views of the pixel's dates, values and QA codes; `MaskedData` and `apply_mask()` are in `ccd/input/masked_series.hpp`, with `MaskedSeries` |
| `standard::coefficient_count()` | the model strategy's `coefficients()`, a `CoefficientRule` (`ccd/strategies/coefficients.hpp`) |
| `Clock`, `elapsed_ms`, `print_change_model(s)` | none |
| `ccd/types.hpp`, `ccd/pmask.hpp`, `ccd/procedure/standard.hpp`, `ccd/record.hpp`, `ccd/harmonic/...`, `ccd/preprocess/...` | `ccd/core/types.hpp`, `ccd/input/processing_mask.hpp`, none (the machine is internal; `ccd/machine/suspended.hpp` holds the record's `Suspended`), `ccd/results/record.hpp`, `ccd/input/...`, `ccd/input/mask/...` (`preprocess/preprocess.hpp` is `input/mask/chain.hpp`, the plans and their chains; `ccd/ccd.hpp` includes every public header) |
| `filter::remove_duplicate_dates`, `apply_thermal_filter`, `apply_saturation_filter`, `apply_green_median_filter(..., filter_range, ...)` | `filter::apply_dedupe`, `apply_thermal_range`, `apply_band_range`, `apply_green_median(..., green_offset, ...)` (`ccd/input/mask/filter.hpp`), each named for its preprocess step |
| `ccd/procedure/pixel_procedure.hpp` | `ccd/input/procedure.hpp` (`PixelProcedure`, with `PROCEDURES` and `name()`; `PreprocessPlan::of()` is a procedure's plan); `fit_whole_series` is in `ccd/strategies/edge_fit/lasso.hpp`, the `mode` stage's `PixelMode` in `ccd/strategies/stages.hpp` |
| `ccd/procedure/fit.hpp`, `ccd/procedure/kernels.hpp` | the header of each function's owner: `ccd/results/segment.hpp` (`ChangeModel`, `BandFit`, `FitResult`, `CurveQA`), `ccd/core/window.hpp` (`Window`, `span`), `ccd/strategies/stages.hpp` (`Variogram`), `ccd/strategies/model.hpp` (`KeepResiduals`), `ccd/strategies/model/lasso.hpp` (`fit_bands`, `peek_residuals`, `band_fit`), `ccd/strategies/edge_fit/lasso.hpp` (`catch_segment`, `fit_whole_series`), `ccd/strategies/noise_floor/variogram.hpp` (`adjusted_variogram`), `ccd/strategies/window_screen/tmask.hpp` (`tmask`), `ccd/strategies/pyccd/peek_statistics.hpp` (`adjust_peek`, `adjust_change_threshold`, `change_magnitude`, `SeasonalOrder`, `seasonal_rmse`), `ccd/core/maths.hpp` (`chi2_quantile`); `find_closest_doy`, `calculate_variogram`, `check_variogram`, `enough_time` and `enough_samples` are internal or gone |

`RunSpec` is `{input, algorithm, record, masks, observer, observe}`; `algorithm` is a
compiled configuration (`ccd::Algorithm`, `run/algorithm.hpp`), the `ccdc` preset's
(`Algorithm::preset("ccdc")`) unless set. The values and QA are views of any supported element
type (`PixelSpectral`, `PixelQa`, `CubeSpectral`, `CubeQa`).

```cpp
// 0.10.0
ccd::HarmonicOptions h;
ccd::LassoOptions l;
h.CHANGE_PROBABILITY = 0.995;
h.CHANGE_THRESHOLD = ccd::chi2_quantile(0.995, 5);
h.RETRAIN_SPAN_FACTOR = 2.0;
ccd::FitResult r = ccd::detect(dates, spectral, qas, h, l);
ccd::CubeResult c = ccd::detect_cube(dates, cube, cube_qas, h, l, 8);   // masks kept
```

```cpp
// 1.0
ccd::config::Config config = ccd::strategies::library_registry().preset("ccdc");
config.shared["statistics_end"] = ccd::config::integer(736694);    // 0.10.0's cutoff
config.shared["change_probability"] = ccd::config::real(0.995);
config.stages["refit"].params["factor"] = ccd::config::real(2.0);
ccd::RunSpec spec;                                                   // spec.input: Collection-1 DN
spec.algorithm = std::make_shared<const ccd::Algorithm>(config);
spec.masks = true;                                                   // 0.10.0 kept them
ccd::FitResult r = ccd::detect(spec, dates, spectral, qas).result;
ccd::CubeOutput c = ccd::detect_cube(spec, dates, cube, cube_qas, 8);
```
