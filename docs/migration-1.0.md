# Migrating to 1.0

In 1.0 the options describe the input and the configuration is the algorithm. `HarmonicOptions`
keeps what it says about the input: the QA codes, `CLEAR_OBSERVATION_THRESHOLD`, the thermal
transform and value ranges, the reflectance ranges, `MEDIAN_GREEN_FILTER`,
`GREEN_FILTER_SCOPE`, `STAT_ORD`, `NUM_BANDS`, `DETECTION_BANDS`, `SATURATION_BANDS` and the
band indices. Every algorithm option of 0.10.0, and `LassoOptions` with it, is a parameter of a
stage of the configuration ([config.md](config.md)), set as `stage__parameter`.

- **Results are bit-identical.** A 0.10.0 run and the 1.0 run of the same input, with its
  options moved as below, give the same segments, coefficients, RMSEs and magnitudes to the bit.
  A call with `profile=` and the default options needs no change at all.
- **0.10.0 resume records do not continue.** The resume digest covers the options' fields and
  the configuration, so a 0.10.0 record's digest is not a 1.0 run's: `resume()` and
  `resume_cube_columns()` refuse it with `ValueError` ("the resume record was made with other
  options, configuration, plan or QA table"), before any pixel runs. Run the detection once on 1.0 over the
  whole series with `resume_record=True` and store its records; they continue from then on.

## Python

### What runs unchanged

A call with a profile and the default configuration runs as written, with the same output:

```python
out = ccdcpp.detect_cube_columns(dates, s, q, threads=n, profile=p, resume_record=True)
update = ccdcpp.resume_cube_columns(dates, s, q, threads=n, profile=p,
                                    resume=out["resume"], resume_header=out["resume_header"],
                                    segments={k: out[k] for k in SEGMENT_COLUMNS})
```

### Removed names

| 0.10.0 | 1.0 |
|---|---|
| `ccdcpp.LassoOptions`, the `loptions` argument of every entry point | `config=`: the Lasso's `max_iter`, `alpha`, `tolerance` and `coef_floor` are parameters of `model`, `stability_fit` and `edge_fit` |
| the algorithm fields of `HarmonicOptions` (`MEOW_SIZE`, `PEEK_SIZE`, `CHANGE_THRESHOLD`, `TMASK_BANDS`, ...) | configuration parameters (the table below) |
| `SensorProfile.options()` returning `(HarmonicOptions, LassoOptions)` | `options()` returns the `HarmonicOptions`; `config()` returns the profile's configuration |
| `SensorProfile(cadence=...)` keyed by option name, `ccdcpp.CADENCE_FIELDS` | `cadence` keyed by `stage__parameter`; any parameter `ccdcpp.strategies()` lists |
| `threads` as the sixth positional argument of `detect_cube`, `detect_cube_columns`, `resume_cube_columns` | the fifth (after `hoptions`); `threads=n` works in both |

A call without a profile runs the `ccdc` configuration unless it gives `config=`; a call with
one runs `profile.config()`.

### Default options

```python
# 0.10.0
out = ccdcpp.detect(dates, dn, codes, ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions())
# 1.0
out = ccdcpp.detect(dates, dn, codes, ccdcpp.HarmonicOptions())
```

### Lasso constants for physical units

The Lasso's `alpha` and `coef_floor` are in the input's units; without a profile, set them in
the three stages that fit:

```python
# 0.10.0
lo = ccdcpp.LassoOptions()
lo.alpha, lo.coef_floor = 1e-4, 1e-4
out = ccdcpp.detect_cube_columns(dates=dates, spectral=cube, qas=codes, hoptions=h, loptions=lo,
                                 threads=4)
# 1.0
cfg = ccdcpp.config("ccdc").replace(**{f"{stage}__{name}": 1e-4
                                       for stage in ("model", "stability_fit", "edge_fit")
                                       for name in ("alpha", "coef_floor")})
out = ccdcpp.detect_cube_columns(dates=dates, spectral=cube, qas=codes, hoptions=h, config=cfg,
                                 threads=4)
```

`h` keeps its physical thresholds (`REFL_MAX`, `THERM_MIN`, `THERM_MAX`, `MEDIAN_GREEN_FILTER`,
the thermal transform) as before. `presets.landsat_c1_physical().config()` is the same
configuration.

### Algorithm options

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
t = ccdcpp.chi2_quantile(0.995, 5)
cfg = ccdcpp.config("ccdc").replace(
    change_test__change_probability=0.995, change_test__change_threshold=t,
    stability_rule__change_probability=0.995, stability_rule__change_threshold=t,
    lookback__change_probability=0.995, lookback__change_threshold=t,
    refit__factor=2.0,
    start_search__meow_size=16, edge_fit__min_observations=16,
    window_screen__bands=[1, 2, 4],
    model__max_iter=500, stability_fit__max_iter=500, edge_fit__max_iter=500)
out = ccdcpp.detect(dates, dn, codes, ccdcpp.HarmonicOptions(), config=cfg)
```

With a profile, start from its configuration: `config=p.config().replace(refit__factor=2.0)`.

### A profile's cadence

```python
# 0.10.0
dense = dataclasses.replace(presets.hls(), cadence={"PEEK_REFERENCE_DAYS": 8.0,
                                                     "RETRAIN_MIN_OBS": 36})
# 1.0
dense = dataclasses.replace(presets.hls(), cadence={"change_test__peek_reference_days": 8.0,
                                                     "refit__min_obs": 36,
                                                     "change_test__model_rmse_up_to": 36})
```

### Reading a profile's options

```python
# 0.10.0
h, lo = presets.landsat_c2().options()
tmask, alpha = list(h.TMASK_BANDS), lo.alpha
# 1.0
h, cfg = presets.landsat_c2().options(), presets.landsat_c2().config()
tmask, alpha = cfg.params("window_screen")["bands"], cfg.params("model")["alpha"]
```

## The options' fields and the parameters they became

Where one 0.10.0 field set several parameters, set each of them to keep the 0.10.0 run.

| 0.10.0 field | Default | 1.0 parameters |
|---|---|---|
| `MEOW_SIZE` | 12 | `start_search__meow_size`, `edge_fit__min_observations` |
| `PEEK_SIZE` | 6 | `change_test__peek_size` (`stability_rule__peek` and `lookback__peek` are 0, the change test's peek) |
| `DAY_DELTA` | 365 | `start_search__day_delta` |
| `CLEAR_PCT_THRESHOLD` | 0.25 | `mode__clear_pct_threshold` |
| `SNOW_PCT_THRESHOLD` | 0.75 | `mode__snow_pct_threshold` |
| `OUTLIER_THRESHOLD` | 35.8882 | `outlier__threshold`, `lookback__outlier_threshold` |
| `CHANGE_THRESHOLD` | 15.0863 | `change_test__change_threshold`, `stability_rule__change_threshold`, `lookback__change_threshold` |
| `CHANGE_PROBABILITY` | 0.99 | `change_test__change_probability`, `stability_rule__change_probability`, `lookback__change_probability` |
| `T_CONST` | 4.89 | `window_screen__t_const` |
| `TMASK_BANDS` | [1, 4] | `window_screen__bands` |
| `PEEK_REFERENCE_DAYS` | 16.0 | `change_test__peek_reference_days` |
| `VARIOGRAM_MIN_LAG_DAYS` | 30 | `noise_floor__min_lag_days` |
| `RETRAIN_MIN_OBS` | 24 | `refit__min_obs`, `change_test__model_rmse_up_to` |
| `RETRAIN_SPAN_FACTOR` | 1.33 | `refit__factor` |
| `SEASONAL_RMSE_COUNT` | 24 | `change_test__seasonal_count` |
| `SEASONAL_RMSE_DIVISOR` | 4.0 | `change_test__seasonal_divisor` |
| `COEFFICIENT_MIN` | 4 | `model__coefficient_min`, `stability_fit__coefficient_min`, `edge_fit__coefficients` |
| `COEFFICIENT_MID` | 6 | `model__coefficient_mid`, `stability_fit__coefficient_mid` |
| `COEFFICIENT_MAX` | 8 | `model__coefficient_max`, `stability_fit__coefficient_max` |
| `NUM_OBS_FACTOR` | 3 | `model__num_obs_factor`, `stability_fit__num_obs_factor` |
| `LassoOptions.max_iter`, `alpha`, `tolerance`, `coef_floor` | 1000, 1.0, 1e-4, 1.0 | the same names in `model`, `stability_fit` and `edge_fit` |

A 0.10.0 option set as a configuration, by this table:

```python
OPTION_PARAMS = {
    "MEOW_SIZE": ("start_search__meow_size", "edge_fit__min_observations"),
    "PEEK_SIZE": ("change_test__peek_size",),
    "DAY_DELTA": ("start_search__day_delta",),
    "CLEAR_PCT_THRESHOLD": ("mode__clear_pct_threshold",),
    "SNOW_PCT_THRESHOLD": ("mode__snow_pct_threshold",),
    "OUTLIER_THRESHOLD": ("outlier__threshold", "lookback__outlier_threshold"),
    "CHANGE_THRESHOLD": ("change_test__change_threshold", "stability_rule__change_threshold",
                         "lookback__change_threshold"),
    "CHANGE_PROBABILITY": ("change_test__change_probability", "stability_rule__change_probability",
                           "lookback__change_probability"),
    "T_CONST": ("window_screen__t_const",),
    "TMASK_BANDS": ("window_screen__bands",),
    "PEEK_REFERENCE_DAYS": ("change_test__peek_reference_days",),
    "VARIOGRAM_MIN_LAG_DAYS": ("noise_floor__min_lag_days",),
    "RETRAIN_MIN_OBS": ("refit__min_obs", "change_test__model_rmse_up_to"),
    "RETRAIN_SPAN_FACTOR": ("refit__factor",),
    "SEASONAL_RMSE_COUNT": ("change_test__seasonal_count",),
    "SEASONAL_RMSE_DIVISOR": ("change_test__seasonal_divisor",),
    "COEFFICIENT_MIN": ("model__coefficient_min", "stability_fit__coefficient_min",
                        "edge_fit__coefficients"),
    "COEFFICIENT_MID": ("model__coefficient_mid", "stability_fit__coefficient_mid"),
    "COEFFICIENT_MAX": ("model__coefficient_max", "stability_fit__coefficient_max"),
    "NUM_OBS_FACTOR": ("model__num_obs_factor", "stability_fit__num_obs_factor"),
}
OPTION_PARAMS.update({name: tuple(f"{stage}__{name}" for stage in ("model", "stability_fit", "edge_fit"))
                      for name in ("max_iter", "alpha", "tolerance", "coef_floor")})


def config_from_options(fields, base="ccdc"):
    """``fields``: 0.10.0 option names (and LassoOptions' four) -> values."""
    return ccdcpp.config(base).replace(**{key: value for name, value in fields.items()
                                          for key in OPTION_PARAMS[name]})


cfg = config_from_options({"CHANGE_PROBABILITY": 0.995,
                           "CHANGE_THRESHOLD": ccdcpp.chi2_quantile(0.995, 5),
                           "COEFFICIENT_MAX": 6, "T_CONST": 6.0})
```

## C++

### Removed names

| 0.10.0 | 1.0 |
|---|---|
| `detect(d, s, q, h, l, table, plan)` | `run_pixel({h, cfg, table, plan}, d, s, q).result` |
| `detect_recorded(...)` | `run_pixel` with `RunSpec::record = true` |
| `resume(d, s, q, h, l, header, record, segments)` | `run_pixel({h}, d, s, q, PixelResume{header, record, segments})` |
| `detect_cube(d, s, q, h, l, threads, keep_masks, table, plan, record)` | `run_cube({h, cfg, table, plan, record, keep_masks}, d, s, q, threads)` |
| `resume_cube(..., from, ...)` | `run_cube(..., threads, &from)` |
| `Recorded`, `Resumed`; `DetectScratch` | `PixelRun`; `Worker` |
| `LassoOptions` in a run (`RunSpec::loptions`, the entry points' `l`) | the `lasso` strategies' `max_iter`, `alpha`, `tolerance`, `coef_floor` in `RunSpec::config` (`LassoOptions` remains the solver's own options type, inside those strategies) |
| the algorithm fields of `HarmonicOptions` | `RunSpec::config` parameters (the table above) |
| `standard::coefficient_count()` | the model strategy's `coefficients()`, a `CoefficientRule` (`include/ccd/strategies/coefficients.hpp`; `count(n)`) |
| `standard::Context`'s, `Procedure`'s and `Detector`'s Lasso workspace and solver | none: each strategy holds its own |
| `ccd/types.hpp`, `ccd/procedure/standard.hpp`, `ccd/record.hpp`, ... | `ccd/core/types.hpp`, `ccd/machine/standard.hpp`, `ccd/results/record.hpp`, ... (`ccd/ccd.hpp` includes every header) |

`RunSpec` is `{hoptions, config, qa_table, plan, record, keep_masks, observer, observe}`, its
`config` a `config::ConfigTree` held by value, the `ccdc` preset unless set; `cfg` above is the
configuration, `ccd::strategies::library_registry().preset("ccdc")` for 0.10.0's default
options. `run_pixel` and `run_cube` take every supported element type (`PixelSpectral`,
`PixelQa`, `CubeSpectral`, `CubeQa`) through one call each.

```cpp
// 0.10.0
ccd::HarmonicOptions h;
ccd::LassoOptions l;
h.CHANGE_PROBABILITY = 0.995;
h.CHANGE_THRESHOLD = 16.7496;
l.alpha = l.coef_floor = 1e-4;
ccd::FitResult r = ccd::detect(dates, spectral, qas, h, l);

// 1.0
ccd::RunSpec spec{ccd::HarmonicOptions{}};
for (const char* stage : {"change_test", "stability_rule", "lookback"})
{
    spec.config[stage].params["change_probability"] = ccd::config::real(0.995);
    spec.config[stage].params["change_threshold"] = ccd::config::real(16.7496);
}
for (const char* stage : {"model", "stability_fit", "edge_fit"})
{
    spec.config[stage].params["alpha"] = ccd::config::real(1e-4);
    spec.config[stage].params["coef_floor"] = ccd::config::real(1e-4);
}
ccd::FitResult r = ccd::run_pixel(spec, dates, spectral, qas).result;
```

### Messages

`validate()` checks the input. The algorithm's values are checked by the configuration's
schemas when a run is built, by stage, strategy and parameter, and still throw
`std::invalid_argument` (`ValueError` in Python) before any pixel runs:

| 0.10.0 | 1.0 |
|---|---|
| `PEEK_SIZE is 1; it must be at least 2` | `change_test 'chi2' parameter 'peek_size' is 1; it must be at least 2` |
| `COEFFICIENT_MAX is 9; the harmonic basis has at most 8` | `model 'lasso' parameter 'coefficient_max' is 9; the harmonic basis has at most 8` |
| `TMASK_BANDS is empty` | `window_screen 'tmask' parameter 'bands' is empty` |

A `MEOW_SIZE` of 0, a NaN change or outlier threshold and negative or NaN Lasso constants ran
in 0.10.0; 1.0 refuses them (`start_search 'pyccd' parameter 'meow_size' is 0; it must be at
least 1`, `model 'lasso' parameter 'alpha' is -1; it must be at least 0`).
