# Architecture

ccdcpp runs one machine, the CCDC family's phases and routes. Every decision the machine
makes is a **stage**, decided by the **strategy** the configuration names for it. A run is
an **input** (how to read the data) and a **configuration** (what to do with it, in physical
units); the run translates the configuration to the input once and then runs every pixel
through the same steps. This page is the map: what happens, in which order, and in which
file. [ccdc-state-machine.md](ccdc-state-machine.md) has the machine in detail,
[config.md](config.md) the configuration, [writing-a-strategy.md](writing-a-strategy.md) how
to add a strategy. Paths are relative to the repository root.

## A run, end to end

```
 Python (python/ccdcpp)                     C++ (include/ccd, src/ccd)
 ──────────────────────                     ──────────────────────────
 input=  Input | SensorProfile ──────────► RunSpec::input            how to read the data
 config= Config | preset name  ──────────► RunSpec::algorithm        what to do with it
         (Config compiles an Algorithm)     ccd::Algorithm            checked once, immutable
                                                │ for_input(input), once per input
                                                ▼
                                            config::Resolved          strategies, plan, shared, digest
 detect / detect_cube ───────────────────► ccd::detect / ccd::detect_cube
                                                │ Run: the call's checks; detect_cube: one OpenMP
                                                │ loop, a Worker per thread
                                                ▼
                                            Run::prepare              decode the QA, the mode stage,
                                                │                     the values, the preprocessing chain
                                                ▼
                                            machine::Detector::fit    the machine, or the whole-series
                                                │                     curve (edge_segments)
                                                ▼
 dicts / columns (results.py) ◄──────────── PixelOutput / CubeOutput  segments, masks, records
```

1. **Input and configuration.** `ccd::Input` (`include/ccd/input/input.hpp`) is the band
   roles, units, thermal transform and QA table. `config::Config`
   (`include/ccd/config/config.hpp`) is the shared values, one `{strategy, parameters}` per
   stage and one preprocessing chain per procedure, all in physical terms. In Python they are
   `ccdcpp.Input` (or a `SensorProfile`, `python/ccdcpp/profiles.py`) and `ccdcpp.Config`
   (`python/ccdcpp/config.py`).
2. **Algorithm.** `ccd::Algorithm` (`include/ccd/run/algorithm.hpp`) checks a configuration
   once (`config::check()`) and keeps it in full (`config::canonical()`). A Python `Config`
   compiles one as it is built (`bindings/config.cpp`).
3. **Resolve.** `Algorithm::for_input()` calls `config::resolve()`
   (`include/ccd/config/resolve.hpp`) the first time it sees an input, and caches the result:
   reflectance and degrees Celsius in the input's units, roles as rows, QA names as codes,
   probabilities as chi-squared thresholds, then one strategy object per stage and the
   preprocessing plan, built by name from the registry. The `Resolved` also holds the digest a
   resume record is checked against.
4. **Detect.** `ccd::detect` and `ccd::detect_cube` (`include/ccd/run/run.hpp`) build one `Run`
   per call (`src/ccd/run/run_pixel.hpp`): the input, dates, QA and resume header checked
   before any pixel. `detect_cube` dispatches on the value and QA types once and runs one
   OpenMP loop (`src/ccd/run/cube.cpp`); each thread holds a `Worker`, the buffers and the
   `Detector` it reuses from pixel to pixel.
5. **Prepare a pixel.** `Run::prepare` (`src/ccd/run/run.cpp`): decode the QA, compute the QA
   statistics up to `statistics_end`, ask the `mode` stage for the pixel's `PixelMode`, apply
   its procedure's transform, and run its procedure's chain into the processing mask.
6. **Fit.** `machine::Detector::fit` (`src/ccd/machine/detector.hpp`) runs the machine for a
   Standard pixel, or `edge_segments`' `whole_series()` for a PermanentSnow or
   InsufficientClear one. A resume continues the stored run instead (`Run::resumed`,
   `src/ccd/run/resume.cpp`, and `Detector::resume`).
7. **Outputs.** A `FitResult` (`include/ccd/results/segment.hpp`): the segments and, with
   `RunSpec::masks`, the processing mask; with `RunSpec::record`, a `PixelRecord` and the
   call's `ResumeHeader` (`include/ccd/results/record.hpp`). The binding returns them as dicts
   or columns (`bindings/detect.cpp`, `bindings/results.cpp`, `python/ccdcpp/results.py`).

## The machine

One machine for every method (`src/ccd/machine/detector.cpp`). Its phases and routes are
fixed; each phase asks stages for decisions and routes on what they return. A strategy never
steers the machine: it returns a value from a closed set (`TestOutcome`, `OutlierOutcome`,
`LookbackVerdict`, `Exhaustion`, ...), and a new outcome is a change to the machine.

| Phase | What it does | Stages it asks |
|---|---|---|
| Prepare | the early exits, the peek plan, the noise, the first candidate window | `start_search`, `change_test` (`prepare()`), `noise`, `stability_rule`, `lookback` (`prepare()`) |
| Initialize | one candidate start window: screen it, fit it, judge it; grow or shift it | `start_search`, `window_screen`, `stability_fit`, `stability_rule` |
| Lookback | one observation before a stable window: include, drop, or a change | `lookback` |
| StartCatch | the Start segment before the first stable window | `edge_segments` |
| Extend | one peek test after the window: change, outlier, series end, or grow | `refit`, `model`, `change_test`, `outlier`, `segment_report` |
| Restart | the cursor after a segment | `start_search` |
| EndCatch | the End segment after the last segment | `edge_segments` |

The phase diagram, its guards and the exit reasons are in
[ccdc-state-machine.md](ccdc-state-machine.md#2-the-machine).

## The stages

`ccd::STAGES` (`include/ccd/strategies/stages.hpp`) lists the 13 stages in the order a pixel
meets them; a configuration file lists them so. Each stage is an interface in that header,
and each strategy is one class in `include/ccd/strategies/<stage>/<name>.hpp` and
`src/ccd/strategies/<stage>/<name>.cpp`.

| Stage | Interface: methods | Its job | `ccdc`'s strategy |
|---|---|---|---|
| `mode` | `ModeRule`: `choose()` | the pixel's procedure from its QA statistics, with extra mask steps and a coefficient cap (`PixelMode`) | `pyccd` |
| `noise` | `Noise`: `estimate()` | each band's noise floor, which every test scales by | `variogram` |
| `start_search` | `StartSearch`: `too_few()`, `first()`, `room()`, `begin()`, `exhausted()`, `forced()`, `short_span()`, `usable()`, `min_window()` | where a candidate start window is, and when the search has none left | `pyccd` |
| `window_screen` | `WindowScreen`: `screen()` | the outliers of a candidate window, dropped before it is fitted | `tmask` |
| `stability_fit` | `StabilityFit`: `make_model()`, `coefficients()`, `provides()` | the fit a candidate window is judged on | `lasso` |
| `stability_rule` | `StabilityRule`: `prepare()`, `stable()`, `needs()`, `make_scratch()` | whether that fit is stable | `pyccd` |
| `lookback` | `Lookback`: `prepare()`, `judge()`, `needs()`, `make_scratch()` | whether the observation before a stable window joins it | `pyccd` |
| `model` | `MonitoringModel`: `make_model()`, `coefficients()`, `provides()` | the fit Extend tests each peek against | `lasso` |
| `change_test` | `ChangeTest`: `prepare()`, `test()`, `break_at()`, `magnitude()`, `needs()`, `provides()`, `statistics()`, `make_scratch()` | the pixel's peek; whether a peek is a change; the break and magnitudes | `chi2` |
| `outlier` | `OutlierRule`: `judge()`, `needs()`, `statistics()` | whether the first peek observation is dropped, or the series ends | `floored` |
| `refit` | `RefitSchedule`: `due()` | when Extend refits the model | `span-factor` |
| `segment_report` | `SegmentReport`: `segment()`, `unfitted()`, `make_scratch()` | the coefficients a finished segment reports | `model-fit` |
| `edge_segments` | `EdgeSegments`: `start_due()`, `end_due()`, `catch_window()`, `whole_series()`, `make_scratch()` | the Start and End segments, and the whole-series curve | `lasso` |

The contract every stage shares:

- **Immutable strategies.** A strategy is built once per resolution from its `Params` and
  shared by every thread; its methods are `const`. What changes from call to call lives in a
  per-thread object it makes: a `Model` (`make_model()`, `include/ccd/strategies/model.hpp`)
  or a `Scratch` (`make_scratch()`), which the machine keeps in a `PerThread`
  (`include/ccd/strategies/per_thread.hpp`).
- **Declared needs.** At an Extend peek every stage reads one `Peek`, and the change test
  writes a `PeekResult` the outlier rule reads. What a strategy provides and needs is
  declared (`Capabilities`, `include/ccd/strategies/capabilities.hpp`; statistics by name), and
  a configuration whose strategy needs what no earlier stage provides is refused when it is
  checked.
- **Batch calls.** A fit or a peek takes all its bands in one call: a virtual call per peek or
  band, never per value.

What each `ccdc` strategy decides, rule by rule, is in
[ccdc-state-machine.md](ccdc-state-machine.md#stages-and-strategies).

## Registry, configuration, resolve

Three pieces, each with one job:

| Piece | File | Job |
|---|---|---|
| registry | `include/ccd/config/registry.hpp` | the catalogue: each stage's strategy types by name, the preprocessing step types by name, the presets. `Registry::add<Interface, Strategy>(name)`, `add_step<Step>(name)`, `add_preset(name, config)` |
| configuration | `include/ccd/config/config.hpp`, `shared.hpp`, `preprocess.hpp` | a value: the shared values, the stages, the preprocessing, in physical terms. Nothing in it is resolved |
| schema | `include/ccd/config/params.hpp` | each strategy's and step's parameters, written once (`Field`): name, member, unit, bounds, doc, and where the value comes from (its own, a shared value, a chi-squared quantile) |
| resolve | `include/ccd/config/resolve.hpp`, `src/ccd/config/convert.cpp` | a configuration against an input: values converted, shared values filled in, strategies and plan built, the digest |

`ccd::strategies::library_registry()` (`include/ccd/strategies/library.hpp`) is the registry
every `Algorithm` uses by default: `register_core_strategies()`
(`src/ccd/strategies/library.cpp`), which also registers the `ccdc` preset
(`config::ccdc_preset()`, `src/ccd/config/presets.cpp`), and the registration functions of the
extensions the library was built with (CMake's `CCD_EXTENSIONS`). An `Algorithm` takes another
registry as its second argument. `python/ccdcpp/configs/ccdc.toml` is the `ccdc` preset written
out; a test keeps the two equal.

## Preprocessing

Each procedure (`standard`, `snow`, `insufficient_clear`) has a chain: steps in order, each a
step type by name with its parameters. The step types are `keep_qa`, `add_qa`,
`thermal_range`, `band_range`, `green_median` and `one_per_date`
(`include/ccd/input/mask/steps.hpp`); a registry may add more (`Registry::add_step()`, a
`preprocess::CustomStep`). Resolved, the chains are a `preprocess::PreprocessPlan`
(`include/ccd/input/mask/chain.hpp`): per procedure its thermal transform and its steps in the
input's rows, codes and units. `Run::prepare` applies the pixel's chain, then the mode's extra
steps, into the processing mask (`include/ccd/input/processing_mask.hpp`).
`ccdcpp.plan_of(config, input)` shows a plan; the README's
[default plan](../README.md#the-default-plan) is `ccdc`'s.

## The storage and physical boundary

The input describes how the data is stored; the configuration is physical. The run converts
the configuration's values to the input's units once, at resolve, and never touches the data
except for the Standard procedure's thermal transform on a private copy.
[units.md](units.md) has the boundary, the Input's fields and the guards.

## Diagnostics

| Tier | What | How |
|---|---|---|
| outputs | the segments; the processing mask (the observations a fit used); the resume record and its status | always; `RunSpec::masks`; `RunSpec::record` (Python's `processing_mask`, `record=`) |
| reason codes | why each observation is where the run left it: kept in a segment or not, or the preprocessing step or machine stage that removed it (`ccd::Reason`, `include/ccd/input/processing_mask.hpp`); the chain notes each step's removals, `MaskedSeries` each drop's, the run the kept observations no segment holds | `RunSpec::reasons` (Python's `reasons=`, `"mask_reasons"`) |
| diagnostics columns | each pixel's procedure, exit, observation counts and peek; the machine's work on each segment (`ccd::PixelDiagnostics`, `include/ccd/results/diagnostics.hpp`), counted on every pixel and copied out when asked for | `RunSpec::diagnostics` (Python's `diagnostics=`) |
| observer | every phase transition with the cursor, every Extend test with its outcome and statistics, every dropped observation, every segment, the exit | `RunSpec::observer` and `RunSpec::observe`; `ccd::TextObserver` prints them; Python's `trace=True` and `trace_pixels=` return them as rows |
| compiled out | the observer's hooks | CMake's `CCD_OBSERVE=OFF` |

An observed run's outputs, and a run's with reason codes or diagnostics, equal the plain run's,
bit for bit. The observer is in
`include/ccd/results/diagnostics.hpp`; [diagnostics.md](diagnostics.md) has the events.

## Where each piece lives

```
include/ccd/                 the public headers (src/ccd/ mirrors them)
  ccd.hpp                    every public header
  core/                      types, ArrayView, BandList, Window, constants, maths
  input/                     Input, QA tables and statistics, PixelSeries, MaskedSeries,
                             ProcessingMask, procedures
    mask/                    the preprocessing: Chain, Step, PreprocessPlan, the step types
  regression/                the harmonic basis, the Lasso and robust solvers, CoefficientRule
  config/                    Config, Shared, the preprocess section, Field schemas, the
                             Registry, resolve, presets
  strategies/                stages.hpp (the interfaces), model.hpp, capabilities.hpp,
                             per_thread.hpp, library.hpp (the registry)
    <stage>/<name>.hpp       one strategy per file
    pyccd/                   pyccd's peek statistics, read by four pyccd strategies
  machine/                   Suspended: where a stopped machine resumes
  results/                   segments, records, the observer
  run/                       Algorithm, RunSpec, detect, detect_cube
src/ccd/machine/             the machine (ccd::machine: detector.hpp, detector.cpp; internal)
src/ccd/run/                 Run and Worker (run_pixel.hpp; internal), the cube loop, resume
bindings/                    the Python module: config, input, QA tables, preprocess, detect,
                             results
python/ccdcpp/               the package: detect, detect_cube, Config, profiles, presets,
                             results, configs/ccdc.toml, the .pyi stubs
tests/ccd/                   C++ tests, by area; support/ (shared helpers), data/ (CSV
                             pixels), golden/ (bit-exact fixtures), extension/ (the worked
                             example strategy)
tests/python/                the Python tests
tools/                       bench/ (timing and memory), linux-compare/ (the golden driver
                             and the Linux check), new_strategy.py (a strategy's scaffold)
```

## Before and after

Where each decision lived in 0.10.0, and where it lives now:

| Decision | 0.10.0 | 1.0 |
|---|---|---|
| the calls | 20 `detect`, `detect_cube`, `resume`, `resume_cube`, `detect_recorded` overloads (`src/ccd/ccd.cpp`); Python `detect`, `detect_cube`, `detect_cube_columns`, `resume`, `resume_cube_columns` | `ccd::detect`, `ccd::detect_cube` (`include/ccd/run/run.hpp`); Python `detect`, `detect_cube` |
| the algorithm's values | `HarmonicOptions` (`include/ccd/harmonic/harmonic.hpp`) and `LassoOptions`, in the input's units, per profile | `config::Config` in physical units, each value in its strategy's schema; the `ccdc` preset |
| the procedure | `select_procedure()` (`src/ccd/procedure/pixel_procedure.cpp`), `CLEAR_PCT_THRESHOLD`, `SNOW_PCT_THRESHOLD` | `mode` `pyccd` |
| the noise | `adjusted_variogram()` (`src/ccd/procedure/fit.cpp`) in `prepare()` (`src/ccd/procedure/standard.cpp`), `VARIOGRAM_MIN_LAG_DAYS` | `noise` `variogram` |
| the start window | `initialize_criteria()` (`standard.cpp`), `enough_time()`, `enough_samples()` (`fit.cpp`), `MEOW_SIZE`, `DAY_DELTA` | `start_search` `pyccd`; shared `min_observations` |
| the start window's outliers | `tmask()` (`fit.cpp`) in `initialize_step()` (`src/ccd/procedure/kernels.cpp`), `TMASK_BANDS`, `T_CONST` | `window_screen` `tmask` |
| the stability check | `fit_bands()` and `stable()` in `initialize_step()` (`kernels.cpp`, `fit.cpp`) | `stability_fit` `lasso`, `stability_rule` `pyccd` |
| Lookback | `lookback_step()` (`standard.cpp`), `CHANGE_THRESHOLD`, `OUTLIER_THRESHOLD` | `lookback` `pyccd` |
| the refit | inside `extend_step()` (`standard.cpp`), `RETRAIN_MIN_OBS`, `RETRAIN_SPAN_FACTOR` | `refit` `span-factor` |
| the monitoring fit | `fit_bands()` (`kernels.cpp`) with the run's `LassoOptions` | `model` `lasso`; shared `lasso.*` |
| the change test | `detect_change()` inside `extend_step()`; the peek from `measure_cadence()`; `seasonal_rmse()` (`fit.cpp`), `PEEK_SIZE`, `CHANGE_THRESHOLD`, `SEASONAL_RMSE_*` | `change_test` `chi2`; shared `change_probability` |
| the outlier rule | `detect_outlier()` inside `extend_step()`, `OUTLIER_THRESHOLD` | `outlier` `floored`; shared `outlier_probability` |
| the coefficient count | `coefficient_count()` (`standard.cpp`), `COEFFICIENT_MIN`, `_MID`, `_MAX`, `NUM_OBS_FACTOR` | `CoefficientRule` (`include/ccd/regression/coefficients.hpp`); shared `coefficients.*` |
| a segment's fit | `complete_models()`, `write_extended_segment()` (`standard.cpp`) | `segment_report` `model-fit` |
| the Start, End and whole-series curves | `catch_segment()` (`kernels.cpp`), `fit_whole_series()` (`pixel_procedure.cpp`) | `edge_segments` `lasso` |
| the preprocessing | `default_plan(HarmonicOptions)` (`src/ccd/preprocess/preprocess.cpp`), the filters (`src/ccd/harmonic/filter.cpp`) | the configuration's `preprocess` section; the step types (`include/ccd/input/mask/steps.hpp`) |
| the units | each profile's own option values (DN constants on DN inputs) | `Unit` on each field, converted at resolve (`src/ccd/config/convert.cpp`) |
| debugging | `LOG_DEBUG` (`include/ccd/logger.hpp`) | the observer (`include/ccd/results/diagnostics.hpp`) |

The 0.10.0 paths are in the `v0.10.0` tag: `git show v0.10.0:src/ccd/procedure/standard.cpp`.
