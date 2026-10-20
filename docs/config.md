# Configuring the algorithm

ccdcpp runs one fixed machine, the CCDC family's phases and routes, whose decisions are made by
**stages**: the procedure a pixel takes, its noise floor, where the search for a stable start
begins, the outlier screen and the fit and rule a start window is judged by, Lookback, the
monitoring model and when it is refitted, the change test, the outlier rule, and the fits a
segment reports. Each stage is decided by a **strategy** chosen by name, with parameters. A
**configuration** holds the **shared** values several stages read (written once), for every
stage its strategy and that strategy's parameters, and for every procedure its
**preprocessing**: the steps that build the processing mask.

> The input says only how to read the data. The configuration says what to do with it, in
> physical units. The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check,
> exactly as today.
>
> - **Two calls**: `ccdcpp.detect(dates, spectral, qas, *, encoding, config="ccdc", record=False,
>   resume=None, band_names=None, trace=False)` and `ccdcpp.detect_cube(..., *, encoding,
>   config="ccdc", threads=0, record=False, resume=None, band_names=None, trace_pixels=None,
>   masks=False)`, which returns columns (with `masks=True` also every pixel's processing
>   mask, one `(H, W, T)` uint8 array); `ccdcpp.to_pixels(out)` reads them per pixel.
>   `resume=` is an earlier call's output made with `record=True`. C++: `ccd::detect` /
>   `ccd::detect_cube` with `RunSpec{encoding, algorithm, record, masks, ...}`.
> - **Input** (`ccdcpp.Input`, `ccd::Input`; immutable in Python, `Input(**fields)` and
>   `replace(**fields)`): the encoding only: the QA codes and `qa_table` (the decoder of raw QA,
>   or none), `roles`, `reflectance_scale` and `reflectance_offset` (Collection-1 DN 10000 and 0,
>   reflectance 1 and 0), `thermal_scale`/`thermal_offset` with `thermal_at_100c` (the value of
>   100 °C after the transform) and the saturation codes `saturation_code` and `thermal_saturation_code`
>   (stored values). `encoding=` takes an `Input` or a
>   `SensorProfile`; a profile describes the data only.
> - **Configuration** (`ccdcpp.Config`, `config::Config`): `shared` values, the 13 `stages` and
>   the `preprocess` chain per procedure, in physical units (reflectance, degrees Celsius,
>   probabilities, bands by role, QA codes by name). It is checked and compiled when built and
>   resolved once per input; `ccdcpp.plan_of(config, input)` shows the resolved preprocessing.
>   `config=` is a `Config` or a preset's name.
> - **"ccdc" has no statistics cutoff**; pyccd's is
>   `ccdcpp.config("ccdc").replace(shared__statistics_end=736694)`.

The input (`ccdcpp.Input`, or a `SensorProfile` that builds one) says how to read the data. The
configuration is the algorithm, in physical units, and means the same on every input. A run
without one uses the `ccdc` preset (each strategy and step at its defaults, pyccd's algorithm),
with or without a profile. The stages and their strategies are in
[docs/ccdc-state-machine.md](ccdc-state-machine.md#stages-and-strategies).

## From Python

```python
import ccdcpp

cfg = ccdcpp.config("ccdc")          # the preset: pyccd's algorithm, in physical terms
print(cfg)                           # every shared value, stage, parameter and step
# Config(
#   shared: alpha=0.0001, change_probability=0.99, ..., statistics_end=9223372036854775807, ...
#   change_test: chi2(model_rmse_up_to=24, peek_reference_days=16.0, peek_size=6, ...)
#   ...
#   preprocess standard: thermal_transform, keep_qa(codes=['clear', 'water']); ...
# )

# Change values: shared__value=, stage__parameter=, preprocess__...=. A Config is immutable;
# replace() returns a copy, checked and compiled as it is built.
cfg2 = cfg.replace(shared__change_probability=0.95, refit__factor=2.0)

# Swap a stage's strategy (a name ccdcpp.strategies() lists for it): its parameters start
# from the new strategy's defaults.
cfg3 = cfg.replace(refit="span-factor")

# Run with it: detect and detect_cube take config= (a Config or a preset's name).
out = ccdcpp.detect(dates, spectral, qas, encoding=ccdcpp.presets.landsat_c2(), config=cfg2)
cube = ccdcpp.detect_cube(dates, stack, qas, encoding=..., threads=8, config=cfg2)
```

Mistakes are refused by name, before any pixel runs:

```python
cfg.replace(refit__fctor=2.0)
# ValueError: refit 'span-factor' has no parameter 'fctor'; it has min_obs, factor
cfg.replace(refit__factor=0.0)
# ValueError: refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
cfg.replace(window_screen__bands=["green", "coastal"]).check(ccdcpp.Input())
# ValueError: ... holds 'coastal', which the input does not have; its rows are ...
```

What exists and what it takes:

```python
table = ccdcpp.strategies()      # {stage: {strategy: [parameter, ...]}}
for p in table["refit"]["span-factor"]:
    print(p["name"], p["unit"], p["bounds"], p["default"], "-", p["doc"])
# min_obs observations at least 1 24 - observations up to which every test refits
# factor none greater than 0 1.33 - refit once the window spans this many times the last fit's span
ccdcpp.shared_values()           # [{"name", "unit", "bounds", "doc", "default"}, ...]
ccdcpp.steps()                   # {step: [parameter, ...]}
```

A parameter whose source is `shared` or `chi2` reads a shared value (its chi-squared quantile
for the detection-band count, for `chi2`) and is set as `shared__<name>`, not in its stage.
`cfg.hash()` is a digest equal for equal configurations on any platform; `cfg.to_toml()` writes
the configuration as a preset file; a `Config` pickles, so it can go to worker processes.

### Units

A configuration's values are physical: the shared `alpha` and `coef_floor`, band_range's bounds
and green_median's offset in reflectance, thermal_range's bounds in degrees Celsius, the change
and outlier thresholds as probabilities, bands by role and QA codes by name. A run resolves them
against its input once: a reflectance level (unit `reflectance`, band_range's bounds) as
`r * reflectance_scale + reflectance_offset`, a reflectance difference (unit `reflectance_delta`:
`alpha`, `coef_floor`, green_median's offset) as `r * reflectance_scale` (Collection-1 DN 10000
and 0, reflectance 1 and 0), degrees Celsius `c` as `c * 100 * thermal_at_100c / 10000`, roles to
rows, names to codes.
`cfg.resolve(input)` shows the values a run uses; `ccdcpp.plan_of(cfg, input)` the
preprocessing plan. [units.md](units.md) has the input's units and the physical-unit input.

### Preprocessing

```python
from ccdcpp import BandRange, Dedupe, KeepQa, ProcedureChain, ThermalRange

keeps_snow = cfg.replace(preprocess__standard=ProcedureChain(
    steps=(KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), Dedupe()),
    thermal_transform=True,
))
cfg.replace(preprocess__thermal_range__min=-80.0)            # every thermal_range step
print(ccdcpp.plan_of(keeps_snow, ccdcpp.Input()))            # the plan, in the input's units
```

## Presets

A preset is a TOML file: a `[shared]` table, one table per stage and one
`[preprocess.<procedure>]` table per procedure. `ccdcpp.config(name)` reads
`ccdcpp/configs/<name>.toml`, or else the library registry's preset of that name (an
extension's, `Registry::add_preset()`); `ccdcpp.config(path)` reads any file. A file may extend
another and list only what differs:

```toml
# my_variant.toml
extends = "ccdc"

[shared]
change_probability = 0.95

[refit]
strategy = "span-factor"
[refit.params]
factor = 2.0
```

```python
cfg = ccdcpp.config("my_variant.toml")
```

A stage the file names with the base's strategy changes only the parameters it gives; with
another strategy, the stage starts from that strategy's defaults. A procedure the file names
takes the file's `thermal_transform` and `steps` where it gives them. A relative path in
`extends` is the extending file's directory's. The shipped preset is `ccdc` (pyccd's algorithm
with ccdcpp's corrections, no statistics cutoff).

## From C++

```cpp
#include "ccd/run/run.hpp"

using namespace ccd;

config::Config config = strategies::library_registry().preset("ccdc");
config.shared["change_probability"] = config::real(0.95);
config.stages["refit"].params["factor"] = config::real(2.0);

RunSpec spec;                        // spec.encoding is Collection-1 DN, spec.algorithm "ccdc"
spec.algorithm = std::make_shared<const Algorithm>(config);   // checked here
PixelOutput run = detect(spec, dates, spectral, qas);
preprocess::PreprocessPlan plan = plan_of(config, spec.encoding);  // for inspection
```

Three types hold a configuration. `config::Config` is the data: the `shared` values, the
`stages` and the `preprocess` chains, as written, a value you edit. `ccd::Algorithm` is its
compiled form: built once from a `Config` (checked by the schemas then), immutable, shared by
every call and thread, and caching its resolution per input (`for_input()`: resolved and built
the first time it meets an input, reused for every equal one). `RunSpec::algorithm` takes an
`Algorithm`, so a call resolves nothing it has resolved before. The Python `ccdcpp.Config`
holds both: the data (`params()`, `shared()`, `to_toml()`, `replace()`) and the `Algorithm`
compiled from it when it is built, which every `detect` and `detect_cube` with that `Config`
runs.

`strategies::library_registry().names()` lists the strategies, `.schemas()` their parameters,
`.step_schemas()` the preprocessing steps' and `.presets()` the named configurations (`ccdc`,
and any an extension registers with `Registry::add_preset()`).

## Experiments

An experiment is a configuration: a parameter changed, a strategy swapped, or a strategy of
your own registered and named.

| What you want to try | How |
|---|---|
| a stricter or looser change test, everywhere it is read | `cfg.replace(shared__change_probability=0.95)` |
| refit less often | `cfg.replace(refit__factor=2.0)` |
| the model's own RMSE for longer windows | `cfg.replace(change_test__model_rmse_up_to=40)` |
| other Tmask bands | `cfg.replace(window_screen__bands=["green", "red", "swir1"])` |
| fewer outliers dropped | `cfg.replace(shared__outlier_probability=0.99999)` |
| Initialize and Lookback at a peek of 6, whatever the change test's | `cfg.replace(stability_rule__peek=6, lookback__peek=6)` |
| fewer coefficients for long windows | `cfg.replace(shared__coefficient_max=6)` |
| pyccd's statistics cutoff | `cfg.replace(shared__statistics_end=736694)` |
| another preprocessing | `cfg.replace(preprocess__insufficient_clear__green_median__offset=0.05)`, `cfg.replace(preprocess__standard=ccdcpp.ProcedureChain(...))` |
| a sweep | `[ccdcpp.detect_cube(..., config=cfg.replace(refit__factor=f)) for f in (1.2, 1.33, 1.5, 2.0)]` |
| a new decision rule | a C++ strategy class for its stage, registered by name (below) |
| a new preprocessing step | a C++ step type, registered by name (`Registry::add_step()`, `include/ccd/input/mask/steps.hpp`); its `make()` is handed each step's context (`config::StepContext`: the procedure, whether it applies the thermal transform, the position) |

A strategy of your own is a class for its stage's interface
(`include/ccd/strategies/stages.hpp`) with a `Params` struct and its schema, registered once:

```cpp
// A refit schedule that refits every n-th test.
struct EveryN final : RefitSchedule
{
    struct Params { index_t n = 5; };
    static const std::vector<config::Field<Params>>& schema()
    {
        static const std::vector<config::Field<Params>> s = {
            {"n", &Params::n, config::Unit::Observations, config::at_least(1), "refit every n tests"}};
        return s;
    }
    explicit EveryN(const Params& p) : p_(p) {}
    bool due(index_t window_size, scalar_t, index_t, scalar_t) const override
    {
        return window_size % p_.n == 0;
    }
private:
    Params p_;
};

registry.add<RefitSchedule, EveryN>("every-n");   // in an extension's registration function
```

and then `cfg.replace(refit="every-n", refit__n=3)`. A build adds strategies through an
extension directory (CMake's `CCD_EXTENSIONS`; see
[docs/ccdc-state-machine.md](ccdc-state-machine.md#configurations-and-extensions)).

## Resume records

A record carries the digest of the input (its QA table included) and the resolved configuration
(`ResumeHeader::digest`, the resolution's digest): it continues only under the same
ones, and a resume under another input or configuration raises `ValueError` before any pixel
runs.
