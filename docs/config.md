# Configuring the algorithm

ccdcpp runs one fixed machine, the CCDC family's phases and routes, whose decisions are made by
**stages**: the procedure a pixel takes, its noise, where the search for a stable start
begins, the outlier screen and the fit and rule a start window is judged by, Lookback, the
monitoring model and when it is refitted, the change test, the outlier rule, and the fits a
segment reports. Each stage is decided by a **strategy** chosen by name, with parameters. A
**configuration** holds the **shared** values several stages read (written once), for every
stage its strategy and that strategy's parameters, and for every procedure its
**preprocessing**: the steps that build the processing mask. A run without one uses the `ccdc`
preset: pyccd's algorithm with ccdcpp's corrections, each strategy and step at its defaults,
with no statistics cutoff, on any input. The stages and their strategies are in
[ccdc-state-machine.md](ccdc-state-machine.md#stages-and-strategies).

The examples use `dates`, `spectral`, `qas`, `cube` and `cube_qa` from the README's
[quick start](../README.md#quick-start).

## The rule

| | Example |
|---|---|
| The input says only how to read the data. | `presets.landsat_c2().input()` holds reflectance 0..1 (`reflectance_scale` 1), thermal in degrees Celsius / 100, and the decoder of `QA_PIXEL`. |
| The configuration says what to do with it, in physical units. | `ccdc`'s `green_median` drops observations 0.04 reflectance above the median green; its `lasso.alpha` is 1e-4 reflectance. |
| The run translates once: 0–1 reflectance × 10000 = the 0–10000 DN check, exactly as today. | On `ccdcpp.Input()` the `green_median` offset resolves to 400 DN, `lasso.alpha` to 1.0 and `band_range` to `0 < x < 10000`. |

```python
import ccdcpp
from ccdcpp import presets

c2 = presets.landsat_c2().input()
print(c2.reflectance_scale, c2.thermal_at_100c, c2.qa_table is not None)

cfg = ccdcpp.config("ccdc")
print(cfg.preprocess("insufficient_clear")["steps"][4], cfg.shared()["lasso"]["alpha"])

plan = ccdcpp.plan_of(cfg, ccdcpp.Input())
print(plan.insufficient_clear.chain[4])
print(cfg.resolve(ccdcpp.Input())["model"]["alpha"], plan.standard.chain[2])
```

```text
1.0 1.0 True
{'step': 'green_median', 'offset': 0.04, 'scope': 'stat_window'} 0.0001
<Step GreenMedian band 1, drop x >= median(every date) + 400, scope StatWindow>
1.0 <Step BandRange bands [0, 1, 2, 3, 4, 5], 0 < x < 10000>
```

The input is a `ccdcpp.Input`, or a `SensorProfile` that builds one ([units.md](units.md) has the
Input's fields and the boundary between the data and the configuration). A run resolves its
configuration against its input once:

- a reflectance level (unit `reflectance`: `band_range`'s `reflectance` range) as
  `r * reflectance_scale + reflectance_offset`;
- a reflectance difference (unit `reflectance_delta`: `lasso.alpha`, `lasso.coef_floor`,
  `green_median`'s offset) as `r * reflectance_scale`;
- degrees Celsius `c` (`thermal_range`'s `celsius` range) as `c * 100 * thermal_at_100c / 10000`;
- a date `"YYYY-MM-DD"` (`statistics_end`) as its ordinal day;
- a probability to its chi-squared quantile for the detection-band count;
- roles to rows and QA code names to the input's codes.

`cfg.resolve(input)` shows every resolved value, `ccdcpp.plan_of(cfg, input)` the
preprocessing plan.

## From Python

```python
cfg = ccdcpp.config("ccdc")          # the preset, built once and shared
print(cfg.stages()[:4], cfg.strategy("refit"), cfg.params("refit"))
print(cfg.shared()["detection_bands"], cfg.shared()["statistics_end"])

# Change values: shared__name=, stage__parameter=, preprocess__...=. A Config is immutable;
# replace() returns a copy, checked and compiled as it is built.
cfg2 = cfg.replace(shared__change_probability=0.95, refit__factor=2.0)

# Swap a stage's strategy (a name ccdcpp.strategies() lists for it): its parameters start
# from the new strategy's defaults.
cfg3 = cfg.replace(refit="span-factor")

# Run with it: detect and detect_cube take config= (a Config or a preset's name).
pixel = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2(), config=cfg2)
cols = ccdcpp.detect_cube(dates, cube, cube_qa, input=presets.landsat_c2(), config=cfg2,
                          threads=4)
print(len(pixel["models"]), cols["pixels_with_segments"], cfg3 == cfg)
```

```text
['mode', 'noise', 'start_search', 'window_screen'] span-factor {'factor': 1.33, 'min_obs': 24}
['green', 'red', 'nir', 'swir1', 'swir2'] none
2 12 True
```

Mistakes are refused by name with `ValueError`, before any pixel runs. A parameter outside its
own bounds names its stage, strategy and parameter, or `shared` for a shared value; a
combination a strategy refuses names the first stage that reads it; a role is checked against
the input when the configuration is resolved (`check(input)`, or the run):

```python
for bad in (lambda: cfg.replace(refit__fctor=2.0),
            lambda: cfg.replace(refit__factor=0.0),
            lambda: cfg.replace(shared__lasso__alpha=-1.0),
            lambda: cfg.replace(shared__coefficients__min=7),
            lambda: cfg.replace(start_search__meow_size=16),
            lambda: cfg.replace(window_screen__bands=["green", "coastal"]).check(ccdcpp.Input())):
    try:
        bad()
    except ValueError as e:
        print(e)
```

```text
refit 'span-factor' has no parameter 'fctor'; it has min_obs, factor
refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
shared lasso parameter 'alpha' is -1; it must be at least 0
shared coefficients min, mid, max are 7, 6, 8; they must not decrease
start_search 'pyccd' parameter 'meow_size' is the shared value 'min_observations'; set shared 'min_observations' instead
window_screen 'tmask' parameter 'bands' holds 'coastal', which the input does not have; its rows are blue, green, red, nir, swir1, swir2, thermal
```

What exists and what it takes:

```python
table = ccdcpp.strategies()      # {stage: {strategy: [parameter, ...]}}
for p in table["change_test"]["chi2"][:4]:
    print(p["name"], "|", p["unit"], "|", p["source"], "|", p["default"])
print([s["name"] for s in ccdcpp.shared_values()][:4])
print(sorted(ccdcpp.steps()))
```

```text
peek_size | observations | own | 6
peek_reference_days | days | own | 16.0
change_threshold | none | chi2 | None
change_probability | none | shared | None
['detection_bands', 'change_probability', 'outlier_probability', 'min_observations']
['add_qa', 'band_range', 'green_median', 'keep_qa', 'one_per_date', 'thermal_range']
```

A parameter whose source is `shared` or `chi2` reads a shared value (its chi-squared quantile
for the detection-band count, for `chi2`) and is set as `shared__<name>`, or
`shared__<table>__<name>` for a table's (`shared__lasso__alpha`), not in its stage.
`cfg.hash()` is a digest equal for equal configurations on any platform; `cfg.to_toml()` writes
the configuration as a preset file ([below](#presets)); a `Config` pickles, so it can go to
worker processes. `cfg.to_dict()` is the configuration laid out as its file: `"shared"`, one
table per stage (`{"strategy": ..., <parameter>: ...}`) and `"preprocess"`.

### Shared values

| Shared value | `ccdc` | Unit | Read by |
|---|---|---|---|
| `detection_bands` | green, red, nir, swir1, swir2 | roles | every test; their count is the chi-squared degrees of freedom |
| `change_probability` | 0.99 | probability | `change_test`, `stability_rule`, `lookback` (their `change_threshold`, derived) |
| `outlier_probability` | 0.999999 | probability | `outlier`'s `threshold`, `lookback`'s `outlier_threshold` (derived) |
| `lasso.alpha`, `lasso.coef_floor` | 1e-4, 1e-4 | reflectance (difference) | the Lasso of `model`, `stability_fit`, `edge_segments` |
| `lasso.max_iter`, `lasso.tolerance` | 1000, 1e-4 | none | the same three |
| `coefficients.min`, `.mid`, `.max`, `.per` | 4, 6, 8, 3 | none | `model`, `stability_fit` (the coefficient rule); `edge_segments`' `coefficients` (`coefficients.min`) |
| `min_observations` | 12 | observations | `start_search`'s `meow_size`, `edge_segments`' `min_observations`; more than `coefficients.min` |
| `statistics_end` | `"none"` | date | the QA statistics, the cadence, the variogram, `green_median`'s date |

`statistics_end` is `"none"` (the whole series), a date `"YYYY-MM-DD"` or an ordinal day
(`date.toordinal()`): pyccd's cutoff is `replace(shared__statistics_end="2017-12-31")`, the same
run as `replace(shared__statistics_end=736694)`.

### Preprocessing

```python
from ccdcpp import BandRange, OnePerDate, KeepQa, ProcedureChain, ThermalRange

keeps_snow = cfg.replace(preprocess__standard=ProcedureChain(
    steps=(KeepQa(("clear", "water", "snow")), ThermalRange(), BandRange(), OnePerDate()),
    thermal="celsius",
))
colder = cfg.replace(preprocess__thermal_range__celsius=[-80.0, 70.7])   # every thermal_range step
print(ccdcpp.plan_of(keeps_snow, ccdcpp.Input()).standard.chain[0])
print(ccdcpp.plan_of(colder, ccdcpp.Input()).standard.chain[1])  # the plan, in the input's units
```

```text
<Step KeepQa codes {0, 1, 3}>
<Step ThermalRange band 6, -8000 < x < 7070>
```

## Presets

A preset is a TOML file. `ccdcpp.config(name)` reads `ccdcpp/configs/<name>.toml`, or else the
library registry's preset of that name (an extension's, `Registry::add_preset()`), built once;
`ccdcpp.config(path)` reads any file. `cfg.to_toml()` writes one, in this layout (`ccdc`'s
file, shortened):

```toml
# --- Values several stages share ----------------------------------------------------------

[shared]
detection_bands = ["green", "red", "nir", "swir1", "swir2"]
change_probability = 0.99
outlier_probability = 0.999999
min_observations = 12            # observations
statistics_end = "none"          # a date "YYYY-MM-DD", an ordinal day, or "none"

[shared.coefficients]            # how many harmonic coefficients a curve gets
min = 4
mid = 6
max = 8
per = 3

[shared.lasso]                   # the Lasso every fitting stage uses
alpha = 0.0001                   # reflectance
coef_floor = 0.0001              # reflectance
max_iter = 1000
tolerance = 0.0001

# --- Preprocessing: which observations each procedure keeps -------------------------------

[preprocess.standard]
thermal = "celsius"              # through the input's thermal transform (degrees Celsius)
steps = [
    { step = "keep_qa", codes = ["clear", "water"] },
    { step = "thermal_range", celsius = [-93.2, 70.7] },                      # degrees Celsius
    { step = "band_range", bands = "reflective", reflectance = [0.0, 1.0] },  # reflectance
    { step = "one_per_date" },
]

# --- The stages, in the order a pixel meets them ------------------------------------------

[mode]                           # the procedure a pixel runs: standard, snow or insufficient clear
strategy = "pyccd"
clear_pct_threshold = 0.25
snow_pct_threshold = 0.75

[stability_rule]                 # whether that fit is stable enough to start a curve
strategy = "pyccd"
peek = "change_test"             # observations, or "change_test"

[refit]                          # when a growing curve is refitted
strategy = "span-factor"
min_obs = 24                     # observations
factor = 1.33
```

- **One table per stage**, in the order a pixel meets them: `strategy`, and beside it the
  strategy's parameters. A file may give the tables in any order. An unknown key is refused,
  naming it.
- **The shared values** are `[shared]`, with two tables for the values of one rule each:
  `[shared.coefficients]` (`min`, `mid`, `max`, `per`) and `[shared.lasso]` (`alpha`,
  `coef_floor`, `max_iter`, `tolerance`). Python reaches a table's value with double
  underscores, `replace(shared__lasso__alpha=1e-5)`; C++ by its path,
  `config.shared["lasso.alpha"]`.
- **Each procedure's chain** is `[preprocess.<procedure>]`: `thermal` (`"celsius"`, through the
  input's thermal transform, or `"stored"`, as the input holds it) and its `steps` in order,
  each written out in full: nothing is looked up elsewhere.
- **Named values** stand where a number would mean something particular: `statistics_end =
  "none"` (or a date `"2017-12-31"`, or an ordinal), `peek = "change_test"` (or a number of
  observations).
- **A range is a pair keyed by its unit**: `celsius = [-93.2, 70.7]`, `reflectance = [0.0,
  1.0]`, open at both ends unless `inclusive = [true, false]` (or another pair) says so.
- **Comments from the code**: each table says what it decides and each value its unit, from
  the registry's schemas. `to_toml()` leaves out a flag left off at its default (the
  `inclusive` pairs); `to_toml(full=True)` writes every value.

A file may extend another and list only what differs:

```toml
# my_variant.toml
extends = "ccdc"

[shared]
change_probability = 0.95

[refit]
factor = 2.0

[preprocess.insufficient_clear]
steps = [
    { step = "keep_qa", codes = ["clear", "water"] },
    { step = "band_range" },
    { step = "one_per_date" },
    { step = "green_median", offset = 0.05 },
    { step = "one_per_date" },
]
```

```python
variant = ccdcpp.config("my_variant.toml")
print(variant.shared()["change_probability"], variant.params("refit"))
print(variant.preprocess("insufficient_clear")["steps"][3])
print(variant.preprocess("standard") == cfg.preprocess("standard"))
```

```text
0.95 {'factor': 2.0, 'min_obs': 24}
{'step': 'green_median', 'offset': 0.05, 'scope': 'stat_window'}
True
```

A stage the file names keeps its base's strategy and the parameters the file does not give
(`strategy` may be left out, or repeated); with another `strategy`, the stage starts from that
strategy's defaults. A shared table changes the values it gives. A procedure the file names
takes the file's `thermal` and `steps` where it gives them, each step's parameters from the
step's defaults. A relative path in `extends` is the extending file's directory's. The shipped
preset is `ccdc`; `python/ccdcpp/configs/ccdc.toml` is its `to_toml()` under a header comment.
A file in the layout before 1.0 (`[<stage>.params]`, `thermal_transform`, the old names) is
refused, naming what changed; [migration-1.0.md](migration-1.0.md#configuration-files) maps
it.

## From C++

```cpp
#include "ccd/ccd.hpp"

using namespace ccd;

PixelOutput run_variant(ArrayView<const std::int64_t, 1> dates,
                        ArrayView<const double, 2> spectral,
                        ArrayView<const std::uint8_t, 1> qas)
{
    config::Config config = strategies::library_registry().preset("ccdc");
    config.shared["change_probability"] = config::real(0.95);
    config.shared["lasso.alpha"] = config::real(2e-4);            // [shared.lasso] alpha
    config.stages["refit"].params["factor"] = config::real(2.0);

    RunSpec spec;                        // spec.input is Collection-1 DN, spec.algorithm "ccdc"
    spec.algorithm = std::make_shared<const Algorithm>(config);   // checked here
    preprocess::PreprocessPlan plan = plan_of(config, spec.input);   // for inspection
    return detect(spec, dates, spectral, qas);
}
```

Three types hold a configuration. `config::Config` (`include/ccd/config/config.hpp`) is the
data: the `shared` values (a table's by its path, `"lasso.alpha"`), the `stages` and the
`preprocess` chains, as written, a value you edit; `config::canonical()` writes it out in
full, and `config::resolve()` (`include/ccd/config/resolve.hpp`) against an input. `ccd::Algorithm` is its
compiled form: built once from a `Config` (checked by the schemas then), immutable, shared by
every call and thread, and caching its resolution per input (`for_input()`: resolved and built
the first time it meets an input, reused for every equal one; the 16 inputs used last are
kept). `RunSpec::algorithm` takes an `Algorithm`, so a call resolves nothing it has resolved
before. The Python `ccdcpp.Config` holds both: the data (`params()`, `shared()`, `to_toml()`,
`replace()`) and the `Algorithm` compiled from it when it is built, which every `detect` and
`detect_cube` with that `Config` runs.

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
| fewer coefficients for long windows | `cfg.replace(shared__coefficients__max=6)` |
| pyccd's statistics cutoff | `cfg.replace(shared__statistics_end="2017-12-31")` |
| another preprocessing | `cfg.replace(preprocess__insufficient_clear__green_median__offset=0.05)`, `cfg.replace(preprocess__standard=ccdcpp.ProcedureChain(...))` |
| a sweep | `[ccdcpp.detect_cube(..., config=cfg.replace(refit__factor=f)) for f in (1.2, 1.33, 1.5, 2.0)]` |
| a new decision rule | a C++ strategy class for its stage, registered by name (below) |
| a new preprocessing step | a C++ step type, registered by name (`Registry::add_step()`, `include/ccd/input/mask/steps.hpp`); its `make()` is handed each step's context (`config::StepContext`: the procedure, how it reads the thermal row, the position) |

```python
sweep = {f: ccdcpp.detect_cube(dates, cube, cube_qa, input=presets.landsat_c2(), threads=4,
                               config=cfg.replace(refit__factor=f))["pixels_with_segments"]
         for f in (1.2, 1.33, 1.5, 2.0)}
print(sweep)
```

```text
{1.2: 12, 1.33: 12, 1.5: 12, 2.0: 12}
```

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

void register_every_n(config::Registry& registry)   // an extension's registration function
{
    registry.add<RefitSchedule, EveryN>("every-n");
}
```

and then `cfg.replace(refit="every-n", refit__n=3)`. A build adds strategies through an
extension directory (CMake's `CCD_EXTENSIONS`; see
[ccdc-state-machine.md](ccdc-state-machine.md#configurations-and-extensions)).

## Resume records

A record carries the digest of the input (its QA table included) and the resolved configuration
(`ResumeHeader::digest`, `resume_header["digest"]` in Python): it continues only under the same
ones, and a resume under another input or configuration raises `ValueError`
(`std::invalid_argument` in C++) before any pixel runs.
