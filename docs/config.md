# Configuring the algorithm

ccdcpp runs one fixed machine, the CCDC family's phases and routes, whose decisions are made by
**stages**: the procedure a pixel takes, its noise floor, where the search for a stable start
begins, the outlier screen and the fit and rule a start window is judged by, Lookback, the
monitoring model and when it is refitted, the change test, the outlier rule, and the fits a
segment reports. Each stage is decided by a **strategy** chosen by name, with parameters. A
**configuration** names, for every stage, its strategy and that strategy's parameters.

The options (`HarmonicOptions`) describe the input: its bands, units, QA codes and statistics
window. The configuration is the algorithm. A run without one uses the `ccdc` preset (each
strategy at its defaults, pyccd's values, Collection-1 DN); a run with `profile=` and no
`config=` uses `profile.config()`, that preset written for the profile's input. The stages and
their strategies are in [docs/ccdc-state-machine.md](ccdc-state-machine.md#stages-and-strategies).

> **Step 5 state.** The description of the data is `ccdcpp.Input` (C++ `ccd::Input`): the QA
> codes, a role per row (`roles`), `saturation_bands`, `reflectance_unit` (Collection-1 DN 10000,
> reflectance 1) and the bounds and thermal transform, in snake_case. `HarmonicOptions` is gone,
> and the entry points take `input=` where they took `hoptions=`; `profile.input()` replaces
> `profile.options()`. A configuration is `{"shared": ..., "stages": ...}` in physical terms
> (`ccdcpp.Config`, C++ `config::Config`): the values several stages read are written once in
> `shared` (`detection_bands` by role, `change_probability`, `outlier_probability`, the Lasso's
> `alpha`, `coef_floor`, `max_iter`, `tolerance`, the coefficient rule, `stat_ord`,
> `green_filter` and `green_filter_scope`), reflectance values are multiplied by the input's
> `reflectance_unit`, and the change and outlier thresholds are the chi-squared quantiles of the
> probabilities for the detection-band count, all when a run resolves the configuration against
> its input (C++ `ccd::Algorithm`, which caches the resolution per input). Some examples below
> still use the earlier names; the full documentation comes with the two-call API.

## From Python

```python
import ccdcpp

cfg = ccdcpp.config("ccdc")          # the preset: pyccd's algorithm, Collection-1 DN
print(cfg)                           # every stage and parameter
# Config(
#   change_test: chi2(change_probability=0.99, change_threshold=15.086272469388987, ...)
#   edge_fit: lasso(alpha=1.0, coef_floor=1.0, coefficients=4, ...)
#   ...
#   refit: span-factor(factor=1.33, min_obs=24)
#   ...
# )

# Change parameters: stage__parameter=value. A Config is immutable; replace() returns a copy.
cfg2 = cfg.replace(change_test__change_probability=0.95, refit__factor=2.0)

# Swap a stage's strategy (a name ccdcpp.strategies() lists for it): its parameters start
# from the new strategy's defaults.
cfg3 = cfg.replace(refit="span-factor")

# Run with it: every entry point takes config=.
out = ccdcpp.detect(dates, spectral, qas, profile=ccdcpp.presets.landsat_c2(), config=cfg2)
cube = ccdcpp.detect_cube_columns(dates, stack, qas, profile=..., threads=8, config=cfg2)
```

Mistakes are refused by name, before any pixel runs:

```python
cfg.replace(refit__fctor=2.0)
# ValueError: refit 'span-factor' has no parameter 'fctor'; it has min_obs, factor
cfg.replace(refit__factor=0.0).check()
# ValueError: refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
cfg.replace(window_screen__bands=[1, 9]).check(bands=7)
# ValueError: window_screen 'tmask' parameter 'bands' holds band 9; the input has 7 bands
```

What exists and what it takes:

```python
table = ccdcpp.strategies()      # {stage: {strategy: [parameter, ...]}}
for p in table["refit"]["span-factor"]:
    print(p["name"], p["unit"], p["bounds"], p["default"], "-", p["doc"])
# min_obs observations at least 1 24 - observations up to which every test refits
# factor none greater than 0 1.33 - refit once the window spans this many times the last fit's span
```

`cfg.hash()` is a digest equal for equal configurations on any platform; `cfg.to_toml()` writes
the configuration as a preset file; a `Config` pickles, so it can go to worker processes.

### The configuration of a profile

```python
p = ccdcpp.presets.landsat_c2()
cfg = p.config()                     # "ccdc" for the profile's input
print(cfg.params("model")["alpha"], cfg.params("window_screen")["bands"])
# 0.0001 [1, 4]
assert ccdcpp.presets.landsat_c1_dn().config() == ccdcpp.config("ccdc")
# The same run, with and without it:
assert ccdcpp.detect(d, s, q, profile=p)["models"] == ccdcpp.detect(d, s, q, profile=p, config=cfg)["models"]
# Without a profile, the options and the "ccdc" preset:
out = ccdcpp.detect(d, s_dn, codes, ccdcpp.HarmonicOptions())
```

`profile.config(base="ccdc")` writes `base` (any `ccdcpp.config()` source) for the profile: its
`cadence` (`stage__parameter` keys, applied with `Config.replace()`), its `alpha` and
`coef_floor` in `model`, `stability_fit` and `edge_fit`, its Tmask roles as `window_screen`'s
`bands`, and the change and outlier thresholds of its detection band count in `change_test`,
`stability_rule`, `lookback` and `outlier`. A `config=` given with a profile replaces
`profile.config()` whole, so start from it (`p.config().replace(refit__factor=2.0)`) to keep
the profile's units. The profile still supplies the band layout, the QA table, the preprocessing
plan and the options the machine itself reads (detection bands, the statistics window).

### Units

A configuration's parameters are in the input's units, as the options' are: the Lasso's
`alpha` and `coef_floor` (in the `stability_fit`, `model` and `edge_fit` stages) are 1.0 for
Collection-1 DN and 1e-4 for reflectance. `profile.config()` carries the profile's values.

## Presets

A preset is a TOML file, one table per stage. `ccdcpp.config(name)` reads
`ccdcpp/configs/<name>.toml`, or else the library registry's preset of that name (an
extension's, `Registry::add_preset()`); `ccdcpp.config(path)` reads any file. A file may extend another and
list only what differs:

```toml
# my_variant.toml
extends = "ccdc"

[change_test]
strategy = "chi2"
[change_test.params]
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
another strategy, the stage starts from that strategy's defaults. A relative path in `extends`
is the extending file's directory's. The shipped preset is `ccdc`
(pyccd's algorithm with ccdcpp's corrections, Collection-1 DN values).

## From C++

```cpp
#include "ccd/run/run.hpp"

using namespace ccd;

config::ConfigTree tree = strategies::library_registry().preset("ccdc");
tree["change_test"].params["change_probability"] = config::real(0.95);
tree["refit"].params["factor"] = config::real(2.0);

RunSpec spec{HarmonicOptions{}};     // spec.config is the "ccdc" preset
spec.config = tree;
PixelRun run = run_pixel(spec, dates, spectral, qas);
```

`strategies::library_registry().names()` lists the strategies, `.schemas()` their parameters
and `.presets()` the named configurations (`ccdc`, and any an extension registers with
`Registry::add_preset()`).

## Experiments

An experiment is a configuration: a parameter changed, a strategy swapped, or a strategy of
your own registered and named.

| What you want to try | How |
|---|---|
| a stricter or looser change test | `cfg.replace(change_test__change_probability=0.95)` |
| refit less often | `cfg.replace(refit__factor=2.0)` |
| the model's own RMSE for longer windows | `cfg.replace(change_test__model_rmse_up_to=40)` |
| other Tmask bands | `cfg.replace(window_screen__bands=[1, 2, 4])` |
| a stricter start | `cfg.replace(stability_rule__change_threshold=11.0705)` |
| fewer outliers dropped | `cfg.replace(outlier__threshold=30.856, lookback__outlier_threshold=30.856)` |
| Initialize and Lookback at a peek of 6, whatever the change test's | `cfg.replace(stability_rule__peek=6, lookback__peek=6)` |
| fewer coefficients for long windows | `cfg.replace(model__coefficient_max=6)` |
| a sweep | `[ccdcpp.detect_cube_columns(..., config=cfg.replace(refit__factor=f)) for f in (1.2, 1.33, 1.5, 2.0)]` |
| a new decision rule | a C++ strategy class for its stage, registered by name (below) |

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

A record carries the digest of the options and the configuration (`options_digest()`): it
continues only under the same ones, and a resume under another configuration raises
`ValueError` before any pixel runs.
