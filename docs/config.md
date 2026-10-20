# Configuring the algorithm

ccdcpp runs one fixed machine, the CCDC family's phases and routes, whose decisions are made by
**stages**: the procedure a pixel takes, its noise floor, where the search for a stable start
begins, the outlier screen and the fit and rule a start window is judged by, Lookback, the
monitoring model and when it is refitted, the change test, the outlier rule, and the fits a
segment reports. Each stage is decided by a **strategy** chosen by name, with parameters. A
**configuration** names, for every stage, its strategy and that strategy's parameters.

A run without a configuration uses the one its `HarmonicOptions` and `LassoOptions` describe,
so code written before configurations existed runs as it always has. The stages, their
strategies and how each option maps to a parameter are in
[docs/ccdc-state-machine.md](ccdc-state-machine.md#stages-and-strategies).

## From Python

```python
import ccdcpp

cfg = ccdcpp.config("ccdc")          # the preset: the default options' configuration
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

### The configuration the options describe

```python
h, l = ccdcpp.HarmonicOptions(), ccdcpp.LassoOptions()
cfg = ccdcpp.Config.from_options(h, l)
assert cfg == ccdcpp.config("ccdc")
# The same run, with and without it:
assert ccdcpp.detect(d, s, q, h, l)["models"] == ccdcpp.detect(d, s, q, h, l, config=cfg)["models"]
```

With a profile, the configuration replaces the algorithm's choices; the profile still supplies
the band layout, the QA table, the preprocessing plan and the options the machine itself reads
(detection bands, the statistics window, the coefficient counts).

### Units

A configuration's parameters are in the input's units, as the options' are: the Lasso's
`alpha` and `coef_floor` (in the `stability_fit`, `model` and `edge_fit` stages) are 1.0 for
Collection-1 DN and 1e-4 for reflectance. `Config.from_options()` of a profile's options
(`profile.options()`) carries the profile's values.

## Presets

A preset is a TOML file, one table per stage. `ccdcpp.config(name)` reads
`ccdcpp/configs/<name>.toml`; `ccdcpp.config(path)` reads any file. A file may extend another and
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
#include "ccd/config/from_options.hpp"
#include "ccd/run/run.hpp"

using namespace ccd;

config::ConfigTree tree = config::from_options(HarmonicOptions{}, LassoOptions{});
tree["change_test"].params["change_probability"] = config::real(0.95);
tree["refit"].params["factor"] = config::real(2.0);

RunSpec spec{HarmonicOptions{}, LassoOptions{}};
spec.config = std::make_shared<const config::ConfigTree>(tree);
PixelRun run = run_pixel(spec, dates, spectral, qas);
```

`strategies::library_registry().names()` lists the strategies; `.schemas()` their parameters.

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

A record written by a configured run carries the configuration's digest: it continues only under
the same configuration. Records of runs without one are unchanged, so records written by 0.10.0
stay valid.
