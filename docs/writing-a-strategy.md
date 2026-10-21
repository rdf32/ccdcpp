# Writing a strategy

A stage's decision is a strategy: a C++ class for the stage's interface, with a `Params`
struct and its schema, registered by name. A configuration then names it like any other, from
Python, TOML or C++. This page adds one end to end, the outlier rule `residual-cap`, and then a
preprocessing step. [architecture.md](architecture.md) has where the stages sit in a run.

The example is real code: `tests/ccd/extension` is an extension directory the library builds
with `-DCCD_EXTENSIONS=tests/ccd/extension`, and `tests/ccd/strategies/test_residual_cap.cpp`
compiles and runs it in every test build. Every C++ and CMake block below that quotes a file
is checked against that file by the same test, so the page follows the code.

## 1. Pick the stage

`ccd::STAGES` (`include/ccd/strategies/stages.hpp`) lists the stages, and the same header holds
each stage's interface with what it decides. The scaffold lists them with the methods a
strategy writes:

```bash
python tools/new_strategy.py --list
```

```text
mode             ModeRule: choose()
noise            Noise: estimate()
start_search     StartSearch: too_few(), first(), room(), begin(), exhausted(), forced(), short_span(), usable(), min_window()
...
outlier          OutlierRule: needs(), judge()
...
```

`residual-cap` is an `outlier` rule: after a peek that is not a change, it decides whether the
peek's first observation is dropped. Its interface:

<!-- from: include/ccd/strategies/stages.hpp -->
```cpp
enum class OutlierOutcome : std::uint8_t
{
    Keep,
    Drop,
    EndSeries
};

struct OutlierRule
{
    static constexpr const char* stage = "outlier";
    static constexpr const char* purpose = "a single observation that does not fit, dropped";

    virtual ~OutlierRule() = default;
    virtual Capabilities needs() const = 0;
    // The names of the change test's statistics judge() reads; the registry
    // refuses a configuration whose change test does not provide them.
    virtual std::vector<std::string> statistics() const { return {}; }
    virtual OutlierOutcome judge(const Peek& peek, const PeekResult& result) const = 0;
};
```

`judge()` reads the `Peek` (the model, the windows, each detection band's residuals over the
peek, the band noise, the detection bands) and the `PeekResult` the change test wrote (its
comparison RMSE and statistics by name). It returns an outcome; the machine routes on it.

## 2. Scaffold it

```bash
python tools/new_strategy.py outlier residual-cap --into tests/ccd/extension
```

```text
tests/ccd/extension/outlier/residual_cap.hpp
tests/ccd/extension/outlier/residual_cap.cpp
tests/ccd/extension/extension.cmake
```

The header declares `ResidualCap` (the name in CamelCase; `--class` sets another) with an
empty `Params`, its schema, and every method the interface leaves to it. The source has the
schema, a body per method that throws until it is written, and the registration function. The
scaffold reads the interface from `stages.hpp`; the build scaffolds every stage and compiles
the results (`tests/ccd/strategies/test_scaffold.cpp`), so it follows the interfaces.

## 3. Params and the schema

The filled-in class:

<!-- from: tests/ccd/extension/outlier/residual_cap.hpp -->
```cpp
struct ResidualCap final : OutlierRule
{
    struct Params
    {
        scalar_t limit = 2000.0;   // on the default input (Collection-1 DN): 0.2
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit ResidualCap(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    OutlierOutcome judge(const Peek& peek, const PeekResult& result) const override;

private:
    Params p_;
};
```

A parameter is written once, as a `config::Field` (`include/ccd/config/params.hpp`): its name,
the member it sets, its unit, its bounds, a line of documentation, and optionally where its
value comes from, its configuration default and a keyword. From the schema come the checks,
the unit conversion, the digest, `ccdcpp.strategies()` and the messages.

<!-- from: tests/ccd/extension/outlier/residual_cap.cpp -->
```cpp
        {"limit", &Params::limit, config::Unit::ReflectanceDelta, config::greater_than(0),
         "a first peek observation with a residual above this in any detection band is an "
         "outlier", config::Source::Own, nullptr, config::real(0.2)},
```

- **Units.** A configuration is physical; the run converts each value to the input's units
  once. `ReflectanceDelta` is a difference of reflectance, times the input's
  `reflectance_scale`: `limit` 0.2 is 2000 on Collection-1 DN and 0.2 on reflectance. The
  other units are `None`, `Observations`, `Days`, `Reflectance` (a level, with the offset),
  `Date`, `Celsius`, `Bands` (roles to rows) and `QaCodes` (names to codes); `params.hpp` says
  how each converts, and [units.md](units.md) where the boundary is.
- **Defaults.** A member's default is its value on the default input (Collection-1 DN, the
  Landsat rows): what a strategy built from `Params{}` runs with. Where that is not the
  physical value, the field's `default_value` is (`config::real(0.2)` here).
- **Bounds.** `any()`, `at_least()`, `greater_than()`, `between()`, `open_interval()`. A
  value outside them is refused by name when the configuration is built.
- **Shared values.** A value several stages read is written once, in the configuration's
  `shared` section, and a field reads it with `Source::Shared` and its name; `Source::Chi2`
  reads the chi-squared quantile of a shared probability for the detection-band count. A stage
  that sets such a field is refused, naming the shared value. The Lasso model reads the shared
  penalty, and `floored` its threshold from the shared outlier probability:

<!-- from: src/ccd/strategies/model/lasso.cpp -->
```cpp
        {"lasso.alpha", &Params::lasso_alpha, Unit::ReflectanceDelta, config::at_least(0),
         "the L1 penalty", config::Source::Shared, "lasso.alpha"},
```

<!-- from: src/ccd/strategies/outlier/floored.cpp -->
```cpp
        {"threshold", &Params::threshold, Unit::None, config::any(),
         "a first peek observation whose statistic is above this is an outlier", config::Source::Chi2,
         "outlier_probability"},
```

- **Keywords.** A whole number may take a name for a value it means something particular by
  (`config::Keyword`): the pyccd lookback's `peek` is `"change_test"` for 0, the change test's
  peek.

## 4. The decision

<!-- from: tests/ccd/extension/outlier/residual_cap.cpp -->
```cpp
// It reads the model's peek residuals.
Capabilities ResidualCap::needs() const
{
    return Capability::PeekResiduals;
}

OutlierOutcome ResidualCap::judge(const Peek& peek, const PeekResult& /*result*/) const
{
    for (const index_t band : peek.bands)
        if (peek.residuals[band][0] > p_.limit)
            return OutlierOutcome::Drop;
    return OutlierOutcome::Keep;
}
```

The rules every strategy follows:

- **It is immutable.** It is built once per resolution and shared by every thread; its methods
  are `const`. What changes from call to call goes in the per-thread object it makes:
  `make_scratch()` (a `Scratch`) or `make_model()` (a `Model`,
  `include/ccd/strategies/model.hpp`), for the stages whose interface has them.
- **It declares what it reads.** `needs()` names the capabilities it reads from the stages
  before it (`include/ccd/strategies/capabilities.hpp`), and `statistics()` the change test's
  statistics by name; a configuration whose earlier stages do not provide them is refused
  ("outlier 'residual-cap' needs ..., which model '...' does not provide").
- **It returns, the machine routes.** An outcome outside the stage's enum is a change to the
  machine (`src/ccd/machine/detector.cpp`), not to a strategy.

## 5. Register it

<!-- from: tests/ccd/extension/outlier/residual_cap.cpp -->
```cpp
void register_outlier_residual_cap(config::Registry& registry)
{
    registry.add<OutlierRule, ResidualCap>("residual-cap");
}
```

`Registry::add<Interface, Strategy>(name)` (`include/ccd/config/registry.hpp`) registers a
strategy type by name for its interface's stage, and refuses a name the stage already has.
`Registry::add_step<Step>(name)` registers a preprocessing step type (section 9), and
`add_preset(name, config)` a named configuration. Where the registration runs decides who
sees the strategy:

- **An extension.** A directory, relative to the source directory, whose `extension.cmake`
  appends its sources, include directories and registration functions; CMake's
  `CCD_EXTENSIONS` lists the directories, and the library calls each registration function when
  it first builds `ccd::strategies::library_registry()`
  (`src/ccd/strategies/register_extensions.cpp.in`). Every `Algorithm`, and so Python, then has
  the strategy. The scaffold writes this file:

<!-- from: tests/ccd/extension/extension.cmake -->
```cmake
list(APPEND CCD_EXTENSION_INCLUDE_DIRS ${CCD_EXTENSION_DIR})
list(APPEND CCD_EXTENSION_SOURCES ${CCD_EXTENSION_DIR}/outlier/residual_cap.cpp)
list(APPEND CCD_EXTENSION_REGISTRATIONS register_outlier_residual_cap)
```

- **A registry of your own** (C++ only). Register the library's strategies and yours, and
  give the registry to the `Algorithm`; the test does this, without building the extension
  into the library:

<!-- from: tests/ccd/strategies/test_residual_cap.cpp -->
```cpp
const cfg::Registry& registry()
{
    static const cfg::Registry r = []
    {
        cfg::Registry r;
        ccd::strategies::register_core_strategies(r);
        ccd::strategies::register_outlier_residual_cap(r);
        return r;
    }();
    return r;
}
```

- **The library itself.** The header goes in `include/ccd/strategies/<stage>/<name>.hpp`, the
  source in `src/ccd/strategies/<stage>/<name>.cpp` and in `ccd_core`'s sources in
  `CMakeLists.txt`, and the `registry.add` line in `register_core_strategies()`
  (`src/ccd/strategies/library.cpp`). `Registry.ListsTheCoreStrategies`
  (`tests/ccd/strategies/test_registry.cpp`) lists every core strategy.

## 6. Build it

The dev build, a CMake build directory with the extension, the tests and the Python module:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCCD_EXTENSIONS=tests/ccd/extension
cmake --build build -j8
ctest --test-dir build -j8 -LE corpus
```

`CCD_EXTENSIONS` takes `;`-separated directories, each relative to the source directory. On
macOS add the OpenMP flags of the README's [Build and test](../README.md#build-and-test).

The editable Python install, into your own environment: the package's Python files are read
from `python/ccdcpp`, so an edit to them needs nothing, and with `editable.rebuild` an import
rebuilds the extension after a C++ edit (printing CMake's output):

```bash
pip install scikit-build-core pybind11 numpy
pip install --no-build-isolation -e . -Ceditable.rebuild=true -Cbuild-dir=build/editable \
    -Ccmake.define.CCD_EXTENSIONS=tests/ccd/extension
```

The OpenMP flags go in the same way, one `-Ccmake.define.NAME=VALUE` each.

## 7. Select it

From Python, the strategy is a name like any other. The example continues from the README's
[quick-start pixel](../README.md#one-pixel):

```python
print(sorted(ccdcpp.strategies()["outlier"]))
cfg = ccdcpp.config("ccdc").replace(outlier="residual-cap", outlier__limit=0.02)
print(cfg.params("outlier"))
print(cfg.resolve(ccdcpp.Input())["outlier"], cfg.resolve(presets.landsat_c2().input())["outlier"])

plain = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2())
capped = ccdcpp.detect(dates, spectral, qas, input=presets.landsat_c2(), config=cfg)
print(plain["processing_mask"].sum(), capped["processing_mask"].sum(), len(capped["models"]))
```

```text
['floored', 'residual-cap']
{'limit': 0.02}
{'strategy': 'residual-cap', 'limit': 200.0} {'strategy': 'residual-cap', 'limit': 0.02}
376 357 2
```

`replace(outlier="residual-cap")` swaps the strategy, its parameters at their defaults;
`outlier__limit` sets one. The same configuration as a file, `ccdcpp.config("capped.toml")`:

```toml
# capped.toml
extends = "ccdc"

[outlier]
strategy = "residual-cap"
limit = 0.02
```

In C++, a stage's entry is its strategy and parameters:

<!-- from: tests/ccd/strategies/test_residual_cap.cpp -->
```cpp
cfg::Config capped(double limit)
{
    cfg::Config config = golden::pyccd();
    config.stages["outlier"] = {"residual-cap", {{"limit", cfg::real(limit)}}};
    return config;
}
```

and `std::make_shared<const ccd::Algorithm>(config, registry())` compiles it (the second
argument only for a registry of your own).

## 8. Test it

`tests/ccd/strategies/test_residual_cap.cpp` is the pattern: register, resolve, decide, run.
Its tests:

| Test | What it checks |
|---|---|
| `IsRegisteredWithItsPhysicalDefault` | the registry lists it, its default is 0.2, and it resolves to 2000 on DN and 0.2 on reflectance |
| `DropsAResidualAboveTheLimit` | `judge()` on a hand-built `Peek` |
| `ASmallerLimitDropsMore` | `detect()` with it, on a synthetic pixel |
| `ACubeIsItsPixels` | `detect_cube()` on 1 and 4 threads gives `detect()`'s dump of every pixel |
| `RefusesALimitOutsideItsBounds` | the message a bad value gets |

The helpers in `tests/ccd/support` (namespace `golden`, a test's include directory):

- `config.hpp`: `pyccd()` and `ccdc()`, the configurations the references run; `with()`,
  `with_shared()`, `with_step()` to edit them; `spec_of()`, a `RunSpec` with masks;
  `parameter<T>()`, a resolved value;
- `synthetic.hpp`: `make_cube()`, synthetic pixels and cubes of several regimes;
- `units.hpp`: `physical_input()`, the reflectance input, and `to_physical()`;
- `dump.hpp`: `dump()`, a result as hexfloat text, equal exactly when every value is
  bit-equal.

A dump compared with a committed fixture pins a strategy's outputs:
`golden::expect_matches_fixture()` (`fixture.hpp`) writes the fixture when its environment
variable is set and compares against it otherwise, as `tests/ccd/golden/test_bands.cpp` does.
A fixture is exact only on the platform that wrote it, and is written from the reference
build, never from the change under test. A new strategy that the `ccdc` preset does not name
leaves the golden dumps unchanged: the full tier checks that.

A test target is one line in `CMakeLists.txt`, `ccd_add_gtest(name SOURCES ...)`; the
example's also compiles the strategy's source and adds `tests/ccd/extension` to its include
directories.

## 9. A preprocessing step

A step type is a struct with `Params`, a schema and a static `make()`, registered with
`Registry::add_step<Step>(name)`; the configuration's chains name it. `make()` gets the
parameters resolved against the input, the input, and where the step is (its procedure, how
that procedure reads the thermal row, its position: `config::StepContext`), and returns the
step, or none where it does not apply. A step beyond the built-in kinds is a
`preprocess::CustomStep` (`include/ccd/input/mask/chain.hpp`): `apply()` clears the mask bits
of the observations it drops, `validate()` checks the plan against the band count, and
`describe()` names it. `tests/ccd/strategies/test_registry_preprocess.cpp` has one, `drop_day`, which
drops the observations of one date:

<!-- from: tests/ccd/strategies/test_registry_preprocess.cpp -->
```cpp
    static std::optional<ccd::preprocess::Step> make(const Params& params, const ccd::Input& input,
                                                     const ccd::config::StepContext& context)
    {
        contexts().push_back(context);
        if (params.day < 0)
            throw std::invalid_argument("parameter 'day' is negative");
        return ccd::preprocess::Step::of(std::make_shared<const DropDay>(params.day,
                                                                       input.index("green")));
    }
```

A chain names it by its type, as the built-in steps: `{"drop_day", {{"day", ...}}}` among a
procedure's `steps` in C++ (`Preprocess.ARegistryAddsAStepType` inserts one and runs it), or
`{ step = "drop_day", day = ... }` in a TOML chain once an extension registers it. The
library's step types (`include/ccd/input/mask/steps.hpp`) have the same shape. The scaffold
writes strategies only.
