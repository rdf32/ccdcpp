# CCDC as a state machine

Per pixel, CCDC is a small state machine. It has one cursor, the model window over the pixel's
kept observations, and one input it changes as it goes, the processing mask. Every loop in the
algorithm is one state repeating itself until a guard sends it on. ccdcpp writes it that way:
each state is a function that makes one decision and returns the next state, one loop steps any
machine, and the per-pixel record holds everything that survives from one step to the next.

This page describes the code as it is, with the files to read. Paths are relative to the
repository root. The algorithm is pyccd's, with pyccd's intent kept where its code says
otherwise (the rules in section 7); see the README's
[Differences from pyccd](../README.md#differences-from-pyccd) for every difference.

## 1. One pixel, end to end

`Run::pixel` (`src/ccd/run/run.cpp`) is a straight pipeline, not a machine. A `Run` is built
once per call (`src/ccd/run/run_pixel.hpp`, internal): it checks the input, dates and QA, takes the
configuration resolved for the input (its strategies and preprocessing plan) and holds
what every pixel shares; each OpenMP thread holds a `Worker` with the buffers and the change
detector it reuses from pixel to pixel. The steps of one pixel:

1. decode the QA (through the input's `QaTable` when it has one);
2. compute the QA statistics of the observations dated up to the shared `statistics_end`
   (`qa_statistics()`, `include/ccd/input/qa_statistics.hpp`);
3. choose the pixel's procedure: the run's `mode` stage (`ModeRule::choose`,
   `include/ccd/strategies/stages.hpp`);
4. apply that procedure's transform to the pixel's private copy and run its filter chain to
   build the processing mask (`include/ccd/input/mask/preprocess.hpp`);
5. fit the mask with the run's change detector, `standard::Detector::fit`
   (`src/ccd/machine/standard.hpp`).

The procedure choice (the `pyccd` mode's, with its parameters):

| Guard, in order | Procedure | What fits it |
|---|---|---|
| clear percentage >= `clear_pct_threshold` | Standard | the Standard machine (section 2), with the run's strategies |
| else snow percentage >= `snow_pct_threshold` | PermanentSnow | `ccd::fit_whole_series`, curve_qa 54 |
| else | InsufficientClear | `ccd::fit_whole_series`, curve_qa 44 |

The whole-series fit has no loop, so it is a function, not a machine. With fewer than the
`lasso` edge fit's `min_observations` (the shared value, 12) kept observations it returns no segment, with the
processing mask. Otherwise it returns one segment over the whole series, every band fitted with
the edge fit's `coefficients` (the shared `coefficient_min`, 4).

## 2. The Standard machine

```
                              ┌─ size <= meow_size ───────────────────────► Done  [TooFewObservations]
                              ├─ a NaN in the variogram ──────────────────► Done  [VariogramNaN]
   Prepare ───────────────────┼─ no room for a start window ──────────────► EndCatch [NoRoomToInitialize]
                              ▼
       ┌───────────────► Initialize ◄──┐  ShortSpan, AllOutliers, RejectedOutliers: grow
       │                  │    │   └───┘  Unstable, DroppedThenUnstable: shift
       │                  │    └── Exhausted ─────────────────────────────► EndCatch [InitializeExhausted]
       │                  │ Stable, DroppedThenStable
       │                  ▼
       │   window.start > prev_break ?──── no ─────────────┐
       │                  │ yes                            │
       │                  ▼                                │
       │      ┌──► Lookback  Change / Dropped / Included   │
       │      └───────┘  (while window.start > prev_break) │
       │                  │                                │
       │                  ▼                                ▼
       │   no segment yet and window.start - prev_break > peek ?
       │                  │ yes                            │ no
       │                  ▼                                │
       │              StartCatch (the Start segment)       │
       │                  ▼                                ▼
       │   a full peek after the window ? ── no ──────────────────────────► EndCatch [PeekBeyondEnd]
       │                  │ yes
       │                  ▼
       │      ┌──► Extend  Change / Dropped / SeriesEnd / Grown
       │      └───────┘  (while a full peek fits)
       │                  │ series end: segment, change probability 0 ───────────────► EndCatch [SeriesEnd]
       │                  │ Change: segment, change probability 1
       │                  ▼
       └─── room ──── Restart: prev_break = window.stop, window = (stop, stop + meow_size)
                          │ no room
                          ▼
                      EndCatch [NoRoomToInitialize] ── the End segment, if prev_break + peek < size ──► Done
```

`meow_size` (the shared `min_observations`) and `day_delta` here are the `pyccd` start search's parameters
(the `start_search` stage, section 3): the machine asks the run's start search where a candidate
is, whether it has room and whether the search is exhausted.

Bracketed names are the `Exit` value the record keeps: why the machine stopped where it did.
The exit is internal; the trace and the tests read it, the result does not carry it.

### Phases and their guards

Each row is one transition function in `src/ccd/machine/standard.cpp`, declared in
`src/ccd/machine/standard.hpp`. The three phases that repeat themselves do one thing per
call and say what they did as an outcome value, so the trace and the tests see every decision.

| Phase | One call | Next |
|---|---|---|
| `Prepare` (`prepare`) | The early exits; the statistics window; the cadence (the change test's `prepare()`, the pixel's peek); the stability rule's and the lookback's `prepare()`; Initialize's criteria (`initialize_criteria`); the variogram; the cursor at the start search's `first(0)` | Initialize, EndCatch or Done |
| `Initialize` (`initialize`) | One candidate start window: `ccd::initialize_step` returns an `InitializeOutcome` | itself after a grow or a shift; then Lookback, StartCatch, Extend or EndCatch |
| `Lookback` (`lookback`) | One observation before the window: `lookback_step` returns a `LookbackOutcome` | itself while `window.start > prev_break`; then StartCatch, Extend or EndCatch |
| `StartCatch` (`start_catch`) | The first segment's leading observations as a Start segment (curve_qa 14) | Extend or EndCatch |
| `Extend` (`extend`) | One peek test after the window: `extend_step` returns an `ExtendOutcome`; `extend` writes the segment | itself while a peek fits; Restart after a change; EndCatch at the series end |
| `Restart` (`restart`) | The cursor after a change | Initialize or EndCatch |
| `EndCatch` (`end_catch`) | The End segment (curve_qa 24), when enough observations trail the last break | Done |

The guards inside one call, in the order they are tested:

- **Initialize** (`initialize_step`, `src/ccd/machine/standard.cpp`): the start search finds
  no candidate left (`Exhausted`) or a span too short (`ShortSpan`, grow; pyccd: under
  `day_delta`); the window screen (Tmask) flags
  every observation (`AllOutliers`, grow); it flags some and the shortened window is not a
  candidate (the search's `usable()`)
  (`RejectedOutliers`, grow, nothing dropped); then the detection bands are fitted by the
  stability fit and judged by the stability rule (`Stable`, or `Unstable` and a shift). When Tmask's outliers are dropped, the
  same call fits the shortened window, as pyccd does in one pass of its loop; those steps report
  `DroppedThenStable` and `DroppedThenUnstable`.
- **Lookback** (`lookback_step`): the lookback strategy's verdict on the observation before the
  window, which the machine applies: `Change` (the window stays), `Drop` (`Dropped`, the
  observation leaves the series) or `Include` (`Included`, `window.start - 1`). pyccd's: every
  peek statistic above the threshold is a change, the first above its `outlier_threshold` an
  outlier. Its peek window is `lookback_peek`'s: far from the previous break, the adjusted peek of observations
  before the window; near it, down to the previous break and never below. The comparison RMSE
  is the stable start window's: nothing is refitted while the window grows backward, as in
  Zhu's TrendSeasonalFit.
- **Extend** (`extend_step`): first the refit schedule (refit when untrained, at most
  `min_obs` observations, or once the window spans `factor` times the last fit; pyccd's 24 and
  1.33); then the change test (`Change`); the outlier rule (`Dropped`, the observation at
  `window.stop` leaves the series; pyccd's: its statistic above the rule's `threshold`); a peek that
  reached the last observation (`SeriesEnd`: the curve ends at the window, before the
  observation the peek starts at, its break day); otherwise `Grown` (`window.stop + 1`). The
  curve that reaches the series end, by `SeriesEnd` or by a drop that leaves no full peek, is
  the last one: EndCatch follows, not Restart. A refit fits the detection bands, all a test
  reads; the other bands are fitted on the last refit's window when the segment is
  written (`complete_models`; before the drop when a drop ends the curve), which gives the
  models and peek residuals an every-band refit would have.

Initialize fits the stability fit's `coefficient_min` coefficients, Extend's fits follow the
monitoring model's rule (`CoefficientRule`, `include/ccd/strategies/coefficients.hpp`:
`coefficient_min`, `_mid` or `_max` by window length), and both peek bases are `coefficient_max`
wide: a model predicts with zeros above its own count. Extend's fits are the monitoring model's
(the `model` stage, below), which `extend` holds.

The routing between phases is a set of named predicates and routers, so a guard has one place:
`room_to_initialize`, `peek_fits`, `start_catch_due`, `after_stable_start`, `after_lookback`
and `peek_guard`. Every route into EndCatch records its `Exit`.

The transitions assert the invariants the algorithm relies on: Initialize's window holds at
least the start search's `min_window()` observations; Lookback runs only while `window.start > prev_break`; Extend runs
only when a full peek fits.

## 3. The per-pixel record and the context

A transition is a function of two arguments, `(const Context&, State&)`. What it may change is
in the `State`. What it reads but never changes is in the `Context`.

**`standard::Context`** holds the run's resolved shared values (never written below
`Run::pixel`), the pixel's `PixelSeries` (its dates, spectral and QA, with the run's `Input`),
the run's strategies, the observer and the pixel mode's coefficient cap. It holds no solver: each strategy
that fits holds its own. The catch segments (StartCatch, EndCatch) fit with the edge fit's,
Initialize with the stability fit's and Extend with the monitoring model's (the `edge_fit`,
`stability_fit` and `model` stages' parameters), and Lookback reads the stability fit. A direct
caller of `standard::Procedure` takes the strategies of a configuration resolved against its
input (`strategies::library_registry().resolve(config, input).strategies`), as `Run` does.

**`standard::State`** is one pixel's record:

| Field | What it is | Shared or Standard's |
|---|---|---|
| `phase`, `exit` | where the machine is, and why it stopped | each machine has its own enums |
| `series` (`MaskedSeries`, `include/ccd/input/masked_series.hpp`) | the processing mask and the observations it keeps, always in step | shared |
| `cursor` (`Cursor`, `src/ccd/machine/machine.hpp`) | the model window and `prev_break`, in masked coordinates | shared |
| `segments`, `carried` | the segments this run wrote, and the count of a resumed run's stored final ones before them; "no segment yet" is `segments.empty() && carried == 0` | shared |
| `suspended` (`Suspended`) | where the segment loop stopped: what a resume continues from (section 6) | Standard's |
| `cadence` (`Cadence`) | the peek size the change test chose for the pixel (`prepare()`), which the stability rule and the lookback are prepared with | Standard's |
| `variogram` (`Variogram`, `include/ccd/strategies/stages.hpp`) | the per-band noise floor of the statistics window | a shared type, computed by Standard |
| `initialize_criteria` | what a candidate start window is tested against | a shared type, filled by Standard |
| `init_model`, `stability`, `lookback` | the stability fit's `Model` (the stable start window's fits, which Lookback tests against), and the stability rule's and the lookback's per-pixel scratch, each a `PerThread` of its strategy | Standard's |
| `extend` (`ExtendState`) | what Extend carries from one peek test to the next | Standard's |

`MaskedSeries` is the only writer of the mask. Its two drops, `drop` (Lookback, Extend) and
`drop_outliers_if` (Initialize's tentative Tmask removal), move the caller's window with the
observations by one rule: a window bound above the dropped position moves down by one.

The record's buffers keep their capacity across pixels: `State::reset` starts a pixel without
freeing anything. `standard::Procedure` owns one record, `standard::Detector` owns one
`standard::Procedure`, and a `Worker` (`src/ccd/run/run_pixel.hpp`) holds one `Detector` per
OpenMP thread.

### Stages and strategies

The machine's phases and routes are fixed; what a phase decides is a **stage**, decided by the
**strategy** the run's configuration names for it (`include/ccd/strategies/stages.hpp`). A
strategy returns its decision and the machine routes on it; a strategy never steers the machine.
The run's `ccd::Algorithm` builds its strategies once per input, through the registry
(`include/ccd/config/registry.hpp`, `ccd::strategies::library_registry()`), from a configuration
of shared values and one `{strategy, parameters}` per stage resolved against the input, and the
run hands them to the machine in `Context::strategies`. Each strategy declares its parameters
once, as a schema (`include/ccd/config/params.hpp`), which checks a configuration's values
(unknown names, types, ranges) when the `Algorithm` is built, and says where each value comes
from: the stage's own, a shared value, or the chi-squared quantile of a shared probability. A
parameter is a flag, a whole number, a real, a name or a band list (roles, `config::roles({"green",
"swir1"})`, or rows, `config::bands({1, 4})`).

| Stage | Decides | Strategies |
|---|---|---|
| `mode` | the pixel's procedure from its QA statistics: change detection (Standard), or one curve over the whole series (PermanentSnow, InsufficientClear); with what it asks of a Standard pixel's detection (`PixelMode`: the green median filter before it, a cap on its segments' coefficients) | `pyccd`: Standard at `clear_pct_threshold` clear, else PermanentSnow at `snow_pct_threshold` snow, else InsufficientClear |
| `noise_floor` | each band's noise floor from the statistics window, which every test scales by (a NaN ends the pixel) | `variogram`: the first-order variogram at the first lag whose most common separation exceeds `min_lag_days` |
| `start_search` | where the search for a stable start begins (`first()`), whether a series has too few observations, whether there is room to start, a search's plan (`begin()`: the last day it sees, what it does exhausted, end the segment loop, end it with the window screen's drops given back, or force a start, `forced()`, and whether a resume may continue it), when no candidate is left, whether a candidate spans too few days, and whether one the window screen shortened is still a candidate | `pyccd`: candidates of `meow_size` observations needing `meow_size` more after them, spanning `day_delta` days; a search sees the whole series, and an exhausted one ends the loop |
| `window_screen` | the outliers of a candidate start window, which Initialize drops before fitting it | `tmask`: pyccd's Tmask, a robust annual-harmonic fit of `bands` (pyccd: green and swir1), an observation an outlier above `t_const` variograms |
| `stability_fit` | the fit of a candidate start window's detection bands, which the stability rule judges and Lookback tests against (a `Model`), and its coefficients (`coefficients()`) | `lasso`: the monitoring model's Lasso, `coefficient_min` coefficients |
| `stability_rule` | whether that candidate is stable, against a threshold it derives per pixel (`prepare()`, from the change test's `PeekPlan`) | `pyccd`: over the detection bands, the sum of squares of (\|slope × span\| + \|first residual\| + \|last residual\|) / max(variogram, RMSE) below `change_threshold`, adjusted for the peek as chi2's is (with `peek` N, for a peek of N, unadjusted) |
| `lookback` | after a stable start, whether the observation before the window joins it, is dropped or is a change, testing the observations before the window against the stability fit (`prepare()` for the pixel's peek; `judge()` per step) | `pyccd`: `lookback_peek()`'s observations, the statistic against the stability fit's RMSE, a change above the threshold (adjusted for the peek), an outlier above `outlier_threshold`; with `peek` N, a peek of N |
| `model` | the fits Extend tests each peek against: each band's fit on the model window (`fit()`, which may fail, and `extend()` on a window that grew at its end, which a model may fit from the new observations alone), its peek residuals, RMSE and in-window residuals, the segment's coefficients, and how many coefficients a window's fit has (`coefficients()`, a `CoefficientRule`) | `lasso`: pyccd's Lasso on the harmonic basis; `max_iter`, `alpha`, `tolerance`, `coef_floor` are the solver's, `coefficient_min`, `_mid`, `_max` and `num_obs_factor` the rule's (pyccd's 4, 6, 8 at 3 observations per coefficient) |
| `refit` | whether Extend refits the model before a peek test (after the first), from the window's size and span now and at the last refit | `span-factor`: refit while the window holds at most `min_obs` observations, then whenever it spans `factor` times the last fit's span (pyccd's 24 and 1.33) |
| `change_test` | the pixel's peek plan and anything else it derives per pixel (`prepare()`: the peek size, whether it tests only full peeks, the fewest observations it tests a window of), whether an Extend peek is a change (`test()`: `Change`, `NoChange`, or `Undecided` on a shortened peek; writing the peek's comparison RMSE and each observation's statistic into the `PeekResult`, which the outlier rule reads), and the segment's break observation and per-band magnitudes (`break_at()`, `magnitude()`) | `chi2`: pyccd's test, of full peeks only, every window, every statistic above the threshold (adjusted for the pixel's peek); the segment breaks at the last test's first peek observation, each band's magnitude the median of its peek residuals; the comparison RMSE is the model's up to `model_rmse_up_to` observations, then the seasonal RMSE of `seasonal_count` observations over `seasonal_divisor` |
| `outlier` | after a peek that is not a change, whether its first observation is an outlier, which Extend drops, or the peek too short to tell (`EndSeries`) | `floored`: pyccd's rule, the change test's statistic for that observation (noise floored at the variogram) above `threshold` |
| `segment_fit` | the coefficients each band of a segment Extend grew reports, and its coefficient count (`segment()`, under the mode's cap); and a forced start's own segment when the series ends before Extend fits it (`unfitted()`) | `model-fit`: the monitoring model's, as its last refit left them; refitted on the segment's window with the capped count below a cap |
| `edge_fit` | whether the Start and End segments are written (`start_due()`, `end_due()`), their fits (`catch_window()`) and the whole-series curve (`whole_series()`) | `lasso`: every band a Lasso fit of `coefficients` coefficients (pyccd: 4); no whole-series segment under `min_observations` kept observations (pyccd: 12); the Start and End segments when more than a peek of observations is before the first stable start, or after the last segment |

A run's configuration is `RunSpec::algorithm`, by default the registry's `ccdc` preset
(`strategies::library_registry().preset("ccdc")`, written out in `python/ccdcpp/configs/ccdc.toml`):
each core strategy above at its defaults, in physical terms. Where pyccd has one constant, the
configuration holds it once, as a shared value (`include/ccd/config/shared.hpp`) the stages
read, or derives it from one:

| pyccd's constant | Where the configuration holds it |
|---|---|
| the start window, 12 observations | shared `min_observations`: `start_search`'s `meow_size`, `edge_fit`'s `min_observations` |
| the change probability, 0.99 | shared `change_probability`: `change_probability` of `change_test`, `stability_rule` and `lookback` |
| the change threshold, chi2(5) at 0.99 | derived from it: `change_threshold` of `change_test`, `stability_rule` and `lookback` |
| the outlier threshold, chi2(5) at 1 - 1e-6 | derived from shared `outlier_probability`: `outlier`'s `threshold`, `lookback`'s `outlier_threshold` |
| the coefficient counts, 4, 6 and 8 | shared `coefficient_min`, `_mid`, `_max`: `model`'s and `stability_fit`'s; `edge_fit`'s `coefficients` (4) |
| the Lasso's constants | shared `max_iter`, `alpha`, `tolerance`, `coef_floor`: `model`'s, `stability_fit`'s and `edge_fit`'s |
| the refit and model-RMSE window, 24 observations | two parameters, `refit`'s `min_obs` and `change_test`'s `model_rmse_up_to`, which a configuration may set apart |

The stability rule's and the lookback's `peek` is 0, the change test's peek; N gives them a peek
of N, with their `change_threshold` for it, unadjusted. The machine's other decisions are made
inside its phases. The peek size the change test chooses is also what the machine's guards and
the stability rule's and lookback's `prepare()` read.

**The routes other strategies take.** The core strategies take the routes of section 2; the
hooks above give other strategies routes of their own, all on the same machine:

- A change test that reads shortened peeks (`PeekPlan::full` false) has no peek guard: Extend runs
  while observations follow the window, its peek the next `peek` of them or fewer, and the curve
  ends where none is left (its break day the window's last observation) or where the test is
  `Undecided`. A window smaller than its `PeekPlan::min_window` grows untested.
- A change's `break_at()` may be past the peek's first observation: the segment breaks there, and
  the peek's observations before it leave the series after the segment is written.
- The outlier rule's `EndSeries` ends the curve at the window, as a test's `Undecided` does.
- A model whose fit fails (`fit()` false) ends the series without the curve: Done, after a forced
  start's own segment if one is pending.
- A start search's exhausted search may give back what its window screen dropped (`Restore`), or
  force a start (`Force`): the window `forced()` names, with no Lookback and no Start segment,
  straight to Extend. A series that ends before Extend fits it reports the segment fit's
  `unfitted()` segment of that window, through the last observation.
- The mode's `PixelMode` may add mask steps after the procedure's chain (the run applies them
  before the machine runs), and cap its segments' coefficients (the segment fit refits under the
  cap; the detection itself is not capped).
- A run that takes any of these routes is not resumable (`State::resumable`): its record stores
  no suspension, and a resume fits the pixel again.

```cpp
// A change test of shortened peeks, starting at windows of 12 observations or more.
struct Shortened final : ccd::ChangeTest
{
    ccd::PeekPlan prepare(...) const override { return ccd::PeekPlan{18, false, 12}; }
    // ... test() (Undecided when a shortened peek decides nothing), break_at(), ...
};
```

At an Extend peek every stage reads one `Peek` (the model, the windows, the peek's residuals,
the variogram and the detection bands), and the change test writes what it found into a
`PeekResult` (the comparison RMSE and its per-observation statistics by name, chi2's
`change_magnitude`), which the outlier rule reads, selecting a statistic by name (floored's
`statistic`). What each strategy provides and needs is declared as `Capabilities`
(`include/ccd/strategies/capabilities.hpp`: the model's `peek_residuals`, `model_rmse` and
`in_window_residuals`; the change test's `comparison_rmse`) and as statistic names
(`ChangeTest::statistics()`, `OutlierRule::statistics()`), and the registry refuses a
configuration in which a strategy needs what no earlier stage's strategy provides:

```cpp
// A model strategy that provides nothing, registered as "opaque", with chi2:
// std::invalid_argument: change_test 'chi2' needs peek_residuals, model_rmse,
//   in_window_residuals, which model 'opaque' does not provide
```

A stage whose strategy keeps state from one call to the next (the model's fits; the test's
per-pixel threshold and seasonal order) makes one per thread (`make_model()`, `make_scratch()`),
which the State keeps in a `PerThread` (`include/ccd/strategies/per_thread.hpp`) and makes
again only when the run's strategy is another.

The monitoring model's contract is `ccd::Model` (`include/ccd/strategies/model.hpp`): one
thread's fits of one pixel, which the strategy makes (`make_model()`) and the thread keeps across
pixels. Extend asks it to `fit` the window's detection bands (the other bands when the segment is
written), for the `peek_residuals` of each test, for each band's `rmse` and in-window `residuals`
(the comparison RMSE), and for each band's `band_fit` when the segment is written. A fit and a peek take
all the bands they act on and a band's RMSE, residuals and segment fit are one call each, so a
step makes a virtual call per band at most, never one per value.

A configuration, its strategies, and a strategy's decision (`tests/ccd/strategies/test_registry.cpp`
has more):

```cpp
#include "ccd/run/run.hpp"

using namespace ccd;

// The configuration a run uses by default: the shared values, the stages and the
// preprocessing, in physical terms (config::Config, config/registry.hpp).
config::Config config = strategies::library_registry().preset("ccdc");
// config.shared holds detection_bands {"green", "red", "nir", "swir1", "swir2"},
//   change_probability 0.99, outlier_probability 0.999999, alpha and coef_floor 1e-4
//   (reflectance), max_iter 1000, tolerance 1e-4, the coefficient rule 4/6/8/3,
//   min_observations 12 and statistics_end NO_STATISTICS_END; the stages that read them hold no copy.
// config.stages["start_search"] is {"pyccd", {{"day_delta", 365}}}: its meow_size is the
//   shared min_observations.
// config.stages["window_screen"] is {"tmask", {{"bands", {"green", "swir1"}}, {"t_const", 4.89}}}
// config.stages["refit"] is {"span-factor", {{"min_obs", 24}, {"factor", 1.33}}}
// config.stages["outlier"] is {"floored", {{"statistic", "change_magnitude"}}}: its
//   threshold is the chi-squared quantile of the shared outlier_probability.

// Start windows of 18 observations spanning two years.
config.shared["min_observations"] = config::integer(18);
config.stages["start_search"].params["day_delta"] = config::integer(730);

// Tmask over green, red and swir1, and a looser outlier cut.
config.stages["window_screen"].params["bands"] = config::roles({"green", "red", "swir1"});
config.stages["window_screen"].params["t_const"] = config::real(6.0);

// Every change test at 0.95 instead of 0.99, and fewer outliers (0.99999).
config.shared["change_probability"] = config::real(0.95);
config.shared["outlier_probability"] = config::real(0.99999);

// Refit less often: once the window spans twice the last fit's span.
config.stages["refit"].params["factor"] = config::real(2.0);

// A run under it: checked when the Algorithm is built, resolved against the input (units,
// rows, thresholds, the preprocessing plan) once, and part of its resume records' digest.
RunSpec spec;
spec.algorithm = std::make_shared<const Algorithm>(config);
PixelOutput run = detect(spec, dates, spectral, qas);

// A value out of range is refused when the Algorithm is built:
config.stages["refit"].params["factor"] = config::real(0.0);
Algorithm bad(config);
// std::invalid_argument: refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
```

### Configurations and extensions

A run's configuration is `RunSpec::algorithm`, a compiled `config::Config`. It names its
strategies and preprocessing steps in
`ccd::strategies::library_registry()`: the library's own (`register_core_strategies()`, which
also registers the `ccdc` preset) and those of the extensions it was built with, which may
register presets and preprocessing step types of their own (`Registry::add_preset()`; `preset()`, `presets()`; `Registry::add_step()`, a step beyond the built-in kinds being a `preprocess::CustomStep`). An extension is a
directory in CMake's `CCD_EXTENSIONS` whose `extension.cmake` adds its sources, include
directories and registration functions (`CCD_EXTENSION_SOURCES`, `CCD_EXTENSION_INCLUDE_DIRS`,
`CCD_EXTENSION_REGISTRATIONS`); the library calls each registration function when the registry
is first built (`src/ccd/strategies/register_extensions.cpp.in`). A public build has none; the
registry's listing is `library_registry().names()`.

## 4. Shared functions and Standard's own

What any change-detection machine can use, each taking narrow arguments and never a machine's
record:

| Function or type | File |
|---|---|
| `PixelProcedure`, `PROCEDURES`, `name()` | `include/ccd/input/procedure.hpp` |
| `PixelMode` | `include/ccd/machine/pixel_procedure.hpp` |
| `MaskedSeries` | `include/ccd/input/masked_series.hpp` |
| `Cursor`, `drive` | `src/ccd/machine/machine.hpp` |
| `Window`, `span`, `count_through` | `include/ccd/core/window.hpp` |
| `ChangeModel`, `BandFit`, `FitResult`, `CurveQA` | `include/ccd/results/segment.hpp` |
| `Variogram`, with the stage interfaces | `include/ccd/strategies/stages.hpp` |
| `KeepResiduals`, with the `Model` contract | `include/ccd/strategies/model.hpp` |
| `HarmonicBasis` (the harmonic design, column order documented) | `include/ccd/regression/harmonic_basis.hpp` |

Each strategy holds its own code, beside its class:

| Function or type | Owner | File |
|---|---|---|
| `fit_bands` (the one per-band fit loop), `peek_residuals`, `band_fit` | the Lasso model and stability fit | `include/ccd/strategies/model/lasso.hpp` |
| `catch_segment`, `fit_whole_series` | the Lasso edge fit | `include/ccd/strategies/edge_fit/lasso.hpp` |
| `adjusted_variogram` | the variogram noise floor | `include/ccd/strategies/noise_floor/variogram.hpp` |
| `tmask` | the Tmask window screen | `include/ccd/strategies/window_screen/tmask.hpp` |
| `adjust_peek`, `adjust_change_threshold`, `ThresholdPeek`, `ChangeThresholdMemo`, `change_magnitude`, `detect_change`, `detect_outlier`, `SeasonalOrder`, `seasonal_rmse` | pyccd's peek statistics, read by the pyccd strategies of four stages (`chi2`, the `pyccd` stability rule and lookback, `floored`) | `include/ccd/strategies/pyccd/peek_statistics.hpp` |

What only the Standard machine uses lives with it, in `ccd::standard`
(`src/ccd/machine/standard.hpp`): `initialize_criteria`, `initialize_step` with
`InitializeCriteria`, `InitializeStages` and `InitializeOutcome`, the phases, the transitions
and the routing. A function moves to the shared list once a second machine or a strategy uses
it.

## 5. The driver and the trace

`ccd::drive<Machine>` (`src/ccd/machine/machine.hpp`) is the only loop:

```cpp
while (state.phase != Phase::Done)
{
    const Phase next = Machine::step(context, state);
    // with an observer (CCD_OBSERVE): context.observe.observer->event(pixel, the transition)
    LOG_DEBUG(Machine::name << ": " << to_string(state.phase) << " -> "
              << to_string(next) << " | " << summary(state));
    state.phase = next;
}
```

`standard::Machine::step` is a `switch` from the phase to its transition, and `standard::run`
is reset, drive, take the result. A machine supplies its types, its `step`, a name, and
`to_string(Phase)` and `summary(State)` for the trace.

An observer (`RunSpec::observer`, [diagnostics.md](diagnostics.md)) is told each transition with
the cursor as the step left it. The log trace is one DEBUG line per transition through the
library's logger, so the log level is its only switch; below DEBUG (the default) it compiles to nothing. A build with
`-DMAX_LOG_LEVEL=0` prints, for each pixel, lines such as:

```
[DEBUG] Initialize: stable
[DEBUG] standard: Initialize -> Lookback | window (18, 92), prev_break 13, 512 observations, 1 segments, exit None
[DEBUG] Lookback: included
[DEBUG] standard: Lookback -> Lookback | window (17, 92), prev_break 13, 512 observations, 1 segments, exit None
[DEBUG] Lookback: change
[DEBUG] standard: Lookback -> Extend | window (17, 92), prev_break 13, 512 observations, 1 segments, exit None
```

The outcome line comes from the transition, the arrow line from `drive`, after the step: the
summary shows the record as the decision left it.

## 6. Resuming a stopped machine

The machine reads the length of the series only in its size guards, each a comparison of the
size with a position plus a count, and observations added after the end of the series turn no
guard that held into one that fails. So a run on the observations up to a date and a run on a
longer series make the same decisions up to the first size guard that failed in the shorter
one (`PrefixStability.*`, `tests/ccd/golden/test_prefix.cpp`). Every segment written before
that guard is final.

That guard is where the record stops: `standard::Suspended` (`include/ccd/machine/suspended.hpp`) is captured by
every route into EndCatch and by the early exits, before anything after the guard writes. It
holds the exit, the cursor, the fit window's end, a pending grow, the last peek test's
coefficient count and magnitudes, and the final segment count. `ccd::detect()` with `RunSpec::record`
returns it inside a `PixelRecord`, with the pixel's input fingerprint and the days the machine dropped
(`include/ccd/results/record.hpp`).

`standard::resume()` continues from it on the longer series:

1. Prepare runs as for any run, on the series as the filters leave it, with the statistics
   cutoffs capped at the first run's series end, so its cadence, variogram and criteria are the
   stored run's.
2. The stored drops leave the series, the latest first.
3. The record is checked against the series, then the cursor is put back and the guard that
   failed is tested again: Initialize for NoRoomToInitialize and InitializeExhausted, the peek
   guard for PeekBeyondEnd, and for SeriesEnd the open segment's models refitted on its fit
   window (the same fit, bit for bit), then its pending grow and Extend, or the same
   provisional segment again when the series still ends there.

`State::carried` counts the stored final segments, which come before the new ones, so
StartCatch stays the first segment's. A resume (`detect()` or `detect_cube()` with the stored
run) keeps the stored final segments as given
and returns the complete result, equal bit for bit to the run on the whole series
(`ResumeEqualsTheFullRun.*`, `tests/ccd/golden/test_resume.cpp`). A pixel whose stored input
changed, the pixels that take no machine and the early exits are rerun instead.

## 7. Why it is written this way

pyccd (`ccd/procedures.py`) writes the same algorithm as nested loops: `standard_procedure`
holds an outer `while` over the series, and `initialize`, `lookback` and `lookforward` each hold
an inner `while` of their own. That shape comes from reading the algorithm as a sequence of
phases, each run to completion. Read as a state machine, the algorithm is flat: the inner loops
are states that repeat, and the outer loop is the transition back to Initialize. pyccd's shape
leaves the machine implicit:

- a transition is a `continue` or a `break`, so which state comes next is read off control flow;
- states are sentinel values: `models is None` means "not yet trained" in one function and
  "initialization failed" in another;
- the `start` flag duplicates "no segment yet";
- the per-pixel peek size and change threshold are written into the parameter object and read
  back by name inside every function;
- every function rebuilds the masked dates and spectral from the mask on entry, and the window's
  meaning after a removal is explained in comments rather than kept by one operation;
- results travel back up as tuples, so each function is a fragment its caller reassembles.

In ccdcpp each of these is a named thing: a transition returns its next phase; the record has
`extend.trained` and `segments`; `Cadence` holds the per-pixel values and the configuration is
const; `MaskedSeries` keeps the mask and the kept observations in step and owns the coordinate
rule; the result is taken from the record once the machine is done.

### Testability per transition

A test builds a `State` by hand (a short synthetic series, a chosen cursor, models from
`fit_bands`), calls one transition or one step function, and checks the outcome, the next phase
and the record. `tests/ccd/machine/test_machine.cpp` does that for every row of the guard
lists above (`StandardInitialize.*`, `StandardLookback.*`, `StandardExtend.*`), for the routing
(`StandardRouting.*`, `StandardPrepare.*`) and for every `Exit` (`StandardExit.*`).
`StandardMachine.EachStepIsOneDecision` steps whole pixels by hand and checks that every
self-transition changes the window by one decision. The outputs themselves are pinned byte for
byte by the golden fixtures (`tests/ccd/golden/`), including the transitions fixture, built from
pixels that take the transitions the random corpus misses. `PrefixStability.*`
(`tests/ccd/golden/test_prefix.cpp`) holds the machine to its causality: a run on the
observations up to a date keeps every final segment of the run on the whole series.

### Variants touch one transition

Each rule where ccdcpp keeps pyccd's intent over its code sits in one function, so changing one
is that function and its test (the numbers are those of the tests and the CHANGELOG):

| Rule | Where it is |
|---|---|
| 1. Far from the previous break, Lookback peeks the adjusted peek of observations | `strategies::lookback_peek`, first case |
| 2. Near it, Lookback peeks down to the previous break, never below | `strategies::lookback_peek`, second case |
| 3. Lookback compares against the stable start window's RMSE | the `pyccd` lookback's `judge()` |
| 4. Initialize fits the stability fit's `coefficient_min` coefficients; both peek bases are `coefficient_max` wide | `initialize_criteria` (from the stability fit's `CoefficientRule`: `max_coefficients`, which Lookback's peek reads), and Extend's monitoring model (its rule's `max` as its fits' basis width) |
| 5. Every exit, the early ones too, returns the processing mask | `State::take_result`; `fit_whole_series` |
| 6. A curve ends once its peek reaches the last observation, and is the last curve | `extend_step`'s `SeriesEnd`; `extend` |
| 7. Extend refits while the window holds at most the refit's `min_obs` observations | `extend_step`'s refit schedule |

A sensor variant works the same way: a time-based start window or refit rule would change
Initialize's and Extend's guards and nothing else.

### Another procedure

Another change-detection procedure, one that replaces Standard's change test, say, and keeps its
initialization, is a configuration of the same machine: its strategies for the stages it
changes (an extension's, `CCD_EXTENSIONS`), registered by name, and the routes of section 3's
"The routes other strategies take". It needs no machine or detector of its own; a run
names it in `RunSpec::algorithm`, and its suspension is kept only while it takes routes a resume
continues (`State::resumable`).

A segment reports each band's fit as a `BandFit` (`ChangeModel::bands`): intercept,
coefficients, RMSE, magnitude and the solver's iteration count (0 for a fit in closed form), so
a detector that fits by ordinary least squares writes the same record. Inside the Standard
machine Extend's fits are the monitoring model's (`ccd::Model`), whose in-window residuals the
seasonal RMSE reads and whose `band_fit()` gives each band's when a segment is written.

### A later batch on the CPU

Because every transition is one step on a plain record, a driver could hold many pixels and
advance them phase by phase, for example doing every pixel's Extend peek prediction together,
or fitting many windows at once with a Lasso solver vectorised across pixels. Each pixel's
arithmetic would keep its order, so the outputs could stay byte-identical. Pixels in different
phases limit the gain. On a 500 x 500 Landsat cell, Extend steps take about 80% of the
per-phase thread time and Initialize steps about 16%, so those two are where a batch would have
to pay. Nothing here is built; it would need a structure-of-arrays layout for the record.
