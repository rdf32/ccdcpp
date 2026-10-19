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
once per call (`include/ccd/run/run.hpp`): it checks the options, dates, QA and plan and holds
what every pixel shares; each OpenMP thread holds a `Worker` with the buffers and the change
detector it reuses from pixel to pixel. The steps of one pixel:

1. decode the QA (through a `QaTable` when one is given);
2. compute the QA statistics of the observations dated up to `STAT_ORD`
   (`Quality::compute_until`, `include/ccd/input/quality.hpp`);
3. choose the pixel's procedure: the run's `mode` stage (`ModeRule::choose`,
   `include/ccd/strategies/stages.hpp`);
4. apply that procedure's transform to the pixel's private copy and run its filter chain to
   build the processing mask (`include/ccd/input/mask/preprocess.hpp`);
5. fit the mask with the run's change detector, `standard::Detector::fit`
   (`include/ccd/machine/standard.hpp`).

The procedure choice:

| Guard, in order | Procedure | What fits it |
|---|---|---|
| clear percentage >= `CLEAR_PCT_THRESHOLD` | Standard | the run's method (`HarmonicOptions::METHOD`): `ccdc`, the Standard machine (section 2) |
| else snow percentage >= `SNOW_PCT_THRESHOLD` | PermanentSnow | `ccd::fit_whole_series`, curve_qa 54 |
| else | InsufficientClear | `ccd::fit_whole_series`, curve_qa 44 |

The whole-series fit has no loop, so it is a function, not a machine. With fewer than
`MEOW_SIZE` kept observations it returns no segment, with the processing mask. Otherwise it returns one segment over the
whole series, every band fitted with `COEFFICIENT_MIN` coefficients (the `lasso` edge fit's
`coefficients`; its `min_observations` is `MEOW_SIZE`).

## 2. The Standard machine

```
                              ┌─ size <= MEOW_SIZE ───────────────────────► Done  [TooFewObservations]
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
       └─── room ──── Restart: prev_break = window.stop, window = (stop, stop + MEOW_SIZE)
                          │ no room
                          ▼
                      EndCatch [NoRoomToInitialize] ── the End segment, if prev_break + peek < size ──► Done
```

`MEOW_SIZE` and `DAY_DELTA` here are the `pyccd` start search's `meow_size` and `day_delta`
(the `start_search` stage, section 3): the machine asks the run's start search where a candidate
is, whether it has room and whether the search is exhausted.

Bracketed names are the `Exit` value the record keeps: why the machine stopped where it did.
The exit is internal; the trace and the tests read it, the result does not carry it.

### Phases and their guards

Each row is one transition function in `src/ccd/machine/standard.cpp`, declared in
`include/ccd/machine/standard.hpp`. The three phases that repeat themselves do one thing per
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

- **Initialize** (`initialize_step`, `src/ccd/machine/kernels.cpp`): the start search finds
  no candidate left (`Exhausted`) or a span too short (`ShortSpan`, grow; pyccd: under
  `DAY_DELTA`); the window screen (Tmask) flags
  every observation (`AllOutliers`, grow); it flags some and the shortened window is not a
  candidate (the search's `usable()`)
  (`RejectedOutliers`, grow, nothing dropped); then the detection bands are fitted by the
  stability fit and judged by the stability rule (`Stable`, or `Unstable` and a shift). When Tmask's outliers are dropped, the
  same call fits the shortened window, as pyccd does in one pass of its loop; those steps report
  `DroppedThenStable` and `DroppedThenUnstable`.
- **Lookback** (`lookback_step`): the lookback strategy's verdict on the observation before the
  window, which the machine applies: `Change` (the window stays), `Drop` (`Dropped`, the
  observation leaves the series) or `Include` (`Included`, `window.start - 1`). pyccd's: every
  peek statistic above the threshold is a change, the first above `OUTLIER_THRESHOLD` an
  outlier. Its peek window is `lookback_peek`'s: far from the previous break, the adjusted peek of observations
  before the window; near it, down to the previous break and never below. The comparison RMSE
  is the stable start window's: nothing is refitted while the window grows backward, as in
  Zhu's TrendSeasonalFit.
- **Extend** (`extend_step`): first the refit schedule (refit when untrained, at most
  `RETRAIN_MIN_OBS` observations, or once the window spans `RETRAIN_SPAN_FACTOR` times the last
  fit); then the change test (`Change`); the outlier rule (`Dropped`, the observation at
  `window.stop` leaves the series; pyccd's: its statistic above `OUTLIER_THRESHOLD`); a peek that
  reached the last observation (`SeriesEnd`: the curve ends at the window, before the
  observation the peek starts at, its break day); otherwise `Grown` (`window.stop + 1`). The
  curve that reaches the series end, by `SeriesEnd` or by a drop that leaves no full peek, is
  the last one: EndCatch follows, not Restart. A refit fits the detection bands, all a test
  reads; the other bands are fitted on the last refit's window when the segment is
  written (`complete_models`; before the drop when a drop ends the curve), which gives the
  models and peek residuals an every-band refit would have.

Initialize fits `COEFFICIENT_MIN` coefficients, Extend's fits follow `coefficient_count`
(`COEFFICIENT_MIN`, `_MID` or `_MAX` by window length), and both peek bases are `COEFFICIENT_MAX`
wide: a model predicts with zeros above its own count. Extend's fits are the monitoring model's
(the `model` stage, below), which the extension holds.

The routing between phases is a set of named predicates and routers, so a guard has one place:
`room_to_initialize`, `peek_fits`, `start_catch_due`, `after_stable_start`, `after_lookback`
and `peek_guard`. Every route into EndCatch records its `Exit`.

The transitions assert the invariants the algorithm relies on: Initialize's window holds at
least the start search's `min_window()` observations; Lookback runs only while `window.start > prev_break`; Extend runs
only when a full peek fits.

## 3. The per-pixel record and the context

A transition is a function of two arguments, `(const Context&, State&)`. What it may change is
in the `State`. What it reads but never changes is in the `Context`.

**`standard::Context`** holds the caller's `HarmonicOptions` (never written below
`Run::pixel`), the pixel's `HarmonicWorkspace` (its dates, spectral and QA), the Lasso
solver with its workspace, and the run's strategies. The catch segments (StartCatch, EndCatch)
fit with the context's solver; Initialize fits with the stability fit's own and Extend with the
monitoring model's own (the `stability_fit` and `model` stages' parameters), and Lookback reads
the stability fit. A direct caller of `standard::Procedure` builds all of them from the same
`LassoOptions` (`strategies_for(hoptions, loptions)` and `LassoSolver(loptions)`), as `Run` does.

**`standard::State`** is one pixel's record:

| Field | What it is | Shared or Standard's |
|---|---|---|
| `phase`, `exit` | where the machine is, and why it stopped | each machine has its own enums |
| `series` (`MaskedSeries`, `include/ccd/input/masked_series.hpp`) | the processing mask and the observations it keeps, always in step | shared |
| `cursor` (`Cursor`, `include/ccd/machine/machine.hpp`) | the model window and `prev_break`, in masked coordinates | shared |
| `segments`, `carried` | the segments this run wrote, and the count of a resumed run's stored final ones before them; "no segment yet" is `segments.empty() && carried == 0` | shared |
| `suspended` (`Suspended`) | where the segment loop stopped: what a resume continues from (section 6) | Standard's |
| `cadence` (`Cadence`) | the peek size the change test chose for the pixel (`prepare()`), which the stability rule and the lookback are prepared with | Standard's |
| `variogram` (`Variogram`, `include/ccd/machine/fit.hpp`) | the per-band noise floor of the statistics window | a shared type, computed by Standard |
| `initialize_criteria` | what a candidate start window is tested against | a shared type, filled by Standard |
| `init_model`, `stability`, `lookback` | the stability fit's `Model` (the stable start window's fits, which Lookback tests against), and the stability rule's and the lookback's per-pixel scratch, each a `PerThread` of its strategy | Standard's |
| `extension` (`Extension`) | what Extend carries from one peek test to the next | Standard's |

`MaskedSeries` is the only writer of the mask. Its two drops, `drop` (Lookback, Extend) and
`drop_outliers_if` (Initialize's tentative Tmask removal), move the caller's window with the
observations by one rule: a window bound above the dropped position moves down by one.

The record's buffers keep their capacity across pixels: `State::reset` starts a pixel without
freeing anything. `standard::Procedure` owns one record, `standard::Detector` owns one
`standard::Procedure`, and a `Worker` (`include/ccd/run/run.hpp`) holds one `Detector` per
OpenMP thread.

### Stages and strategies

The machine's phases and routes are fixed; what a phase decides is a **stage**, decided by the
**strategy** the run's configuration names for it (`include/ccd/strategies/stages.hpp`). A
strategy returns its decision and the machine routes on it; a strategy never steers the machine.
The run builds its strategies once per call, through the registry
(`include/ccd/config/registry.hpp`, `ccd::strategies::core_registry()`), from a configuration of
one `{strategy, parameters}` per stage, and hands them to the machine in `Context::strategies`.
Each strategy declares its parameters once, as a schema (`include/ccd/config/params.hpp`), which
checks a configuration's values (unknown names, types, ranges) when the strategy is built. A
parameter is a flag, a whole number, a real, a name or a band list (indices into the spectral
rows, `config::bands({1, 4})`).

| Stage | Decides | Strategies |
|---|---|---|
| `mode` | the pixel's procedure from its QA statistics: change detection (Standard), or one curve over the whole series (PermanentSnow, InsufficientClear) | `pyccd`: Standard at `clear_pct_threshold` clear, else PermanentSnow at `snow_pct_threshold` snow, else InsufficientClear |
| `noise_floor` | each band's noise floor from the statistics window, which every test scales by (a NaN ends the pixel) | `variogram`: the first-order variogram at the first lag whose most common separation exceeds `min_lag_days` (`VARIOGRAM_MIN_LAG_DAYS`) |
| `start_search` | where the search for a stable start begins (`first()`), whether a series has too few observations, whether there is room to start, when no candidate is left, whether a candidate spans too few days, and whether one the window screen shortened is still a candidate | `pyccd`: candidates of `meow_size` observations needing `meow_size` more after them, spanning `day_delta` days |
| `window_screen` | the outliers of a candidate start window, which Initialize drops before fitting it | `tmask`: pyccd's Tmask, a robust annual-harmonic fit of `bands` (pyccd: green and swir1), an observation an outlier above `t_const` variograms |
| `stability_fit` | the fit of a candidate start window's detection bands, which the stability rule judges and Lookback tests against (a `Model`) | `lasso`: the monitoring model's Lasso, `COEFFICIENT_MIN` coefficients |
| `stability_rule` | whether that candidate is stable, against a threshold it derives per pixel (`prepare()`, from the change test's peek size) | `pyccd`: over the detection bands, the sum of squares of (\|slope × span\| + \|first residual\| + \|last residual\|) / max(variogram, RMSE) below `change_threshold`, adjusted for the peek as chi2's is |
| `lookback` | after a stable start, whether the observation before the window joins it, is dropped or is a change, testing the observations before the window against the stability fit (`prepare()` for the pixel's peek; `judge()` per step) | `pyccd`: `lookback_peek()`'s observations, the statistic against the stability fit's RMSE, a change above the threshold (adjusted for the peek), an outlier above `outlier_threshold` |
| `model` | the fits Extend tests each peek against: each band's fit on the model window, its peek residuals, RMSE and in-window residuals, and the segment's coefficients | `lasso`: pyccd's Lasso on the harmonic basis; `max_iter`, `alpha`, `tolerance`, `coef_floor` are the solver's |
| `refit` | whether Extend refits the model before a peek test (after the first) | `span-factor`: refit while the window holds at most `min_obs` observations, then whenever it spans `factor` times the last fit's span (pyccd's 24 and 1.33) |
| `change_test` | the pixel's peek size and anything else it derives per pixel (`prepare()`), whether an Extend peek is a change (`test()`, writing the peek's comparison RMSE and each observation's statistic into the `PeekResult`, which the outlier rule reads), and the segment's break observation and per-band magnitudes (`break_at()`, `magnitude()`) | `chi2`: pyccd's test, every statistic above the threshold (adjusted for the pixel's peek); the segment breaks at the last test's first peek observation, each band's magnitude the median of its peek residuals; the comparison RMSE is the model's up to `model_rmse_up_to` observations, then the seasonal RMSE of `seasonal_count` observations over `seasonal_divisor` |
| `outlier` | after a peek that is not a change, whether its first observation is an outlier, which Extend drops | `floored`: pyccd's rule, the change test's statistic for that observation (noise floored at the variogram) above `threshold` |
| `segment_fit` | the coefficients each band of a segment Extend grew reports | `model-fit`: the monitoring model's, as its last refit left them |
| `edge_fit` | the Start and End segments' fits (`catch_window()`) and the whole-series curve (`whole_series()`) | `lasso`: every band a Lasso fit of `coefficients` coefficients (pyccd: `COEFFICIENT_MIN`); no whole-series segment under `min_observations` kept observations (`MEOW_SIZE`) |

A run's configuration is the one its options describe
(`ccd::config::from_options()`):

| Strategy | Parameter | From |
|---|---|---|
| `pyccd` (start search) | `meow_size`, `day_delta` | `MEOW_SIZE`, `DAY_DELTA` |
| `tmask` | `bands`, `t_const` | `TMASK_BANDS`, `T_CONST` |
| `lasso` (both stages) | `max_iter`, `alpha`, `tolerance`, `coef_floor` | `LassoOptions`' fields of those names |
| `pyccd` (stability rule) | `peek_size`, `change_threshold`, `change_probability` | `PEEK_SIZE`, `CHANGE_THRESHOLD`, `CHANGE_PROBABILITY` |
| `span-factor` | `min_obs`, `factor` | `RETRAIN_MIN_OBS`, `RETRAIN_SPAN_FACTOR` |
| `chi2` | `peek_size`, `peek_reference_days`, `change_threshold`, `change_probability` | `PEEK_SIZE`, `PEEK_REFERENCE_DAYS`, `CHANGE_THRESHOLD`, `CHANGE_PROBABILITY` |
| `chi2` | `model_rmse_up_to`, `seasonal_count`, `seasonal_divisor` | `RETRAIN_MIN_OBS`, `SEASONAL_RMSE_COUNT`, `SEASONAL_RMSE_DIVISOR` |
| `floored` | `threshold` | `OUTLIER_THRESHOLD` |
| `pyccd` (mode) | `clear_pct_threshold`, `snow_pct_threshold` | `CLEAR_PCT_THRESHOLD`, `SNOW_PCT_THRESHOLD` |
| `variogram` | `min_lag_days` | `VARIOGRAM_MIN_LAG_DAYS` |
| `lasso` (edge fit) | `coefficients`, `min_observations`, and the Lasso's four | `COEFFICIENT_MIN`, `MEOW_SIZE`, `LassoOptions` |
| `pyccd` (lookback) | `peek_size`, `change_threshold`, `change_probability`, `outlier_threshold` | `PEEK_SIZE`, `CHANGE_THRESHOLD`, `CHANGE_PROBABILITY`, `OUTLIER_THRESHOLD` |

`RETRAIN_MIN_OBS` sets two parameters, the refit's and the test's, as pyccd's one constant does;
a configuration may set them apart. The mapping lives in one file (`include/ccd/config/from_options.hpp`).
The machine's other decisions are made inside its phases. The peek size the change test chooses
is also what the machine's guards and the stability rule's and lookback's `prepare()` read.
Each of the change test, the stability rule and the lookback has its own `change_threshold` and
`change_probability`, and each of the outlier rule and the lookback its own outlier threshold, so
a configuration may set Extend's tests, Initialize's and Lookback's apart.

At an Extend peek every stage reads one `Peek` (the model, the windows, the peek's residuals,
the variogram and the detection bands), and the change test writes what it found into a
`PeekResult` (the comparison RMSE and the per-observation statistic), which the outlier rule
reads. What each strategy provides and needs is declared as `Capabilities`
(`include/ccd/strategies/capabilities.hpp`: the model's `peek_residuals`, `model_rmse` and
`in_window_residuals`; the change test's `comparison_rmse` and `peek_statistic`), and
`build_strategies` refuses a configuration in which a strategy needs what no earlier stage's
strategy provides:

```cpp
// A model strategy that provides nothing, registered as "opaque", with chi2:
// std::invalid_argument: change_test 'chi2' needs peek_residuals, model_rmse,
//   in_window_residuals, which model 'opaque' does not provide
```

A stage whose strategy keeps state from one call to the next (the model's fits; the test's
per-pixel threshold and seasonal order) makes one per thread (`make_model()`, `make_scratch()`),
which the extension keeps in a `PerThread` (`include/ccd/strategies/per_thread.hpp`) and makes
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
#include "ccd/config/from_options.hpp"
#include "ccd/strategies/register_core.hpp"

using namespace ccd;

// The configuration a run under the defaults uses, and its strategies.
config::ConfigTree tree = config::from_options(HarmonicOptions{}, LassoOptions{});
// tree["start_search"] is {"pyccd", {{"meow_size", 12}, {"day_delta", 365}}}
// tree["lookback"] is {"pyccd", {{"peek_size", 6}, {"change_threshold", 15.0863},
//                                {"change_probability", 0.99}, {"outlier_threshold", 35.8882}}}
// tree["window_screen"] is {"tmask", {{"bands", {1, 4}}, {"t_const", 4.89}}}
// tree["stability_fit"] and tree["model"] are {"lasso", {{"max_iter", 1000}, {"alpha", 1.0},
//                                           {"tolerance", 1e-4}, {"coef_floor", 1.0}}}
// tree["stability_rule"] is {"pyccd", {{"peek_size", 6}, {"change_threshold", 15.0863},
//                                      {"change_probability", 0.99}}}
// tree["refit"] is {"span-factor", {{"min_obs", 24}, {"factor", 1.33}}}
// tree["change_test"] is {"chi2", {{"peek_size", 6}, {"peek_reference_days", 16.0},
//                                  {"change_threshold", 15.0863}, {"change_probability", 0.99},
//                                  {"model_rmse_up_to", 24}, {"seasonal_count", 24},
//                                  {"seasonal_divisor", 4.0}}}
// tree["outlier"] is {"floored", {{"threshold", 35.8882}}}

// Start windows of 18 observations spanning two years.
tree["start_search"].params["meow_size"] = config::integer(18);
tree["start_search"].params["day_delta"] = config::integer(730);

// Tmask over green, red and swir1, and a looser outlier cut.
tree["window_screen"].params["bands"] = config::bands({1, 2, 4});
tree["window_screen"].params["t_const"] = config::real(6.0);

// Initialize's and Extend's fits for reflectance input: a 10,000 times smaller penalty
// (the catch segments' fits read LassoOptions until they are stages).
tree["stability_fit"].params["alpha"] = config::real(1e-4);
tree["model"].params["alpha"] = config::real(1e-4);

// A stricter start: stability below chi2(5) at 0.95 instead of 0.99, for the default peek
// (change_threshold) and a longer one (change_probability).
tree["stability_rule"].params["change_threshold"] = config::real(11.0705);
tree["stability_rule"].params["change_probability"] = config::real(0.95);

// Refit less often: once the window spans twice the last fit's span.
tree["refit"].params["factor"] = config::real(2.0);

// The change test compares against the model's own RMSE up to 40 observations (pyccd: 24),
// apart from the refit schedule's 24.
tree["change_test"].params["model_rmse_up_to"] = config::integer(40);

// Extend and Lookback drop fewer observations as outliers: chi2(5) at 0.99999 instead of
// pyccd's 0.999999.
tree["outlier"].params["threshold"] = config::real(30.856);
tree["lookback"].params["outlier_threshold"] = config::real(30.856);

Strategies s = strategies::build_strategies(strategies::core_registry(), tree);
s.refit->due(/*window_size=*/30, /*window_span=*/150.0, /*fit_span=*/100.0);  // false
s.refit->due(30, 200.0, 100.0);                                              // true

// A run under a configuration of its own (RunSpec::config): checked when the run is built,
// band lists against the input's bands, and part of its resume records' digest.
RunSpec spec{HarmonicOptions{}, LassoOptions{}};
spec.config = std::make_shared<const config::ConfigTree>(tree);
PixelRun run = run_pixel(spec, dates, spectral, qas);

// A value out of range is refused when the strategies are built:
tree["refit"].params["factor"] = config::real(0.0);
strategies::build_strategies(strategies::core_registry(), tree);
// std::invalid_argument: refit 'span-factor' parameter 'factor' is 0; it must be greater than 0
```

### Configurations and extensions

A run's configuration is the one its options describe unless `RunSpec::config` holds one. That
configuration names its strategies in `ccd::strategies::library_registry()`: the library's own
(`register_core_strategies()`) and those of the extensions it was built with. An extension is a
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
| `fit_whole_series`, `PixelProcedure` | `include/ccd/machine/pixel_procedure.hpp` |
| `MaskedSeries` | `include/ccd/input/masked_series.hpp` |
| `Cursor`, `drive` | `include/ccd/machine/machine.hpp` |
| `fit_bands` (the one per-band fit loop), `peek_residuals`, `change_magnitude`, `catch_segment`, `initialize_step` with `InitializeCriteria` and `InitializeOutcome` | `include/ccd/machine/kernels.hpp` |
| `HarmonicBasis` (the harmonic design, column order documented) | `include/ccd/regression/harmonic_basis.hpp` |
| `Variogram`, `adjusted_variogram`, `tmask`, `find_closest_doy`, `SeasonalOrder`, `seasonal_rmse`, `span`, `Window`, `ChangeModel`, `BandFit` | `include/ccd/machine/fit.hpp` |

What only the Standard machine uses lives with it, in `ccd::standard`
(`include/ccd/machine/standard.hpp`): `initialize_criteria`, `coefficient_count`, the phases,
the
transitions and the routing. A function moves to the shared list once a second machine or a
strategy uses it.

## 5. The driver and the trace

`ccd::drive<Machine>` (`include/ccd/machine/machine.hpp`) is the only loop:

```cpp
while (state.phase != Phase::Done)
{
    const Phase next = Machine::step(context, state);
    LOG_DEBUG(Machine::name << ": " << to_string(state.phase) << " -> "
              << to_string(next) << " | " << summary(state));
    state.phase = next;
}
```

`standard::Machine::step` is a `switch` from the phase to its transition, and `standard::run`
is reset, drive, take the result. A machine supplies its types, its `step`, a name, and
`to_string(Phase)` and `summary(State)` for the trace.

The trace is one DEBUG line per transition through the library's logger, so the log level is
its only switch; below DEBUG (the default) it compiles to nothing. A build with
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

That guard is where the record stops: `standard::Suspended` (`standard.hpp`) is captured by
every route into EndCatch and by the early exits, before anything after the guard writes. It
holds the exit, the cursor, the fit window's end, a pending grow, the last peek test's
coefficient count and magnitudes, and the final segment count. `ccd::run_pixel()` with `RunSpec::record`
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
StartCatch stays the first segment's. A resume (`run_pixel()` or `run_cube()` with the stored
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
`extension.trained` and `segments`; `Cadence` holds the per-pixel values and the options are
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
| 4. Initialize fits `COEFFICIENT_MIN` coefficients; both peek bases are `COEFFICIENT_MAX` wide | `initialize_criteria` (`max_coefficients`, the stability fit's `max_coef`, which Lookback's peek reads), and Extend's monitoring model (`COEFFICIENT_MAX` as its fits' `max_coef`) |
| 5. Every exit, the early ones too, returns the processing mask | `State::take_result`; `fit_whole_series` |
| 6. A curve ends once its peek reaches the last observation, and is the last curve | `extend_step`'s `SeriesEnd`; `extend` |
| 7. Extend refits while the window holds at most `RETRAIN_MIN_OBS` observations | `extend_step`'s refit schedule |

A sensor variant works the same way: a time-based start window or refit rule would change
Initialize's and Extend's guards and nothing else.

### Another procedure

Another change-detection procedure, one that replaces Standard's change test, say, and keeps its
initialization, is a machine of its own beside the Standard one:

- Its own namespace, with its own `Phase`, `Exit`, `State`, transitions and `Procedure` (the
  machine with its record, as `standard::Procedure` is Standard's). Nothing is inherited.
- It composes the shared parts: `MaskedSeries`, `Cursor`, the segments, `initialize_step` with
  its own `InitializeCriteria`, `HarmonicBasis`, and `drive` with its trace.
- It can hold a `standard::State` and call the Standard machine's work for the phases they share
  (`prepare`, `initialize_outcome`, `lookback_step`, `write_start_segment`, `move_past_segment`,
  `write_end_segment`), with routes of its own.
- Its detector is a sibling of `standard::Detector` with the same `fit` call, and it reuses
  the `mode` stage and the edge fit's `whole_series()` for the pixels that take no machine.
- It is a `ChangeMethod` (`HarmonicOptions::METHOD`), named in `method_name()`, its parameters
  set by `with_method()`; `Run::pixel` picks its detector by the method.

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
