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

`detect_pixel` (`src/ccd/ccd.cpp`) is a straight pipeline, not a machine:

1. decode the QA (through a `QaTable` when one is given);
2. compute the QA statistics of the observations dated up to `STAT_ORD`
   (`Quality::compute_until`, `include/ccd/harmonic/quality.hpp`);
3. choose the pixel's procedure, `ccd::select_procedure`
   (`include/ccd/procedure/pixel_procedure.hpp`);
4. apply that procedure's transform to the pixel's private copy and run its filter chain to
   build the processing mask (`include/ccd/preprocess/preprocess.hpp`);
5. fit the mask with the run's change detector, `standard::Detector::fit`
   (`include/ccd/procedure/standard.hpp`).

The procedure choice:

| Guard, in order | Procedure | What fits it |
|---|---|---|
| clear percentage >= `CLEAR_PCT_THRESHOLD` | Standard | the run's method (`HarmonicOptions::METHOD`): `ccdc`, the Standard machine (section 2) |
| else snow percentage >= `SNOW_PCT_THRESHOLD` | PermanentSnow | `ccd::fit_whole_series`, curve_qa 54 |
| else | InsufficientClear | `ccd::fit_whole_series`, curve_qa 44 |

The whole-series fit has no loop, so it is a function, not a machine. With fewer than
`MEOW_SIZE` kept observations it returns no segment, with the processing mask. Otherwise it returns one segment over the
whole series, every band fitted with `COEFFICIENT_MIN` coefficients.

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

Bracketed names are the `Exit` value the record keeps: why the machine stopped where it did.
The exit is internal; the trace and the tests read it, the result does not carry it.

### Phases and their guards

Each row is one transition function in `src/ccd/procedure/standard.cpp`, declared in
`include/ccd/procedure/standard.hpp`. The three phases that repeat themselves do one thing per
call and say what they did as an outcome value, so the trace and the tests see every decision.

| Phase | One call | Next |
|---|---|---|
| `Prepare` (`prepare`) | The early exits; the statistics window; the cadence (`measure_cadence`); Initialize's criteria (`initialize_criteria`); the variogram; the cursor at `(0, MEOW_SIZE)` | Initialize, EndCatch or Done |
| `Initialize` (`initialize`) | One candidate start window: `ccd::initialize_step` returns an `InitializeOutcome` | itself after a grow or a shift; then Lookback, StartCatch, Extend or EndCatch |
| `Lookback` (`lookback`) | One observation before the window: `lookback_step` returns a `LookbackOutcome` | itself while `window.start > prev_break`; then StartCatch, Extend or EndCatch |
| `StartCatch` (`start_catch`) | The first segment's leading observations as a Start segment (curve_qa 14) | Extend or EndCatch |
| `Extend` (`extend`) | One peek test after the window: `extend_step` returns an `ExtendOutcome`; `extend` writes the segment | itself while a peek fits; Restart after a change; EndCatch at the series end |
| `Restart` (`restart`) | The cursor after a change | Initialize or EndCatch |
| `EndCatch` (`end_catch`) | The End segment (curve_qa 24), when enough observations trail the last break | Done |

The guards inside one call, in the order they are tested:

- **Initialize** (`initialize_step`, `src/ccd/procedure/kernels.cpp`): no candidate left
  (`Exhausted`); a span under `DAY_DELTA` (`ShortSpan`, grow); Tmask flags every observation
  (`AllOutliers`, grow); Tmask flags some and the shortened window is too short or too sparse
  (`RejectedOutliers`, grow, nothing dropped); then the detection bands are fitted and tested
  for stability (`Stable`, or `Unstable` and a shift). When Tmask's outliers are dropped, the
  same call fits the shortened window, as pyccd does in one pass of its loop; those steps report
  `DroppedThenStable` and `DroppedThenUnstable`.
- **Lookback** (`lookback_step`): every peek magnitude above the change threshold (`Change`, the
  window stays); the first magnitude above `OUTLIER_THRESHOLD` (`Dropped`, the observation
  before the window leaves the series); otherwise `Included` (`window.start - 1`). The peek
  window is `lookback_peek`'s: far from the previous break, the adjusted peek of observations
  before the window; near it, down to the previous break and never below. The comparison RMSE
  is the stable start window's: nothing is refitted while the window grows backward, as in
  Zhu's TrendSeasonalFit.
- **Extend** (`extend_step`): first the refit schedule (refit when untrained, at most
  `RETRAIN_MIN_OBS` observations, or once the window spans `RETRAIN_SPAN_FACTOR` times the last
  fit); then every magnitude above the threshold (`Change`); the first above
  `OUTLIER_THRESHOLD` (`Dropped`, the observation at `window.stop` leaves the series); a peek that
  reached the last observation (`SeriesEnd`: the curve ends at the window, before the
  observation the peek starts at, its break day); otherwise `Grown` (`window.stop + 1`). The
  curve that reaches the series end, by `SeriesEnd` or by a drop that leaves no full peek, is
  the last one: EndCatch follows, not Restart. A refit fits the detection bands, all a test
  reads; the other bands are fitted on the last refit's window when the segment is
  written (`complete_models`; before the drop when a drop ends the curve), which gives the
  models and peek residuals an every-band refit would have.

Initialize fits `COEFFICIENT_MIN` coefficients, Extend's fits follow `coefficient_count`
(`COEFFICIENT_MIN`, `_MID` or `_MAX` by window length), and both peek bases are `COEFFICIENT_MAX`
wide: a model predicts with zeros above its own count.

The routing between phases is a set of named predicates and routers, so a guard has one place:
`room_to_initialize`, `peek_fits`, `start_catch_due`, `after_stable_start`, `after_lookback`
and `peek_guard`. Every route into EndCatch records its `Exit`.

The transitions assert the invariants the algorithm relies on: Initialize's window holds at
least `MEOW_SIZE` observations; Lookback runs only while `window.start > prev_break`; Extend runs
only when a full peek fits.

## 3. The per-pixel record and the context

A transition is a function of two arguments, `(const Context&, State&)`. What it may change is
in the `State`. What it reads but never changes is in the `Context`.

**`standard::Context`** holds the caller's `HarmonicOptions` (never written below
`detect_pixel`), the pixel's `HarmonicWorkspace` (its dates, spectral and QA), and the Lasso
solver with its workspace.

**`standard::State`** is one pixel's record:

| Field | What it is | Shared or Standard's |
|---|---|---|
| `phase`, `exit` | where the machine is, and why it stopped | each machine has its own enums |
| `series` (`MaskedSeries`, `include/ccd/procedure/masked_series.hpp`) | the processing mask and the observations it keeps, always in step | shared |
| `cursor` (`Cursor`, `include/ccd/procedure/machine.hpp`) | the model window and `prev_break`, in masked coordinates | shared |
| `segments`, `carried` | the segments this run wrote, and the count of a resumed run's stored final ones before them; "no segment yet" is `segments.empty() && carried == 0` | shared |
| `suspended` (`Suspended`) | where the segment loop stopped: what a resume continues from (section 6) | Standard's |
| `cadence` (`Cadence`) | the peek size and change threshold, scaled to the pixel's revisit interval | Standard's |
| `variogram` (`Variogram`, `include/ccd/procedure/fit.hpp`) | the per-band noise floor of the statistics window | a shared type, computed by Standard |
| `initialize_criteria` | what a candidate start window is tested against | a shared type, filled by Standard |
| `init_models` | the stable start window's fits, which Lookback tests against | Standard's |
| `extension` (`Extension`) | what Extend carries from one peek test to the next | Standard's |
| `peek_basis`, `lookback` | scratch buffers, kept for their capacity | Standard's |

`MaskedSeries` is the only writer of the mask. Its two drops, `drop` (Lookback, Extend) and
`drop_outliers_if` (Initialize's tentative Tmask removal), move the caller's window with the
observations by one rule: a window bound above the dropped position moves down by one.

The record's buffers keep their capacity across pixels: `State::reset` starts a pixel without
freeing anything. `standard::Procedure` owns one record, `standard::Detector` owns one
`standard::Procedure`, and `DetectScratch` (`include/ccd/ccd.hpp`) holds one `Detector` per
OpenMP thread.

## 4. Shared functions and Standard's own

What any change-detection machine can use, each taking narrow arguments and never a machine's
record:

| Function or type | File |
|---|---|
| `select_procedure`, `fit_whole_series`, `PixelProcedure` | `include/ccd/procedure/pixel_procedure.hpp` |
| `MaskedSeries` | `include/ccd/procedure/masked_series.hpp` |
| `Cursor`, `drive` | `include/ccd/procedure/machine.hpp` |
| `fit_bands` (the one per-band fit loop), `catch_segment`, `initialize_step` with `InitializeCriteria` and `InitializeOutcome` | `include/ccd/procedure/kernels.hpp` |
| `HarmonicBasis` (the harmonic design, column order documented) | `include/ccd/regression/harmonic_basis.hpp` |
| `Variogram`, `adjusted_variogram`, `tmask`, `stable`, `find_closest_doy`, `SeasonalOrder`, `seasonal_rmse`, `span`, `Window`, `ChangeModel`, `BandFit` | `include/ccd/procedure/fit.hpp` |

What only the Standard machine uses lives with it, in `ccd::standard`
(`include/ccd/procedure/standard.hpp`): `Cadence` and `measure_cadence`, `initialize_criteria`,
`lookback_peek`, `peek_residuals`, `change_magnitude`, `coefficient_count`, the phases, the
transitions and the routing. A function moves to the shared list only once a second machine uses
it.

## 5. The driver and the trace

`ccd::drive<Machine>` (`include/ccd/procedure/machine.hpp`) is the only loop:

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
coefficient count and magnitudes, and the final segment count. `ccd::detect_recorded()` returns
it inside a `PixelRecord`, with the pixel's input fingerprint and the days the machine dropped
(`include/ccd/record.hpp`).

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
StartCatch stays the first segment's. `ccd::resume()` keeps the stored final segments as given
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
and the record. `tests/ccd/procedure/test_machine.cpp` does that for every row of the guard
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
| 1. Far from the previous break, Lookback peeks the adjusted peek of observations | `lookback_peek`, first case |
| 2. Near it, Lookback peeks down to the previous break, never below | `lookback_peek`, second case |
| 3. Lookback compares against the stable start window's RMSE | `lookback_step` |
| 4. Initialize fits `COEFFICIENT_MIN` coefficients; both peek bases are `COEFFICIENT_MAX` wide | `initialize_criteria`; the `peek_basis.build` calls in `lookback_step` and `extend_step` |
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
  `select_procedure` and `fit_whole_series` for the pixels that take no machine.
- It is a `ChangeMethod` (`HarmonicOptions::METHOD`), named in `method_name()`, its parameters
  set by `with_method()`; `detect_pixel` picks its detector by the method.

A segment reports each band's fit as a `BandFit` (`ChangeModel::bands`): intercept,
coefficients, RMSE, magnitude and the solver's iteration count (0 for a fit in closed form), so
a detector that fits by ordinary least squares writes the same record. Inside the Standard
machine the fits stay `LassoResult`s, whose residuals the seasonal RMSE reads; `band_fit()`
converts each when a segment is written.

### A later batch on the CPU

Because every transition is one step on a plain record, a driver could hold many pixels and
advance them phase by phase, for example doing every pixel's Extend peek prediction together,
or fitting many windows at once with a Lasso solver vectorised across pixels. Each pixel's
arithmetic would keep its order, so the outputs could stay byte-identical. Pixels in different
phases limit the gain. On a 500 x 500 Landsat cell, Extend steps take about 80% of the
per-phase thread time and Initialize steps about 16%, so those two are where a batch would have
to pay. Nothing here is built; it would need a structure-of-arrays layout for the record.
