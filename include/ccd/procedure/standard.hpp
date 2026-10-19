#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ccd/harmonic/quality.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/kernels.hpp"
#include "ccd/procedure/machine.hpp"
#include "ccd/procedure/masked_series.hpp"
#include "ccd/procedure/pixel_procedure.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/per_thread.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

namespace standard
{

//==============================================================================
//
// Cadence
//
// The Standard procedure's per-pixel cadence, set in Prepare: the peek size
// the run's change test chose for the pixel (ChangeTest::prepare()), which
// Extend, Lookback and the machine's guards read, and the threshold Lookback
// compares against. That threshold is pyccd's, from the options
// (adjust_change_threshold() of PEEK_SIZE, CHANGE_THRESHOLD, the detection
// band count and CHANGE_PROBABILITY for the peek), until Lookback is a stage
// with its own.
//
//==============================================================================

struct Cadence
{
    index_t peek;                // observations a peek holds
    scalar_t change_threshold;   // Lookback's
};

// What Initialize tests a candidate window against: the options' MEOW_SIZE,
// DAY_DELTA and DETECTION_BANDS, and stability fits of
// COEFFICIENT_MIN coefficients in a basis COEFFICIENT_MAX wide. Built once
// per pixel, in Prepare.
InitializeCriteria initialize_criteria(const HarmonicOptions& options);

//------------------------------------------------------------------------------
// The Standard procedure's kernels
//
// Plain functions of the procedure's steps, over the pixel's MaskedSeries.
// Their arithmetic, and the order of it, is pyccd's.
//------------------------------------------------------------------------------

// Extend's coefficient count for a window of `observations` observations:
// COEFFICIENT_MIN below COEFFICIENT_MID * NUM_OBS_FACTOR observations,
// COEFFICIENT_MID below COEFFICIENT_MAX * NUM_OBS_FACTOR, else
// COEFFICIENT_MAX.
index_t coefficient_count(
    index_t observations,
    const HarmonicOptions& options
);

//==============================================================================
//
// The Standard machine
//
// pyccd's standard procedure as a per-pixel state machine. drive()
// (machine.hpp) steps it from Prepare to Done; each call of a phase's
// transition makes one decision on the State and returns the next phase.
// Initialize tests one candidate window per call, Lookback one observation
// before the window, Extend one peek after it; each repeats itself until a
// guard sends it on.
//
//     Prepare ──┬─────────────────────────────────────────────► Done
//               │  (too few observations, or a NaN variogram)
//               ▼
//     ┌──► Initialize ◄┐ grow, shift ─────── exhausted ──┐
//     │         │ └────┘                                │
//     │         │ stable                                │
//     │         ▼                                       │
//     │     Lookback ◄┐ include, drop                   │
//     │         │ └───┘ (while window.start > prev_break)
//     │         ▼                                       │
//     │     StartCatch (first segment only)             │
//     │         │ no full peek after the window ────────┤
//     │         ▼                                       │
//     │     Extend ◄┐ grow, drop (while a peek fits)    │
//     │         │ └─┘    series end ────────────────────┤
//     │         ▼ change                                ▼
//     └── Restart ── no room to initialize ────────► EndCatch ──► Done
//
// Prepare enters Initialize, or EndCatch when there is no room; StartCatch
// and the peek guard follow Lookback, or the stable start directly.
//
// Invariants the transitions assert: in Initialize the window holds at least
// MEOW_SIZE observations; in Lookback window.start > prev_break; in Extend a
// full peek fits after the window (peek_fits()).
//
//==============================================================================

enum class Phase : std::uint8_t
{
    Prepare,       // the early exits, the statistics window, cadence, variogram
    Initialize,    // one candidate start window (ccd::initialize_step)
    Lookback,      // one observation before the window, towards prev_break
    StartCatch,    // the leading observations of the first segment
    Extend,        // one peek test after the window
    Restart,       // the cursor after a segment
    EndCatch,      // the trailing observations
    Done
};

const char* to_string(Phase phase);

// Why the machine stopped where it did: the early exits, and what ended the
// segment loop (the routes into EndCatch). Internal: tests and the trace read
// it, the result does not carry it.
enum class Exit : std::uint8_t
{
    None,                 // still running
    TooFewObservations,   // Prepare: at most MEOW_SIZE observations
    VariogramNaN,         // Prepare: a NaN in the variogram
    InitializeExhausted,  // Initialize: no stable window before the end
    PeekBeyondEnd,        // no full peek after the stable window
    NoRoomToInitialize,   // no MEOW_SIZE window after the cursor
    SeriesEnd             // Extend: a curve reached the series end
};

const char* to_string(Exit exit);

// Where the machine stopped, for a later resume (record.hpp): the record at
// the first size guard that failed, before anything after it wrote. Every
// segment written by then is final; a longer series makes the same
// decisions up to here, then passes the guard this one failed.
//
//   exit            the guard (Exit::None while running); after an early
//                   exit, and for the whole-series fit, a longer series is
//                   run from the start
//   window,         the cursor at the guard, masked coordinates
//   prev_break
//   fit_stop        Extend's fit window's stop (SeriesEnd only; the fit
//                   window starts at window.start)
//   grow            SeriesEnd after a test whose peek reached the last
//                   observation: that step's grow is pending (false after a
//                   drop that left no full peek). A resume that ignored it
//                   would test that peek again, with the same outcome, and
//                   grow then: one step more, the same result
//   num_coef,       SeriesEnd: the last test's coefficient count and each
//   magnitude       band's magnitude, which write the provisional segment
//                   again when a longer series still ends there
//   final_segments  the segments written before the guard, those of the run
//                   it resumed included
struct Suspended
{
    Exit exit = Exit::None;
    Window window;
    index_t prev_break = 0;
    index_t fit_stop = 0;
    bool grow = false;
    index_t num_coef = 0;
    std::vector<scalar_t> magnitude;
    index_t final_segments = 0;
};

// Lookback's peek, read backward from the observation before the window:
// positions start, start - 1, ..., stop + 1 (stop is exclusive and may be -1).
struct ReversePeek
{
    std::ptrdiff_t start;
    std::ptrdiff_t stop;
};

// Two cases, for a window starting at `window_start` > prev_break; neither
// reads below prev_break, into the previous segment:
//
//   * window_start - prev_break > peek: start - 1 down to start - peek, the
//     adjusted peek (Cadence::peek) of observations, as Extend's;
//   * else: start - 1 down to prev_break (stop is -1 when prev_break is 0).
ReversePeek lookback_peek(index_t window_start, index_t prev_break, index_t peek);

// What one Lookback step did, in the order it tests them.
enum class LookbackOutcome : std::uint8_t
{
    Change,    // every peek magnitude above the change threshold: the window
               // stays
    Dropped,   // the observation before the window is an outlier: dropped,
               // the window moving down one with the series
    Included   // it fits: window.start - 1
};

const char* to_string(LookbackOutcome outcome);

// Lookback's buffers, kept for their capacity; every one is rewritten
// before it is read.
struct LookbackScratch
{
    std::vector<std::int64_t> peek_dates;
    std::vector<scalar_t> peek_spectral;
    std::vector<std::vector<scalar_t>> residuals;   // by band; detection bands written
    std::vector<scalar_t> comp_rmse;                // by position in DETECTION_BANDS
    std::vector<scalar_t> magnitude;
};

// What Extend carries from one peek test to the next, within one segment.
// The buffers keep their capacity across segments and pixels.
struct Extension
{
    // Whether `model`'s fits are this segment's: false until the segment's first
    // refit, and again once its segment is written (pyccd's `models is None`).
    bool trained = false;

    // The window and span of the last refit, which the refit schedule and
    // the seasonal RMSE read.
    Window fit_window;
    scalar_t fit_span = 0;

    // Of the last peek test; the segment reads both: num_coef is its
    // curve_qa, which may not be the count `model` was fitted with, and
    // peek.start its break observation.
    index_t num_coef = 0;
    Window peek;

    // The fits, one per band, made by the run's monitoring-model strategy
    // and kept across pixels. A refit fits the detection bands, all a test
    // reads; the others are fitted on the same window when the segment is
    // written (complete), which takes band_fit() of each.
    PerThread<MonitoringModel, Model> model;
    bool complete = false;

    // The refits made, never reset: a test's buffers built on a fit (the
    // change test's) know it by its number.
    index_t fits = 0;

    // Indexed by band: the residuals over the last peek, whose medians are
    // the segment's magnitudes; the detection bands' at each test, the
    // others' once they are fitted.
    std::vector<std::vector<scalar_t>> residuals;

    // The change test's buffers, made by the run's change-test strategy.
    PerThread<ChangeTest, Scratch> test;

    // What the change test found at the last peek, which the outlier rule
    // reads.
    PeekResult result;
};

// What one Extend step did, in the order it tests them.
enum class ExtendOutcome : std::uint8_t
{
    Change,    // the change test's Change: the segment ends at the window
    Dropped,   // the first peek observation, at window.stop, is an outlier:
               // dropped, the window keeping its coordinates
    SeriesEnd, // no change, and the peek reached the last observation: the
               // curve ends at the window
    Grown      // window.stop + 1
};

const char* to_string(ExtendOutcome outcome);

//------------------------------------------------------------------------------
// State
//
// One pixel's record: everything that survives from one step to the next.
// Its buffers keep their capacity across pixels; reset() starts a pixel.
//------------------------------------------------------------------------------
struct State
{
    Phase phase = Phase::Prepare;
    Exit exit = Exit::None;

    // The shared parts (machine.hpp).
    MaskedSeries series;
    Cursor cursor;
    std::vector<ChangeModel> segments;   // written by this run

    // The final segments of the run this one resumed, which come before
    // `segments` (0 for a run from the start). "No segment yet", pyccd's
    // `start`, is segments.empty() && carried == 0.
    index_t carried = 0;

    // Standard's own: set in Prepare, read by every later phase.
    Cadence cadence{};
    ChangeThresholdMemo thresholds;             // the cadence's threshold
    Variogram variogram;
    InitializeCriteria initialize_criteria{};

    // Initialize's fits of the detection bands, made by the run's
    // stability-fit strategy: the stable window's, which Lookback tests
    // against. The stability rule's per-pixel state.
    PerThread<StabilityFit, Model> init_model;
    PerThread<StabilityRule, Scratch> stability;

    // Extend's carried state.
    Extension extension;

    // Scratch.
    LookbackScratch lookback;

    // Where the segment loop stopped (set with every Exit).
    Suspended suspended;

    // Starts a pixel: Prepare, no exit, the series of `workspace` that
    // `mask` keeps, the cursor at 0, no segments and none carried, the
    // cadence and Initialize's criteria cleared, no extension trained, no
    // suspension.
    void reset(const HarmonicWorkspace& workspace, ProcessingMask mask);

    // The pixel's result once Done: the segments (none after an early exit)
    // and the processing mask. Moves them out.
    FitResult take_result();
};

// One trace line's view of the record: cursor, observations, segments, exit.
std::string summary(const State& state);

//------------------------------------------------------------------------------
// Context
//
// What the transitions read and use but never change the meaning of: the
// caller's options (never written), the pixel, and the Lasso solver with its
// workspace.
//------------------------------------------------------------------------------
struct Context
{
    const HarmonicOptions& options;
    const HarmonicWorkspace& pixel;
    LassoWorkspace& lasso;
    LassoSolver& solver;
    // The run's strategies, one per stage (strategies/stages.hpp).
    const Strategies& strategies;
};

// The state's stability fit and stability-rule scratch, of the run's
// strategies (made when the state's are another run's).
Model& stability_fit(const Context& context, State& state);
Scratch& stability_scratch(const Context& context, State& state);

//------------------------------------------------------------------------------
// Transitions: one per phase but Done. Each makes one decision and returns
// the next phase; a route into EndCatch sets state.exit to why.
//------------------------------------------------------------------------------

// Done (TooFewObservations) when the series holds at most MEOW_SIZE
// observations. Otherwise sets the cadence, Initialize's criteria and the
// variogram from the statistics window (the observations dated <= STAT_ORD); Done
// (VariogramNaN) on a NaN in it. Puts the window at (0, MEOW_SIZE) and
// prev_break at 0. Next: Initialize, or EndCatch (NoRoomToInitialize).
Phase prepare(const Context& context, State& state);

// The work of the phases, apart from their routes: what another machine
// composing this one's phases calls (docs/ccdc-state-machine.md, "Another
// procedure").
//
//   initialize_outcome()  initialize()'s step, without the route;
//   write_start_segment() the Start segment over (prev_break, window.start);
//   move_past_segment()   restart()'s cursor move;
//   write_end_segment()   the End segment over (prev_break, size), unguarded.
InitializeOutcome initialize_outcome(const Context& context, State& state);
void write_start_segment(const Context& context, State& state);
void move_past_segment(const Context& context, State& state);
void write_end_segment(const Context& context, State& state);

// One ccd::initialize_step() on the window against the pixel's
// initialize_criteria (COEFFICIENT_MIN-coefficient fits), with the run's
// stability fit (into init_model) and stability rule. Next: Initialize after a grow or a shift; after_stable_start()
// when stable; EndCatch (InitializeExhausted) when no candidate is left.
Phase initialize(const Context& context, State& state);

// One Lookback step: tests the observations of lookback_peek() against
// init_model, whose RMSE is the comparison RMSE, and makes the
// decision on the record. Pre: window.start > prev_break. In order:
//
//   * every peek magnitude above the cadence's threshold: Change, the window
//     stays;
//   * the first above OUTLIER_THRESHOLD: Dropped, the observation before the
//     window dropped from the series and the window moved down with it;
//   * otherwise Included: window.start - 1.
//
// Lookback's tests are pyccd's, from the options, until Lookback is a stage.
LookbackOutcome lookback_step(const Context& context, State& state);

// lookback_step(), then the route: after a drop or an include, Lookback
// while window.start > prev_break; after a change, or once window.start
// reaches prev_break, after_lookback().
Phase lookback(const Context& context, State& state);

// Writes the Start segment over (prev_break, window.start) with
// COEFFICIENT_MIN coefficients. Next: Extend, or EndCatch (PeekBeyondEnd).
Phase start_catch(const Context& context, State& state);

// One Extend step, one peek test after the window. Pre: peek_fits().
// Refits the model over the window when the extension is not trained or the
// refit schedule is due; then, in order:
//
//   * the change test's Change: Change, the window stays;
//   * the outlier rule's Drop: Dropped, the first peek observation (at
//     window.stop) dropped, the window keeping its coordinates;
//   * no change, and the peek reached the last observation (window.stop +
//     peek == size): SeriesEnd, the window as it is;
//   * otherwise Grown, the window one longer.
//
// It writes no segment: extend() does.
ExtendOutcome extend_step(const Context& context, State& state);

// extend_step(), then the route. After a change, writes the segment with
// change probability 1; next Restart. After a drop or a grow, next Extend
// while a peek fits. At the series end (SeriesEnd, or a drop that left no
// full peek) writes the segment with change probability 0, moves prev_break
// to the window's stop and goes to EndCatch (SeriesEnd): the curve that
// reaches the end is the last one. The segment's curve_qa and break day are
// the last test's: its break day is the first observation after its end.
Phase extend(const Context& context, State& state);

// prev_break moves to window.stop and the window to (stop, stop +
// MEOW_SIZE). Next: Initialize, or EndCatch (NoRoomToInitialize).
Phase restart(const Context& context, State& state);

// Writes the End segment over (prev_break, size) with COEFFICIENT_MIN
// coefficients when prev_break + peek < size. Next: Done.
Phase end_catch(const Context& context, State& state);

//------------------------------------------------------------------------------
// Routing: the guards between phases, in the order the procedure tests them.
//------------------------------------------------------------------------------

// Room for a start window after the cursor: window.stop + MEOW_SIZE <= size.
bool room_to_initialize(const Context& context, const State& state);

// A full peek after the window: window.stop + peek <= size.
bool peek_fits(const State& state);

// The first segment's leading observations need a Start segment: no
// segment yet and window.start - prev_break > peek.
bool start_catch_due(const State& state);

// After a stable start: Lookback when window.start > prev_break, else
// after_lookback().
Phase after_stable_start(const Context& context, State& state);

// After Lookback: StartCatch when start_catch_due(), else peek_guard().
Phase after_lookback(const Context& context, State& state);

// The peek guard: Extend when peek_fits(), else EndCatch (PeekBeyondEnd).
Phase peek_guard(const Context& context, State& state);

//------------------------------------------------------------------------------
// The machine, for drive(), and one pixel's run of it.
//------------------------------------------------------------------------------
struct Machine
{
    using Context = standard::Context;
    using State = standard::State;
    using Phase = standard::Phase;

    static constexpr const char* name = "standard";

    // The transition of state.phase, which is not Done.
    static Phase step(const Context& context, State& state);
};

// Resets `state` for `mask`, drives it to Done and returns its result.
FitResult run(const Context& context, State& state, ProcessingMask mask);

// Continues a run that stopped at `from` on a longer series: `mask` is the
// filter mask of the stored run's series with the observations after it,
// and the context's statistics cutoffs are the stored run's, so Prepare's
// values are its values. Prepare runs as usual, on the series as the filters
// leave it; then the stored run's drops (`excluded_days`, each a day the
// filters keep) leave the series, the cursor is put back and the guard that
// failed is tested again: Initialize
// for NoRoomToInitialize (when there is room) and InitializeExhausted, the
// peek guard for PeekBeyondEnd, and for SeriesEnd the open segment's models
// refitted on its fit window, then its pending grow and Extend when a full
// peek now fits, or the same provisional segment again. Drives to Done; the
// result's segments are this run's, after the carried final ones.
// `from.exit` is one of those four.
FitResult resume(
    const Context& context,
    State& state,
    ProcessingMask mask,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days
);

//==============================================================================
//
// Procedure
//
// pyccd's standard procedure: the Standard machine and the per-pixel record
// it steps. The record is a member so that its buffers keep their capacity
// across the pixels one object fits; run() resets it for each pixel, so a
// reused object gives every pixel the result a fresh one does. A second
// machine has its own Procedure in its own namespace.
//
//==============================================================================

class Procedure
{

public:
    //----------------------------------------------------------------------
    // Fits the observations of hworkspace that `mask` keeps (the Standard
    // procedure's processing mask over its series, built by the caller:
    // Run::pixel builds it with the procedure's filter chain), through
    // standard::run(). The result carries the mask, after an early exit
    // too.
    //----------------------------------------------------------------------
    FitResult run(
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const Strategies& strategies
    );

    // run() from where a stored run stopped (standard::resume()).
    FitResult resume(
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const Strategies& strategies,
        const Suspended& from,
        const std::vector<std::int64_t>& excluded_days
    );

    // Where the last run() stopped.
    const Suspended& suspended() const
    {
        return state_.suspended;
    }

private:

    State state_;
};

//==============================================================================
//
// Detector
//
// The Standard change detector, the one a run uses for every pixel of a call
// (another method's detector is a sibling class with the same call, and the
// run picks one by its METHOD). Each pixel's procedure is chosen before
// it, by the shared ccd::select_procedure() (pixel_procedure.hpp); fit()
// then fits the pixel's processing mask with this machine for Standard, and
// with the shared ccd::fit_whole_series() for PermanentSnow and
// InsufficientClear.
//
// Holds the machine's record: one Detector per thread reuses its buffers
// across the pixels it fits.
//
//==============================================================================

class Detector
{
public:

    // Fits the observations of `hworkspace` that `mask` keeps with
    // `procedure`: Procedure::run() or fit_whole_series().
    FitResult fit(
        PixelProcedure procedure,
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const Strategies& strategies
    );

    // The Standard machine from where a stored run stopped (Procedure::
    // resume()).
    FitResult resume(
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const Strategies& strategies,
        const Suspended& from,
        const std::vector<std::int64_t>& excluded_days
    );

    // Where the last Standard fit() stopped; stale after a whole-series fit,
    // which does not run the machine.
    const Suspended& suspended() const
    {
        return procedure_.suspended();
    }

private:

    Procedure procedure_;
};

} // namespace standard

} // namespace ccd