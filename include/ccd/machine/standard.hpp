#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ccd/machine/fit.hpp"
#include "ccd/machine/kernels.hpp"
#include "ccd/machine/machine.hpp"
#include "ccd/input/masked_series.hpp"
#include "ccd/machine/pixel_procedure.hpp"
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
// Extend and the machine's guards read, and the stability rule and Lookback
// are prepared with; the fewest observations a peek it tests holds, and a
// window it tests (its PeekPlan's full and min_window).
//
//==============================================================================

struct Cadence
{
    index_t peek;         // observations a peek holds
    index_t shortest;     // peek for a test of full peeks only, else 0
    index_t min_window;   // a smaller model window grows untested
};

// What Initialize fits a candidate window with: the shared detection bands,
// and stability fits of the stability fit's rule's min coefficients in a
// basis its max wide. Built once per pixel, in Prepare.
InitializeCriteria initialize_criteria(const config::Shared& shared,
                                       const CoefficientRule& stability);

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
// the start search's min_window() observations; in Lookback window.start > prev_break; in Extend a
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
    TooFewObservations,   // Prepare: too few observations to start (the start
                          // search's too_few(); pyccd: at most meow_size)
    VariogramNaN,         // Prepare: a NaN in the variogram
    InitializeExhausted,  // Initialize: no stable window before the end
    PeekBeyondEnd,        // no full peek after the stable window
    NoRoomToInitialize,   // no room for the start search's first candidate
                          // after the cursor
    SeriesEnd,            // Extend: a curve reached the series end
    NoFit                 // Extend: the series ended before the model could
                          // be fitted on the window, or it could not be
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

// What one Lookback step did: the lookback's verdict, applied.
enum class LookbackOutcome : std::uint8_t
{
    Change,    // the lookback's Change: the window stays
    Dropped,   // the observation before the window is an outlier: dropped,
               // the window moving down one with the series
    Included   // it fits: window.start - 1
};

const char* to_string(LookbackOutcome outcome);

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
    SeriesEnd, // no change, and the series ends: the peek reached the last
               // observation, or none is left, or the test or the outlier
               // rule could not tell; the curve ends at the window
    NoFit,     // the series ended before the model could be fitted on the
               // window, or it could not be: the curve is not written
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

    // Standard's own: set in Prepare, read by every later phase. Extend's
    // coefficients are the monitoring model's rule's.
    Cadence cadence{};
    CoefficientRule coefficients{};
    Variogram variogram;
    InitializeCriteria initialize_criteria{};

    // Initialize's fits of the detection bands, made by the run's
    // stability-fit strategy: the stable window's, which Lookback tests
    // against. The stability rule's per-pixel state.
    PerThread<StabilityFit, Model> init_model;
    PerThread<StabilityRule, Scratch> stability;
    PerThread<Lookback, Scratch> lookback;   // the lookback's per-pixel state
    PerThread<EdgeFit, Scratch> edges;       // the edge fit's buffers
    PerThread<SegmentFit, Scratch> segment_fit;

    // The processing mask when the current start's search began, kept for a
    // start search that restores it when exhausted (Exhaustion::Restore).
    ProcessingMask search_mask;

    // A forced start (Exhaustion::Force) until Extend fits it: the segment
    // the series reports if it ends first.
    std::optional<Window> forced;

    // The current start search's plan (StartSearch::begin()).
    SearchPlan search;

    // Whether the last of the `carried` segments ended in a change (a
    // resume's; the End segment's rule reads it while this run wrote none).
    bool carried_change = false;

    // Whether a resume can continue this run from its suspension: false once
    // it takes a route the record does not keep (shortened peeks, a forced
    // start, a restored search, drops before a change, no fit; or a mode's
    // filter or cap), and then Procedure::suspended() is no suspension.
    bool resumable = true;

    // Extend's carried state.
    Extension extension;

    // Where the segment loop stopped (set with every Exit).
    Suspended suspended;

    // Starts a pixel: Prepare, no exit, the series of `pixel` that
    // `mask` keeps, the cursor at 0, no segments and none carried, the
    // cadence and Initialize's criteria cleared, no extension trained, no
    // suspension.
    void reset(const PixelSeries& pixel, ProcessingMask mask);

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
// run's shared values (the detection bands, the statistics cutoff; never
// written), the pixel, and the run's strategies (each holds its own solver
// and buffers in the State's scratch).
//------------------------------------------------------------------------------
struct Context
{
    const config::Shared& shared;
    const PixelSeries& pixel;
    // The run's strategies, one per stage (strategies/stages.hpp).
    const Strategies& strategies;
    // Where the pixel reports what it does (results/diagnostics.hpp).
    Observe observe{};
    // The pixel's mode's cap on its segments' coefficients (PixelMode; 0:
    // none).
    index_t max_coefficients = 0;
};

// The state's cursor and series length, as an observer's transition reports
// them.
ObserverEvent observed(const State& state);

// The observer's event of a segment written (Kind::Segment).
ObserverEvent segment_event(const ChangeModel& segment);

// The state's stability fit and stability-rule scratch, of the run's
// strategies (made when the state's are another run's).
Model& stability_fit(const Context& context, State& state);
Scratch& stability_scratch(const Context& context, State& state);

// Reports the pixel's exit (state.exit) to the context's observer, if any.
void report_exit(const Context& context, const State& state);

// The state's lookback and edge-fit scratch, of the run's strategies.
Scratch& lookback_scratch(const Context& context, State& state);
Scratch& edge_scratch(const Context& context, State& state);

//------------------------------------------------------------------------------
// Transitions: one per phase but Done. Each makes one decision and returns
// the next phase; a route into EndCatch sets state.exit to why.
//------------------------------------------------------------------------------

// Done (TooFewObservations) when the start search finds too few observations
// (too_few()). Otherwise sets the cadence, Initialize's criteria and the
// variogram from the statistics window (the observations dated <= statistics_end); Done
// (VariogramNaN) on a NaN in it. Puts the window at the search's first(0)
// and prev_break at 0. Next: Initialize (begin_search()), or EndCatch
// (NoRoomToInitialize).
Phase prepare(const Context& context, State& state);

// Into Initialize, a search from prev_break: its plan from the start
// search's begin() (state.search; the pixel's first search is the one from
// 0), and the processing mask it starts from for a search that restores it.
// Next: Initialize.
Phase begin_search(const Context& context, State& state);

// The work of the phases, apart from their routes, which tests and a trace
// call one at a time.
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
// initialize_criteria (the stability fit's min-coefficient fits), with the run's
// stability fit (into init_model) and stability rule. Next: Initialize after
// a grow or a shift; after_stable_start() when stable. When no candidate is
// left, as the search's plan says: EndCatch (InitializeExhausted), the series as it
// was or (Restore) as the search began; or (Force) the search's forced
// start, its window the forced one, then the peek guard (no Lookback and no
// Start segment).
Phase initialize(const Context& context, State& state);

// One Lookback step: the run's lookback judges the observation before the
// window against init_model, and the step applies its verdict on the
// record. Pre: window.start > prev_break.
//
//   * Change: the window stays;
//   * Drop (Dropped): that observation leaves the series and the window
//     moves down with it;
//   * Include (Included): window.start moves to it.
LookbackOutcome lookback_step(const Context& context, State& state);

// lookback_step(), then the route: after a drop or an include, Lookback
// while window.start > prev_break; after a change, or once window.start
// reaches prev_break, after_lookback().
Phase lookback(const Context& context, State& state);

// Writes the Start segment over (prev_break, window.start) with
// the edge fit's. Next: Extend, or EndCatch (PeekBeyondEnd).
Phase start_catch(const Context& context, State& state);

// One Extend step, one peek test after the window. Pre: peek_fits(). A
// window smaller than the change test's min_window() grows untested (NoFit
// when no observation is left). Otherwise refits the model over the window
// when the extension is not trained or the refit schedule is due (NoFit
// when it cannot be fitted); the peek is the next `peek` observations,
// fewer at the series end (SeriesEnd when none is left); then, in order:
//
//   * the change test's Change: Change, the window stays;
//   * its Undecided: SeriesEnd;
//   * the outlier rule's Drop: Dropped, the first peek observation (at
//     window.stop) dropped, the window keeping its coordinates;
//   * its EndSeries: SeriesEnd;
//   * no change, and no peek the test reads is left after the next
//     observation (window.stop + shortest >= size): SeriesEnd, the window
//     as it is;
//   * otherwise Grown, the window one longer.
//
// It writes no segment: extend() does.
ExtendOutcome extend_step(const Context& context, State& state);

// extend_step(), then the route. After a change, writes the segment with
// change probability 1, and the peek's observations before the change
// test's break_at() leave the series; next Restart. After a drop or a grow,
// next Extend while a peek fits. At the series end (SeriesEnd, or a drop
// that left no full peek) writes the segment with change probability 0,
// moves prev_break to the window's stop and goes to EndCatch (SeriesEnd):
// the curve that reaches the end is the last one. The segment's curve_qa is
// the segment fit's; its break day is the change's first observation, or at
// the series end the first observation after the window (its last when
// none is left). After NoFit, Done (SeriesEnd) with a forced start's own
// segment (SegmentFit::unfitted(), through the last observation) or none.
Phase extend(const Context& context, State& state);

// prev_break moves to window.stop and the window to the start search's
// first(stop). Next: Initialize, or EndCatch (NoRoomToInitialize).
Phase restart(const Context& context, State& state);

// Writes the End segment over (prev_break, size) when the edge fit's
// end_due() says. Next: Done.
Phase end_catch(const Context& context, State& state);

//------------------------------------------------------------------------------
// Routing: the guards between phases, in the order the procedure tests them.
//------------------------------------------------------------------------------

// Room for a start window after the cursor: the start search's room().
bool room_to_initialize(const Context& context, const State& state);

// A peek the change test reads after the window: window.stop + shortest <=
// size (a full peek for a test of full peeks only).
bool peek_fits(const State& state);

// The first segment's leading observations need a Start segment: no
// segment yet, and the edge fit's start_due() of window.start - prev_break.
bool start_catch_due(const Context& context, const State& state);

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
// result's segments are this run's, after the carried final ones, the last
// of which ended in a change when `carried_change`.
// `from.exit` is one of those four.
FitResult resume(
    const Context& context,
    State& state,
    ProcessingMask mask,
    const Suspended& from,
    const std::vector<std::int64_t>& excluded_days,
    bool carried_change = false
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
    // Fits the observations of series that `mask` keeps (the Standard
    // procedure's processing mask over its series, built by the caller:
    // Run::pixel builds it with the procedure's filter chain), through
    // standard::run(). The result carries the mask, after an early exit
    // too.
    //----------------------------------------------------------------------
    FitResult run(
        const PixelSeries& series,
        ProcessingMask mask,
        const Strategies& strategies,
        Observe observe = {},
        index_t max_coefficients = 0
    );

    // run() from where a stored run stopped (standard::resume()).
    FitResult resume(
        const PixelSeries& series,
        ProcessingMask mask,
        const Strategies& strategies,
        const Suspended& from,
        const std::vector<std::int64_t>& excluded_days,
        Observe observe = {},
        bool carried_change = false
    );

    // Where the last run() stopped; no suspension (Exit::None, a resume
    // reruns the pixel) when the run is not resumable.
    const Suspended& suspended() const;

private:

    State state_;
};

//==============================================================================
//
// Detector
//
// The change detector a run uses for every pixel of a call, whatever its
// configuration (another procedure is other strategies). Each pixel's procedure is chosen before
// it, by the run's mode (ModeRule); fit() then fits the pixel's processing
// mask with this machine for Standard, and with the run's edge fit
// (EdgeFit::whole_series()) for PermanentSnow and InsufficientClear.
//
// Holds the machine's record: one Detector per thread reuses its buffers
// across the pixels it fits.
//
//==============================================================================

class Detector
{
public:

    // Fits the observations of `series` that `mask` keeps (with the
    // mode's extra mask steps applied) with the mode's procedure:
    // Procedure::run() with its cap, or the edge fit's whole_series().
    FitResult fit(
        const PixelMode& mode,
        const PixelSeries& series,
        ProcessingMask mask,
        const Strategies& strategies,
        Observe observe = {}
    );

    // The Standard machine from where a stored run stopped (Procedure::
    // resume()).
    FitResult resume(
        const PixelSeries& series,
        ProcessingMask mask,
        const Strategies& strategies,
        const Suspended& from,
        const std::vector<std::int64_t>& excluded_days,
        Observe observe = {},
        bool carried_change = false
    );

    // Where the last Standard fit() stopped; stale after a whole-series fit,
    // which does not run the machine.
    const Suspended& suspended() const
    {
        return procedure_.suspended();
    }

private:

    Procedure procedure_;
    PerThread<EdgeFit, Scratch> edges_;   // the whole-series fit's
};

} // namespace standard

} // namespace ccd