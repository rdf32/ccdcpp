#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ccd/config/shared.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/types.hpp"
#include "ccd/core/window.hpp"
#include "ccd/machine/suspended.hpp"

#include "ccd/input/masked_series.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/results/diagnostics.hpp"
#include "ccd/results/segment.hpp"
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

//==============================================================================
//
// Initialize
//
// One step of the search for a stable start window, as a plain function of
// what it reads: the Initialize phase calls it once per candidate window.
//
//==============================================================================

// What Initialize fits a candidate window with, besides its strategies. The
// machine fills it from its run's shared values (initialize_criteria());
// initialize_step() does not read them directly. Values throughout (a
// BandList is a small fixed-capacity array), so a criteria object never
// refers to what it was filled from.
struct InitializeCriteria
{
    BandList detection_bands;   // the bands fitted and tested for stability
    index_t num_coefficients;   // of the stability fits
    index_t max_coefficients;   // the widest model's: their peek basis width

    // The last day the search sees (StartSearch::last_day()): its
    // exhausted() reads the observations dated through it.
    std::int64_t last_day = std::numeric_limits<std::int64_t>::max();
};

// The strategies one Initialize step runs, of the run's stages: the start
// search, the window screen, the stability fit (its Model, which holds the
// fits) and the stability rule with its scratch, prepared for the pixel.
struct InitializeStages
{
    const StartSearch& search;
    const WindowScreen& screen;
    Model& fit;
    const StabilityRule& rule;
    const Scratch& rule_scratch;
};

// What one step of the start-window search did, in the order the step tests
// them.
//
// The last two are one step that changes two things: Tmask's outliers are
// dropped from the series (the window shortened by as many), and the
// shortened window is fitted and tested in the same step, as pyccd does in
// one pass of its loop. Their routing is Unstable's and Stable's.
enum class InitializeOutcome : std::uint8_t
{
    Exhausted,            // the start search has no candidate left:
                          // nothing tested or changed
    ShortSpan,            // the window spans too few days: grown
    AllOutliers,          // Tmask flags every observation in it: grown
    RejectedOutliers,     // what Tmask's outliers leave is too short or too
                          // few: grown, nothing dropped
    Unstable,             // nothing flagged; fitted, not stable: shifted
    Stable,               // nothing flagged; fitted and stable
    DroppedThenUnstable,  // outliers dropped; fitted, not stable: shifted
    DroppedThenStable     // outliers dropped; fitted and stable
};

const char* to_string(InitializeOutcome outcome);

// One step of the search for a stable start window: one candidate window.
// `window` holds at least the search's min_window() observations.
//
// In this order:
//
//   * the start search finds no candidate left: Exhausted;
//   * it finds the window spans too few days: the window grows by one;
//   * the window screen flags every observation of the window: it grows by
//     one (AllOutliers);
//   * it flags some: they are dropped from `series` if what is left of
//     the window is still a candidate (the search's usable();
//     MaskedSeries::drop_outliers_if(), which shortens the window), and the
//     step goes on; otherwise the window grows by one with nothing dropped
//     (RejectedOutliers);
//   * the detection bands are fitted over the window by the stability fit
//     (residuals kept): Stable (DroppedThenStable after a drop) when the
//     stability rule finds the fit stable, with `window` the stable window
//     and the fit its fits; otherwise the window shifts by one (Unstable,
//     DroppedThenUnstable).
//
// Only the detection bands are fitted, and only on the four fitted
// outcomes.
InitializeOutcome initialize_step(
    MaskedSeries& series,
    Window& window,
    const InitializeCriteria& criteria,
    const Variogram& variogram,
    const InitializeStages& stages
);

// What Initialize fits a candidate window with: the shared detection bands,
// and stability fits of the stability fit's rule's min coefficients in a
// basis its max wide. Built once per pixel, in Prepare.
InitializeCriteria initialize_criteria(const config::Shared& shared,
                                       const CoefficientRule& stability);

//==============================================================================
//
// The Standard machine
//
// pyccd's standard procedure as a per-pixel state machine. The Detector
// steps it from Prepare to Done (step()); each call of a phase's transition
// makes one decision on the State and returns the next phase.
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
    Initialize,    // one candidate start window (initialize_step)
    Lookback,      // one observation before the window, towards prev_break
    StartCatch,    // the leading observations of the first segment
    Extend,        // one peek test after the window
    Restart,       // the cursor after a segment
    EndCatch,      // the trailing observations
    Done
};

const char* to_string(Phase phase);

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
struct ExtendState
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
// Cursor
//
// Where the machine is in the pixel's series, in masked coordinates
// (positions in MaskedSeries::dates()):
//
//   window       the model window, the observations a model is fitted to
//   prev_break   the position where the previous segment stopped, the
//                break observation of the last change; 0 before the first
//
// A MaskedSeries::drop() moves `window` with the observations; `prev_break`
// stays where it is.
//------------------------------------------------------------------------------
struct Cursor
{
    Window window;
    index_t prev_break = 0;
};

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

    // The pixel's kept observations, where the window is, and the segments.
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
    PerThread<EdgeSegments, Scratch> edges;   // edge_segments' buffers, the
                                              // whole-series fit's too
    PerThread<SegmentReport, Scratch> segment_report;

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
    // filter or cap), and then Detector::suspended() is no suspension.
    bool resumable = true;

    // Extend's carried state.
    ExtendState extend;

    // Where the segment loop stopped (set with every Exit).
    Suspended suspended;

    // Makes the objects of the run's `strategies` the State holds: each
    // PerThread's of its strategy, kept when that strategy made it and made
    // again when another did (another run's). The phases read them
    // (*init_model, *extend.test, ...) from the pixel's Prepare on; the
    // Detector binds before each pixel.
    void bind(const Strategies& strategies);

    // Starts a pixel: Prepare, no exit, the series of `pixel` that
    // `mask` keeps, the cursor at 0, no segments and none carried, the
    // cadence and Initialize's criteria cleared, Extend untrained, no
    // suspension.
    void reset(const PixelSeries& pixel, ProcessingMask mask);

    // The pixel's result once Done: the segments (none after an early exit)
    // and the processing mask. Moves them out.
    FitResult take_result();
};

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
// edge_segments'. Next: Extend, or EndCatch (PeekBeyondEnd).
Phase start_catch(const Context& context, State& state);

// One Extend step, one peek test after the window. Pre: peek_fits(). A
// window smaller than the change test's min_window() grows untested (NoFit
// when no observation is left). Otherwise refits the model over the window
// when Extend is not trained or the refit schedule is due (NoFit
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
// segment_report's; its break day is the change's first observation, or at
// the series end the first observation after the window (its last when
// none is left). After NoFit, Done (SeriesEnd) with a forced start's own
// segment (SegmentReport::unfitted(), through the last observation) or none.
Phase extend(const Context& context, State& state);

// prev_break moves to window.stop and the window to the start search's
// first(stop). Next: Initialize, or EndCatch (NoRoomToInitialize).
Phase restart(const Context& context, State& state);

// Writes the End segment over (prev_break, size) when edge_segments'
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
// segment yet, and edge_segments' start_due() of window.start - prev_break.
bool start_catch_due(const Context& context, const State& state);

// After a stable start: Lookback when window.start > prev_break, else
// after_lookback().
Phase after_stable_start(const Context& context, State& state);

// After Lookback: StartCatch when start_catch_due(), else peek_guard().
Phase after_lookback(const Context& context, State& state);

// The peek guard: Extend when peek_fits(), else EndCatch (PeekBeyondEnd).
Phase peek_guard(const Context& context, State& state);

// The transition of state.phase, which is not Done: the machine's one step,
// which the Detector repeats until Done.
Phase step(const Context& context, State& state);

//==============================================================================
//
// Detector
//
// The change detector a run uses for every pixel of a call, whatever its
// configuration (another procedure is other strategies). Each pixel's
// procedure is chosen before it, by the run's mode (ModeRule); fit() then fits
// the pixel's processing mask with the Standard machine for Standard, and with
// the run's edge_segments strategy (EdgeSegments::whole_series()) for
// PermanentSnow and InsufficientClear.
//
// Holds the machine's record: one Detector per thread reuses its buffers
// across the pixels it fits. Each fit resets the record, so a reused Detector
// gives every pixel the result a fresh one does.
//
//==============================================================================

class Detector
{
public:

    // Fits the observations of `series` that `mask` keeps (the procedure's
    // processing mask, with the mode's extra mask steps applied: Run::pixel
    // builds it with the procedure's filter chain) with the mode's
    // procedure, under the run's `shared` values: the Standard machine,
    // driven from Prepare to Done with the mode's cap on the segments'
    // coefficients, or edge_segments' whole_series(). The result carries the
    // mask, after an early exit too.
    FitResult fit(
        const PixelMode& mode,
        const PixelSeries& series,
        ProcessingMask mask,
        const config::Shared& shared,
        const Strategies& strategies,
        Observe observe = {}
    );

    // Continues a Standard run that stopped at `from` on a longer series:
    // `mask` is the filter mask of the stored run's series with the
    // observations after it, and `shared`'s statistics cutoffs are the stored
    // run's, so Prepare's values are its values. Prepare runs as usual, on
    // the series as the filters leave it; then the stored run's drops
    // (`excluded_days`, each a day the filters keep) leave the series, the
    // cursor is put back and the guard that failed is tested again:
    // Initialize for NoRoomToInitialize (when there is room) and
    // InitializeExhausted, the peek guard for PeekBeyondEnd, and for
    // SeriesEnd the open segment's models refitted on its fit window, then
    // its pending grow and Extend when a full peek now fits, or the same
    // provisional segment again. Drives to Done; the result's segments are
    // this run's, after the carried final ones, the last of which ended in a
    // change when `carried_change`. `from.exit` is one of those four.
    FitResult resume(
        const PixelSeries& series,
        ProcessingMask mask,
        const config::Shared& shared,
        const Strategies& strategies,
        const Suspended& from,
        const std::vector<std::int64_t>& excluded_days,
        Observe observe = {},
        bool carried_change = false
    );

    // Where the last Standard fit() or resume() stopped; no suspension
    // (Exit::None, a resume reruns the pixel) when the run is not resumable.
    // Stale after a whole-series fit, which does not run the machine.
    const Suspended& suspended() const;

    // Why the last Standard fit() or resume() ended its segment loop.
    Exit exit() const
    {
        return state_.exit;
    }

private:

    State state_;
};

} // namespace standard

} // namespace ccd