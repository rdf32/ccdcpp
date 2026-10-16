#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ccd/harmonic/filter.hpp"
#include "ccd/harmonic/quality.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/kernels.hpp"
#include "ccd/procedure/machine.hpp"
#include "ccd/procedure/masked_series.hpp"
#include "ccd/regression/harmonic_basis.hpp"

namespace ccd
{

namespace standard
{

//==============================================================================
//
// Cadence
//
// The Standard procedure's per-pixel cadence: the peek window, scaled to the
// pixel's revisit interval, and the change threshold that goes with it.
// Derived once per pixel from the statistics-window dates (measure_cadence)
// and read by every later step in place of the caller's PEEK_SIZE and
// CHANGE_THRESHOLD, which are never written. Standard's own record: a
// procedure that does not count observations has no use for it.
//
//==============================================================================

struct Cadence
{
    // adjust_peek(): PEEK_SIZE scaled by PEEK_REFERENCE_DAYS over the median
    // gap, never below PEEK_SIZE. Observations.
    index_t peek;

    // adjust_change_threshold() for `peek`: CHANGE_THRESHOLD when peek is
    // PEEK_SIZE, lower for a longer peek.
    scalar_t change_threshold;
};

// The cadence of a pixel whose statistics window (the observations dated <=
// STAT_ORD) has dates `statistics_dates`.
Cadence measure_cadence(
    ArrayView<const std::int64_t, 1> statistics_dates,
    const HarmonicOptions& options
);

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

// |y - prediction| of models[b] at each peek observation, for every b in
// `bands`, into residuals[b] (resized to the peek's length; the other
// entries are not touched). `peek_basis` is built over the peek's dates with
// at least the models' coefficient count, and `peek_spectral` (B, peek
// length) holds the peek's observations in the same order.
void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    const BandList& bands,
    std::vector<std::vector<scalar_t>>& residuals
);

// peek_residuals() of every row of `peek_spectral`; residuals has B entries.
void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    std::vector<std::vector<scalar_t>>& residuals
);

// The change magnitude of each peek observation j, into `magnitude`:
//
//     sum over i of (residuals[bands[i]][j] / max(variogram[bands[i]], rmse[i]))^2
//
// in the order of `bands`, a band whose max(...) is not above 0 adding
// nothing. rmse[i] is the comparison RMSE of bands[i]; every residuals[b]
// named has the peek's length. Empty for an empty `bands`.
void change_magnitude(
    const std::vector<std::vector<scalar_t>>& residuals,
    const BandList& bands,
    const Variogram& variogram,
    const std::vector<scalar_t>& rmse,
    std::vector<scalar_t>& magnitude
);

//==============================================================================
//
// The Standard machine
//
// pyccd's standard procedure as a per-pixel state machine. drive()
// (machine.hpp) steps it from Prepare to Done; each phase's transition does
// its work on the State and returns the next phase. Lookback, Extend
// and Initialize still run their whole loop in one call.
//
//     Prepare ──┬─────────────────────────────────────────────► Done
//               │  (too few observations, or a NaN variogram)
//               ▼
//     ┌──► Initialize ── exhausted ───────────────────┐
//     │         │ stable                              │
//     │         ▼                                     │
//     │     Lookback (when window.start > prev_break) │
//     │         ▼                                     │
//     │     StartCatch (first segment only)           │
//     │         │ no full peek after the window ──────┤
//     │         ▼                                     │
//     │     Extend                                    │
//     │         ▼                                     ▼
//     └── Restart ── no room to initialize ──────► EndCatch ──► Done
//
// Prepare enters Initialize, or EndCatch when there is no room; StartCatch
// and the peek guard follow Lookback, or the stable start directly.
//
//==============================================================================

enum class Phase : std::uint8_t
{
    Prepare,       // the early exits, the statistics window, cadence, variogram
    Initialize,    // the start-window search (ccd::initialize)
    Lookback,      // the backward extension towards prev_break
    StartCatch,    // the leading observations of the first segment
    Extend,        // the forward extension and its change test
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
    NoRoomToInitialize    // no MEOW_SIZE window after the cursor
};

const char* to_string(Exit exit);

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
    std::vector<ChangeModel> segments;   // pyccd's `start` is segments.empty()

    // Standard's own: set in Prepare, read by every later phase.
    Cadence cadence{};
    Variogram variogram;

    // Initialize's stable fits of the detection bands (indexed by band),
    // which Lookback tests against.
    std::vector<LassoResult> init_models;

    // Scratch.
    HarmonicBasis peek_basis;
    LookbackScratch lookback;

    // Starts a pixel: Prepare, no exit, the series of `workspace` that
    // `mask` keeps, the cursor at 0, no segments.
    void reset(const HarmonicWorkspace& workspace, ProcessingMask mask);

    // The pixel's result once Done: the segments and the processing mask,
    // or no segments and an empty mask after an early exit (quirk 5, kept
    // until 0.8.0). Moves them out.
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
};

//------------------------------------------------------------------------------
// Transitions: one per phase but Done. Each returns the next phase; a route
// into EndCatch sets state.exit to why.
//------------------------------------------------------------------------------

// Done (TooFewObservations) when the series holds at most MEOW_SIZE
// observations. Otherwise sets the cadence and the variogram from the
// statistics window (the observations dated <= STAT_ORD); Done
// (VariogramNaN) on a NaN in it. Puts the window at (0, MEOW_SIZE) and
// prev_break at 0. Next: Initialize, or EndCatch (NoRoomToInitialize).
Phase prepare(const Context& context, State& state);

// ccd::initialize() from the window, with the 4-coefficient stability fits
// pyccd hard-codes (quirk 4) and the cadence's change threshold, into
// init_models. Next: after_stable_start(), or EndCatch
// (InitializeExhausted).
Phase initialize(const Context& context, State& state);

// Extends the window backward towards prev_break one observation at a time,
// testing the observations before it against init_models, whose RMSE is the
// comparison RMSE (quirk 3). Stops at prev_break or on a change; drops an
// outlier from the series and tests again. The peek reads PEEK_SIZE - 1
// observations far from prev_break (quirk 1) and, near the series start,
// down to position 0 even below prev_break (quirk 2). Next: after_lookback().
Phase lookback(const Context& context, State& state);

// Writes the Start segment over (prev_break, window.start) with
// COEFFICIENT_MIN coefficients. Next: Extend, or EndCatch
// (PeekBeyondEnd).
Phase start_catch(const Context& context, State& state);

// Grows the window one observation at a time while a full peek fits in the
// series, refitting every band on the refit schedule and testing the peek
// after the window; drops an outlier at the first peek observation and tests
// again. Writes the segment: change probability 1 on a change, 0 at the
// series end. Next: Restart.
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

// After Lookback: StartCatch when start_catch_due(), else
// peek_guard().
Phase after_lookback(const Context& context, State& state);

// The peek guard: Extend when peek_fits(), else EndCatch
// (PeekBeyondEnd).
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

} // namespace standard

//==============================================================================
//
// StandardProcedure
//
// pyccd's standard procedure: the facade of the Standard machine
// (standard::run()), with the FitProcedure signatures.
//
//==============================================================================

class StandardProcedure final : public FitProcedure
{

public:
    //----------------------------------------------------------------------
    // Fits the observations `mask` keeps, as FitProcedure::run() does.
    //----------------------------------------------------------------------
    FitResult run(
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    );

    //----------------------------------------------------------------------
    // run() on the mask filter::standard() builds from hworkspace.
    //----------------------------------------------------------------------
    FitResult run(
        const HarmonicWorkspace& hworkspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    )
    {
        return run(
            hworkspace,
            filter::standard(hworkspace),
            lworkspace,
            solver
        );
    }

protected:

    //----------------------------------------------------------------------
    // Harmonic complexity: standard::coefficient_count()
    //----------------------------------------------------------------------
    index_t coefficient_count(
        ArrayView<const std::int64_t, 1> dates,
        const HarmonicOptions& options
    ) const override
    {
        return standard::coefficient_count(dates.size(), options);
    }

    //----------------------------------------------------------------------
    // Result QA
    //----------------------------------------------------------------------

    CurveQA curve_qa() const override
    {
        return CurveQA::Start;
    }

private:

    //----------------------------------------------------------------------
    // The machine's record, a member so that its buffers keep their
    // capacity across the pixels one procedure object runs.
    //----------------------------------------------------------------------
    standard::State state_;
};

namespace standard
{

//==============================================================================
//
// Detector
//
// The Standard change detector, the one a run uses for every pixel of a call
// (the only detector so far: a second one, BFP, is a sibling class with the
// same two calls, and the run picks one). Per pixel it chooses the procedure
// from the pixel's QA statistics, select(), and after the caller has built
// the processing mask with that procedure's plan, fits it, fit(): this
// machine for Standard, the whole-series fit for PermanentSnow and
// InsufficientClear. The per-pixel choice is the detector's own.
//
// Holds the machine's record: one Detector per thread reuses its buffers
// across the pixels it fits.
//
//==============================================================================

class Detector
{
public:

    // pyccd's choice: Standard when the clear percentage reaches
    // CLEAR_PCT_THRESHOLD; otherwise PermanentSnow when the snow percentage
    // reaches SNOW_PCT_THRESHOLD; otherwise InsufficientClear.
    PixelProcedure select(
        const Quality::Statistics& statistics,
        const HarmonicOptions& options
    ) const;

    // Fits the observations of `hworkspace` that `mask` keeps with
    // `procedure`, as that procedure's run() does.
    FitResult fit(
        PixelProcedure procedure,
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    );

private:

    StandardProcedure standard_;
};

} // namespace standard

inline scalar_t span(
    ArrayView<const std::int64_t,1> dates,
    const Window& window
)
{
    assert(window.stop > window.start);

    return static_cast<scalar_t>(
        dates(window.stop - 1) -
        dates(window.start)
    );
}

} // namespace ccd