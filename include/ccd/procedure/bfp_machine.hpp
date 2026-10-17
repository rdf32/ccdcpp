#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "ccd/procedure/bfp.hpp"
#include "ccd/procedure/pixel_procedure.hpp"
#include "ccd/procedure/standard.hpp"

namespace ccd
{
namespace bfp
{

//==============================================================================
//
// The BFP machine
//
// The Standard machine (standard.hpp) with its change test replaced by the
// BFP test (bfp.hpp). It composes the Standard machine's record and phases:
// Prepare, Initialize (initialize_outcome()), Lookback (lookback_step()), the
// Start and End segments and Restart's cursor move are the Standard
// machine's own; Lookforward replaces Extend, and the routes are its own:
//
//   Prepare     -> Initialize; EndCatch (no room); Done (early exits)
//   Initialize  -> Initialize; after a stable start: Lookback while
//                  window.start > prev_break, else the start guard;
//                  EndCatch when no candidate is left
//   Lookback    -> Lookback; the start guard
//   start guard -> StartCatch when no segment is written yet and
//                  window.start - prev_break >= BFP.START_CATCH_MIN, else
//                  Lookforward
//   StartCatch  -> Lookforward
//   Lookforward -> Lookforward (an observation appended or dropped);
//                  Restart (a change); Done (the series end)
//   Restart     -> Initialize; EndCatch (no room)
//   EndCatch    -> Done, after the End segment when the last segment ended
//                  in a change and BFP.END_CATCH_MIN observations follow it
//
// There is no full-peek guard: the test runs on the observations left when
// fewer than BFP.MAX_PEEK remain, and the series ends where a shortened peek
// leaves the test undecided (JuliaCCD's lookForward).
//
//==============================================================================

enum class Phase : std::uint8_t
{
    Prepare,
    Initialize,
    Lookback,
    StartCatch,
    Lookforward,
    Restart,
    EndCatch,
    Done
};

const char* to_string(Phase phase);

//------------------------------------------------------------------------------
// LeastSquares
//
// Ordinary least squares on the harmonic model by running sums: X'X, X'y and
// y'y over a segment's observations, each added once, so a refit at every
// observation costs a q x q factorisation. The trend column is time from the
// segment's first day in years (the normal equations stay well conditioned;
// the fitted values, residuals, SSR and leverages are those of the ordinal
// day's column), and a fit is reported in the HarmonicBasis convention
// (band_fit()): the slope per day and the intercept at day 0.
//
// Columns, the first q of them fitted: 1, years, cos(wt), sin(wt),
// cos(2wt), sin(2wt), cos(3wt), sin(3wt), w = 2 pi / AVG_DAYS_YR.
//------------------------------------------------------------------------------
class LeastSquares
{
public:

    static constexpr index_t COLUMNS = 8;

    void reset(std::int64_t origin, index_t bands);

    // Adds the observation of `day` with value spectral(b, t) in each band.
    void add(std::int64_t day, ArrayView<const scalar_t, 2> spectral, index_t t);

    index_t count() const { return n_; }

    // Fits q coefficients in every band: false when X'X[q,q] is not positive
    // definite (repeated dates leave the columns dependent).
    bool fit(index_t q);

    // After fit(): the band's SSR, the residual at (day, y), the column of
    // `day` whitened by the factor of X'X (L^-1 x, q entries) and the band's
    // fit as a segment reports it.
    scalar_t ssr(index_t band) const { return ssr_[band]; }
    scalar_t residual(index_t band, std::int64_t day, scalar_t y) const;
    void whitened(std::int64_t day, scalar_t* out) const;
    BandFit band_fit(index_t band, scalar_t rmse, scalar_t magnitude) const;

    index_t terms() const { return q_; }

private:

    std::array<scalar_t, COLUMNS> row(std::int64_t day) const;

    std::int64_t origin_ = 0;
    index_t bands_ = 0;
    index_t n_ = 0;
    index_t q_ = 0;

    std::array<scalar_t, COLUMNS * COLUMNS> xtx_{};
    std::vector<scalar_t> xty_;   // bands x COLUMNS
    std::vector<scalar_t> yty_;   // bands

    std::array<scalar_t, COLUMNS * COLUMNS> chol_{};   // q x q lower
    std::vector<scalar_t> beta_;  // bands x COLUMNS
    std::vector<scalar_t> ssr_;   // bands
};

//------------------------------------------------------------------------------
// Lookforward
//------------------------------------------------------------------------------

// What one Lookforward step did, in the order it tests them.
enum class LookforwardOutcome : std::uint8_t
{
    Change,      // the test found a change: the segment ends at the window
    SeriesEnd,   // no observation left, or a shortened peek undecided
    Dropped,     // no change; the first peek observation is an outlier and
                 // leaves the series
    Appended     // no change: the first peek observation joins the window
};

const char* to_string(LookforwardOutcome outcome);

// Lookforward's carried state: the segment's sums and the test's scratch.
struct Extension
{
    // Whether `sums` hold this segment's window: false until its first
    // Lookforward step, and again once its segment is written.
    bool trained = false;
    LeastSquares sums;

    // The fit's coefficient count, the observations it was fitted on (it
    // is refitted when `sums` hold more) and the detection bands' MSE.
    index_t terms = 0;
    index_t fitted = 0;
    std::vector<scalar_t> mse;

    // The last test: its peek's dates, the detection bands' residuals over
    // it (bands x N), and after a change the subset size that decided and
    // the change's first observation in the peek.
    std::vector<std::int64_t> peek_dates;
    std::vector<scalar_t> residuals;
    index_t decided_at = 0;
    index_t break_start = 0;

    // After a change, every band's median residual from its first
    // observation to the deciding one: the segment's magnitudes.
    std::vector<scalar_t> magnitude;

    Mixture mixture;
    std::vector<scalar_t> whitened;   // MAX_PEEK x q
};

//------------------------------------------------------------------------------
// State, Context
//------------------------------------------------------------------------------
struct State
{
    Phase phase = Phase::Prepare;

    // The Standard machine's record: the series, the cursor, the segments,
    // Prepare's cadence and variogram, Initialize's models.
    standard::State core;

    Extension extension;

    // Starts a pixel, as standard::State::reset().
    void reset(const HarmonicWorkspace& workspace, ProcessingMask mask);

    FitResult take_result() { return core.take_result(); }
};

std::string summary(const State& state);

using Context = standard::Context;

//------------------------------------------------------------------------------
// Transitions
//------------------------------------------------------------------------------

// One test position, the observation at window.stop and the peek from it.
// Fits the window (refitted whenever it holds new observations) with the
// coefficient count of its size, then tests the next min(BFP.MAX_PEEK,
// remaining) observations, in order:
//
//   * no observation left: SeriesEnd;
//   * the test finds a change at peek size m: Change. With BreakStart::
//     Select the change starts at the longest run of observations whose own
//     P is below SELECT_THRESHOLD that ends the m (at least
//     SELECT_MIN_OBSERVATIONS of them; otherwise the first): the ones
//     before it leave the series;
//   * a shortened peek (fewer than MAX_PEEK left) undecided: SeriesEnd;
//   * otherwise the outlier test on the first peek observation: Dropped or
//     Appended.
LookforwardOutcome lookforward_step(const Context& context, State& state);

// lookforward_step(), then the route: after a change, writes the segment
// (change probability 1, its break day the change's first observation, its
// magnitudes the median residual of each band from there to the deciding
// observation) and goes to Restart; at the series end writes it with change
// probability 0 and is Done; otherwise Lookforward.
Phase lookforward(const Context& context, State& state);

struct Machine
{
    using Context = bfp::Context;
    using State = bfp::State;
    using Phase = bfp::Phase;

    static constexpr const char* name = "bfp";

    static Phase step(const Context& context, State& state);
};

FitResult run(const Context& context, State& state, ProcessingMask mask);

// The BFP method's detector: the BFP machine for the Standard procedure, the
// shared whole-series fit for the others.
class Detector
{
public:

    FitResult fit(
        PixelProcedure procedure,
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    );

private:

    State state_;
};

} // namespace bfp
} // namespace ccd
