#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/core/band_list.hpp"

#include "ccd/machine/fit.hpp"
#include "ccd/input/masked_series.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace ccd
{

class Model;           // strategies/model.hpp
struct StartSearch;    // strategies/stages.hpp
struct WindowScreen;   // strategies/stages.hpp
struct StabilityRule;  // strategies/stages.hpp
class Scratch;         // strategies/per_thread.hpp

//==============================================================================
//
// The procedures' shared kernels
//
// Plain functions that a procedure composes: the per-band Lasso fit of a
// window, its residuals over a peek, the catch segment written from it, and
// one step of the start-window search (Initialize). Each takes what it reads
// as arguments,
// never a procedure's state, and none fixes a coefficient count or a
// threshold of its own: the caller passes them.
//
// Every fit over a window builds the HarmonicBasis of that window's dates
// (column order in ccd/regression/harmonic_basis.hpp) in the LassoWorkspace.
//
//==============================================================================

// Whether fit_bands() keeps each fit's residuals in LassoScore::residuals.
// The stability rule and Extend's seasonal RMSE read them; a
// returned segment's are never read.
enum class KeepResiduals : bool
{
    No = false,
    Yes = true
};

// Fits row b of `spectral` against `dates` for every b in `bands`, in that
// order, with a num_coefficients-coefficient Lasso model, into fits[b]:
//
//     fits[b].model           the fitted model
//     fits[b].score.rmse      rmse() of its fitted values, num_coefficients
//                             degrees of freedom
//     fits[b].score.magn      0
//     fits[b].score.residuals y - fitted (keep == Yes) or empty (No)
//
// `spectral` is (B, n) and `dates` (n): one window's observations. fits must
// have an entry for every band named; the others are not touched. Each band's
// fit is independent of the others (LassoSolver::fit() resets the workspace).
void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    const BandList& bands,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
);

// fit_bands() of every row of `spectral`, 0 to B - 1; fits is resized to B.
void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
);

// The segment that catches `window` of a series (`dates` (n), `spectral`
// (B, n)) with one num_coefficients-coefficient fit of every band, no
// residuals kept:
//
//     start_day          dates[window.start]
//     end_day            dates[window.stop - 1]
//     break_day          dates[window.stop], or the last date when the
//                        window reaches the end of the series
//     observation_count  window.size()
//     change_probability 0
//     curve_qa           curve_qa
//     bands[b]           band_fit() of the band's fit: magnitude 0
ChangeModel catch_segment(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    const Window& window,
    index_t num_coefficients,
    CurveQA curve_qa,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
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

//------------------------------------------------------------------------------
// Initialize
//------------------------------------------------------------------------------

// What Initialize fits a candidate window with, besides its strategies. A
// procedure fills it from its run's shared values; nothing here reads them
// directly. Values throughout (a BandList is a small fixed-capacity array),
// so a criteria object never refers to what it was filled from.
struct InitializeCriteria
{
    BandList detection_bands;   // the bands fitted and tested for stability
    index_t num_coefficients;   // of the stability fits
    index_t max_coefficients;   // the widest model's: their peek basis width

    // The last day the search sees (StartSearch::last_day()): its
    // exhausted() reads the observations dated through it.
    std::int64_t last_day = std::numeric_limits<std::int64_t>::max();
};

// The observations of `dates` dated up to `day`: dates is sorted.
index_t count_through(ArrayView<const std::int64_t, 1> dates, std::int64_t day);

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

} // namespace ccd
