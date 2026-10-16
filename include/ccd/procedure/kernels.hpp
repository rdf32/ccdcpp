#pragma once

#include <cstdint>
#include <vector>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/band_list.hpp"

#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/series.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace ccd
{

//==============================================================================
//
// The procedures' shared kernels
//
// Plain functions that a procedure composes: the per-band Lasso fit of a
// window, the catch segment written from it, and the start-window search
// (Initialise). Each takes what it reads as arguments, never a procedure's
// state, and none fixes a coefficient count or a threshold of its own: the
// caller passes them.
//
// Every fit over a window builds the HarmonicBasis of that window's dates
// (column order in ccd/regression/harmonic_basis.hpp) in the LassoWorkspace.
//
//==============================================================================

// Whether fit_bands() keeps each fit's residuals in LassoScore::residuals.
// Initialise's stability check and Extend's seasonal RMSE read them; a
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
//     bands[b].score     rmse, magnitude 0
ChangeModel catch_segment(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    const Window& window,
    index_t num_coefficients,
    CurveQA curve_qa,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
);

//------------------------------------------------------------------------------
// Initialise
//------------------------------------------------------------------------------

// What the start-window search tests a window against, besides the series
// and its variogram. A procedure fills it from its options and its own
// derived values; nothing here is read from the options directly.
struct InitializeCriteria
{
    index_t meow_size;                 // fewest observations in a window
    std::int64_t day_delta;            // shortest span, days
    const BandList& tmask_bands;       // Tmask's bands
    scalar_t t_const;                  // Tmask's residual threshold, in variograms
    const BandList& detection_bands;   // the bands fitted and tested for stability
    index_t num_coefficients;          // of the stability fits
    scalar_t change_threshold;         // stable() below this
};

// Searches for a stable start window from `window`, which holds at least
// criteria.meow_size observations.
//
// One candidate per step, in this order: a window spanning fewer than
// day_delta days grows by one; Tmask's outliers in it are dropped from
// `series` if what is left still spans day_delta days and holds meow_size
// observations (Series::drop_outliers_if()), and the window grows instead
// when Tmask flags every observation or what is left fails; the detection
// bands are fitted (fit_bands(), residuals kept) into models[band], and the
// window shifts by one unless stable() holds.
//
// Returns true with `window` the stable window and `models` its fits, or
// false once window.stop + meow_size reaches series.size() (exhausted), with
// `window` where the search stopped. Only the detection bands' entries of
// `models` are written.
bool initialize(
    Series& series,
    Window& window,
    const InitializeCriteria& criteria,
    const Variogram& variogram,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& models
);

} // namespace ccd
