#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/results/segment.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

//==============================================================================
//
// The per-band Lasso fits
//
// The fits of the Lasso strategies (the model and stability fit below, the
// edge_segments "lasso"): every band of a window fitted with a Lasso model, its
// residuals over a peek, and a band's fit as a segment reports it. Each
// takes what it reads as arguments, and none fixes a coefficient count of
// its own: the caller passes it.
//
// Every fit over a window builds the HarmonicBasis of that window's dates
// (column order in ccd/regression/harmonic_basis.hpp) in the LassoWorkspace.
//
//==============================================================================

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

// fit_bands() of every row of `spectral` (every_band()); fits is resized to B.
void fit_bands(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t num_coefficients,
    KeepResiduals keep,
    LassoWorkspace& lworkspace,
    LassoSolver& solver,
    std::vector<LassoResult>& fits
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

// The BandFit of a Lasso fit: its model's intercept, coefficients and
// iteration count, its RMSE and its magnitude.
BandFit band_fit(const LassoResult& fit);

namespace strategies
{

//------------------------------------------------------------------------------
// model and stability_fit "lasso": pyccd's monitoring and stability fits. Each band is a Lasso regression on
// the harmonic basis of the window's dates (regression/harmonic_basis.hpp),
// solved by coordinate descent (regression/lasso_solver.hpp) with these
// parameters; its RMSE has the model's coefficient count as its degrees of
// freedom. Its coefficients follow the rule of coefficient_min, _mid, _max
// and num_obs_factor (CoefficientRule).
//------------------------------------------------------------------------------
struct Lasso final : MonitoringModel, StabilityFit
{
    struct Params
    {
        index_t max_iter = 1000;
        scalar_t alpha = 1.0;
        scalar_t tolerance = 1e-4;
        scalar_t coef_floor = 1.0;
        index_t coefficient_min = 4;
        index_t coefficient_mid = 6;
        index_t coefficient_max = 8;
        index_t num_obs_factor = 3;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit Lasso(const Params& p);

    Capabilities provides() const override;
    std::unique_ptr<Model> make_model() const override;
    CoefficientRule coefficients() const override;

    // The solver's options these parameters are.
    LassoOptions solver_options() const;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
