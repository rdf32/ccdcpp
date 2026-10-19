#pragma once

#include <cstdint>
#include <vector>

#include "ccd/array_view.hpp"
#include "ccd/band_list.hpp"
#include "ccd/types.hpp"

#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/kernels.hpp"

namespace ccd
{

//==============================================================================
//
// The model contract
//
// One pixel's fitted models, one per band: the monitoring model's, which
// Extend tests each peek against (fitted on the model window when the refit
// schedule says so), and the stability fit's, which the stability rule
// judges and Lookback tests against. Either is asked for a peek's residuals,
// each band's RMSE and in-window residuals, and each band's coefficients.
//
// A Model holds one pixel's fits and the buffers its fits and predictions
// reuse: one per thread, made by the run's strategy for its stage
// (MonitoringModel::make_model(), StabilityFit::make_model(),
// strategies/stages.hpp) and kept across pixels. A fit and a peek take every band they act on at once; a band's
// RMSE, residuals and segment fit are one call each.
//
//==============================================================================

class Model
{
public:

    virtual ~Model() = default;

    // Fits row b of `spectral` (B, n) against `dates` (n), one window's
    // observations, for every b in `bands`, in that order, with num_coef
    // coefficients; the pixel's models have at most max_coef. Keeps each
    // band's in-window residuals (residuals()) when `keep` is Yes. The other
    // bands' fits are not touched.
    virtual void fit(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        index_t num_coef,
        index_t max_coef,
        const BandList& bands,
        KeepResiduals keep
    ) = 0;

    // |y - prediction| of band b's fit at each peek observation, for every b
    // in `bands`, into residuals[b] (resized to the peek's length; the other
    // entries are not touched). `dates` (p) and `spectral` (B, p) are the
    // peek's observations.
    virtual void peek_residuals(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        const BandList& bands,
        std::vector<std::vector<scalar_t>>& residuals
    ) = 0;

    // Band b's fit: its RMSE over the window it was fitted on, its residuals
    // there (y - fitted, by window position; kept only by a fit with
    // KeepResiduals::Yes), and the segment's coefficients (magnitude 0).
    virtual scalar_t rmse(index_t band) const = 0;
    virtual const std::vector<scalar_t>& residuals(index_t band) const = 0;
    virtual BandFit band_fit(index_t band) const = 0;
};

} // namespace ccd
