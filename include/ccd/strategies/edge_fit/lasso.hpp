#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// edge_fit "lasso": pyccd's. Every band of the window, or of the whole
// series, a Lasso fit of `coefficients` coefficients (catch_segment(),
// fit_whole_series()), solved with the Lasso parameters; the whole series
// has no segment with fewer than min_observations kept observations. The
// Start and End segments are written when more than a peek of observations
// is before the first stable start, or after the last segment, and more
// than `coefficients`, so that the fit's RMSE has a degree of freedom (a
// peek_size below coefficient_min leaves those observations without one).
//------------------------------------------------------------------------------
struct LassoEdges final : EdgeFit
{
    struct Params
    {
        index_t coefficients = 4;
        index_t min_observations = 12;
        index_t max_iter = 1000;
        scalar_t alpha = 1.0;
        scalar_t tolerance = 1e-4;
        scalar_t coef_floor = 1.0;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit LassoEdges(const Params& p) : p_(p) {}

    std::unique_ptr<Scratch> make_scratch() const override;
    bool start_due(index_t before, index_t peek) const override
    {
        return before > peek && before > p_.coefficients;
    }
    bool end_due(index_t after, index_t peek, bool) const override
    {
        return after > peek && after > p_.coefficients;
    }
    ChangeModel catch_window(ArrayView<const std::int64_t, 1> dates,
                             ArrayView<const scalar_t, 2> spectral, const Window& window,
                             CurveQA curve_qa, Scratch& scratch) const override;
    FitResult whole_series(PixelProcedure procedure, const PixelSeries& pixel,
                           ProcessingMask mask, Scratch& scratch) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
