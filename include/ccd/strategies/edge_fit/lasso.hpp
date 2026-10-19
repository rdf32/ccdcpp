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
// has no segment with fewer than min_observations kept observations.
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
    ChangeModel catch_window(ArrayView<const std::int64_t, 1> dates,
                             ArrayView<const scalar_t, 2> spectral, const Window& window,
                             CurveQA curve_qa, Scratch& scratch) const override;
    FitResult whole_series(PixelProcedure procedure, const HarmonicWorkspace& pixel,
                           ProcessingMask mask, Scratch& scratch) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
