#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/core/window.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/processing_mask.hpp"
#include "ccd/input/procedure.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/results/segment.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

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

// The whole-series fit of a PermanentSnow or InsufficientClear pixel: one
// segment over every observation `mask` keeps, every band fitted with
// num_coefficients coefficients:
//
//     start_day, end_day  the first and last date of the whole series (not
//                         only the kept observations), as pyccd
//     break_day           end_day
//     observation_count   mask.count()
//     change_probability  0
//     curve_qa            PersistentSnow (54) or InsufficientClear (44)
//     bands[b].score      rmse, magnitude 0, no residuals
//
// The result carries `mask`, also when there are fewer than min_observations
// kept observations and so no segment.
//
// `procedure` is PermanentSnow or InsufficientClear; `mask` is that
// procedure's processing mask over series's series, built by the
// caller (Run::pixel builds it with the procedure's filter chain).
FitResult fit_whole_series(
    PixelProcedure procedure,
    const PixelSeries& series,
    ProcessingMask mask,
    index_t min_observations,
    index_t num_coefficients,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
);

namespace strategies
{

//------------------------------------------------------------------------------
// edge_segments "lasso": pyccd's. Every band of the window, or of the whole
// series, a Lasso fit of `coefficients` coefficients (catch_segment(),
// fit_whole_series()), solved with the Lasso parameters; the whole series
// has no segment with fewer than min_observations kept observations. The
// Start and End segments are written when more than a peek of observations
// is before the first stable start, or after the last segment, and more
// than `coefficients`, so that the fit's RMSE has a degree of freedom (a
// peek_size below the shared coefficients.min leaves those observations
// without one).
//------------------------------------------------------------------------------
struct LassoEdges final : EdgeSegments
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
