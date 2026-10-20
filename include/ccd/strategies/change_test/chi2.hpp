#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// change_test "chi2": pyccd's test. The peek is peek_size observations,
// scaled by peek_reference_days over the pixel's median revisit gap
// (adjust_peek()); the threshold is change_threshold, lowered for a longer
// peek to keep the default peek's false-alarm rate (adjust_change_threshold(),
// the chi-squared quantile of change_probability with a degree of freedom per
// detection band).
//
// It tests full peeks only, every window. A segment breaks at its last
// test's first peek observation, each band's magnitude the median of that
// test's peek residuals.
//
// It provides the peek's comparison RMSE and one statistic per observation,
// "change_magnitude": the sum over the detection bands of (residual /
// max(variogram, comparison RMSE))^2 (change_magnitude()); the peek is a
// change when every observation's is above the threshold. The
// comparison RMSE is the model's own while the window holds at most
// model_rmse_up_to observations, then the seasonal RMSE: the norm of the
// model's residuals at the seasonal_count fit-window observations nearest the
// peek's last in the year, over seasonal_divisor (seasonal_rmse()).
//------------------------------------------------------------------------------
struct Chi2 final : ChangeTest
{
    struct Params
    {
        index_t peek_size = 6;
        scalar_t peek_reference_days = 16.0;
        scalar_t change_threshold = 15.086272469388987l;
        scalar_t change_probability = 0.99;
        index_t model_rmse_up_to = 24;
        index_t seasonal_count = 24;
        scalar_t seasonal_divisor = 4.0;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit Chi2(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    Capabilities provides() const override;
    std::vector<std::string> statistics() const override;
    std::unique_ptr<Scratch> make_scratch() const override;
    PeekPlan prepare(ArrayView<const std::int64_t, 1> statistics_dates, index_t detection_bands,
                     Scratch& scratch) const override;
    TestOutcome test(const Peek& peek, Scratch& scratch, PeekResult& result) const override;
    index_t break_at(const Peek& peek, const Scratch& scratch) const override;
    scalar_t magnitude(const Peek& peek, const Scratch& scratch, index_t band) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
