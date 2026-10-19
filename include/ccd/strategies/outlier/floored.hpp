#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// outlier "floored": pyccd's rule. The first peek observation is an outlier
// when the change test's statistic for it, whose noise is floored at the
// variogram (chi2's max(variogram, comparison RMSE)), is above `threshold`.
//------------------------------------------------------------------------------
struct Floored final : OutlierRule
{
    struct Params
    {
        scalar_t threshold = 35.888186879610423L;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit Floored(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    OutlierOutcome judge(const Peek& peek, const PeekResult& result) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
