#pragma once

#include <string>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// outlier "floored": pyccd's rule. The first peek observation is an outlier
// when the change test's statistic named `statistic` for it is above
// `threshold`: chi2's "change_magnitude" (the default), whose noise is
// floored at the variogram (max(variogram, comparison RMSE)); the floor is
// the test's.
//------------------------------------------------------------------------------
struct Floored final : OutlierRule
{
    struct Params
    {
        std::string statistic = "change_magnitude";
        scalar_t threshold = 35.888186879610423L;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit Floored(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    std::vector<std::string> statistics() const override;
    OutlierOutcome judge(const Peek& peek, const PeekResult& result) const override;
    // The first peek observation's statistic and the threshold.
    std::vector<scalar_t> explain(const PeekResult& result) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
