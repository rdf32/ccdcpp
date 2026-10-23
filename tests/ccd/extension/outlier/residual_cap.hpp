#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{
class Registry;
}

namespace strategies
{

//------------------------------------------------------------------------------
// outlier "residual-cap": the first peek observation is an outlier when its
// residual in any detection band is above `limit`, a difference of
// reflectance. The worked example of docs/writing-a-strategy.md.
//------------------------------------------------------------------------------
struct ResidualCap final : OutlierRule
{
    struct Params
    {
        scalar_t limit = 2000.0;   // on the default DataSpec (Collection-1 DN): 0.2
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit ResidualCap(const Params& p) : p_(p) {}

    Capabilities needs() const override;
    OutlierOutcome judge(const Peek& peek, const PeekResult& result) const override;

private:
    Params p_;
};

// Registers ResidualCap as outlier "residual-cap".
void register_outlier_residual_cap(config::Registry& registry);

} // namespace strategies
} // namespace ccd
