#include "residual_cap.hpp"

#include "ccd/config/registry.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<ResidualCap::Params>>& ResidualCap::schema()
{
    static const std::vector<config::Field<Params>> s = {
        {"limit", &Params::limit, config::Unit::ReflectanceDelta, config::greater_than(0),
         "a first peek observation with a residual above this in any detection band is an "
         "outlier", config::Source::Own, nullptr, config::real(0.2)},
    };
    return s;
}

// It reads the model's peek residuals.
Capabilities ResidualCap::needs() const
{
    return Capability::PeekResiduals;
}

OutlierOutcome ResidualCap::judge(const Peek& peek, const PeekResult& /*result*/) const
{
    for (const index_t band : peek.bands)
        if (peek.residuals[band][0] > p_.limit)
            return OutlierOutcome::Drop;
    return OutlierOutcome::Keep;
}

void register_outlier_residual_cap(config::Registry& registry)
{
    registry.add<OutlierRule, ResidualCap>("residual-cap");
}

} // namespace strategies
} // namespace ccd
