#include "ccd/strategies/outlier/floored.hpp"

#include <cassert>

#include "ccd/core/logger.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<Floored::Params>>& Floored::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"threshold", &Params::threshold, Unit::None, config::any(),
         "a first peek observation whose statistic is above this is an outlier"},
    };
    return s;
}

Capabilities Floored::needs() const
{
    return Capability::PeekStatistic;
}

OutlierOutcome Floored::judge(const Peek&, const PeekResult& result) const
{
    assert(!result.statistic.empty());
    if (detect_outlier(result.statistic[0], p_.threshold))
    {
        LOG_DEBUG("mag: " << result.statistic[0] << " thresh: " << p_.threshold);
        return OutlierOutcome::Drop;
    }
    return OutlierOutcome::Keep;
}

} // namespace strategies
} // namespace ccd
