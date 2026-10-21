#include "ccd/strategies/outlier/floored.hpp"

#include <cassert>

#include "ccd/strategies/pyccd/peek_statistics.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<Floored::Params>>& Floored::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"statistic", &Params::statistic, Unit::None, config::any(),
         "the change test's per-observation statistic it reads, by name (chi2's "
         "change_magnitude)"},
        {"threshold", &Params::threshold, Unit::None, config::any(),
         "a first peek observation whose statistic is above this is an outlier", config::Source::Chi2,
         "outlier_probability"},
    };
    return s;
}

Capabilities Floored::needs() const
{
    return {};
}

std::vector<std::string> Floored::statistics() const
{
    return {p_.statistic};
}

OutlierOutcome Floored::judge(const Peek&, const PeekResult& result) const
{
    // The registry builds no configuration whose change test lacks it.
    const std::vector<scalar_t>* statistic = result.find(p_.statistic);
    assert(statistic && !statistic->empty());
    if (detect_outlier((*statistic)[0], p_.threshold))
    {
        return OutlierOutcome::Drop;
    }
    return OutlierOutcome::Keep;
}

} // namespace strategies
} // namespace ccd
