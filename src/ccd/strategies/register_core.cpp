#include "ccd/strategies/register_core.hpp"

#include <stdexcept>

#include "ccd/strategies/change_test/chi2.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/refit/span_factor.hpp"

namespace ccd
{
namespace strategies
{

void register_core_strategies(config::Registry& registry)
{
    registry.add<MonitoringModel, Lasso>("lasso");
    registry.add<RefitSchedule, SpanFactor>("span-factor");
    registry.add<ChangeTest, Chi2>("chi2");
}

const config::Registry& core_registry()
{
    static const config::Registry registry = []
    {
        config::Registry r;
        register_core_strategies(r);
        return r;
    }();
    return registry;
}

namespace
{

const config::StageConfig& stage_of(const config::ConfigTree& config, const char* stage)
{
    const auto it = config.find(stage);
    if (it == config.end())
        throw std::invalid_argument(std::string("the configuration has no '") + stage +
                                    "' stage");
    return it->second;
}

} // namespace

Strategies build_strategies(const config::Registry& registry, const config::ConfigTree& config)
{
    Strategies s;
    s.model = registry.make<MonitoringModel>(stage_of(config, MonitoringModel::stage));
    s.refit = registry.make<RefitSchedule>(stage_of(config, RefitSchedule::stage));
    s.change_test = registry.make<ChangeTest>(stage_of(config, ChangeTest::stage));
    return s;
}

} // namespace strategies
} // namespace ccd
