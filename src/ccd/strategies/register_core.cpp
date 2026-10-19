#include "ccd/strategies/register_core.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

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

// A count as a configuration holds it. Past INT64_MAX it is INT64_MAX: no
// window and no solver counts that far, so the two mean the same.
config::Value count(index_t n)
{
    return config::integer(static_cast<std::int64_t>(
        std::min<index_t>(n, std::numeric_limits<std::int64_t>::max())));
}

} // namespace

Strategies build_strategies(const config::Registry& registry, const config::ConfigTree& config)
{
    Strategies s;
    s.model = registry.make<MonitoringModel>(stage_of(config, MonitoringModel::stage));
    s.refit = registry.make<RefitSchedule>(stage_of(config, RefitSchedule::stage));
    return s;
}

Strategies strategies_for(const HarmonicOptions& hoptions, const LassoOptions& loptions)
{
    return build_strategies(core_registry(), config_from_options(hoptions, loptions));
}

config::ConfigTree config_from_options(const HarmonicOptions& hoptions,
                                       const LassoOptions& loptions)
{
    config::ConfigTree tree;
    tree[MonitoringModel::stage] = {
        "lasso",
        {{"max_iter", count(loptions.max_iter)},
         {"alpha", config::real(loptions.alpha)},
         {"tolerance", config::real(loptions.tolerance)},
         {"coef_floor", config::real(loptions.coef_floor)}}};
    tree[RefitSchedule::stage] = {
        "span-factor",
        {{"min_obs", count(hoptions.RETRAIN_MIN_OBS)},
         {"factor", config::real(hoptions.RETRAIN_SPAN_FACTOR)}}};
    return tree;
}

} // namespace strategies
} // namespace ccd
