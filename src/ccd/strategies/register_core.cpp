#include "ccd/strategies/register_core.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "ccd/strategies/refit/span_factor.hpp"

namespace ccd
{
namespace strategies
{

void register_core_strategies(config::Registry& registry)
{
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

} // namespace

Strategies build_strategies(const config::Registry& registry, const config::ConfigTree& config)
{
    Strategies s;
    s.refit = registry.make<RefitSchedule>(stage_of(config, RefitSchedule::stage));
    return s;
}

Strategies strategies_for(const HarmonicOptions& options)
{
    return build_strategies(core_registry(), config_from_options(options));
}

config::ConfigTree config_from_options(const HarmonicOptions& options)
{
    config::ConfigTree tree;
    tree[RefitSchedule::stage] = {
        "span-factor",
        // A window never holds more than INT64_MAX observations, so a larger
        // RETRAIN_MIN_OBS refits at every step, as INT64_MAX does.
        {{"min_obs", config::integer(static_cast<std::int64_t>(std::min<index_t>(
                         options.RETRAIN_MIN_OBS, std::numeric_limits<std::int64_t>::max())))},
         {"factor", config::real(options.RETRAIN_SPAN_FACTOR)}}};
    return tree;
}

} // namespace strategies
} // namespace ccd
