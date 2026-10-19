#include "ccd/strategies/register_core.hpp"

#include <stdexcept>
#include <string>

#include "ccd/strategies/change_test/chi2.hpp"
#include "ccd/strategies/edge_fit/lasso.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/mode/pyccd.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/outlier/floored.hpp"
#include "ccd/strategies/refit/span_factor.hpp"
#include "ccd/strategies/segment_fit/model_fit.hpp"
#include "ccd/strategies/stability_rule/pyccd.hpp"
#include "ccd/strategies/start_search/pyccd.hpp"
#include "ccd/strategies/window_screen/tmask.hpp"

namespace ccd
{
namespace strategies
{

void register_core_strategies(config::Registry& registry)
{
    registry.add<ModeRule, PyccdMode>("pyccd");
    registry.add<StartSearch, PyccdSearch>("pyccd");
    registry.add<WindowScreen, Tmask>("tmask");
    registry.add<StabilityFit, Lasso>("lasso");
    registry.add<StabilityRule, PyccdStability>("pyccd");
    registry.add<Lookback, PyccdLookback>("pyccd");
    registry.add<MonitoringModel, Lasso>("lasso");
    registry.add<RefitSchedule, SpanFactor>("span-factor");
    registry.add<ChangeTest, Chi2>("chi2");
    registry.add<OutlierRule, Floored>("floored");
    registry.add<SegmentFit, ModelFit>("model-fit");
    registry.add<EdgeFit, LassoEdges>("lasso");
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
    s.mode = registry.make<ModeRule>(stage_of(config, ModeRule::stage));
    s.start_search = registry.make<StartSearch>(stage_of(config, StartSearch::stage));
    s.window_screen = registry.make<WindowScreen>(stage_of(config, WindowScreen::stage));
    s.stability_fit = registry.make<StabilityFit>(stage_of(config, StabilityFit::stage));
    s.stability_rule = registry.make<StabilityRule>(stage_of(config, StabilityRule::stage));
    s.lookback = registry.make<Lookback>(stage_of(config, Lookback::stage));
    s.model = registry.make<MonitoringModel>(stage_of(config, MonitoringModel::stage));
    s.refit = registry.make<RefitSchedule>(stage_of(config, RefitSchedule::stage));
    s.change_test = registry.make<ChangeTest>(stage_of(config, ChangeTest::stage));
    s.outlier = registry.make<OutlierRule>(stage_of(config, OutlierRule::stage));
    s.segment_fit = registry.make<SegmentFit>(stage_of(config, SegmentFit::stage));
    s.edge_fit = registry.make<EdgeFit>(stage_of(config, EdgeFit::stage));

    // Every strategy's needs, from the stages before it.
    const auto named = [&](const char* stage)
    { return std::string(stage) + " '" + config.at(stage).strategy + "'"; };
    const auto require = [](Capabilities needs, Capabilities provided, const std::string& who,
                            const std::string& providers)
    {
        const Capabilities missing = needs.missing_from(provided);
        if (!missing.empty())
            throw std::invalid_argument(who + " needs " + missing.names() + ", which " +
                                        providers + " does not provide");
    };
    require(s.stability_rule->needs(), s.stability_fit->provides(), named(StabilityRule::stage),
            named(StabilityFit::stage));
    require(s.lookback->needs(), s.stability_fit->provides(), named(Lookback::stage),
            named(StabilityFit::stage));
    const Capabilities model = s.model->provides();
    require(s.change_test->needs(), model, named(ChangeTest::stage), named(MonitoringModel::stage));
    require(s.outlier->needs(), model | s.change_test->provides(), named(OutlierRule::stage),
            named(MonitoringModel::stage) + " with " + named(ChangeTest::stage));
    return s;
}

} // namespace strategies
} // namespace ccd
