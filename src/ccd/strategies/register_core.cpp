#include "ccd/strategies/register_core.hpp"

#include <string>

#include "ccd/input/mask/steps.hpp"
#include "ccd/strategies/change_test/chi2.hpp"
#include "ccd/strategies/edge_fit/lasso.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/mode/pyccd.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/noise_floor/variogram.hpp"
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
    registry.add<NoiseFloor, VariogramFloor>("variogram");
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

    registry.add_step<preprocess::KeepQa>("keep_qa");
    registry.add_step<preprocess::AddQa>("add_qa");
    registry.add_step<preprocess::ThermalRange>("thermal_range");
    registry.add_step<preprocess::BandRange>("band_range");
    registry.add_step<preprocess::GreenMedian>("green_median");
    registry.add_step<preprocess::Dedupe>("dedupe");

    // "ccdc": pyccd's algorithm, in physical terms: the shared values'
    // defaults (alpha 1e-4, probabilities, detection bands by role, no
    // statistics cutoff), each strategy's, and pyccd's preprocessing, each
    // step at its defaults (clear and water kept, the thermal range -93.2 to
    // 70.7 degrees Celsius, reflectance in (0, 1), the green median 0.04
    // above). Every procedure's chain begins with the same four steps,
    // keep_qa, thermal_range, band_range and dedupe; Standard then ends,
    // reading thermal through the input's transform; PermanentSnow adds snow
    // back and dedupes; InsufficientClear runs the green median and dedupes.
    config::Config ccdc;
    ccdc.shared = config::shared_defaults();
    for (const auto& [stage, strategy] : std::initializer_list<std::pair<const char*, const char*>>{
             {ModeRule::stage, "pyccd"}, {NoiseFloor::stage, "variogram"},
             {StartSearch::stage, "pyccd"}, {WindowScreen::stage, "tmask"},
             {StabilityFit::stage, "lasso"}, {StabilityRule::stage, "pyccd"},
             {Lookback::stage, "pyccd"}, {MonitoringModel::stage, "lasso"},
             {RefitSchedule::stage, "span-factor"}, {ChangeTest::stage, "chi2"},
             {OutlierRule::stage, "floored"}, {SegmentFit::stage, "model-fit"},
             {EdgeFit::stage, "lasso"}})
        ccdc.stages[stage] = {strategy, registry.defaults(stage, strategy)};
    const auto chain = [&](bool thermal_transform, std::initializer_list<const char*> after)
    {
        config::ProcedureConfig procedure;
        procedure.thermal_transform = thermal_transform;
        for (const char* step : {"keep_qa", "thermal_range", "band_range", "dedupe"})
            procedure.steps.push_back({step, registry.step_defaults(step)});
        for (const char* step : after)
            procedure.steps.push_back({step, registry.step_defaults(step)});
        return procedure;
    };
    ccdc.preprocess["standard"] = chain(true, {});
    ccdc.preprocess["snow"] = chain(false, {"add_qa", "dedupe"});
    ccdc.preprocess["insufficient_clear"] = chain(false, {"green_median", "dedupe"});
    registry.add_preset("ccdc", std::move(ccdc));
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

const config::Registry& library_registry()
{
    static const config::Registry registry = []
    {
        config::Registry r;
        register_core_strategies(r);
        register_extension_strategies(r);
        return r;
    }();
    return registry;
}

} // namespace strategies
} // namespace ccd
