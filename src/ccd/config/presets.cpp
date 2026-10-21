#include "ccd/config/presets.hpp"

#include <initializer_list>
#include <utility>

#include "ccd/config/shared.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{

Config ccdc_preset(const Registry& registry)
{
    Config ccdc;
    ccdc.shared = shared_defaults();
    for (const auto& [stage, strategy] : std::initializer_list<std::pair<const char*, const char*>>{
             {ModeRule::stage, "pyccd"}, {NoiseFloor::stage, "variogram"},
             {StartSearch::stage, "pyccd"}, {WindowScreen::stage, "tmask"},
             {StabilityFit::stage, "lasso"}, {StabilityRule::stage, "pyccd"},
             {Lookback::stage, "pyccd"}, {MonitoringModel::stage, "lasso"},
             {RefitSchedule::stage, "span-factor"}, {ChangeTest::stage, "chi2"},
             {OutlierRule::stage, "floored"}, {SegmentFit::stage, "model-fit"},
             {EdgeFit::stage, "lasso"}})
        ccdc.stages[stage] = {strategy, registry.defaults(stage, strategy)};
    const auto chain = [&](bool thermal_transform, std::initializer_list<const char*> steps)
    {
        ProcedureConfig procedure;
        procedure.thermal_transform = thermal_transform;
        for (const char* step : steps)
            procedure.steps.push_back({step, registry.step_defaults(step)});
        return procedure;
    };
    ccdc.preprocess["standard"] = chain(true, {"keep_qa", "thermal_range", "band_range", "dedupe"});
    ccdc.preprocess["snow"] = chain(
        false, {"keep_qa", "thermal_range", "band_range", "dedupe", "add_qa", "dedupe",
                "thermal_range"});
    ccdc.preprocess["insufficient_clear"] = chain(
        false, {"keep_qa", "thermal_range", "band_range", "dedupe", "green_median", "dedupe"});
    return ccdc;
}

} // namespace config
} // namespace ccd
