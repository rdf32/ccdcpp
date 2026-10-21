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
             {ModeRule::stage, "pyccd"}, {Noise::stage, "variogram"},
             {StartSearch::stage, "pyccd"}, {WindowScreen::stage, "tmask"},
             {StabilityFit::stage, "lasso"}, {StabilityRule::stage, "pyccd"},
             {Lookback::stage, "pyccd"}, {MonitoringModel::stage, "lasso"},
             {RefitSchedule::stage, "span-factor"}, {ChangeTest::stage, "chi2"},
             {OutlierRule::stage, "floored"}, {SegmentReport::stage, "model-fit"},
             {EdgeSegments::stage, "lasso"}})
        ccdc.stages[stage] = {strategy, registry.defaults(stage, strategy)};
    const auto chain = [&](Thermal thermal, std::initializer_list<const char*> steps)
    {
        ProcedureConfig procedure;
        procedure.thermal = thermal;
        for (const char* step : steps)
            procedure.steps.push_back({step, registry.step_defaults(step)});
        return procedure;
    };
    ccdc.preprocess["standard"] =
        chain(Thermal::Celsius, {"keep_qa", "thermal_range", "band_range", "one_per_date"});
    ccdc.preprocess["snow"] =
        chain(Thermal::Stored, {"keep_qa", "thermal_range", "band_range", "one_per_date", "add_qa",
                                "one_per_date", "thermal_range"});
    ccdc.preprocess["insufficient_clear"] =
        chain(Thermal::Stored, {"keep_qa", "thermal_range", "band_range", "one_per_date",
                                "green_median", "one_per_date"});
    return ccdc;
}

} // namespace config
} // namespace ccd
