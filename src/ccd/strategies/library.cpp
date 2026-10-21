#include "ccd/strategies/library.hpp"

#include <string>

#include "ccd/config/presets.hpp"
#include "ccd/input/mask/steps.hpp"
#include "ccd/strategies/change_test/chi2.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/mode/pyccd.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/noise/variogram.hpp"
#include "ccd/strategies/outlier/floored.hpp"
#include "ccd/strategies/refit/span_factor.hpp"
#include "ccd/strategies/segment_report/model_fit.hpp"
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
    registry.add<Noise, VariogramNoise>("variogram");
    registry.add<StartSearch, PyccdSearch>("pyccd");
    registry.add<WindowScreen, Tmask>("tmask");
    registry.add<StabilityFit, Lasso>("lasso");
    registry.add<StabilityRule, PyccdStability>("pyccd");
    registry.add<Lookback, PyccdLookback>("pyccd");
    registry.add<MonitoringModel, Lasso>("lasso");
    registry.add<RefitSchedule, SpanFactor>("span-factor");
    registry.add<ChangeTest, Chi2>("chi2");
    registry.add<OutlierRule, Floored>("floored");
    registry.add<SegmentReport, ModelFit>("model-fit");
    registry.add<EdgeSegments, LassoEdges>("lasso");

    registry.add_step<preprocess::KeepQa>("keep_qa");
    registry.add_step<preprocess::AddQa>("add_qa");
    registry.add_step<preprocess::ThermalRange>("thermal_range");
    registry.add_step<preprocess::BandRange>("band_range");
    registry.add_step<preprocess::GreenMedian>("green_median");
    registry.add_step<preprocess::OnePerDate>("one_per_date");

    registry.add_preset("ccdc", config::ccdc_preset(registry));
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
