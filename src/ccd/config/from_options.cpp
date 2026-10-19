#include "ccd/config/from_options.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "ccd/strategies/register_core.hpp"

namespace ccd
{
namespace config
{

namespace
{

// A count as a configuration holds it. Past INT64_MAX it is INT64_MAX: no
// window and no solver counts that far, so the two mean the same.
Value count(index_t n)
{
    return integer(static_cast<std::int64_t>(
        std::min<index_t>(n, std::numeric_limits<std::int64_t>::max())));
}

} // namespace

Strategies strategies_for(const HarmonicOptions& hoptions, const LassoOptions& loptions)
{
    return strategies::build_strategies(strategies::core_registry(),
                                        from_options(hoptions, loptions));
}

ConfigTree from_options(const HarmonicOptions& hoptions, const LassoOptions& loptions)
{
    ConfigTree tree;
    tree[MonitoringModel::stage] = {
        "lasso",
        {{"max_iter", count(loptions.max_iter)},
         {"alpha", real(loptions.alpha)},
         {"tolerance", real(loptions.tolerance)},
         {"coef_floor", real(loptions.coef_floor)}}};
    tree[RefitSchedule::stage] = {
        "span-factor",
        {{"min_obs", count(hoptions.RETRAIN_MIN_OBS)},
         {"factor", real(hoptions.RETRAIN_SPAN_FACTOR)}}};
    tree[ChangeTest::stage] = {
        "chi2",
        {{"peek_size", count(hoptions.PEEK_SIZE)},
         {"peek_reference_days", real(hoptions.PEEK_REFERENCE_DAYS)},
         {"change_threshold", real(hoptions.CHANGE_THRESHOLD)},
         {"change_probability", real(hoptions.CHANGE_PROBABILITY)},
         {"model_rmse_up_to", count(hoptions.RETRAIN_MIN_OBS)},
         {"seasonal_count", count(hoptions.SEASONAL_RMSE_COUNT)},
         {"seasonal_divisor", real(hoptions.SEASONAL_RMSE_DIVISOR)}}};
    return tree;
}

} // namespace config
} // namespace ccd
