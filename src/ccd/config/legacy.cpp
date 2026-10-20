#include "ccd/config/legacy.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <variant>

#include "ccd/strategies/stages.hpp"

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

// Reads one parameter of a recovered tree into `out`; false when the stage
// has no such parameter of that type.
template<typename T>
bool read(const ConfigTree& tree, const char* stage, const char* name, T& out)
{
    const auto s = tree.find(stage);
    if (s == tree.end())
        return false;
    const auto v = s->second.params.find(name);
    if (v == s->second.params.end())
        return false;
    if constexpr (std::is_same_v<T, scalar_t>)
    {
        if (!std::holds_alternative<double>(v->second))
            return false;
        out = std::get<double>(v->second);
    }
    else if constexpr (std::is_same_v<T, BandList>)
    {
        if (!std::holds_alternative<std::vector<std::int64_t>>(v->second))
            return false;
        const auto& list = std::get<std::vector<std::int64_t>>(v->second);
        if (list.size() > CCD_MAX_BANDS)
            return false;
        out = BandList{};
        for (const std::int64_t b : list)
        {
            if (b < 0)
                return false;
            out.push_back(static_cast<index_t>(b));
        }
    }
    else
    {
        if (!std::holds_alternative<std::int64_t>(v->second))
            return false;
        const std::int64_t i = std::get<std::int64_t>(v->second);
        if constexpr (std::is_unsigned_v<T>)
        {
            if (i < 0)
                return false;
        }
        out = static_cast<T>(i);
    }
    return true;
}

} // namespace

ConfigTree legacy_config(const Legacy& o)
{
    ConfigTree tree;
    const Values lasso = {{"max_iter", count(o.lasso.max_iter)},
                          {"alpha", real(o.lasso.alpha)},
                          {"tolerance", real(o.lasso.tolerance)},
                          {"coef_floor", real(o.lasso.coef_floor)}};
    // The model fits: the solver's parameters and the coefficient rule.
    Values fits = lasso;
    fits["coefficient_min"] = count(o.COEFFICIENT_MIN);
    fits["coefficient_mid"] = count(o.COEFFICIENT_MID);
    fits["coefficient_max"] = count(o.COEFFICIENT_MAX);
    fits["num_obs_factor"] = count(o.NUM_OBS_FACTOR);
    tree[ModeRule::stage] = {
        "pyccd",
        {{"clear_pct_threshold", real(o.CLEAR_PCT_THRESHOLD)},
         {"snow_pct_threshold", real(o.SNOW_PCT_THRESHOLD)}}};
    tree[NoiseFloor::stage] = {"variogram", {{"min_lag_days", integer(o.VARIOGRAM_MIN_LAG_DAYS)}}};
    tree[StartSearch::stage] = {
        "pyccd", {{"meow_size", count(o.MEOW_SIZE)}, {"day_delta", integer(o.DAY_DELTA)}}};
    tree[WindowScreen::stage] = {
        "tmask", {{"bands", bands(o.TMASK_BANDS)}, {"t_const", real(o.T_CONST)}}};
    tree[StabilityFit::stage] = {"lasso", fits};
    tree[StabilityRule::stage] = {
        "pyccd",
        {{"change_threshold", real(o.CHANGE_THRESHOLD)},
         {"change_probability", real(o.CHANGE_PROBABILITY)},
         {"peek", integer(0)}}};
    tree[Lookback::stage] = {
        "pyccd",
        {{"change_threshold", real(o.CHANGE_THRESHOLD)},
         {"change_probability", real(o.CHANGE_PROBABILITY)},
         {"outlier_threshold", real(o.OUTLIER_THRESHOLD)},
         {"peek", integer(0)}}};
    tree[MonitoringModel::stage] = {"lasso", fits};
    tree[RefitSchedule::stage] = {
        "span-factor",
        {{"min_obs", count(o.RETRAIN_MIN_OBS)}, {"factor", real(o.RETRAIN_SPAN_FACTOR)}}};
    tree[ChangeTest::stage] = {
        "chi2",
        {{"peek_size", count(o.PEEK_SIZE)},
         {"peek_reference_days", real(o.PEEK_REFERENCE_DAYS)},
         {"change_threshold", real(o.CHANGE_THRESHOLD)},
         {"change_probability", real(o.CHANGE_PROBABILITY)},
         {"model_rmse_up_to", count(o.RETRAIN_MIN_OBS)},
         {"seasonal_count", count(o.SEASONAL_RMSE_COUNT)},
         {"seasonal_divisor", real(o.SEASONAL_RMSE_DIVISOR)}}};
    tree[OutlierRule::stage] = {"floored", {{"threshold", real(o.OUTLIER_THRESHOLD)}}};
    tree[SegmentFit::stage] = {"model-fit", {}};
    Values edges = lasso;
    edges["coefficients"] = count(o.COEFFICIENT_MIN);
    edges["min_observations"] = count(o.MEOW_SIZE);
    tree[EdgeFit::stage] = {"lasso", edges};
    return tree;
}

std::optional<Legacy> recover(const ConfigTree& tree)
{
    Legacy o;
    std::int64_t max_iter = 0;
    const bool read_all =
        read(tree, StartSearch::stage, "meow_size", o.MEOW_SIZE) &&
        read(tree, ChangeTest::stage, "peek_size", o.PEEK_SIZE) &&
        read(tree, StartSearch::stage, "day_delta", o.DAY_DELTA) &&
        read(tree, ModeRule::stage, "clear_pct_threshold", o.CLEAR_PCT_THRESHOLD) &&
        read(tree, ModeRule::stage, "snow_pct_threshold", o.SNOW_PCT_THRESHOLD) &&
        read(tree, OutlierRule::stage, "threshold", o.OUTLIER_THRESHOLD) &&
        read(tree, ChangeTest::stage, "change_threshold", o.CHANGE_THRESHOLD) &&
        read(tree, WindowScreen::stage, "t_const", o.T_CONST) &&
        read(tree, ChangeTest::stage, "peek_reference_days", o.PEEK_REFERENCE_DAYS) &&
        read(tree, NoiseFloor::stage, "min_lag_days", o.VARIOGRAM_MIN_LAG_DAYS) &&
        read(tree, RefitSchedule::stage, "min_obs", o.RETRAIN_MIN_OBS) &&
        read(tree, RefitSchedule::stage, "factor", o.RETRAIN_SPAN_FACTOR) &&
        read(tree, ChangeTest::stage, "seasonal_count", o.SEASONAL_RMSE_COUNT) &&
        read(tree, ChangeTest::stage, "seasonal_divisor", o.SEASONAL_RMSE_DIVISOR) &&
        read(tree, ChangeTest::stage, "change_probability", o.CHANGE_PROBABILITY) &&
        read(tree, MonitoringModel::stage, "coefficient_min", o.COEFFICIENT_MIN) &&
        read(tree, MonitoringModel::stage, "coefficient_mid", o.COEFFICIENT_MID) &&
        read(tree, MonitoringModel::stage, "coefficient_max", o.COEFFICIENT_MAX) &&
        read(tree, MonitoringModel::stage, "num_obs_factor", o.NUM_OBS_FACTOR) &&
        read(tree, WindowScreen::stage, "bands", o.TMASK_BANDS) &&
        read(tree, MonitoringModel::stage, "max_iter", max_iter) &&
        read(tree, MonitoringModel::stage, "alpha", o.lasso.alpha) &&
        read(tree, MonitoringModel::stage, "tolerance", o.lasso.tolerance) &&
        read(tree, MonitoringModel::stage, "coef_floor", o.lasso.coef_floor);
    if (!read_all || max_iter < 0)
        return std::nullopt;
    o.lasso.max_iter = static_cast<index_t>(max_iter);
    // Every other value, and every strategy, must be the one these describe.
    if (!(legacy_config(o) == tree))
        return std::nullopt;
    return o;
}

ConfigTree preset(const std::string& name)
{
    if (name == "ccdc")
        return legacy_config(Legacy{});
    throw std::invalid_argument("no built-in preset '" + name + "'; there is \"ccdc\"");
}

} // namespace config
} // namespace ccd
