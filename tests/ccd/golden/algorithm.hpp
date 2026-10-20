#pragma once

// The algorithm options the tests name their variants by, and the
// configuration each describes: pyccd's strategies with those values (the
// defaults are the "ccdc" preset's). A test's way to write a configuration
// by the values it varies; not part of the library.

#include <algorithm>
#include <cstdint>
#include <limits>

#include "ccd/ccd.hpp"
#include "ccd/strategies/stages.hpp"

namespace golden
{

using ccd::index_t;
using ccd::scalar_t;
using ccd::BandList;
using ccd::LassoOptions;

struct Algorithm
{
    index_t MEOW_SIZE = 12;
    index_t PEEK_SIZE = 6;
    std::int64_t DAY_DELTA = 365;
    scalar_t CLEAR_PCT_THRESHOLD = 0.25;
    scalar_t SNOW_PCT_THRESHOLD = 0.75;
    scalar_t OUTLIER_THRESHOLD = 35.888186879610423L;
    scalar_t CHANGE_THRESHOLD = 15.086272469388987l;
    scalar_t T_CONST = 4.89;
    scalar_t PEEK_REFERENCE_DAYS = 16.0;
    std::int64_t VARIOGRAM_MIN_LAG_DAYS = 30;
    index_t RETRAIN_MIN_OBS = 24;
    scalar_t RETRAIN_SPAN_FACTOR = 1.33;
    index_t SEASONAL_RMSE_COUNT = 24;
    scalar_t SEASONAL_RMSE_DIVISOR = 4.0;
    scalar_t CHANGE_PROBABILITY = 0.99;
    index_t COEFFICIENT_MIN = 4;
    index_t COEFFICIENT_MID = 6;
    index_t COEFFICIENT_MAX = 8;
    index_t NUM_OBS_FACTOR = 3;
    BandList TMASK_BANDS = {1, 4};
    LassoOptions lasso;
};

namespace detail
{

// A count as a configuration holds it, INT64_MAX past it.
inline ccd::config::Value count(ccd::index_t n)
{
    return ccd::config::integer(static_cast<std::int64_t>(
        std::min<ccd::index_t>(n, std::numeric_limits<std::int64_t>::max())));
}

} // namespace detail

inline ccd::config::ConfigTree config_of(const Algorithm& o)
{
    ccd::config::ConfigTree tree;
    const ccd::config::Values lasso = {{"max_iter", detail::count(o.lasso.max_iter)},
                          {"alpha", ccd::config::real(o.lasso.alpha)},
                          {"tolerance", ccd::config::real(o.lasso.tolerance)},
                          {"coef_floor", ccd::config::real(o.lasso.coef_floor)}};
    // The model fits: the solver's parameters and the coefficient rule.
    ccd::config::Values fits = lasso;
    fits["coefficient_min"] = detail::count(o.COEFFICIENT_MIN);
    fits["coefficient_mid"] = detail::count(o.COEFFICIENT_MID);
    fits["coefficient_max"] = detail::count(o.COEFFICIENT_MAX);
    fits["num_obs_factor"] = detail::count(o.NUM_OBS_FACTOR);
    tree[ccd::ModeRule::stage] = {
        "pyccd",
        {{"clear_pct_threshold", ccd::config::real(o.CLEAR_PCT_THRESHOLD)},
         {"snow_pct_threshold", ccd::config::real(o.SNOW_PCT_THRESHOLD)}}};
    tree[ccd::NoiseFloor::stage] = {"variogram", {{"min_lag_days", ccd::config::integer(o.VARIOGRAM_MIN_LAG_DAYS)}}};
    tree[ccd::StartSearch::stage] = {
        "pyccd", {{"meow_size", detail::count(o.MEOW_SIZE)}, {"day_delta", ccd::config::integer(o.DAY_DELTA)}}};
    tree[ccd::WindowScreen::stage] = {
        "tmask", {{"bands", ccd::config::bands(o.TMASK_BANDS)}, {"t_const", ccd::config::real(o.T_CONST)}}};
    tree[ccd::StabilityFit::stage] = {"lasso", fits};
    tree[ccd::StabilityRule::stage] = {
        "pyccd",
        {{"change_threshold", ccd::config::real(o.CHANGE_THRESHOLD)},
         {"change_probability", ccd::config::real(o.CHANGE_PROBABILITY)},
         {"peek", ccd::config::integer(0)}}};
    tree[ccd::Lookback::stage] = {
        "pyccd",
        {{"change_threshold", ccd::config::real(o.CHANGE_THRESHOLD)},
         {"change_probability", ccd::config::real(o.CHANGE_PROBABILITY)},
         {"outlier_threshold", ccd::config::real(o.OUTLIER_THRESHOLD)},
         {"peek", ccd::config::integer(0)}}};
    tree[ccd::MonitoringModel::stage] = {"lasso", fits};
    tree[ccd::RefitSchedule::stage] = {
        "span-factor",
        {{"min_obs", detail::count(o.RETRAIN_MIN_OBS)}, {"factor", ccd::config::real(o.RETRAIN_SPAN_FACTOR)}}};
    tree[ccd::ChangeTest::stage] = {
        "chi2",
        {{"peek_size", detail::count(o.PEEK_SIZE)},
         {"peek_reference_days", ccd::config::real(o.PEEK_REFERENCE_DAYS)},
         {"change_threshold", ccd::config::real(o.CHANGE_THRESHOLD)},
         {"change_probability", ccd::config::real(o.CHANGE_PROBABILITY)},
         {"model_rmse_up_to", detail::count(o.RETRAIN_MIN_OBS)},
         {"seasonal_count", detail::count(o.SEASONAL_RMSE_COUNT)},
         {"seasonal_divisor", ccd::config::real(o.SEASONAL_RMSE_DIVISOR)}}};
    tree[ccd::OutlierRule::stage] = {"floored", {{"threshold", ccd::config::real(o.OUTLIER_THRESHOLD)}}};
    tree[ccd::SegmentFit::stage] = {"model-fit", {}};
    ccd::config::Values edges = lasso;
    edges["coefficients"] = detail::count(o.COEFFICIENT_MIN);
    edges["min_observations"] = detail::count(o.MEOW_SIZE);
    tree[ccd::EdgeFit::stage] = {"lasso", edges};
    return tree;
}

} // namespace golden
