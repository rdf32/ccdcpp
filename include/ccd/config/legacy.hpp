#pragma once

#include <cstdint>
#include <optional>

#include "ccd/config/registry.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/types.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// Legacy
//
// The algorithm's parameters as 0.10's HarmonicOptions and LassoOptions held
// them, and the configuration they describe. A run's configuration is a
// ConfigTree; Legacy is what keeps 0.10's resume records valid (a record's
// digest, results/record.hpp, is 0.10's formula when the run's configuration
// is one Legacy describes) and how tests name 0.10's option variants. Its
// defaults are pyccd's, the "ccdc" preset (preset()).
//
//==============================================================================

struct Legacy
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

// The configuration `legacy` describes: pyccd's strategies with its values.
ConfigTree legacy_config(const Legacy& legacy);

// The Legacy whose configuration `tree` is, if any (every stage pyccd's, the
// values one Legacy sets in several stages equal).
std::optional<Legacy> recover(const ConfigTree& tree);

// The configuration of a preset built in ("ccdc": legacy_config(Legacy{})).
// Throws std::invalid_argument for another name.
ConfigTree preset(const std::string& name);

} // namespace config
} // namespace ccd
