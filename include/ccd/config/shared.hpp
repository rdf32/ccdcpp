#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "ccd/config/params.hpp"

namespace ccd
{

namespace config
{

// The shared statistics_end that puts every date inside the statistics
// window: the largest ordinal. A configuration writes "no cutoff" as this number (TOML
// and Python carry it as the integer 9223372036854775807).
constexpr std::int64_t NO_STATISTICS_END = std::numeric_limits<std::int64_t>::max();

//==============================================================================
//
// Shared
//
// The values of a configuration that several stages read: written once, in
// the configuration's `shared` section, and read by every strategy and
// preprocessing step whose schema names them (Source::Shared, Source::Chi2)
// and by the machine. No stage holds a copy.
//
//   detection_bands      bands (roles)  the change magnitude's bands, in the
//                                       order it sums them; their count is
//                                       the chi-squared degrees of freedom of
//                                       the derived thresholds
//   change_probability   none           change_test, stability_rule and
//                                       lookback: their change_probability,
//                                       and their change_threshold, its
//                                       chi-squared quantile
//   outlier_probability  none           outlier's threshold and lookback's
//                                       outlier_threshold, its quantile
//   alpha, coef_floor    reflectance    the Lasso's (model, stability_fit,
//   max_iter, tolerance  none           edge_segments)
//   coefficient_min/_mid/_max,          the coefficient rule of model and
//   num_obs_factor       none           stability_fit; edge_segments'
//                                       coefficients is coefficient_min
//   min_observations     observations   the fewest observations a curve is
//                                       fitted on: start_search's window
//                                       (pyccd's meow_size) and the
//                                       observations it needs after it, and
//                                       edge_segments' least for a whole-series
//                                       curve
//   statistics_end       ordinal day    the statistics cutoff: the procedure
//                                       choice, the cadence, the variogram
//                                       and the green median step read only
//                                       the observations dated <=
//                                       statistics_end; NO_STATISTICS_END
//                                       for none
//
// A Shared holds them resolved against an input: in its units, the bands as
// its rows. Its member defaults are the "ccdc" values on the default Input
// (Collection-1 DN, the Landsat rows); the configuration's defaults are the
// physical ones, shared_schema()'s.
//
//==============================================================================

struct Shared
{
    BandList detection_bands = {1, 2, 3, 4, 5};
    double change_probability = 0.99;
    double outlier_probability = 1.0 - 1e-6;
    double alpha = 1.0;
    double coef_floor = 1.0;
    index_t max_iter = 1000;
    double tolerance = 1e-4;
    index_t coefficient_min = 4;
    index_t coefficient_mid = 6;
    index_t coefficient_max = 8;
    index_t num_obs_factor = 3;
    index_t min_observations = 12;
    std::int64_t statistics_end = NO_STATISTICS_END;
};

bool operator==(const Shared& a, const Shared& b);
inline bool operator!=(const Shared& a, const Shared& b) { return !(a == b); }

// The shared values' schema, with their physical defaults ("ccdc"'s).
const std::vector<Field<Shared>>& shared_schema();

// Every shared value at its physical default, by name.
Values shared_defaults();

} // namespace config
} // namespace ccd
