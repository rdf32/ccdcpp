#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/regression/coefficients.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace ccd
{

namespace config
{

// The shared statistics_end that puts every date inside the statistics
// window: the largest ordinal. A configuration writes it "none"
// (statistics_end's Keyword).
constexpr std::int64_t NO_STATISTICS_END = std::numeric_limits<std::int64_t>::max();

//==============================================================================
//
// Shared
//
// The values of a configuration that several stages read: written once, in
// the configuration's `shared` section, and read by every strategy and
// preprocessing step whose schema names them (Source::Shared, Source::Chi2)
// and by the machine. No stage holds a copy. Two tables group the values of
// one rule each, [shared.coefficients] and [shared.lasso]; a configuration
// names a table's value by its path, "lasso.alpha":
//
//     config.shared["lasso.alpha"] = config::real(1e-5);
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
//   min_observations     observations   the fewest observations a curve is
//                                       fitted on: start_search's window
//                                       (pyccd's meow_size) and the
//                                       observations it needs after it, and
//                                       edge_segments' least for a
//                                       whole-series curve
//   statistics_end       date           the statistics cutoff: the procedure
//                                       choice, the cadence, the variogram
//                                       and the green median step read only
//                                       the observations dated <=
//                                       statistics_end; "none"
//                                       (NO_STATISTICS_END) for none
//   coefficients.min, .mid, .max, .per  the coefficient rule of model and
//                        none           stability_fit (CoefficientRule);
//                                       edge_segments reads min
//   lasso.alpha, .coef_floor            the Lasso of model, stability_fit
//                        reflectance    and edge_segments (LassoOptions)
//   lasso.max_iter, .tolerance  none
//
// A Shared holds them resolved against a data spec: in its units, the bands as
// its rows. Its member defaults are the "ccdc" values on the default DataSpec
// (Collection-1 DN, the Landsat rows); the configuration's defaults are the
// physical ones, shared_parameters()'.
//
//==============================================================================

struct Shared
{
    BandList detection_bands = {1, 2, 3, 4, 5};
    double change_probability = 0.99;
    double outlier_probability = 1.0 - 1e-6;
    index_t min_observations = 12;
    std::int64_t statistics_end = NO_STATISTICS_END;
    CoefficientRule coefficients;   // [shared.coefficients]
    LassoOptions lasso;             // [shared.lasso]
};

bool operator==(const Shared& a, const Shared& b);
inline bool operator!=(const Shared& a, const Shared& b) { return !(a == b); }

// The schemas of the values outside the tables and of each table's, with
// their physical defaults ("ccdc"'s).
const std::vector<Field<Shared>>& shared_schema();
const std::vector<Field<CoefficientRule>>& coefficients_schema();
const std::vector<Field<LassoOptions>>& lasso_schema();

// A table of the shared section: its name and what its values decide.
struct SharedTable
{
    const char* name;
    const char* doc;
};

// [shared.coefficients] and [shared.lasso].
const std::vector<SharedTable>& shared_tables();

// Every shared value, by the name a configuration gives it: the values
// outside the tables, then each table's as "<table>.<name>", each with its
// physical default.
const std::vector<ParameterInfo>& shared_parameters();

// Every shared value at its physical default, by name.
Values shared_defaults();

// The values of `shared`, by name.
Values shared_values(const Shared& shared);

// The values `values` give, by name, into `shared`, as apply() does: a
// table's named "shared <table> parameter '<name>'" in a refusal.
void apply_shared(const Values& values, Shared& shared);

} // namespace config
} // namespace ccd
