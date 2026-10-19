#include "ccd/strategies/noise_floor/variogram.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<VariogramFloor::Params>>& VariogramFloor::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"min_lag_days", &Params::min_lag_days, Unit::Days, config::greater_than(0),
         "the variogram's lag is the first whose most common separation exceeds this"},
    };
    return s;
}

std::vector<scalar_t> VariogramFloor::estimate(ArrayView<const std::int64_t, 1> dates,
                                               ArrayView<const scalar_t, 2> spectral) const
{
    return adjusted_variogram(dates, spectral, p_.min_lag_days);
}

} // namespace strategies
} // namespace ccd
