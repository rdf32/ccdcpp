#include "ccd/strategies/window_screen/tmask.hpp"

#include <stdexcept>

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<Tmask::Params>>& Tmask::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"bands", &Params::bands, Unit::Bands, config::any(),
         "the bands Tmask fits (pyccd: green and swir1)"},
        {"t_const", &Params::t_const, Unit::None, config::any(),
         "an observation is an outlier above this many variograms"},
    };
    return s;
}

Tmask::Tmask(const Params& p) : p_(p)
{
    if (p_.bands.empty())
        throw std::invalid_argument("window_screen 'tmask' parameter 'bands' is empty");
}

ProcessingMask Tmask::screen(ArrayView<const std::int64_t, 1> dates,
                             ArrayView<const scalar_t, 2> spectral,
                             const Variogram& variogram) const
{
    return tmask(dates, spectral, variogram, p_.bands, p_.t_const);
}

} // namespace strategies
} // namespace ccd
