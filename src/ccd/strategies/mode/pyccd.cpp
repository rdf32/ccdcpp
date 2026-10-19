#include "ccd/strategies/mode/pyccd.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<PyccdMode::Params>>& PyccdMode::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"clear_pct_threshold", &Params::clear_pct_threshold, Unit::None, config::any(),
         "the clear fraction a change-detection pixel reaches"},
        {"snow_pct_threshold", &Params::snow_pct_threshold, Unit::None, config::any(),
         "the snow fraction a permanent-snow pixel reaches"},
    };
    return s;
}

PixelProcedure PyccdMode::choose(const Quality::Statistics& statistics) const
{
    if (!statistics.enough_clear(p_.clear_pct_threshold)) {
        if (statistics.enough_snow(p_.snow_pct_threshold)) {
            return PixelProcedure::PermanentSnow;
        } else {
            return PixelProcedure::InsufficientClear;
        }
    }
    return PixelProcedure::Standard;
}

} // namespace strategies
} // namespace ccd
