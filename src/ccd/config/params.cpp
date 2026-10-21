#include "ccd/config/params.hpp"

#include <set>
#include <sstream>

namespace ccd
{
namespace config
{

std::string number(double v)
{
    std::ostringstream out;
    out << v;
    return out.str();
}

std::string describe(const Bounds& b)
{
    const bool lo = std::isfinite(b.lo), hi = std::isfinite(b.hi);
    if (lo && hi)
        return std::string(b.lo_inclusive ? "in [" : "in (") + number(b.lo) + ", " +
               number(b.hi) + (b.hi_inclusive ? "]" : ")");
    if (lo)
        return std::string(b.lo_inclusive ? "at least " : "greater than ") + number(b.lo);
    if (hi)
        return std::string(b.hi_inclusive ? "at most " : "less than ") + number(b.hi);
    return "a number";
}

void check_band_list(const BandList& list, index_t bands, const std::string& where)
{
    std::set<index_t> seen;
    for (const index_t band : list)
    {
        if (band >= bands)
            throw std::invalid_argument(where + " holds band " + std::to_string(band) +
                                        "; the input has " + std::to_string(bands) + " bands");
        if (!seen.insert(band).second)
            throw std::invalid_argument(where + " holds band " + std::to_string(band) +
                                        " twice");
    }
}

} // namespace config
} // namespace ccd
