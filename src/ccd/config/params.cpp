#include "ccd/config/params.hpp"

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

} // namespace config
} // namespace ccd
