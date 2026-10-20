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

std::string describe(Source source, const char* from)
{
    switch (source)
    {
        case Source::Own:    return "own";
        case Source::Shared: return std::string("shared '") + from + "'";
        case Source::Chi2:   return std::string("chi2 of shared '") + from + "'";
    }
    return "?";
}

} // namespace config
} // namespace ccd
