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

std::string listed(const std::vector<std::string>& names)
{
    std::string out;
    for (const std::string& n : names)
        out += (out.empty() ? "" : ", ") + n;
    return out;
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

namespace
{

// A number of `value`, an integer read as a real.
double number_at(const Value& value, const std::string& at)
{
    if (const auto* v = std::get_if<double>(&value))
        return *v;
    if (const auto* v = std::get_if<std::int64_t>(&value))
        return static_cast<double>(*v);
    throw std::invalid_argument(at + " is a number");
}

void check_bounds(double v, const Bounds& bounds, const std::string& at)
{
    if (!bounds.contains(v))
        throw std::invalid_argument(at + " is " + number(v) + "; it must be " +
                                    (std::isinf(v) ? "a finite number " : "") + describe(bounds));
}

// A whole number of `value` (a number, and an integer).
std::int64_t whole_at(const Value& value, const std::string& at)
{
    number_at(value, at);
    if (!std::holds_alternative<std::int64_t>(value))
        throw std::invalid_argument(at + " is a whole number");
    return std::get<std::int64_t>(value);
}

} // namespace

void assign(bool& member, const Value& value, const Bounds&, const std::string& at)
{
    if (!std::holds_alternative<bool>(value))
        throw std::invalid_argument(at + " is a flag (true or false)");
    member = std::get<bool>(value);
}

void assign(std::string& member, const Value& value, const Bounds&, const std::string& at)
{
    if (!std::holds_alternative<std::string>(value))
        throw std::invalid_argument(at + " is a name");
    member = std::get<std::string>(value);
}

void assign(std::vector<std::int64_t>& member, const Value& value, const Bounds& bounds,
            const std::string& at)
{
    if (!std::holds_alternative<std::vector<std::int64_t>>(value))
        throw std::invalid_argument(at + " is a list of QA code names or codes");
    for (const std::int64_t code : std::get<std::vector<std::int64_t>>(value))
        if (!bounds.contains(static_cast<double>(code)))
            throw std::invalid_argument(at + " holds " + std::to_string(code) +
                                        "; a code must be " + describe(bounds));
    member = std::get<std::vector<std::int64_t>>(value);
}

void assign(BandList& member, const Value& value, const Bounds&, const std::string& at)
{
    if (!std::holds_alternative<std::vector<std::int64_t>>(value))
        throw std::invalid_argument(at + " is a list of band roles or indices");
    BandList list;
    for (const std::int64_t band : std::get<std::vector<std::int64_t>>(value))
    {
        if (band < 0)
            throw std::invalid_argument(at + " holds band " + std::to_string(band) +
                                        "; a band index is at least 0");
        try
        {
            list.push_back(static_cast<index_t>(band));
        }
        catch (const std::invalid_argument& e)
        {
            throw std::invalid_argument(at + ": " + e.what());
        }
    }
    member = list;
}

void assign(double& member, const Value& value, const Bounds& bounds, const std::string& at)
{
    const double v = number_at(value, at);
    check_bounds(v, bounds, at);
    member = v;
}

void assign(index_t& member, const Value& value, const Bounds& bounds, const std::string& at)
{
    const std::int64_t i = whole_at(value, at);
    if (i < 0)
        throw std::invalid_argument(at + " is " + std::to_string(i) + "; it must be at least 0");
    check_bounds(static_cast<double>(i), bounds, at);
    member = static_cast<index_t>(i);
}

void assign(std::int64_t& member, const Value& value, const Bounds& bounds,
            const std::string& at)
{
    const std::int64_t i = whole_at(value, at);
    check_bounds(static_cast<double>(i), bounds, at);
    member = i;
}

void assign(Interval& member, const Value& value, const Bounds& bounds, const std::string& at)
{
    std::vector<double> ends;
    if (const auto* reals = std::get_if<std::vector<double>>(&value))
        ends = *reals;
    else if (const auto* wholes = std::get_if<std::vector<std::int64_t>>(&value))
        ends.assign(wholes->begin(), wholes->end());
    if (ends.size() != 2)
        throw std::invalid_argument(at + " is a pair of numbers, [min, max]");
    for (const double v : ends)
        check_bounds(v, bounds, at);
    member = {ends[0], ends[1]};
}

void assign(Inclusive& member, const Value& value, const Bounds&, const std::string& at)
{
    const auto* ends = std::get_if<std::vector<bool>>(&value);
    if (!ends || ends->size() != 2)
        throw std::invalid_argument(at + " is a pair of flags, [min, max]");
    member = {(*ends)[0], (*ends)[1]};
}

void check_band_list(const BandList& list, index_t bands, const std::string& where)
{
    std::set<index_t> seen;
    for (const index_t band : list)
    {
        if (band >= bands)
            throw std::invalid_argument(where + " holds band " + std::to_string(band) +
                                        "; the data spec has " + std::to_string(bands) + " bands");
        if (!seen.insert(band).second)
            throw std::invalid_argument(where + " holds band " + std::to_string(band) +
                                        " twice");
    }
}

} // namespace config
} // namespace ccd
