#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "ccd/core/band_list.hpp"
#include "ccd/core/types.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// Strategy parameters
//
// Every strategy declares its parameters once, as a schema: each field's
// name, the member it sets, its unit, its allowed range and a line of
// documentation, and where its value comes from. From the schema come the
// strategy's construction from a configuration (apply()) and the
// configuration of a strategy (to_values()): nothing about a parameter is
// written twice.
//
// A schema is a plain vector of Field, written next to the strategy:
//
//     struct Params { index_t min_obs = 24; double factor = 1.33; double alpha = 1.0; };
//     static const std::vector<config::Field<Params>>& schema()
//     {
//         static const std::vector<config::Field<Params>> s = {
//             {"min_obs", &Params::min_obs, Unit::Observations, at_least(1), "..."},
//             {"factor",  &Params::factor,  Unit::None, greater_than(0), "..."},
//             {"alpha",   &Params::alpha,   Unit::Reflectance, at_least(0), "...",
//              Source::Shared, "alpha"},
//         };
//         return s;
//     }
//
// A member's default is its value on the default Input (Collection-1 DN, the
// Landsat rows), which is what a strategy built directly from Params{} runs
// with. The configuration's own default for a field is the member's, or the
// field's default_value where the member cannot say it in physical terms (a
// band list by role).
//
//==============================================================================

// What a parameter's value is measured in: dimensionless, a count of
// observations, days, reflectance (multiplied by Input::reflectance_unit when
// a configuration is resolved), or bands (roles, or rows of the spectral
// array).
enum class Unit : std::uint8_t
{
    None,
    Observations,
    Days,
    Reflectance,
    Bands
};

// One parameter's value, as a configuration holds it: a flag, a whole
// number, a real, a name, a list of band indices, or a list of band roles. A
// resolved configuration holds no roles: they are rows of its input.
using Value = std::variant<bool, std::int64_t, double, std::string, std::vector<std::int64_t>,
                           std::vector<std::string>>;

// A strategy's parameters, by name.
using Values = std::map<std::string, Value>;

// Values of each type, made explicitly: a double or an integer given to a
// Value directly may convert to its bool.
inline Value real(double v) { return Value(std::in_place_type<double>, v); }
inline Value integer(std::int64_t v) { return Value(std::in_place_type<std::int64_t>, v); }
inline Value flag(bool v) { return Value(std::in_place_type<bool>, v); }
inline Value bands(const BandList& v)
{
    return Value(std::in_place_type<std::vector<std::int64_t>>, v.begin(), v.end());
}
inline Value roles(std::vector<std::string> v)
{
    return Value(std::in_place_type<std::vector<std::string>>, std::move(v));
}

// The allowed values of a number: lo <(=) value <(=) hi. NaN never passes.
struct Bounds
{
    double lo = -INFINITY;
    double hi = INFINITY;
    bool lo_inclusive = true;
    bool hi_inclusive = true;

    bool contains(double v) const
    {
        return (lo_inclusive ? v >= lo : v > lo) && (hi_inclusive ? v <= hi : v < hi);
    }
};

inline Bounds any() { return {}; }
inline Bounds at_least(double lo) { return {lo, INFINITY, true, true}; }
inline Bounds greater_than(double lo) { return {lo, INFINITY, false, true}; }
inline Bounds between(double lo, double hi) { return {lo, hi, true, true}; }
inline Bounds open_interval(double lo, double hi) { return {lo, hi, false, false}; }

// "at least 1", "greater than 0", "in [0, 1]", ...
std::string describe(const Bounds& b);

// A number as a message shows it: 1.33, 24, 1e-09.
std::string number(double v);

// Where a field's value comes from:
//
//   Own      the stage's parameters (or the configuration's default);
//   Shared   the configuration's shared value named `from` (config/shared.hpp),
//            in the input's units and rows;
//   Chi2     chi2_quantile(shared value `from`, the number of detection
//            bands): a threshold for the detection bands' change statistic.
//
// A Shared or Chi2 field is not a parameter of its stage: a stage that sets
// one is refused, naming the shared value to set instead.
enum class Source : std::uint8_t
{
    Own,
    Shared,
    Chi2
};

template<class P>
struct Field
{
    using Member = std::variant<double P::*, index_t P::*, std::int64_t P::*, bool P::*,
                                BandList P::*, std::string P::*>;

    const char* name;
    Member member;
    Unit unit;
    Bounds bounds;
    const char* doc;
    Source source = Source::Own;
    const char* from = nullptr;
    std::optional<Value> default_value = std::nullopt;
};

// "own", "shared 'alpha'", "chi2 of shared 'change_probability'".
std::string describe(Source source, const char* from);

//------------------------------------------------------------------------------
// The parameters `values` give `params`, starting from its defaults. Throws
// std::invalid_argument, naming `where` (e.g. "refit 'span-factor'") and the
// field, on a name the schema does not hold, a value of the wrong type
// (an integer is accepted for a real), a whole number its member cannot
// hold, a value outside the field's bounds, a band list with an index below 0
// or not below CCD_MAX_BANDS, or longer than CCD_MAX_BANDS, and a band list
// by role (Registry::resolve() maps roles to rows first).
//------------------------------------------------------------------------------
template<class P>
void apply(const std::vector<Field<P>>& schema, const Values& values, P& params,
           const std::string& where)
{
    for (const auto& entry : values)
    {
        const std::string& name = entry.first;
        const Value& value = entry.second;
        const Field<P>* field = nullptr;
        for (const auto& f : schema)
            if (name == f.name)
                field = &f;
        if (!field)
        {
            std::string known;
            for (const auto& f : schema)
                known += std::string(known.empty() ? "" : ", ") + f.name;
            throw std::invalid_argument(where + " has no parameter '" + name + "'" +
                                        (known.empty() ? "" : "; it has " + known));
        }

        const std::string at = where + " parameter '" + name + "'";
        std::visit(
            [&](auto member)
            {
                using T = std::remove_cv_t<std::remove_reference_t<decltype(params.*member)>>;
                if constexpr (std::is_same_v<T, bool>)
                {
                    if (!std::holds_alternative<bool>(value))
                        throw std::invalid_argument(at + " is a flag (true or false)");
                    params.*member = std::get<bool>(value);
                }
                else if constexpr (std::is_same_v<T, std::string>)
                {
                    if (!std::holds_alternative<std::string>(value))
                        throw std::invalid_argument(at + " is a name");
                    params.*member = std::get<std::string>(value);
                }
                else if constexpr (std::is_same_v<T, BandList>)
                {
                    if (!std::holds_alternative<std::vector<std::int64_t>>(value))
                        throw std::invalid_argument(at + " is a list of band roles or indices");
                    BandList list;
                    for (const std::int64_t band : std::get<std::vector<std::int64_t>>(value))
                    {
                        if (band < 0)
                            throw std::invalid_argument(at + " holds band " +
                                                        std::to_string(band) +
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
                    params.*member = list;
                }
                else
                {
                    double v;
                    if (std::holds_alternative<double>(value))
                        v = std::get<double>(value);
                    else if (std::holds_alternative<std::int64_t>(value))
                        v = static_cast<double>(std::get<std::int64_t>(value));
                    else
                        throw std::invalid_argument(at + " is a number");
                    if constexpr (!std::is_same_v<T, double>)
                    {
                        if (!std::holds_alternative<std::int64_t>(value))
                            throw std::invalid_argument(at + " is a whole number");
                        const std::int64_t i = std::get<std::int64_t>(value);
                        if constexpr (std::is_unsigned_v<T>)
                        {
                            if (i < 0)
                                throw std::invalid_argument(at + " is " + std::to_string(i) +
                                                            "; it must be at least 0");
                        }
                    }
                    if (!field->bounds.contains(v))
                        throw std::invalid_argument(at + " is " + number(v) + "; it must be " +
                                                    describe(field->bounds));
                    if constexpr (std::is_same_v<T, double>)
                        params.*member = v;
                    else
                        params.*member = static_cast<T>(std::get<std::int64_t>(value));
                }
            },
            field->member);
    }
}

// Every parameter of `params`, by name.
template<class P>
Values to_values(const std::vector<Field<P>>& schema, const P& params)
{
    Values out;
    for (const auto& f : schema)
        std::visit(
            [&](auto member)
            {
                using T = std::remove_cv_t<std::remove_reference_t<decltype(params.*member)>>;
                if constexpr (std::is_same_v<T, bool>)
                    out[f.name] = flag(params.*member);
                else if constexpr (std::is_same_v<T, BandList>)
                    out[f.name] = bands(params.*member);
                else if constexpr (std::is_same_v<T, std::string>)
                    out[f.name] = Value(params.*member);
                else if constexpr (std::is_same_v<T, double>)
                    out[f.name] = real(params.*member);
                else
                    out[f.name] = integer(static_cast<std::int64_t>(params.*member));
            },
            f.member);
    return out;
}

} // namespace config
} // namespace ccd
