#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
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
//             {"alpha",   &Params::alpha,   Unit::ReflectanceDelta, at_least(0), "...",
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

// What a parameter's value is measured in, and how a configuration resolved
// against an input converts it:
//
//   None, Observations, Days   unchanged;
//   Reflectance                a reflectance level (a bound of the data's
//                              values): r x Input::reflectance_scale +
//                              Input::reflectance_offset, the data's value
//                              of reflectance r;
//   ReflectanceDelta           a difference or magnitude of reflectance (the
//                              Lasso's alpha and coef_floor, the green
//                              median's offset): times
//                              Input::reflectance_scale only;
//   Celsius                    degrees Celsius, to the thermal row's unit
//                              after the input's transform: c x 100 (hundredths
//                              of a degree, pyccd's unit) times
//                              Input::thermal_at_100c / 10000, so -93.2 and 70.7
//                              are -9320 and 7070 on Collection-1 DN and -0.932
//                              and 0.707 on Celsius / 100, exactly;
//   Bands                      band roles to the input's rows (rows as given);
//                              the name REFLECTIVE to every row whose role is
//                              not "thermal" or "other", in row order;
//   QaCodes                    QA code names ("fill", "clear", "water",
//                              "shadow", "snow", "cloud", "cirrus1",
//                              "cirrus2", "occlusion") to the input's codes
//                              (codes 0-255 as given).
enum class Unit : std::uint8_t
{
    None,
    Observations,
    Days,
    Reflectance,
    ReflectanceDelta,
    Celsius,
    Bands,
    QaCodes
};

// The band list of every reflective row (Unit::Bands).
constexpr const char* REFLECTIVE = "reflective";

// One parameter's value, as a configuration holds it: a flag, a whole
// number, a real, a name, a list of whole numbers (band rows, QA codes), or a
// list of names (band roles, QA code names). A resolved configuration holds
// no names in a list: they are its input's rows and codes.
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
inline Value names(std::vector<std::string> v)
{
    return Value(std::in_place_type<std::vector<std::string>>, std::move(v));
}

// The allowed values of a number: lo <(=) value <(=) hi. NaN never passes,
// and an infinity does not where a bound is a number (numeric()): any()
// alone admits them.
struct Bounds
{
    double lo = -INFINITY;
    double hi = INFINITY;
    bool lo_inclusive = true;
    bool hi_inclusive = true;

    bool numeric() const
    {
        return std::isfinite(lo) || std::isfinite(hi);
    }

    bool contains(double v) const
    {
        if (std::isinf(v) && numeric())
            return false;
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

// Names as a message lists them: "a, b, c".
std::string listed(const std::vector<std::string>& names);

// Where a field's value comes from:
//
//   Own      the stage's parameters (or the configuration's default);
//   Shared   the configuration's shared value named `from` (config/shared.hpp),
//            in the input's units and rows;
//   Chi2     chi2_quantile(shared value `from`, the number of detection
//            bands): a threshold for the detection bands' change statistic.
//
// A Shared or Chi2 field is not a parameter of its stage: a stage that sets
// one is refused, naming the shared value to set instead. A configured run
// always sets it from the shared value, so its member default is only what a
// strategy built from Params{} runs with.
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
                                BandList P::*, std::string P::*, std::vector<std::int64_t> P::*>;

    const char* name;
    Member member;
    Unit unit;
    Bounds bounds;
    const char* doc;
    Source source = Source::Own;
    const char* from = nullptr;
    std::optional<Value> default_value = std::nullopt;
};

// One value into a member of each type a Field may point to, refusing (std::
// invalid_argument naming `at`) a value of the wrong type (an integer is
// accepted for a real), a whole number the member cannot hold, a number or a
// code outside `bounds`, a band list with an index below 0 or not below
// CCD_MAX_BANDS, or longer than CCD_MAX_BANDS, and a band list by role.
void assign(bool& member, const Value& value, const Bounds& bounds, const std::string& at);
void assign(std::string& member, const Value& value, const Bounds& bounds, const std::string& at);
void assign(std::vector<std::int64_t>& member, const Value& value, const Bounds& bounds,
            const std::string& at);
void assign(BandList& member, const Value& value, const Bounds& bounds, const std::string& at);
void assign(double& member, const Value& value, const Bounds& bounds, const std::string& at);
void assign(index_t& member, const Value& value, const Bounds& bounds, const std::string& at);
void assign(std::int64_t& member, const Value& value, const Bounds& bounds,
            const std::string& at);

//------------------------------------------------------------------------------
// The parameters `values` give `params`, starting from its defaults: each by
// assign() into its member, named "<where> parameter '<name>'". Throws
// std::invalid_argument, naming `where` (e.g. "refit 'span-factor'") and the
// field, on a name the schema does not hold and on what assign() refuses (a
// band list by role among it: config::resolve() maps roles to rows first).
//------------------------------------------------------------------------------
template<class P>
void apply(const std::vector<Field<P>>& schema, const Values& values, P& params,
           const std::string& where)
{
    for (const auto& [name, value] : values)
    {
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
        std::visit([&](auto member) { assign(params.*member, value, field->bounds, at); },
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
                else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
                    out[f.name] = Value(params.*member);
                else if constexpr (std::is_same_v<T, double>)
                    out[f.name] = real(params.*member);
                else
                    out[f.name] = integer(static_cast<std::int64_t>(params.*member));
            },
            f.member);
    return out;
}

// One parameter of a schema, as listings show it (the registry's schemas(),
// the Python binding, the docs). A parameter its stage cannot set (a Shared
// or Chi2 source) has no default.
struct ParameterInfo
{
    std::string name;
    Unit unit;
    std::string bounds;   // describe() of its bounds
    std::string doc;
    Source source;
    std::string from;     // the shared value it reads; empty for Own
    std::optional<Value> default_value;
};

// The parameters of `schema`, each Own one with its configuration default
// (its default_value, or else its member's in P{}).
template<class P>
std::vector<ParameterInfo> info_of(const std::vector<Field<P>>& schema)
{
    std::vector<ParameterInfo> info;
    const Values members = to_values(schema, P{});
    for (const auto& field : schema)
    {
        std::optional<Value> fallback;
        if (field.source == Source::Own)
            fallback = field.default_value ? *field.default_value : members.at(field.name);
        info.push_back(ParameterInfo{field.name, field.unit, describe(field.bounds), field.doc,
                                     field.source, field.from ? field.from : "", fallback});
    }
    return info;
}

// Throws std::invalid_argument, naming `where`, when `list` names a band an
// input of `bands` bands does not have, or repeats one.
void check_band_list(const BandList& list, index_t bands, const std::string& where);

// check_band_list() of every band list of `params` ("<where> parameter
// '<name>'").
template<class P>
void check_band_lists(const std::vector<Field<P>>& schema, const P& params, index_t bands,
                      const std::string& where)
{
    for (const auto& field : schema)
        if (const auto* m = std::get_if<BandList P::*>(&field.member))
            check_band_list(params.**m, bands, where + " parameter '" + field.name + "'");
}


} // namespace config
} // namespace ccd
