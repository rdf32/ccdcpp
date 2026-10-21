#include "convert.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

#include "ccd/core/maths.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// A configuration's values in an input's terms
//
// One conversion per Unit (config/params.hpp), each of a value as written:
// a number of another unit, or a list by name, passes unchanged. Without an
// input (check()) each returns the value as written, and none for a list by
// name once its names are known ones.
//
//==============================================================================

namespace
{

// The QA code names and the Input fields that hold their codes.
const std::vector<std::pair<std::string, std::uint8_t Input::*>>& qa_code_names()
{
    static const std::vector<std::pair<std::string, std::uint8_t Input::*>> names = {
        {"fill", &Input::qa_fill},       {"clear", &Input::qa_clear},
        {"water", &Input::qa_water},     {"shadow", &Input::qa_shadow},
        {"snow", &Input::qa_snow},       {"cloud", &Input::qa_cloud},
        {"cirrus1", &Input::qa_cirrus1}, {"cirrus2", &Input::qa_cirrus2},
        {"occlusion", &Input::qa_occlusion},
    };
    return names;
}

// A number of a Value, an integer read as a real; none for another type.
std::optional<double> number_of(const Value& value)
{
    if (const auto* v = std::get_if<double>(&value))
        return *v;
    if (const auto* v = std::get_if<std::int64_t>(&value))
        return static_cast<double>(*v);
    return std::nullopt;
}

// A reflectance level as the input holds it: r x reflectance_scale +
// reflectance_offset.
Value to_reflectance(const Value& value, const Input& input)
{
    // Two statements, so the product is rounded before the offset is added
    // (no fused multiply-add): at offset 0 a level is the product itself
    // (but for -0, which becomes +0).
    if (const auto v = number_of(value))
    {
        const double scaled = *v * input.reflectance_scale;
        return real(scaled + input.reflectance_offset);
    }
    return value;
}

// A reflectance difference as the input holds it: times reflectance_scale.
Value to_reflectance_delta(const Value& value, const Input& input)
{
    if (const auto v = number_of(value))
        return real(*v * input.reflectance_scale);
    return value;
}

// Degrees Celsius in the thermal row's unit after the input's transform
// (thermal_at_100c).
Value to_thermal_unit(const Value& value, const Input& input)
{
    if (const auto v = number_of(value))
        return real(*v * 100.0 * input.thermal_at_100c / 10000.0);
    return value;
}

// QA code names as the input's codes; codes as given.
std::optional<Value> to_codes(const Value& value, const std::string& at, const Input* input)
{
    const auto* names = std::get_if<std::vector<std::string>>(&value);
    if (!names)
        return value;
    std::vector<std::int64_t> codes;
    for (const std::string& name : *names)
    {
        const auto& known = qa_code_names();
        const auto it = std::find_if(known.begin(), known.end(),
                                     [&](const auto& n) { return n.first == name; });
        if (it == known.end())
        {
            std::vector<std::string> all;
            for (const auto& n : known)
                all.push_back(n.first);
            throw std::invalid_argument(at + " holds '" + name +
                                        "', which is not a QA code name; the names are " +
                                        listed(all));
        }
        if (input)
            codes.push_back(static_cast<std::int64_t>(input->*(it->second)));
    }
    if (!input)
        return std::nullopt;
    return Value(std::move(codes));
}

// A band list by role, or REFLECTIVE, as the input's rows; rows as given.
std::optional<Value> to_rows(const Value& value, const std::string& at, const Input* input)
{
    if (const auto* selector = std::get_if<std::string>(&value))
    {
        if (*selector != REFLECTIVE)
            throw std::invalid_argument(at + " is '" + *selector +
                                        "'; a band list is roles, rows or \"" + REFLECTIVE +
                                        "\"");
        if (!input)
            return std::nullopt;
        std::vector<std::int64_t> rows;
        for (index_t row = 0; row < input->bands(); ++row)
            if (input->roles[row] != "thermal" && input->roles[row] != "other")
                rows.push_back(static_cast<std::int64_t>(row));
        return Value(std::move(rows));
    }
    const auto* roles = std::get_if<std::vector<std::string>>(&value);
    if (!roles)
        return value;
    if (roles->empty())
        return Value(std::vector<std::int64_t>{});
    std::vector<std::int64_t> rows;
    for (const std::string& role : *roles)
    {
        if (!is_role(role))
            throw std::invalid_argument(at + " holds '" + role +
                                        "', which is not a role; the roles are " +
                                        listed(ROLES));
        if (role == "other")
            throw std::invalid_argument(at + " holds 'other', which no band list names");
        if (!input)
            continue;
        const index_t row = input->index(role);
        if (row == NO_BAND)
            throw std::invalid_argument(at + " holds '" + role +
                                        "', which the input does not have; its rows are " +
                                        listed(input->roles));
        rows.push_back(static_cast<std::int64_t>(row));
    }
    if (!input)
        return std::nullopt;
    return Value(std::move(rows));
}

// `value` of a field in `unit`, as `input` holds it (null: as written); `at`
// names the field in messages.
std::optional<Value> convert(Unit unit, const Value& value, const std::string& at,
                             const Input* input)
{
    switch (unit)
    {
        case Unit::Reflectance:
            return input ? to_reflectance(value, *input) : value;
        case Unit::ReflectanceDelta:
            return input ? to_reflectance_delta(value, *input) : value;
        case Unit::Celsius:
            return input ? to_thermal_unit(value, *input) : value;
        case Unit::QaCodes:
            return to_codes(value, at, input);
        case Unit::Bands:
            return to_rows(value, at, input);
        case Unit::None:
        case Unit::Observations:
        case Unit::Days:
            break;
    }
    return value;
}

// Throws for a name of `given` that `info` does not hold or that its stage
// cannot set.
void check_names(const std::vector<ParameterInfo>& info, const Values& given,
                 const std::string& where, const char* noun)
{
    for (const auto& [name, value] : given)
    {
        const auto field = std::find_if(info.begin(), info.end(),
                                        [&](const ParameterInfo& p) { return p.name == name; });
        if (field == info.end())
        {
            std::vector<std::string> own;
            for (const ParameterInfo& p : info)
                if (p.source == Source::Own)
                    own.push_back(p.name);
            throw std::invalid_argument(where + " has no " + noun + " '" + name + "'" +
                                        (own.empty() ? "" : "; it has " + listed(own)));
        }
        if (field->source != Source::Own)
            throw std::invalid_argument(
                where + " " + noun + " '" + name + "' is " +
                (field->source == Source::Shared ? "the shared value '" + field->from + "'"
                                                 : "derived from the shared value '" +
                                                       field->from + "'") +
                "; set shared '" + field->from + "' instead");
    }
}

} // namespace

Values section_values(const std::vector<ParameterInfo>& info, const Values& given,
                      const std::string& where, const SectionContext& with, const char* noun)
{
    check_names(info, given, where, noun);
    Values out;
    for (const ParameterInfo& p : info)
    {
        switch (p.source)
        {
            case Source::Own:
            {
                const auto it = given.find(p.name);
                const Value& value = it != given.end() ? it->second : *p.default_value;
                const std::string at = where + " " + noun + " '" + p.name + "'";
                if (auto v = convert(p.unit, value, at, with.input))
                    out[p.name] = std::move(*v);
                break;
            }
            case Source::Shared:
                out[p.name] = with.shared.at(p.from);
                break;
            case Source::Chi2:
                out[p.name] = real(chi2_quantile(std::get<double>(with.shared.at(p.from)),
                                                 with.detection_bands));
                break;
        }
    }
    return out;
}

} // namespace config
} // namespace ccd
