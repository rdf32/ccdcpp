#include "ccd/input/data_spec.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "ccd/config/params.hpp"

#include "../core/fnv.hpp"

namespace ccd
{

const std::vector<std::string> ROLES = {
    "blue", "green", "red", "nir", "swir1", "swir2", "thermal", "coastal",
    "red_edge_1", "red_edge_2", "red_edge_3", "nir_broad", "other",
};

bool is_role(const std::string& role)
{
    return std::find(ROLES.begin(), ROLES.end(), role) != ROLES.end();
}

index_t DataSpec::index(const std::string& role) const
{
    const auto it = std::find(roles.begin(), roles.end(), role);
    return it == roles.end() ? NO_BAND : static_cast<index_t>(it - roles.begin());
}

namespace
{

bool same_table(const QaTablePtr& a, const QaTablePtr& b)
{
    if (a == b)
        return true;
    return a && b && a->digest == b->digest && a->code == b->code;
}

} // namespace

bool operator==(const DataSpec& a, const DataSpec& b)
{
    return same_table(a.qa_table, b.qa_table) && a.qa_fill == b.qa_fill && a.qa_clear == b.qa_clear && a.qa_water == b.qa_water &&
           a.qa_shadow == b.qa_shadow && a.qa_snow == b.qa_snow && a.qa_cloud == b.qa_cloud &&
           a.qa_cirrus1 == b.qa_cirrus1 && a.qa_cirrus2 == b.qa_cirrus2 &&
           a.qa_occlusion == b.qa_occlusion && a.roles == b.roles &&
           a.reflectance_scale == b.reflectance_scale &&
           a.reflectance_offset == b.reflectance_offset &&
           a.thermal_scale == b.thermal_scale && a.thermal_offset == b.thermal_offset &&
           a.thermal_at_100c == b.thermal_at_100c &&
           a.saturation_code == b.saturation_code &&
           a.thermal_saturation_code == b.thermal_saturation_code;
}

std::uint64_t digest(const DataSpec& o)
{
    // -0 and +0 are one value: they compare equal (operator==), and every use
    // of a field (a product, a sum, the identity transform's test) treats
    // them alike, so the digest, as the hash of equal DataSpecs, is the same.
    const auto real = [](scalar_t v) { return v == 0 ? scalar_t(0) : v; };
    detail::Fnv h;
    for (const std::uint8_t v : {o.qa_fill, o.qa_clear, o.qa_water, o.qa_shadow, o.qa_snow,
                                 o.qa_cloud, o.qa_cirrus1, o.qa_cirrus2, o.qa_occlusion})
        h.integer(v);
    h.flag(o.qa_table != nullptr);
    if (o.qa_table)
        h.integer(static_cast<std::int64_t>(o.qa_table->digest));
    h.integer(static_cast<std::int64_t>(o.roles.size()));
    for (const std::string& role : o.roles)
        h.text(role);
    for (const scalar_t v : {o.reflectance_scale, o.reflectance_offset, o.thermal_scale,
                             o.thermal_offset, o.thermal_at_100c})
        h.real(real(v));
    for (const std::optional<scalar_t>& code : {o.saturation_code, o.thermal_saturation_code})
    {
        h.flag(code.has_value());
        if (code)
            h.real(real(*code));
    }
    return h.value();
}

namespace
{

std::string count_text(index_t bands)
{
    return "spectral has " + std::to_string(bands) + " bands";
}

} // namespace

void validate(const DataSpec& spec, index_t bands)
{
    if (bands != spec.bands())
    {
        throw std::invalid_argument(
            count_text(bands) + "; the data spec has " + std::to_string(spec.bands()) +
            " (a row per role)"
        );
    }

    if (bands > CCD_MAX_BANDS)
    {
        throw std::invalid_argument(
            count_text(bands) + "; ccdcpp accepts at most " +
            std::to_string(CCD_MAX_BANDS)
        );
    }

    for (index_t i = 0; i < spec.roles.size(); ++i)
    {
        const std::string& role = spec.roles[i];
        if (!is_role(role))
        {
            std::string known;
            for (const std::string& r : ROLES)
                known += (known.empty() ? "" : ", ") + r;
            throw std::invalid_argument(
                "roles holds '" + role + "', which is not a role; the roles are " + known
            );
        }
        if (role != "other" && spec.index(role) != i)
        {
            throw std::invalid_argument(
                "roles holds '" + role + "' more than once"
            );
        }
    }

    if (spec.index("green") == NO_BAND)
    {
        throw std::invalid_argument(
            "roles has no 'green'; the green median filter needs a green band"
        );
    }

    if (!(spec.reflectance_scale > 0) || !std::isfinite(spec.reflectance_scale))
    {
        throw std::invalid_argument(
            "reflectance_scale is " + config::number(spec.reflectance_scale) +
            "; it must be a positive number"
        );
    }

    if (!std::isfinite(spec.reflectance_offset))
    {
        throw std::invalid_argument(
            "reflectance_offset is " + config::number(spec.reflectance_offset) +
            "; it must be a finite number"
        );
    }

    if (!(spec.thermal_scale > 0) || !std::isfinite(spec.thermal_scale))
    {
        throw std::invalid_argument(
            "thermal_scale is " + config::number(spec.thermal_scale) +
            "; it must be a positive number"
        );
    }

    if (!std::isfinite(spec.thermal_offset))
    {
        throw std::invalid_argument(
            "thermal_offset is " + config::number(spec.thermal_offset) +
            "; it must be a finite number"
        );
    }

    if (!(spec.thermal_at_100c > 0) || !std::isfinite(spec.thermal_at_100c))
    {
        throw std::invalid_argument(
            "thermal_at_100c is " + config::number(spec.thermal_at_100c) +
            "; it must be a positive number"
        );
    }

    for (const auto& [name, code] : {std::pair<const char*, const std::optional<scalar_t>&>{
                                         "saturation_code", spec.saturation_code},
                                     {"thermal_saturation_code", spec.thermal_saturation_code}})
    {
        if (code && std::isnan(*code))
        {
            throw std::invalid_argument(
                std::string(name) + " is nan; it is a number, the value a saturated "
                "observation holds, or none"
            );
        }
    }
}

void validate_dates(ArrayView<const std::int64_t, 1> dates)
{
    for (index_t i = 1; i < dates.size(); ++i)
    {
        if (dates(i) < dates(i - 1))
        {
            throw std::invalid_argument(
                "dates must be non-decreasing; dates[" + std::to_string(i) +
                "] = " + std::to_string(dates(i)) + " follows dates[" +
                std::to_string(i - 1) + "] = " + std::to_string(dates(i - 1))
            );
        }
    }
}

} // namespace ccd
