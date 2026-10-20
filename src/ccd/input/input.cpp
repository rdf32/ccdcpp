#include "ccd/input/input.hpp"

#include <algorithm>

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

index_t Input::index(const std::string& role) const
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

bool operator==(const Input& a, const Input& b)
{
    return same_table(a.qa_table, b.qa_table) && a.qa_fill == b.qa_fill && a.qa_clear == b.qa_clear && a.qa_water == b.qa_water &&
           a.qa_shadow == b.qa_shadow && a.qa_snow == b.qa_snow && a.qa_cloud == b.qa_cloud &&
           a.qa_cirrus1 == b.qa_cirrus1 && a.qa_cirrus2 == b.qa_cirrus2 &&
           a.qa_occlusion == b.qa_occlusion && a.roles == b.roles &&
           a.reflectance_unit == b.reflectance_unit &&
           a.reflectance_offset == b.reflectance_offset &&
           a.thermal_scale == b.thermal_scale && a.thermal_offset == b.thermal_offset &&
           a.thermal_unit == b.thermal_unit &&
           a.saturated == b.saturated && a.thermal_saturated == b.thermal_saturated;
}

std::uint64_t digest(const Input& o)
{
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
    for (const scalar_t v : {o.reflectance_unit, o.reflectance_offset, o.thermal_scale,
                             o.thermal_offset, o.thermal_unit})
        h.real(v);
    for (const std::optional<scalar_t>& code : {o.saturated, o.thermal_saturated})
    {
        h.flag(code.has_value());
        if (code)
            h.real(*code);
    }
    return h.value();
}

} // namespace ccd
