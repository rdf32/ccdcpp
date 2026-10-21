#include "ccd/ccd.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/fnv.hpp"

namespace ccd 
{

namespace
{

std::string count_text(index_t bands)
{
    return "spectral has " + std::to_string(bands) + " bands";
}

} // namespace

void validate(const Input& input, index_t bands)
{
    if (bands != input.bands())
    {
        throw std::invalid_argument(
            count_text(bands) + "; the input has " + std::to_string(input.bands()) +
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

    for (index_t i = 0; i < input.roles.size(); ++i)
    {
        const std::string& role = input.roles[i];
        if (!is_role(role))
        {
            std::string known;
            for (const std::string& r : ROLES)
                known += (known.empty() ? "" : ", ") + r;
            throw std::invalid_argument(
                "roles holds '" + role + "', which is not a role; the roles are " + known
            );
        }
        if (role != "other" && input.index(role) != i)
        {
            throw std::invalid_argument(
                "roles holds '" + role + "' more than once"
            );
        }
    }

    if (input.index("green") == NO_BAND)
    {
        throw std::invalid_argument(
            "roles has no 'green'; the green median filter needs a green band"
        );
    }

    if (!(input.reflectance_scale > 0) || !std::isfinite(input.reflectance_scale))
    {
        throw std::invalid_argument(
            "reflectance_scale is " + config::number(input.reflectance_scale) +
            "; it must be a positive number"
        );
    }

    if (!std::isfinite(input.reflectance_offset))
    {
        throw std::invalid_argument(
            "reflectance_offset is " + config::number(input.reflectance_offset) +
            "; it must be a finite number"
        );
    }

    if (!(input.thermal_scale > 0) || !std::isfinite(input.thermal_scale))
    {
        throw std::invalid_argument(
            "thermal_scale is " + config::number(input.thermal_scale) +
            "; it must be a positive number"
        );
    }

    if (!std::isfinite(input.thermal_offset))
    {
        throw std::invalid_argument(
            "thermal_offset is " + config::number(input.thermal_offset) +
            "; it must be a finite number"
        );
    }

    if (!(input.thermal_at_100c > 0) || !std::isfinite(input.thermal_at_100c))
    {
        throw std::invalid_argument(
            "thermal_at_100c is " + config::number(input.thermal_at_100c) +
            "; it must be a positive number"
        );
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

std::uint64_t input_fingerprint(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qa_codes,
    std::uint8_t qa_fill,
    std::int64_t through
)
{
    return detail::fingerprint_of<scalar_t>(dates, spectral, qa_codes, qa_fill, through);
}

} // namespace ccd
