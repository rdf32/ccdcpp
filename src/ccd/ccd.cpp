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

// `name` holds only indices below `bands`, each once.
void validate_list(const char* name, const BandList& list, index_t bands)
{
    for (index_t i = 0; i < list.size(); ++i)
    {
        if (list[i] >= bands)
        {
            throw std::invalid_argument(
                std::string(name) + " holds band " + std::to_string(list[i]) +
                "; " + count_text(bands)
            );
        }

        for (index_t j = 0; j < i; ++j)
        {
            if (list[j] == list[i])
            {
                throw std::invalid_argument(
                    std::string(name) + " holds band " +
                    std::to_string(list[i]) + " more than once"
                );
            }
        }
    }
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

    validate_list("saturation_bands", input.saturation_bands, bands);

    const index_t thermal = input.index("thermal");
    if (thermal != NO_BAND && input.saturation_bands.contains(thermal))
    {
        throw std::invalid_argument(
            "saturation_bands holds the thermal row (" + std::to_string(thermal) +
            "); the thermal band has its own bounds"
        );
    }

    if (!(input.reflectance_unit > 0) || !std::isfinite(input.reflectance_unit))
    {
        throw std::invalid_argument(
            "reflectance_unit is " + config::number(input.reflectance_unit) +
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

namespace
{

void digest(detail::Fnv& h, const preprocess::ProcedurePlan& plan)
{
    h.flag(plan.transform.has_value());
    if (plan.transform)
    {
        h.integer(static_cast<std::int64_t>(plan.transform->band));
        h.real(plan.transform->scale);
        h.real(plan.transform->offset);
    }

    h.integer(static_cast<std::int64_t>(plan.chain.size()));
    for (index_t i = 0; i < plan.chain.size(); ++i)
    {
        const preprocess::Step& step = plan.chain.steps[i];
        h.integer(static_cast<std::int64_t>(step.kind));
        for (std::size_t c = 0; c < step.codes.size(); ++c)
            h.flag(step.codes.test(c));
        h.bands(step.bands);
        h.integer(static_cast<std::int64_t>(step.band));
        h.real(step.range.lo.value);
        h.flag(step.range.lo.inclusive);
        h.real(step.range.hi.value);
        h.flag(step.range.hi.inclusive);
        h.real(step.green_offset);
        h.integer(step.max_date);
        h.integer(static_cast<std::int64_t>(step.scope));
    }
}

} // namespace

std::uint64_t options_digest(
    const config::Resolved& resolved,
    const preprocess::PreprocessPlan& plan
)
{
    detail::Fnv h;
    h.integer(static_cast<std::int64_t>(resolved.digest));
    for (const preprocess::ProcedurePlan* procedure :
         {&plan.standard, &plan.snow, &plan.insufficient_clear})
        digest(h, *procedure);
    return h.value();
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
