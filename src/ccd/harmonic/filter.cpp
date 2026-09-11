#include "ccd/harmonic/filter.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <optional>

#include "ccd/constants.hpp"
#include "ccd/logger.hpp"

namespace ccd
{

namespace filter
{

ProcessingMask standard(
    const HarmonicWorkspace& workspace
) {
    ProcessingMask mask(
        workspace.dates().size()
    );

    const auto dates    = workspace.dates();
    const auto qas      = workspace.qas();
    const auto& options = workspace.options();

    for (index_t i = 0; i < dates.size(); ++i) {

        const auto qa = qas(i);
        // This is an Union (OR) operation so 
        // set true where conditional is valid
        if (
            qa == options.QA_WATER ||
            qa == options.QA_CLEAR
        ) {
            mask.set(i, true);
        }
    }
    LOG_DEBUG("Processing Mask after QA: " << mask.as_ints());

    apply_thermal_filter(
        workspace.spectral(),
        options.THERMAL_IDX,
        options.THERM_MIN,
        options.THERM_MAX,
        mask
    );

    LOG_DEBUG("Processing Mask after thermal: " << mask.as_ints());

    apply_saturation_filter(
        workspace.spectral(),
        options.REFL_MIN,
        options.REFL_MAX,
        mask
    );

    LOG_DEBUG("Processing Mask after saturation: " << mask.as_ints());

    remove_duplicate_dates(
        dates,
        mask
    );

    LOG_DEBUG("Processing Mask after duplicates: " << mask.as_ints());

    return mask;
}

ProcessingMask snow(
    const HarmonicWorkspace& workspace
)
{
    ProcessingMask mask =
        standard(workspace);

    const auto qas = workspace.qas();
    const auto& options = workspace.options();

    for(index_t i = 0; i < qas.size(); ++i)
    {   
        // This is an Union (OR) operation so 
        // set true where conditional is valid
        if(qas(i) == options.QA_SNOW) 
        {
            mask.set(i, true);
        }
    }

    remove_duplicate_dates(
        workspace.dates(),
        mask
    );

    return mask;
}

ProcessingMask insufficientclear(
    const HarmonicWorkspace& workspace
)
{
    ProcessingMask mask = 
        standard(workspace);

    const auto& options = workspace.options();

    apply_green_median_filter(
        workspace.dates(),
        workspace.spectral(),
        options.GREEN_IDX,
        options.STAT_ORD,
        options.MEDIAN_GREEN_FILTER,
        mask
    );

    remove_duplicate_dates(
        workspace.dates(),
        mask
    );

    return mask;
}


void apply_thermal_filter(
    ArrayView<const scalar_t, 2> spectral,
    index_t thermal_band,
    scalar_t min_value,
    scalar_t max_value,
    ProcessingMask& mask
)
{
    // should actually have it check the "layout" and make sure its (B, T)
    assert(spectral.extent(0) < spectral.extent(1));

    ArrayView<const scalar_t, 1> thermal =
        spectral.slice(fixed(thermal_band), all());

    for (index_t i = 0; i < thermal.size(); ++i) {
        scalar_t value = thermal(i);
        const bool valid =
            value > min_value &&
            value < max_value;
        // This is an Intersection (AND) operation so set false where 
        // conditional is not valid retains original mask
        if(!valid) {     
            mask.set(i, false);
        }
    }
}

void apply_saturation_filter(
    ArrayView<const scalar_t, 2> spectral,
    scalar_t min_value,
    scalar_t max_value,
    ProcessingMask& mask
)
{
    // returns mask where true is UNSATURATED where false SATURATED

    constexpr index_t NUM_SATURATION_BANDS = 6;
    assert(spectral.extent(0) >= NUM_SATURATION_BANDS);

    for (index_t obs = 0; obs < spectral.extent(1); ++obs) {

        bool valid = true;
        for(index_t band = 0; band < NUM_SATURATION_BANDS; ++band) {
            const scalar_t value = spectral(band, obs);

            // One predicate, not two. pyccd's collection-1 bound is
            // [0, 10000] and collection-2's is [7273, 43636]; both mean
            // "reflectance in [0, 1]", and a caller working in reflectance
            // passes [0.0, 1.0]. The two commented-out branches this
            // replaced were the same test written in two units.
            if (value < min_value || value > max_value) {
                valid = false;
                break;
            }
        }
        // This is an Intersection (AND) operation so set false where 
        // conditional is not valid retains original mask
        if (!valid) {
            mask.set(obs, false);
        }
    }
}

void remove_duplicate_dates(
    ArrayView<const std::int64_t, 1> dates,
    ProcessingMask& mask
)
{
    std::optional<std::int64_t> previous;

    for(index_t i = 0; i < dates.size(); ++i) {
        if (!mask.test(i))
            continue;

        if (previous && *previous == dates(i)) {
            mask.set(i, false);
        } else {
            previous = dates(i);
        }
    }

}
void apply_green_median_filter(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t green_band,
    std::int64_t max_date,
    scalar_t filter_range,
    ProcessingMask& mask
)
{
    //----------------------------------------------------------------------
    // Collect green values used for median calculation
    //
    // Equivalent to:
    //
    // green = observations[:, standard_mask][green_idx]
    // green = green[dates <= max_ord]
    //
    //----------------------------------------------------------------------

    std::vector<scalar_t> green_values;

    green_values.reserve(
        mask.count()
    );

    for(index_t i = 0; i < dates.size(); ++i)
    {
        if(!mask.test(i))
            continue;

        if(dates(i) <= max_date)
        {
            green_values.push_back(
                spectral(green_band, i)
            );
        }
    }

    if(green_values.empty())
        return;

    //----------------------------------------------------------------------
    // Compute threshold
    //----------------------------------------------------------------------
    const scalar_t threshold =
        median(green_values) + filter_range;

    //----------------------------------------------------------------------
    // Apply filter
    //
    // Equivalent to:
    //
    // green_mask = green < median + filter_range
    //
    //----------------------------------------------------------------------
    for (index_t i = 0; i < dates.size(); ++i)
    {
        if(!mask.test(i))
            continue;


        const scalar_t green =
            spectral(green_band, i);

        // This is an Intersection (AND) operation so set false where 
        // conditional is not valid retains original mask
        if(green >= threshold)
        {
            mask.set(i,false);
        }
    }
}

} // namespace filter

} // namespace ccd
