#include "ccd/harmonic/filter.hpp"

#include <vector>
#include <algorithm>
#include <optional>

#include "ccd/constants.hpp"

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

    const auto dates   = workspace.dates();
    const auto qas     = workspace.qas();
    const auto options = workspace.options();

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

    apply_thermal_filter(
        workspace.spectral(),
        options.THERMAL_IDX,
        mask
    );

    apply_saturation_filter(
        workspace.spectral(),
        mask
    );

    remove_duplicate_dates(
        dates,
        mask
    );

    return mask;
}

ProcessingMask snow(
    const HarmonicWorkspace& workspace
)
{
    ProcessingMask mask =
        standard(workspace);

    const auto qas = workspace.qas();
    const auto options = workspace.options();

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

    const auto options = workspace.options();

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
            value > MIN_CELSIUS &&
            value < MAX_CELSIUS;
        // This is an Intersection (AND) operation so set false where 
        // conditional is not valid retains original mask
        if(!valid) {     
            mask.set(i, false);
        }
    }
}

void apply_saturation_filter(
    ArrayView<const scalar_t, 2> spectral,
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

            if (value <= 0.0 || value >= 10000.0) {
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
