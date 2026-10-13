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

    // An input without a thermal band has nothing for this screen to test,
    // which is the same mask as a thermal band that is always in range.
    if (options.THERMAL_IDX != NO_BAND)
    {
        apply_thermal_filter(
            workspace.spectral(),
            options.THERMAL_IDX,
            thermal_range(options),
            mask
        );
    }

    LOG_DEBUG("Processing Mask after thermal: " << mask.as_ints());

    apply_saturation_filter(
        workspace.spectral(),
        options.SATURATION_BANDS,
        reflectance_range(options),
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
        options.GREEN_FILTER_SCOPE,
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
    const Range& range,
    ProcessingMask& mask
)
{
    // `spectral` is (B, T). Its extents cannot tell (B, T) from (T, B) --
    // a series may be shorter than its band count -- so the band count is
    // checked at the entry points (validate) and this checks only that
    // the thermal band is on the band axis.
    assert(thermal_band < spectral.extent(0));

    ArrayView<const scalar_t, 1> thermal =
        spectral.slice(fixed(thermal_band), all());

    for (index_t i = 0; i < thermal.size(); ++i) {
        const bool valid = in_range(thermal(i), range);
        // This is an Intersection (AND) operation so set false where 
        // conditional is not valid retains original mask
        if(!valid) {     
            mask.set(i, false);
        }
    }
}

void apply_saturation_filter(
    ArrayView<const scalar_t, 2> spectral,
    const BandList& bands,
    const Range& range,
    ProcessingMask& mask
)
{
    // returns mask where true is UNSATURATED where false SATURATED

    for (const index_t band : bands)
        assert(band < spectral.extent(0));

    for (index_t obs = 0; obs < spectral.extent(1); ++obs) {

        bool valid = true;
        for (const index_t band : bands) {
            // One predicate for every collection. pyccd's collection-1
            // bound is (0, 10000) and collection-2's is (7273, 43636); both
            // mean "reflectance in (0, 1)", and a caller working in
            // reflectance passes (0.0, 1.0).
            if (!in_range(spectral(band, obs), range)) {
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
    GreenFilterScope scope,
    ProcessingMask& mask
)
{
    if (scope == GreenFilterScope::Off)
        return;

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
    // over the masked observations dated <= max_date for StatWindow, and
    // over every masked observation for All.
    //----------------------------------------------------------------------
    const bool window_only = scope == GreenFilterScope::StatWindow;

    for (index_t i = 0; i < dates.size(); ++i)
    {
        if(!mask.test(i))
            continue;

        if (window_only && dates(i) > max_date)
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
