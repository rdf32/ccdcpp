#pragma once

#include "ccd/types.hpp"
#include "ccd/maths.hpp"
#include "ccd/pmask.hpp"
#include "ccd/array_view.hpp"
#include "ccd/harmonic/harmonic.hpp"

namespace ccd
{

namespace filter
{

//------------------------------------------------------------------------------
//
// Standard CCD filtering.
//
// Keeps observations that are:
//   - QA clear
//   - QA water
//   - valid thermal
//   - unsaturated
//   - unique acquisition dates
//
//------------------------------------------------------------------------------

ProcessingMask standard(
    const HarmonicWorkspace& workspace
);

// Snow filtering -- applies standard filtering with UNION (OR) Snow QA
ProcessingMask snow(
    const HarmonicWorkspace& workspace
);

// InsufficientClear filtering -- applies standard filtering with UNION (OR) Snow QA
ProcessingMask insufficientclear(
    const HarmonicWorkspace& workspace
);

//------------------------------------------------------------------------------
//
// Remove duplicate acquisition dates.
//
// Keeps the first observation for each date.
//
//------------------------------------------------------------------------------

void remove_duplicate_dates(
    ArrayView<const std::int64_t, 1> dates,
    ProcessingMask& mask
);


//------------------------------------------------------------------------------
//
// Remove observations with invalid thermal values.
//
//------------------------------------------------------------------------------

void apply_thermal_filter(
    ArrayView<const scalar_t, 2> spectral,
    index_t thermal_band,
    ProcessingMask& mask
);


//------------------------------------------------------------------------------
//
// Remove saturated observations.
//
//------------------------------------------------------------------------------

void apply_saturation_filter(
    ArrayView<const scalar_t, 2> spectral,
    ProcessingMask& mask
);


//------------------------------------------------------------------------------
//
// Insufficient clear procedure green filter.
//
// Removes observations where:
//
// green > median(green before STAT_ORD) + filter_range
//
//------------------------------------------------------------------------------

void apply_green_median_filter(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t green_band,
    std::int64_t max_date,
    scalar_t filter_range,
    ProcessingMask& mask
);

} // namespace filter

} // namespace ccd