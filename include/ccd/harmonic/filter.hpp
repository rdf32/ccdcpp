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
// `min_value` / `max_value` are exclusive bounds in the unit of `spectral`'s
// thermal band AT THE POINT THIS RUNS -- which for the Standard procedure is
// after ccd::detect applied HarmonicOptions::THERMAL_SCALE/THERMAL_OFFSET, and
// for the PermanentSnow and InsufficientClear procedures is the caller's own
// unit, because nothing converted it. See the "Input unit convention" block in
// harmonic.hpp for why that asymmetry is deliberate.
//
// The bounds are parameters rather than the MIN_CELSIUS/MAX_CELSIUS globals so
// that a caller working in degrees Celsius can pass [-93.20, 70.70] instead of
// Celsius x 100. Defaults reproduce the globals exactly.
//
//------------------------------------------------------------------------------

void apply_thermal_filter(
    ArrayView<const scalar_t, 2> spectral,
    index_t thermal_band,
    scalar_t min_value,
    scalar_t max_value,
    ProcessingMask& mask
);


//------------------------------------------------------------------------------
//
// Remove saturated observations.
//
// `min_value` / `max_value` are inclusive bounds on each of the six optical
// bands. pyccd's Collection-1 values are [0, 10000] and Collection-2's are
// [7273, 43636]; both express the single predicate `reflectance in [0, 1]`
// (C2 surface reflectance is DN * 2.75e-5 - 0.2), so a caller working in
// reflectance passes [0.0, 1.0] and there is no collection branch left.
//
//------------------------------------------------------------------------------

void apply_saturation_filter(
    ArrayView<const scalar_t, 2> spectral,
    scalar_t min_value,
    scalar_t max_value,
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