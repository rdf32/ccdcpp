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
// A bound of a valid range, and the range test the thermal and saturation
// filters share.
//
// in_range() keeps `value` when it lies above `lo` and below `hi`, an
// inclusive end also keeping a value equal to it. Each end is written so that
// a NaN fails an exclusive end and passes an inclusive one, so:
//
//   both inclusive   !(value < lo || value > hi)   NaN passes
//   both exclusive   value > lo && value < hi      NaN is rejected
//   mixed                                          NaN is rejected
//
//------------------------------------------------------------------------------

struct Bound
{
    scalar_t value;
    bool inclusive;
};

struct Range
{
    Bound lo;
    Bound hi;
};

inline bool in_range(scalar_t value, const Range& range)
{
    const bool above =
        range.lo.inclusive ? !(value < range.lo.value) : value > range.lo.value;
    const bool below =
        range.hi.inclusive ? !(value > range.hi.value) : value < range.hi.value;
    return above && below;
}

// The saturation range of `options`: REFL_MIN/REFL_MAX and their flags.
inline Range reflectance_range(const HarmonicOptions& options)
{
    return Range{
        Bound{options.REFL_MIN, options.REFL_MIN_INCLUSIVE},
        Bound{options.REFL_MAX, options.REFL_MAX_INCLUSIVE}
    };
}

// The thermal range of `options`: THERM_MIN/THERM_MAX and their flags.
inline Range thermal_range(const HarmonicOptions& options)
{
    return Range{
        Bound{options.THERM_MIN, options.THERM_MIN_INCLUSIVE},
        Bound{options.THERM_MAX, options.THERM_MAX_INCLUSIVE}
    };
}

//------------------------------------------------------------------------------
//
// The processing mask of each procedure's chain in
// preprocess::default_plan(workspace.options()), over the workspace's
// spectral block as it is (no transform is applied here).
//
// standard keeps observations that are:
//   - QA clear or QA water
//   - valid thermal
//   - unsaturated
//   - the first of their acquisition date
//
// snow is standard with every QA snow observation added back (UNION, OR),
// then deduplicated again.
//
// insufficientclear is standard followed by the green median filter, then
// deduplicated again.
//
// Each builds the default plan per call; ccd::detect builds it once per call
// instead and runs preprocess::apply() per pixel.
//
//------------------------------------------------------------------------------

ProcessingMask standard(
    const HarmonicWorkspace& workspace
);

ProcessingMask snow(
    const HarmonicWorkspace& workspace
);

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
// An observation is kept when its thermal value is in_range() of `range`
// (HarmonicOptions::THERM_MIN/THERM_MAX and their flags, exclusive on both
// ends by default). The bounds are in the unit of `spectral`'s
// thermal band AT THE POINT THIS RUNS -- which for the Standard procedure is
// after ccd::detect applied HarmonicOptions::THERMAL_SCALE/THERMAL_OFFSET, and
// for the PermanentSnow and InsufficientClear procedures is the caller's own
// unit, because nothing converted it. See the "Input unit convention" block in
// harmonic.hpp for why that asymmetry is deliberate.
//
// The bounds are parameters rather than the MIN_CELSIUS/MAX_CELSIUS globals so
// that a caller working in degrees Celsius can pass [-93.20, 70.70] instead of
// Celsius x 100. The option defaults are the globals.
//
//------------------------------------------------------------------------------

void apply_thermal_filter(
    ArrayView<const scalar_t, 2> spectral,
    index_t thermal_band,
    const Range& range,
    ProcessingMask& mask
);


//------------------------------------------------------------------------------
//
// Remove saturated observations.
//
// An observation is saturated when any of `bands` is not in_range() of
// `range` (HarmonicOptions::REFL_MIN/REFL_MAX and their flags, exclusive on
// both ends by default); the bands are tested in list order.
// pyccd's Collection-1 values are (0, 10000) and Collection-2's are
// (7273, 43636); both express the single predicate `reflectance in (0, 1)`
// (C2 surface reflectance is DN * 2.75e-5 - 0.2), so a caller working in
// reflectance passes (0.0, 1.0) and there is no collection branch left.
//
//------------------------------------------------------------------------------

void apply_saturation_filter(
    ArrayView<const scalar_t, 2> spectral,
    const BandList& bands,
    const Range& range,
    ProcessingMask& mask
);


//------------------------------------------------------------------------------
//
// Insufficient clear procedure green filter.
//
// Removes masked observations where
//
//   green >= median(green of the masked observations dated <= max_date)
//            + filter_range
//
// at any date (scope All), only at dates <= max_date (StatWindow), or none
// (Off). See GreenFilterScope in harmonic.hpp.
//
//------------------------------------------------------------------------------

void apply_green_median_filter(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    index_t green_band,
    std::int64_t max_date,
    scalar_t filter_range,
    GreenFilterScope scope,
    ProcessingMask& mask
);

} // namespace filter

} // namespace ccd