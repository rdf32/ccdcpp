#pragma once

#include "ccd/core/types.hpp"
#include "ccd/core/maths.hpp"
#include "ccd/input/pmask.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/input/harmonic.hpp"
#include "ccd/input/input.hpp"

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

// The saturation range of `input`: refl_min/refl_max and their flags.
inline Range reflectance_range(const Input& input)
{
    return Range{
        Bound{input.refl_min, input.refl_min_inclusive},
        Bound{input.refl_max, input.refl_max_inclusive}
    };
}

// The thermal range of `input`: therm_min/therm_max and their flags.
inline Range thermal_range(const Input& input)
{
    return Range{
        Bound{input.therm_min, input.therm_min_inclusive},
        Bound{input.therm_max, input.therm_max_inclusive}
    };
}

//------------------------------------------------------------------------------
//
// The processing mask of each procedure's chain in
// preprocess::default_plan(workspace.input(), workspace.shared()), over the workspace's
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
// Each builds the default plan per call; a Run builds it once per call
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
// (Input::therm_min/therm_max and their flags, exclusive on both ends by
// default). The bounds are in the unit of `spectral`'s thermal band AT THE
// POINT THIS RUNS -- which for the Standard procedure is after Run::pixel
// applied Input::thermal_scale/thermal_offset, and for the PermanentSnow and
// InsufficientClear procedures is the caller's own unit, because nothing
// converted it. See the "Units" block in input.hpp for why that asymmetry is
// deliberate.
//
// The bounds are parameters rather than the MIN_CELSIUS/MAX_CELSIUS globals so
// that a caller working in degrees Celsius can pass [-93.20, 70.70] instead of
// Celsius x 100. The Input defaults are the globals.
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
// `range` (Input::refl_min/refl_max and their flags, exclusive on
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
// (Off). See GreenFilterScope in config/shared.hpp.
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