#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ccd/core/array_view.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/maths.hpp"
#include "ccd/core/types.hpp"
#include "ccd/input/pmask.hpp"

namespace ccd
{

//==============================================================================
//
// GreenFilterScope
//
// Which observations the green median filter drops. The threshold is the
// same in each: the median green of the masked observations dated <=
// max_date (the statistics cutoff), plus the filter's offset. A masked
// observation whose green is >= the threshold is dropped
//
//   All          at any date;
//   StatWindow   only when dated <= max_date, which is the scope pyccd's
//                insufficient_clear_filter computes the mask for (the
//                default);
//   Off          never: the filter does not run.
//
// When no masked observation is dated <= max_date there is no median and
// nothing is dropped. The configuration names them "all", "stat_window" and
// "off".
//
//==============================================================================

enum class GreenFilterScope : std::uint8_t
{
    All,
    StatWindow,
    Off
};

// "all", "stat_window", "off".
const char* name(GreenFilterScope scope);

// The scope of that name, or none.
std::optional<GreenFilterScope> green_filter_scope(const std::string& name);

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
// (the thermal_range step's bounds, exclusive on both ends by default). The
// bounds are in the unit of the thermal row after the input's transform; the
// row is tested as `spectral` holds it AT THE POINT THIS RUNS -- transformed
// in a procedure that applies the transform (pyccd's Standard), the caller's
// own unit in one that does not (pyccd's PermanentSnow and
// InsufficientClear). See the "Units" block in input.hpp for why that
// asymmetry is deliberate.
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
// `range` (the band_range step's bounds, exclusive on both ends by default);
// the bands are tested in list order.
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
// (Off). See GreenFilterScope above.
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