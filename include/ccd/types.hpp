#pragma once

#include <cstddef>
#include <cstdint>

namespace ccd
{

//==============================================================================
// Indexing
//==============================================================================

using index_t = std::size_t;


//==============================================================================
// Scalar Types
//==============================================================================
//
// CCD uses floating point math everywhere for regression,
// harmonic modeling, and predictions.
//
// Keep this centralized so switching to float or long double
// later is a one-line change.
//

using scalar_t = double;


//==============================================================================
// Time Types
//==============================================================================
//
// Dates are stored as integer ordinal values.
// Example:
//   days since epoch
//   YYYYDDD
//   modified Julian day
//
// The exact representation belongs to the caller.
//

using time_t = std::int64_t;


//==============================================================================
// Pixel / Raster Types
//==============================================================================

using pixel_value_t = std::uint16_t;

using qa_value_t = std::uint8_t;


//==============================================================================
// Regression Types
//==============================================================================

using coefficient_t = scalar_t;
using observation_t = scalar_t;
using prediction_t  = scalar_t;
using residual_t    = scalar_t;


//==============================================================================
// Mask Types
//==============================================================================

using mask_value_t = std::uint8_t;


} // namespace ccd