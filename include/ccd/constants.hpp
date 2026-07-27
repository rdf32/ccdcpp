#pragma once

#include "ccd/types.hpp"


namespace ccd
{

//==============================================================================
// Mathematical Constants
//==============================================================================

constexpr scalar_t PI =
    static_cast<scalar_t>(
        3.141592653589793238462643383279502884L
    );


//==============================================================================
// Temporal Constants
//==============================================================================
//
// Average Gregorian year length.
// Used for harmonic regression:
//
// sin(2*pi*t / AVG_DAYS_YR)
// cos(2*pi*t / AVG_DAYS_YR)
//

constexpr scalar_t AVG_DAYS_YR =
    static_cast<scalar_t>(365.2425);


//==============================================================================
// Numerical Constants
//==============================================================================

constexpr scalar_t EPSILON =
    static_cast<scalar_t>(1e-12);


constexpr scalar_t KELVIN_SCALE = 10.0;
constexpr scalar_t KELVIN_OFFSET = 27315.0;

constexpr scalar_t MIN_CELSIUS = -9320.0;
constexpr scalar_t MAX_CELSIUS = 7070.0;

} // namespace ccd