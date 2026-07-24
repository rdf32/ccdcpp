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


} // namespace ccd