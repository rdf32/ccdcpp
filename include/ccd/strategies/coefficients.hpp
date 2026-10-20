#pragma once

#include <stdexcept>
#include <string>

#include "ccd/core/constants.hpp"
#include "ccd/core/types.hpp"

namespace ccd
{

//------------------------------------------------------------------------------
// CoefficientRule
//
// How many coefficients a model fits a window of n observations with: `min`
// below mid * per observations, `mid` below max * per, else `max` (pyccd's
// 4, 6 and 8 at 3 observations per coefficient). A stability fit fits `min`;
// `max` is also the width of every peek basis (a fit predicts with zeros
// above its own count). A strategy whose parameters hold a rule checks it
// when it is built (check()).
//------------------------------------------------------------------------------
struct CoefficientRule
{
    index_t min = 4;
    index_t mid = 6;
    index_t max = 8;
    index_t per = 3;

    index_t count(index_t observations) const
    {
        const index_t span = observations / per;
        if (span < mid)
            return min;
        if (span < max)
            return mid;
        return max;
    }

    // Refuses a rule the harmonic basis cannot hold, as a schema refuses a
    // parameter: `where` is the strategy ("model 'lasso'"). Each count is at
    // least 1 by the strategy's schema.
    void check(const std::string& where) const
    {
        if (!(min <= mid && mid <= max))
            throw std::invalid_argument(
                where + " parameters 'coefficient_min', 'coefficient_mid', 'coefficient_max' are " +
                std::to_string(min) + ", " + std::to_string(mid) + ", " + std::to_string(max) +
                "; they must not decrease");
        if (max > CCD_MAX_COEFS + 1)
            throw std::invalid_argument(
                where + " parameter 'coefficient_max' is " + std::to_string(max) +
                "; the harmonic basis has at most " + std::to_string(CCD_MAX_COEFS + 1));
    }
};

} // namespace ccd
