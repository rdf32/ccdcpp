#pragma once

#include <cstdint>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

// The first-order variogram at the smallest lag whose most common
// separation exceeds min_lag_days, over the pairs further apart than
// min_lag_days; the first-order variogram at lag 1 when no lag qualifies.
// min_lag_days is the variogram noise's min_lag_days.
std::vector<scalar_t> adjusted_variogram(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    std::int64_t min_lag_days
);

namespace strategies
{

//------------------------------------------------------------------------------
// noise "variogram": pyccd's. The first-order variogram at the smallest
// lag whose most common separation exceeds min_lag_days, over the pairs
// further apart than that (adjusted_variogram()).
//------------------------------------------------------------------------------
struct VariogramNoise final : Noise
{
    struct Params
    {
        std::int64_t min_lag_days = 30;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit VariogramNoise(const Params& p) : p_(p) {}

    std::vector<scalar_t> estimate(ArrayView<const std::int64_t, 1> dates,
                                   ArrayView<const scalar_t, 2> spectral) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
