#include "ccd/strategies/noise/variogram.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "../../core/logger.hpp"
#include "ccd/core/maths.hpp"

namespace ccd
{

namespace
{

//------------------------------------------------------------------------------
// Calculate first-order variogram (madogram)
// Equivalent to:
// np.median(np.abs(np.diff(observations)), axis=1)
//------------------------------------------------------------------------------
std::vector<scalar_t> calculate_variogram(
    ArrayView<const scalar_t, 2> spectral
)
{
    const index_t bands = spectral.extent(0);
    const index_t count = spectral.extent(1);

    std::vector<scalar_t> variogram(
        bands,
        std::numeric_limits<scalar_t>::quiet_NaN()
    );

    if (count < 2) {
        return variogram;
    }

    std::vector<scalar_t> scratch;
    scratch.reserve(count - 1);

    for (index_t band = 0; band < bands; ++band)
    {
        scratch.clear();

        for (index_t i = 1; i < count; ++i)
        {
            scratch.push_back(
                std::abs(
                    spectral(band, i) -
                    spectral(band, i - 1)
                )
            );
        }

        variogram[band] = median(scratch);
    }

    return variogram;
}

// The most common of dates(i + lag) - dates(i), the smallest on a tie.
std::int64_t majority_separation(
    ArrayView<const std::int64_t, 1> dates,
    index_t lag
)
{
    const index_t count = dates.size();
    std::unordered_map<std::int64_t, index_t> histogram;

    std::int64_t majority_value = 0;
    index_t majority_count = 0;

    for (index_t i = 0; i + lag < count; ++i)
    {
        const std::int64_t delta =
            dates(i + lag) - dates(i);

        const index_t freq =
            ++histogram[delta];

        if (freq > majority_count || (freq == majority_count && delta < majority_value))
        {
            majority_count = freq;
            majority_value = delta;
        }
    }

    return majority_value;
}

} // namespace

//------------------------------------------------------------------------------
// Modified variogram.
//
// Attempts to use observations separated by more than min_lag_days.
//
// Equivalent to adjusted_variogram() in Python.
//------------------------------------------------------------------------------
std::vector<scalar_t> adjusted_variogram(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    std::int64_t min_lag_days
)
{
    std::vector<scalar_t> variogram =
        calculate_variogram(spectral);

    const index_t bands = spectral.extent(0);
    const index_t count = spectral.extent(1);
    assert(dates.size() == count);

    if (count < 2) {
        return variogram;
    }
    //----------------------------------------------------------------------
    // Increase lag until the most common separation exceeds min_lag_days.
    //----------------------------------------------------------------------
    for (index_t lag = 1; lag < count; ++lag)
    {
        const std::int64_t majority_value = majority_separation(dates, lag);

        LOG_DEBUG("lag = " << lag << ", majority = " << majority_value);

        if (majority_value <= min_lag_days) {
            continue;
        }
        //------------------------------------------------------------------
        // Recompute variogram using only pairs more than min_lag_days apart.
        //------------------------------------------------------------------
        LOG_DEBUG("SELECTED LAG: " << lag);

        std::vector<scalar_t> scratch;
        for (index_t band = 0; band < bands; ++band)
        {
            scratch.clear();
            for (index_t i = 0; i + lag < count; ++i)
            {
                const std::int64_t delta =
                    dates(i + lag) - dates(i);

                if (delta <= min_lag_days) {
                    continue;
                }
                scratch.push_back(
                    std::abs(
                        spectral(band, i + lag) -
                        spectral(band, i)
                    )
                );
            }
            if (!scratch.empty()) {
                variogram[band] = median(scratch);
            }
        }
        break;
    }

    return variogram;
}

namespace strategies
{

const std::vector<config::Field<VariogramNoise::Params>>& VariogramNoise::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"min_lag_days", &Params::min_lag_days, Unit::Days, config::greater_than(0),
         "the variogram's lag is the first whose most common separation exceeds this"},
    };
    return s;
}

std::vector<scalar_t> VariogramNoise::estimate(ArrayView<const std::int64_t, 1> dates,
                                               ArrayView<const scalar_t, 2> spectral) const
{
    return adjusted_variogram(dates, spectral, p_.min_lag_days);
}

} // namespace strategies
} // namespace ccd
