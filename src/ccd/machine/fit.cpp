#include "ccd/machine/fit.hpp"

#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

#include "ccd/core/maths.hpp"
#include "ccd/core/constants.hpp"
#include "../core/logger.hpp"


namespace ccd
{

BandFit band_fit(const LassoResult& fit)
{
    BandFit out;
    out.intercept = fit.model.intercept();
    out.coefficients = fit.model.coefficients();
    out.rmse = fit.score.rmse;
    out.magnitude = fit.score.magn;
    out.iterations = fit.model.iterations();
    return out;
}

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

namespace
{

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

//------------------------------------------------------------------------------
// Verify variogram.
//
// Python:
//
// if any(np.isnan(vario)):
//     return False
//
//------------------------------------------------------------------------------
bool check_variogram(
    const std::vector<scalar_t>& variogram
)
{
    for (index_t i = 0; i < variogram.size(); ++i)
    {
        if (std::isnan(variogram[i])) {
            return false;
        }
    }
    return true;
}

// timeseries window operations
bool enough_time(
    ArrayView<const std::int64_t, 1> dates_window,
    std::int64_t day_delta
) {
    const auto first_date = dates_window(0);
    const auto last_date = dates_window(dates_window.size() - 1);

    return (last_date - first_date) >= day_delta;
}

bool enough_samples(
    ArrayView<const std::int64_t, 1> dates_window,
    index_t meow_size
) {
    return dates_window.size() >= meow_size;
}

ProcessingMask tmask(
    ArrayView<const std::int64_t, 1> dates_window,
    ArrayView<const scalar_t, 2> spect_window,
    const Variogram& variogram,
    const BandList& bands,
    scalar_t t_const
) {
    const index_t n =
        spect_window.extent(1);
    //----------------------------------------------------------
    // Output mask
    // true = outlier
    //----------------------------------------------------------
    ProcessingMask outliers(n);

    Eigen::MatrixXd X =
        tmask_basis(dates_window);
    
    Eigen::VectorXd y(n);
    
    RobustOptions options;
    options.max_iter = 5;
    RobustSolver solver(options);

    // Every band is fitted against the same X.
    RobustDesign design(X);

    //----------------------------------------------------------
    // Test each spectral band
    //----------------------------------------------------------
    for(const auto band : bands)
    {

        for (index_t i = 0; i < n; ++i)
        {
            y(i) = spect_window(band, i);
        }
        //------------------------------------------------------
        // Fit
        //------------------------------------------------------
        RobustModel model = solver.fit(
            design,
            y
        );

        //------------------------------------------------------
        // Residual threshold
        //------------------------------------------------------
        const scalar_t threshold =
            variogram[band] * t_const;

        auto preds = model.predict(X);

        for(index_t i = 0; i < n; ++i)
        {
            const scalar_t residual = std::abs(preds[i] - y(i));

            if(
                residual
                >
                threshold
            )
            {
                outliers.set(i,true);
            }
        }
        
    }

    return outliers;
}


} // namespace ccd
