#include "ccd/procedure/fit.hpp"

#include <iostream>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>

#include "ccd/maths.hpp"
#include "ccd/constants.hpp"
#include "ccd/logger.hpp"


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

index_t adjust_peek(
    ArrayView<const std::int64_t, 1> dates,
    index_t defpeek,
    scalar_t reference_days
)
{
    if (dates.size() < 2)
        return defpeek;

    // The gaps as scalar_t: np.median of an even count is the mean of the
    // two middle gaps, which is a half day when they differ by one.
    std::vector<scalar_t> diffs;
    diffs.reserve(dates.size() - 1);

    for(index_t i = 0; i < dates.size() - 1; ++i)
    {   
        const int64_t curr = dates(i);
        const int64_t next = dates(i + 1);
        diffs.push_back(static_cast<scalar_t>(next - curr));
    }

    const auto delta =
        median(diffs) + 0.001;

    const auto adjusted =
        static_cast<index_t>(
            std::round(defpeek * reference_days / delta)
        );

    return std::max(
        adjusted,
        defpeek
    );
}

scalar_t adjust_change_threshold(
    index_t peek,
    index_t defpeek,
    scalar_t defthresh,
    index_t df,
    scalar_t probability
)
{
    if(peek <= defpeek)
        return defthresh;


    const scalar_t pt =
        1.0 -
        std::pow(
            1.0 - probability,
            static_cast<double>(defpeek)/peek
        );


    return chi2_quantile(pt, df);
}

scalar_t chi2_quantile(scalar_t p, index_t df)
{
    boost::math::chi_squared dist(static_cast<scalar_t>(df));

    return boost::math::quantile(dist, p);
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

    if (count < 2) {
        return variogram;
    }
    //----------------------------------------------------------------------
    // Increase lag until the most common separation exceeds min_lag_days.
    //----------------------------------------------------------------------
    for (index_t lag = 1; lag < count; ++lag)
    {
        std::unordered_map<std::int64_t, index_t> histogram;

        std::int64_t majority_value = 0;
        index_t majority_count = 0;

        for (index_t i = 0; i + lag < count; ++i)
        {
            const std::int64_t delta =
                dates(i + lag) - dates(i);

            const index_t freq =
                ++histogram[delta];

            // if (freq > majority_count)
            // {
            //     majority_count = freq;
            //     majority_value = delta;
            // }
            if (freq > majority_count || (freq == majority_count && delta < majority_value))
            {
                majority_count = freq;
                majority_value = delta;
            }
        }

        LOG_DEBUG(
            "lag = " << lag 
            << ", majority = " << majority_value 
            << ", count = " << majority_count
        );

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

bool stable(
    ArrayView<const std::int64_t, 1> dates_window,
    const Variogram& variogram,
    const std::vector<LassoResult>& results,
    scalar_t change_threshold,
    const BandList& detection_bands
) {
    
    scalar_t euclidean_norm = 0.0;
    for (const auto idx: detection_bands) {

        const auto& model = results[idx].model;
        const auto& score = results[idx].score;

         const scalar_t rmse_norm =
            std::max(variogram[idx], score.rmse);

        const scalar_t slope =
            model.coefficients()[0] *
            (dates_window(dates_window.size() - 1) - dates_window(0));

        const scalar_t first_resid =
            score.residuals[0];

        const scalar_t last_resid =
            score.residuals[
                score.residuals.size() - 1
            ];

        const scalar_t check_val =
            (std::abs(slope)
            + std::abs(first_resid)
            + std::abs(last_resid))
            / rmse_norm;

        euclidean_norm += (check_val * check_val); // sum of squares
    }

    return (euclidean_norm < change_threshold);
}

//----------------------------------------------------------------------
// Returns true if every observation in the peek window exceeds the
// change threshold.
//
// Equivalent to:
//
//     np.min(magnitudes) > change_threshold
//----------------------------------------------------------------------
bool detect_change(
    const std::vector<scalar_t>& magnitudes,
    scalar_t change_threshold
)
{
    if (magnitudes.empty())
    {
        return false;
    }

    return *std::min_element(
        magnitudes.begin(),
        magnitudes.end()
    ) > change_threshold;
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

std::vector<index_t> find_closest_doy(
    ArrayView<const std::int64_t, 1> dates,
    index_t date_index,
    const Window& window,
    index_t count
)
{
    struct Candidate
    {
        scalar_t distance;
        index_t index;
    };

    std::vector<Candidate> candidates;

    candidates.reserve(window.size());

    const scalar_t target =
        static_cast<scalar_t>(
            dates(date_index)
        );

    constexpr scalar_t DAYS_PER_YEAR = 365.25;

    for(index_t i = window.start;
        i < window.stop;
        ++i)
    {
        const scalar_t delta =
            static_cast<scalar_t>(
                dates(i)
            ) - target;

        const scalar_t seasonal =
            std::abs(
                std::round(delta / DAYS_PER_YEAR) *
                DAYS_PER_YEAR -
                delta
            );

        candidates.push_back(
            {
                seasonal,
                i - window.start
            }
        );
    }

    count =
        std::min(
            count,
            static_cast<index_t>(
                candidates.size()
            )
        );

    //----------------------------------------------------------
    // The `count` nearest, nearest first, ties in window order.
    //
    // Candidates are built in increasing index, so ordering by
    // (distance, index) is the stable sort by distance; only
    // the first `count` of that order are needed.
    //----------------------------------------------------------
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + count,
        candidates.end(),
        [](const auto& a,
           const auto& b)
        {
            if (a.distance != b.distance)
                return a.distance < b.distance;
            return a.index < b.index;
        });

    std::vector<index_t> indices;

    indices.reserve(count);

    for(index_t i = 0; i < count; ++i)
    {
        indices.push_back(
            candidates[i].index
        );
    }

    return indices;
}

scalar_t seasonal_rmse(
    const LassoResult& model,
    const std::vector<index_t>& indices,
    scalar_t divisor
)
{
    scalar_t sum = 0.0;
    LOG_DEBUG("closest resids");
    for(auto idx : indices)
    {   
        const scalar_t r =
            model.score.residuals[idx];
        LOG_DEBUG("idx=" << idx << " residual=" << r);
        sum += r * r;
    }
    LOG_DEBUG("C++ sum_sq: " << sum);
    LOG_DEBUG("C++ norm:   " << std::sqrt(sum));
    LOG_DEBUG("C++ rmse:   " << std::sqrt(sum) / divisor);
    return std::sqrt(sum) / divisor;
}

} // namespace ccd


// when optimizing -- TODO
// implement encapsulated project buffers within a workspace
//  object to reduce number of memory allocations