#include "ccd/procedure/fit.hpp"

#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>

#include "ccd/maths.hpp"


namespace ccd
{

FitResult FitProcedure::run(
    HarmonicWorkspace& workspace,
    LassoSolver& solver
) { 
    const auto& options = workspace.options();
    //----------------------------------------------------------
    // Build processing mask
    //----------------------------------------------------------
    ProcessingMask mask =
        select_observations(workspace);

    //----------------------------------------------------------
    // Not enough observations
    //----------------------------------------------------------
    if (mask.count() < options.MEOW_SIZE)
    {
        return {};
    }

    //----------------------------------------------------------
    // Mask dates, spectral given ProcessingMask
    //----------------------------------------------------------
    MaskedData masked_data = 
        apply_mask(workspace, mask);

    const auto masked_dates = 
        masked_data.dates_view();

    const auto masked_spectral = 
        masked_data.spectral_view();

    //----------------------------------------------------------
    // Determine harmonic complexity
    //----------------------------------------------------------
    const index_t num_coef =
        coefficient_count(masked_dates, options);

    //----------------------------------------------------------
    // Build harmonic basis
    //----------------------------------------------------------
    auto X_storage = 
        lasso_basis(masked_dates, num_coef);

    auto X = ArrayView<const scalar_t, 2>::contiguous(
        X_storage.data(),
        {masked_dates.size(), 7} // always shape (num_obs, 7) for lasso basis
    );

    //----------------------------------------------------------
    // Build result
    //----------------------------------------------------------
    FitResult results;
    ChangeModel result;
    result.start_day = workspace.dates()(0);
    result.end_day   = workspace.dates()(workspace.dates().size() - 1);
    result.break_day = result.end_day;

    result.observation_count = mask.count();
    result.change_probability = 0.0;
    result.curve_qa = curve_qa();

    //----------------------------------------------------------
    // Fit every spectral band
    //----------------------------------------------------------
    for (index_t band = 0; band < masked_spectral.extent(0); ++band)
    {
        auto y = masked_spectral.slice(
            fixed(band), 
            all()
        );

        LassoModel model = solver.fit(
            X,
            y
        );

        auto preds = model.predict(
            X
        );

        auto metrics = score(
            y,
            preds,
            num_coef,
            true
        );
        metrics.magn = 0.0;
        LassoResult band_result = {
            model,
            metrics
        };

        result.bands[band] = band_result;
    }

    results.models.push_back(result);
    results.mask = mask;

    return results;
}

index_t adjust_peek(
    ArrayView<const std::int64_t, 1> dates,
    index_t defpeek
)
{
    if (dates.size() < 2)
        return defpeek;

    std::vector<std::int64_t> diffs;
    diffs.reserve(dates.size() - 1);

    for(index_t i = 0; i < dates.size() - 1; ++i)
    {   
        const int64_t curr = dates(i);
        const int64_t next = dates(i + 1);
        diffs.push_back(next - curr);
    }

    const auto delta =
        median(diffs) + 0.001;

    const auto adjusted =
        static_cast<index_t>(
            std::round(defpeek * 16.0 / delta)
        );

    return std::max(
        adjusted,
        defpeek
    );
}

scalar_t adjust_change_threshold(
    index_t peek,
    index_t defpeek,
    scalar_t defthresh
)
{
    if(peek <= defpeek)
        return defthresh;


    const scalar_t pt =
        1.0 -
        std::pow(
            1.0-0.99,
            static_cast<double>(defpeek)/peek
        );


    boost::math::chi_squared dist(5);

    return boost::math::quantile(dist, pt);
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
// Attempts to use observations separated by >30 days.
//
// Equivalent to adjusted_variogram() in Python.
//------------------------------------------------------------------------------
std::vector<scalar_t> adjusted_variogram(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral
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
    // Increase lag until the most common separation exceeds 30 days.
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

            if (freq > majority_count)
            {
                majority_count = freq;
                majority_value = delta;
            }
        }

        if (majority_value <= 30) {
            continue;
        }
        //------------------------------------------------------------------
        // Recompute variogram using only pairs >30 days apart.
        //------------------------------------------------------------------
        std::vector<scalar_t> scratch;
        for (index_t band = 0; band < bands; ++band)
        {
            scratch.clear();
            for (index_t i = 0; i + lag < count; ++i)
            {
                const std::int64_t delta =
                    dates(i + lag) - dates(i);

                if (delta <= 30) {
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

    // if (window.size() == 0)
    //     return false;

    // auto subset =
    //     dates.slice(
    //         range(window.start, window.stop)
    //     );
    const auto first_date = dates_window(0);
    const auto last_date = dates_window(dates_window.size() - 1);

    return (last_date - first_date) >= day_delta;
}

bool enough_samples(
    ArrayView<const std::int64_t, 1> dates_window,
    index_t meow_size
) {
    // if(window.size() == 0)
    //     return false;

    // auto subset =
    //     dates.slice(
    //         range(window.start, window.stop)
    //     );
    return dates_window.size() >= meow_size;
}

bool stable(
    ArrayView<const std::int64_t, 1> dates_window,
    std::vector<scalar_t>& variogram,
    std::vector<LassoResult>& results,
    scalar_t change_threshold,
    std::vector<index_t>& detection_bands
) {

    // if(window.size() == 0)
    //     return false;

    // auto dates_window =
    //     dates.slice(
    //         range(window.start, window.stop)
    //     );
    
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

// //----------------------------------------------------------------------
// // Returns true if every observation in the peek window exceeds the
// // change threshold.
// //
// // Equivalent to:
// //
// //     np.min(magnitudes) > change_threshold
// //----------------------------------------------------------------------

// inline bool detect_change(
//     const std::vector<scalar_t>& magnitudes,
//     scalar_t change_threshold
// )
// {
//     if (magnitudes.empty())
//     {
//         return false;
//     }

//     return *std::min_element(
//         magnitudes.begin(),
//         magnitudes.end()
//     ) > change_threshold;
// }

// //----------------------------------------------------------------------
// // Returns true if the current observation should be treated as an outlier.
// //
// // Equivalent to:
// //
// //     magnitude > outlier_threshold
// //----------------------------------------------------------------------

// inline bool detect_outlier(
//     scalar_t magnitude,
//     scalar_t outlier_threshold
// )
// {
//     return magnitude > outlier_threshold;
// }

// inline std::vector<scalar_t> change_magnitude(
//     const std::vector<std::vector<scalar_t>>& residuals,
//     const std::vector<scalar_t>& variogram,
//     const std::vector<scalar_t>& rmse
// )
// {
//     assert(residuals.size() == variogram.size());
//     assert(residuals.size() == rmse.size());

//     if(residuals.empty())
//         return {};

//     const index_t count = 
//         residuals.front().size();

//     std::vector<scalar_t> magnitude(count, 0.0);

//     for(index_t band = 0; band < residuals.size(); ++band)
//     {
//         const scalar_t norm =
//             std::max(
//                 variogram[band],
//                 rmse[band]
//             );

//         if(norm <= 0.0)
//             continue;

//         for (index_t i = 0; i < count; ++i)
//         {
//             const scalar_t value =
//                 residuals[band][i] / norm;

//             magnitude[i] += value * value;
//         }
//     }

//     return magnitude;
// }

} // namespace ccd


// when optimizing -- TODO
// implement encapsulated project buffers within a workspace
//  object to reduce number of memory allocations