#include "ccd/machine/fit.hpp"

#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <boost/math/distributions/chi_squared.hpp>

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

scalar_t ChangeThresholdMemo::operator()(index_t peek, index_t defpeek, scalar_t defthresh,
                                         index_t df, scalar_t probability)
{
    if (!(valid_ && peek == peek_ && defpeek == defpeek_ && df == df_ &&
          defthresh == defthresh_ && probability == probability_))
    {
        value_ = adjust_change_threshold(peek, defpeek, defthresh, df, probability);
        peek_ = peek;
        defpeek_ = defpeek;
        df_ = df;
        defthresh_ = defthresh;
        probability_ = probability;
        valid_ = true;
    }
    return value_;
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

namespace
{

constexpr std::int64_t QUARTERS_PER_YEAR = 1461;

// 4 * day mod 1461, in [0, 1461).
std::uint16_t quarter_phase(std::int64_t day)
{
    const std::int64_t r = (4 * day) % QUARTERS_PER_YEAR;
    return static_cast<std::uint16_t>(r < 0 ? r + QUARTERS_PER_YEAR : r);
}

} // namespace

void SeasonalOrder::build(ArrayView<const std::int64_t, 1> dates, const Window& window)
{
    by_phase_.clear();
    by_phase_.reserve(window.size());
    for (index_t i = 0; i < window.size(); ++i)
        by_phase_.push_back({quarter_phase(dates(window.start + i)), i});
    std::sort(by_phase_.begin(), by_phase_.end(), [](const Entry& a, const Entry& b)
              { return a.phase != b.phase ? a.phase < b.phase : a.index < b.index; });
}

void SeasonalOrder::closest(std::int64_t target_day, index_t count, std::vector<index_t>& out) const
{
    //----------------------------------------------------------
    // The seasonal distance, in quarter days, is the distance
    // from 4 |delta| to the nearest multiple of 1461: an exact
    // integer, at most 730, which orders as the double distance
    // does; it is min(m, 1461 - m) for m the phase difference
    // mod 1461.
    // Walking forward from the target's phase the forward
    // difference only grows, walking backward the backward one;
    // taking the nearer of the two walks each time reaches each
    // observation first by its shorter way (the two ways sum to
    // 1461, odd, so never tie), in order of distance. Each
    // distance's observations are put in window order.
    //----------------------------------------------------------
    out.clear();
    const index_t n = by_phase_.size();
    count = std::min(count, n);
    if (count == 0)
        return;

    const std::int64_t target = quarter_phase(target_day);
    const auto start = std::lower_bound(
        by_phase_.begin(), by_phase_.end(), target,
        [](const Entry& e, std::int64_t phase) { return e.phase < phase; });
    index_t ahead = static_cast<index_t>(start - by_phase_.begin()) % n;   // next forward
    index_t behind = (ahead + n - 1) % n;                                  // next backward
    index_t taken = 0;

    const auto forward = [&](index_t k)
    {
        const std::int64_t d = by_phase_[k].phase - target;
        return d < 0 ? d + QUARTERS_PER_YEAR : d;
    };
    const auto backward = [&](index_t k)
    {
        const std::int64_t d = target - by_phase_[k].phase;
        return d < 0 ? d + QUARTERS_PER_YEAR : d;
    };

    while (out.size() < count && taken < n)
    {
        const std::int64_t distance = std::min(forward(ahead), backward(behind));
        group_.clear();
        while (taken < n && forward(ahead) == distance)
        {
            group_.push_back(by_phase_[ahead].index);
            ahead = (ahead + 1) % n;
            ++taken;
        }
        while (taken < n && backward(behind) == distance)
        {
            group_.push_back(by_phase_[behind].index);
            behind = (behind + n - 1) % n;
            ++taken;
        }
        std::sort(group_.begin(), group_.end());
        out.insert(out.end(), group_.begin(), group_.end());
    }
    out.resize(count);
}

scalar_t seasonal_rmse(
    const std::vector<scalar_t>& residuals,
    const std::vector<index_t>& indices,
    scalar_t divisor
)
{
    scalar_t sum = 0.0;
    for(auto idx : indices)
    {
        const scalar_t r =
            residuals[idx];
        sum += r * r;
    }
    return std::sqrt(sum) / divisor;
}

} // namespace ccd
