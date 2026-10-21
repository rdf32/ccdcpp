#include "ccd/strategies/pyccd/peek_statistics.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "ccd/core/maths.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

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

void change_magnitude(
    const std::vector<std::vector<scalar_t>>& residuals,
    const BandList& bands,
    const BandNoise& variogram,
    const std::vector<scalar_t>& rmse,
    std::vector<scalar_t>& magnitude
)
{
    assert(rmse.size() >= bands.size());

    magnitude.clear();

    if (bands.empty())
        return;

    const index_t count =
        residuals[bands.front()].size();

    magnitude.assign(count, 0.0);
    for (index_t i = 0; i < bands.size(); ++i)
    {
        const scalar_t norm =
            std::max(
                variogram[bands[i]],
                rmse[i]
            );

        if (norm <= 0.0)
            continue;

        const auto& band_residuals = residuals[bands[i]];

        for (index_t j = 0; j < count; ++j)
        {
            const scalar_t value =
                band_residuals[j] / norm;

            magnitude[j] += value * value;
        }
    }
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
