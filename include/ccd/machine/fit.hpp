#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/core/window.hpp"
#include "ccd/input/pmask.hpp"
#include "ccd/results/segment.hpp"

#include "ccd/input/pixel_series.hpp"

#include "ccd/regression/lasso_solver.hpp"
#include "ccd/regression/robust_solver.hpp"


namespace ccd
{

// The BandFit of a Lasso fit: its model's intercept, coefficients and
// iteration count, its RMSE and its magnitude.
BandFit band_fit(const LassoResult& fit);

// full timeseries

// The peek size for `dates`: defpeek * reference_days over the median gap
// between consecutive dates, rounded, and never below `defpeek`. `defpeek`
// for fewer than two dates. reference_days is
// chi2's peek_reference_days.
index_t adjust_peek(
    ArrayView<const std::int64_t, 1> dates,
    index_t defpeek,
    scalar_t reference_days
);

// The change threshold for a peek window of `peek` observations where the
// default window is `defpeek`: the chi-squared quantile, with `df` degrees of
// freedom (the detection band count), of the per-observation probability
// 1 - (1 - probability)^(defpeek / peek), which keeps the window's
// false-alarm rate at the default's. `defthresh` when peek <= defpeek.
// probability is the strategy's change_probability.
scalar_t adjust_change_threshold(
    index_t peek,
    index_t defpeek,
    scalar_t defthresh,
    index_t df,
    scalar_t probability
);

// adjust_change_threshold() remembering its last arguments and value: a
// run's pixels mostly share a peek, so the quantile is computed again only
// when the arguments change.
class ChangeThresholdMemo
{
public:
    scalar_t operator()(index_t peek, index_t defpeek, scalar_t defthresh, index_t df,
                        scalar_t probability);

private:
    bool valid_ = false;
    index_t peek_ = 0, defpeek_ = 0, df_ = 0;
    scalar_t defthresh_ = 0, probability_ = 0, value_ = 0;
};

std::vector<scalar_t> calculate_variogram(
    ArrayView<const scalar_t, 2> spectral
);

// The first-order variogram at the smallest lag whose most common
// separation exceeds min_lag_days, over the pairs further apart than
// min_lag_days; calculate_variogram() when no lag qualifies. min_lag_days is
// the variogram noise floor's min_lag_days.
std::vector<scalar_t> adjusted_variogram(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    std::int64_t min_lag_days
);

bool check_variogram(
    const std::vector<scalar_t>& variogram
);

//------------------------------------------------------------------------------
// Variogram
//
// One pixel's per-band noise floor: adjusted_variogram() of its
// statistics-window observations, one value per spectral row, in row order.
// Computed once per pixel and read, never written, by Tmask, the stability
// check and the change magnitudes. Not tied to a procedure.
//------------------------------------------------------------------------------
struct Variogram
{
    std::vector<scalar_t> bands;

    scalar_t operator[](index_t band) const
    {
        return bands[band];
    }

    // No band is NaN (check_variogram()). A procedure stops without
    // segments on a variogram that is not usable.
    bool usable() const
    {
        return check_variogram(bands);
    }
};

// timeseries window operations
bool enough_time(
    ArrayView<const std::int64_t, 1> dates_window,
    std::int64_t day_delta
);

bool enough_samples(
    ArrayView<const std::int64_t, 1> dates_window,
    index_t meow_size
);

// window models
bool detect_change(
    const std::vector<scalar_t>& magnitudes,
    scalar_t change_threshold
);

inline bool detect_outlier(
    scalar_t magnitude,
    scalar_t outlier_threshold
)
{
    return magnitude > outlier_threshold;
}

ProcessingMask tmask(
    ArrayView<const std::int64_t, 1> dates_window,
    ArrayView<const scalar_t, 2> spect_window,
    const Variogram& variogram,
    const BandList& bands,
    scalar_t t_const
);

// The observations of one window nearest a target in day of year, at many
// targets: the seasonal RMSE's selection (pyccd's find_closest_doy(), with a
// stable sort). The seasonal distance of an observation from the target is
// |round(delta / 365.25) * 365.25 - delta|, delta the days between them.
// build() orders the window's observations by their phase in the year,
// once; closest() then walks out from a target's phase, so a selection
// costs its count (and the ties at its last distance) rather than the
// window's size.
class SeasonalOrder
{
public:
    void build(ArrayView<const std::int64_t, 1> dates, const Window& window);

    // The `count` positions of build()'s window nearest target_day, nearest
    // first, ties in window order (all of them when count exceeds the
    // window's size).
    void closest(std::int64_t target_day, index_t count, std::vector<index_t>& out) const;

private:
    struct Entry
    {
        std::uint16_t phase;   // 4 * date mod 1461, quarter days into the year
        index_t index;         // position in the window
    };
    std::vector<Entry> by_phase_;   // by phase, then index
    mutable std::vector<index_t> group_;
};

// The norm of a fit's in-window `residuals` at `indices`, over `divisor`
// (chi2's seasonal_divisor).
scalar_t seasonal_rmse(
    const std::vector<scalar_t>& residuals,
    const std::vector<index_t>& indices,
    scalar_t divisor
);

} // namespace ccd