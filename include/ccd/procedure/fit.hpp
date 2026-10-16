#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/pmask.hpp"

#include "ccd/harmonic/harmonic.hpp"

#include "ccd/regression/lasso_solver.hpp"
#include "ccd/regression/robust_solver.hpp"


namespace ccd
{

enum class CurveQA : std::uint8_t {
    MinCoefficients = 4,
    MidCoefficients = 6,
    MaxCoefficients = 8,

    Start             = 14,
    End               = 24,
    InsufficientClear = 44,
    PersistentSnow    = 54,

    // future CCD states
    Stable            = 64,
    Recovery          = 74,
    Disturbance       = 84
};

struct ChangeModel {

    std::int64_t start_day;
    std::int64_t end_day;
    std::int64_t break_day;

    index_t observation_count;

    scalar_t change_probability;

    CurveQA curve_qa;

    // One fit per row of the input's spectral array, in row order.
    std::vector<LassoResult> bands;
};

struct FitResult
{
    std::vector<ChangeModel> models;
    ProcessingMask mask;
};

struct Window
{
    index_t start = 0;
    index_t stop  = 0;

    Window() = default;

    Window(index_t start_, index_t stop_)
        : start(start_), stop(stop_){}

    index_t size() const
    {
        return stop - start;
    }

    void grow()
    {
        ++stop;
    }

    void shift()
    {
        ++start;
        ++stop;
    }
};

// The days `window` spans in `dates`: its last date minus its first.
inline scalar_t span(
    ArrayView<const std::int64_t, 1> dates,
    const Window& window
)
{
    assert(window.stop > window.start);

    return static_cast<scalar_t>(
        dates(window.stop - 1) -
        dates(window.start)
    );
}

// full timeseries

// The peek size for `dates`: defpeek * reference_days over the median gap
// between consecutive dates, rounded, and never below `defpeek`. `defpeek`
// for fewer than two dates. reference_days is
// HarmonicOptions::PEEK_REFERENCE_DAYS.
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
// probability is HarmonicOptions::CHANGE_PROBABILITY.
scalar_t adjust_change_threshold(
    index_t peek,
    index_t defpeek,
    scalar_t defthresh,
    index_t df,
    scalar_t probability
);

// The chi-squared quantile at probability p with df degrees of freedom: the
// distribution adjust_change_threshold() reads its threshold from.
scalar_t chi2_quantile(scalar_t p, index_t df);

std::vector<scalar_t> calculate_variogram(
    ArrayView<const scalar_t, 2> spectral
);

// The first-order variogram at the smallest lag whose most common
// separation exceeds min_lag_days, over the pairs further apart than
// min_lag_days; calculate_variogram() when no lag qualifies. min_lag_days is
// HarmonicOptions::VARIOGRAM_MIN_LAG_DAYS.
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

bool stable(
    ArrayView<const std::int64_t, 1> dates_window,
    const Variogram& variogram,
    const std::vector<LassoResult>& results,
    scalar_t change_threshold,
    const BandList& detection_bands
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

std::vector<index_t> find_closest_doy(
    ArrayView<const std::int64_t, 1> dates,
    index_t date_index,
    const Window& window,
    index_t count
);

// The norm of model's residuals at `indices`, over `divisor`
// (HarmonicOptions::SEASONAL_RMSE_DIVISOR).
scalar_t seasonal_rmse(
    const LassoResult& model,
    const std::vector<index_t>& indices,
    scalar_t divisor
);

} // namespace ccd