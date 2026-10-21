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

ProcessingMask tmask(
    ArrayView<const std::int64_t, 1> dates_window,
    ArrayView<const scalar_t, 2> spect_window,
    const Variogram& variogram,
    const BandList& bands,
    scalar_t t_const
);

} // namespace ccd