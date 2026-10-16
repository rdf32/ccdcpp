#pragma once

#include "ccd/harmonic/filter.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/kernels.hpp"
#include "ccd/procedure/series.hpp"
#include "ccd/regression/harmonic_basis.hpp"

namespace ccd
{

namespace standard
{

//==============================================================================
//
// Cadence
//
// The Standard procedure's per-pixel cadence: the peek window, scaled to the
// pixel's revisit interval, and the change threshold that goes with it.
// Derived once per pixel from the statistics-window dates (measure_cadence)
// and read by every later step in place of the caller's PEEK_SIZE and
// CHANGE_THRESHOLD, which are never written. Standard's own record: a
// procedure that does not count observations has no use for it.
//
//==============================================================================

struct Cadence
{
    // adjust_peek(): PEEK_SIZE scaled by PEEK_REFERENCE_DAYS over the median
    // gap, never below PEEK_SIZE. Observations.
    index_t peek;

    // adjust_change_threshold() for `peek`: CHANGE_THRESHOLD when peek is
    // PEEK_SIZE, lower for a longer peek.
    scalar_t change_threshold;
};

// The cadence of a pixel whose statistics window (the observations dated <=
// STAT_ORD) has dates `statistics_dates`.
Cadence measure_cadence(
    ArrayView<const std::int64_t, 1> statistics_dates,
    const HarmonicOptions& options
);

//------------------------------------------------------------------------------
// The Standard procedure's kernels
//
// Plain functions of the procedure's steps, over the pixel's Series. Their
// arithmetic, and the order of it, is pyccd's.
//------------------------------------------------------------------------------

// Extend's coefficient count for a window of `observations` observations:
// COEFFICIENT_MIN below COEFFICIENT_MID * NUM_OBS_FACTOR observations,
// COEFFICIENT_MID below COEFFICIENT_MAX * NUM_OBS_FACTOR, else
// COEFFICIENT_MAX.
index_t coefficient_count(
    index_t observations,
    const HarmonicOptions& options
);

// |y - prediction| of models[b] at each peek observation, for every b in
// `bands`, into residuals[b] (resized to the peek's length; the other
// entries are not touched). `peek_basis` is built over the peek's dates with
// at least the models' coefficient count, and `peek_spectral` (B, peek
// length) holds the peek's observations in the same order.
void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    const BandList& bands,
    std::vector<std::vector<scalar_t>>& residuals
);

// peek_residuals() of every row of `peek_spectral`; residuals has B entries.
void peek_residuals(
    const HarmonicBasis& peek_basis,
    ArrayView<const scalar_t, 2> peek_spectral,
    const std::vector<LassoResult>& models,
    std::vector<std::vector<scalar_t>>& residuals
);

// The change magnitude of each peek observation j, into `magnitude`:
//
//     sum over i of (residuals[bands[i]][j] / max(variogram[bands[i]], rmse[i]))^2
//
// in the order of `bands`, a band whose max(...) is not above 0 adding
// nothing. rmse[i] is the comparison RMSE of bands[i]; every residuals[b]
// named has the peek's length. Empty for an empty `bands`.
void change_magnitude(
    const std::vector<std::vector<scalar_t>>& residuals,
    const BandList& bands,
    const Variogram& variogram,
    const std::vector<scalar_t>& rmse,
    std::vector<scalar_t>& magnitude
);

// Lookback: extends `window` backward towards prev_break one observation at
// a time, testing the observations before it against `models`, the
// initialisation fits (their RMSE is the comparison RMSE). Stops at
// prev_break or on a change (returns true); drops an outlier from `series`
// and tests again. `peek_basis` is scratch.
bool lookback(
    Series& series,
    Window& window,
    index_t prev_break,
    const std::vector<LassoResult>& models,
    const Cadence& cadence,
    const Variogram& variogram,
    const HarmonicOptions& options,
    HarmonicBasis& peek_basis
);

// Extend: grows `window` one observation at a time while a full peek fits
// in the series, refitting every band on the refit schedule and testing the
// peek after the window; drops an outlier at the first peek observation and
// tests again. Returns the segment, change probability 1 on a change and 0
// at the series end. `peek_basis` is scratch.
ChangeModel lookforward(
    Series& series,
    Window& window,
    const Cadence& cadence,
    const Variogram& variogram,
    const HarmonicOptions& options,
    HarmonicBasis& peek_basis,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
);

} // namespace standard

//==============================================================================
//
// StandardProcedure
//
// pyccd's standard procedure: the start-window search (ccd::initialize),
// the backward extension (standard::lookback), the start and end catches
// (ccd::catch_segment) and the forward extension with its change test
// (standard::lookforward), over the pixel's Series.
//
//==============================================================================

class StandardProcedure final : public FitProcedure
{

public:
    //----------------------------------------------------------------------
    // Fits the observations `mask` keeps, as FitProcedure::run() does.
    //----------------------------------------------------------------------
    FitResult run(
        const HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    );

    //----------------------------------------------------------------------
    // run() on the mask filter::standard() builds from hworkspace.
    //----------------------------------------------------------------------
    FitResult run(
        const HarmonicWorkspace& hworkspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    )
    {
        return run(
            hworkspace,
            filter::standard(hworkspace),
            lworkspace,
            solver
        );
    }

protected:

    //----------------------------------------------------------------------
    // Harmonic complexity: standard::coefficient_count()
    //----------------------------------------------------------------------
    index_t coefficient_count(
        ArrayView<const std::int64_t, 1> dates,
        const HarmonicOptions& options
    ) const override
    {
        return standard::coefficient_count(dates.size(), options);
    }

    //----------------------------------------------------------------------
    // Result QA
    //----------------------------------------------------------------------

    CurveQA curve_qa() const override
    {
        return CurveQA::Start;
    }

private:

    //----------------------------------------------------------------------
    // The pixel's processing mask and kept observations, which the steps
    // share and change only through it, and the peek bases' storage.
    // Members so that their buffers keep their capacity across the pixels
    // one procedure object runs.
    //----------------------------------------------------------------------
    Series series_;
    HarmonicBasis peek_basis_;
};

inline scalar_t span(
    ArrayView<const std::int64_t,1> dates,
    const Window& window
)
{
    assert(window.stop > window.start);

    return static_cast<scalar_t>(
        dates(window.stop - 1) -
        dates(window.start)
    );
}

} // namespace ccd