#pragma once

#include "ccd/harmonic/filter.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/series.hpp"

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

} // namespace standard

//==============================================================================
//
// StandardProcedure
//
// pyccd's standard procedure: the start-window search (initialize), the
// backward extension (lookback), the start and end catches (catch_model)
// and the forward extension with its change test (lookforward), over the
// pixel's Series.
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

    // subfunctions
    bool initialize(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const Variogram& variogram,
        scalar_t change_threshold,
        Window& window,
        std::vector<LassoResult>& models
    );

    bool lookback(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        const standard::Cadence& cadence,
        const Variogram& variogram,
        Window& window,
        const std::vector<LassoResult>& models,
        index_t prev_break
    );

    ChangeModel catch_model(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const Window& window,
        CurveQA curve_qa
    );

    ChangeModel lookforward(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        const standard::Cadence& cadence,
        const Variogram& variogram,
        Window& window
    );
    //----------------------------------------------------------------------
    // Harmonic complexity
    //----------------------------------------------------------------------
    index_t coefficient_count(
        ArrayView<const std::int64_t, 1> dates,
        const HarmonicOptions& options
    ) const override
    {   
        const index_t span =
            dates.size() / options.NUM_OBS_FACTOR;

        if (span < options.COEFFICIENT_MID)
        {
            return options.COEFFICIENT_MIN;
        }
        else if (span < options.COEFFICIENT_MAX)
        {
            return options.COEFFICIENT_MID;
        }
        else
        {
            return options.COEFFICIENT_MAX;
        }
    }

    //----------------------------------------------------------------------
    // Result QA
    //----------------------------------------------------------------------

    CurveQA curve_qa() const override
    {
        return CurveQA::Start;
    }

    // |y - preds| into `residuals`, reusing its storage.
    void calc_residuals(
        ArrayView<const scalar_t, 1> y,
        const std::vector<scalar_t>& preds,
        std::vector<scalar_t>& residuals
    );

private:

    //----------------------------------------------------------------------
    // The pixel's processing mask and kept observations, which the
    // subfunctions share and change only through it. A member so that its
    // buffers keep their capacity across the pixels one procedure object
    // runs.
    //----------------------------------------------------------------------
    Series series_;
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