#pragma once

#include "ccd/harmonic/filter.hpp"
#include "ccd/procedure/fit.hpp"

namespace ccd
{

//==============================================================================
//
// PermanentSnow
//
// PermanentSnow CCD fitting procedure.
//
// Snow procedure for when there is a significant amount snow represented
// in the quality information

// This method essentially fits a 4 coefficient model across all the
// observations
//
//==============================================================================

class StandardProcedure final : public FitProcedure
{

public:
    //----------------------------------------------------------------------
    // Fits the observations `mask` keeps, as FitProcedure::run() does.
    //----------------------------------------------------------------------
    FitResult run(
        HarmonicWorkspace& hworkspace,
        ProcessingMask mask,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    );

    //----------------------------------------------------------------------
    // run() on the mask filter::standard() builds from hworkspace.
    //----------------------------------------------------------------------
    FitResult run(
        HarmonicWorkspace& hworkspace,
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
        std::vector<scalar_t>& variogram,
        ProcessingMask& mask,
        Window& window,
        std::vector<LassoResult>& models
    );

    bool lookback(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        std::vector<scalar_t>& variogram,
        ProcessingMask& mask,
        Window& window,
        std::vector<LassoResult>& models,
        index_t prev_break
    );

    ChangeModel catch_model(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        ProcessingMask& mask,
        Window& window,
        CurveQA curve_qa
    );

    ChangeModel lookforward(
        const HarmonicWorkspace& workspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver,
        std::vector<scalar_t>& variogram,
        ProcessingMask& mask,
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

    // Results
    std::vector<scalar_t> calc_residuals(
        ArrayView<const scalar_t, 1> y,
        std::vector<scalar_t>& preds
    );

    // calc_residuals() into `residuals`, reusing its storage.
    void calc_residuals(
        ArrayView<const scalar_t, 1> y,
        const std::vector<scalar_t>& preds,
        std::vector<scalar_t>& residuals
    );

private:

    //----------------------------------------------------------------------
    // Per-pixel workspace.
    //
    // masked_ is apply_mask(workspace, mask) for the mask the subfunctions
    // share. run() builds it, and every subfunction that changes the mask
    // updates it in the same step, so none of them re-masks the full time
    // series on entry. candidate_ holds initialize()'s tentative outlier
    // removal until it is accepted. The buffers keep their capacity, so a
    // procedure object reused across pixels stops allocating them.
    //----------------------------------------------------------------------
    MaskedData masked_;
    MaskedData candidate_;
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