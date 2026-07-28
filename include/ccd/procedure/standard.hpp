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
    FitResult run(
        HarmonicWorkspace& workspace,
        LassoSolver& solver
    );

protected:

    //----------------------------------------------------------------------
    // Select observations
    //
    // Uses the standard QA filtering.
    //----------------------------------------------------------------------

    ProcessingMask select_observations(
        const HarmonicWorkspace& workspace
    ) const override
    {   
        return filter::standard(workspace);
    }

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

};

std::vector<scalar_t> calc_residuals(
    ArrayView<const scalar_t, 1> y,
    std::vector<scalar_t>& preds
);

bool initialize(
    const HarmonicWorkspace& workspace,
    LassoSolver& solver,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window,
    std::vector<LassoResult>& models
);

bool lookback(
    const HarmonicWorkspace& workspace,
    std::vector<scalar_t>& variogram,
    ProcessingMask& mask,
    Window& window,
    std::vector<LassoResult>& models,
    index_t prev_break
);

ChangeModel catch_model(
    const HarmonicWorkspace& workspace,
    LassoSolver& solver,
    ProcessingMask& mask,
    Window& window,
    CurveQA curve_qa
);

// bool lookforward

// bool catch

} // namespace ccd