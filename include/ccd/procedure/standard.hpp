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

} // namespace ccd