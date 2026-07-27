#pragma once

#include "ccd/harmonic/filter.hpp"
#include "ccd/procedure/fit.hpp"

namespace ccd
{

//==============================================================================
//
// InsufficientClear
//
// InsufficientClear CCD fitting procedure.
//
// insufficient clear procedure for when there is an insufficient quality
// observations

// This method essentially fits a 4 coefficient model across all the
// observations
//
//==============================================================================

class InsufficientClear final : public FitProcedure
{
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
        return filter::insufficientclear(workspace);
    }

    //----------------------------------------------------------------------
    // Harmonic complexity
    //----------------------------------------------------------------------
    index_t coefficient_count(
        ArrayView<const std::int64_t, 1> dates,
        const HarmonicOptions& options
    ) const override
    {   
        return options.COEFFICIENT_MIN;
    }

    //----------------------------------------------------------------------
    // Result QA
    //----------------------------------------------------------------------

    CurveQA curve_qa() const override
    {
        return CurveQA::InsufficientClear;
    }

};

} // namespace ccd