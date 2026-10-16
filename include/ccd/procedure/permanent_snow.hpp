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

class PermanentSnow final : public FitProcedure
{
public:

    using FitProcedure::run;

    //----------------------------------------------------------------------
    // run() on the mask filter::snow() builds from hworkspace.
    //----------------------------------------------------------------------

    FitResult run(
        const HarmonicWorkspace& hworkspace,
        LassoWorkspace& lworkspace,
        LassoSolver& solver
    )
    {
        return run(
            hworkspace,
            filter::snow(hworkspace),
            lworkspace,
            solver
        );
    }

protected:

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
        return CurveQA::PersistentSnow;
    }

};

} // namespace ccd