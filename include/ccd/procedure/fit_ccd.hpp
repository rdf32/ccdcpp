#pragma once

#include <array>
#include <vector>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/pmask.hpp"

#include "ccd/harmonic/harmonic.hpp"
#include "ccd/harmonic/filter.hpp"

#include "ccd/regression/ccd_lasso.hpp"


namespace ccd
{

enum class CCDCurveQA : std::uint8_t {
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

struct CCDChangeModel {

    std::int64_t start_day;
    std::int64_t end_day;
    std::int64_t break_day;

    index_t observation_count;

    scalar_t change_probability;

    CCDCurveQA curve_qa;

    std::array<CCDLassoResult, 7> bands;
};

struct CCDFitResult
{
    std::vector<CCDChangeModel> models;
    ProcessingMask mask;
};

class CCDFitProcedure
{
public:

    virtual ~CCDFitProcedure() = default;

    //----------------------------------------------------------------------
    // Execute complete fitting procedure
    //----------------------------------------------------------------------

    CCDFitResult CCDFitProcedure::run(
        HarmonicWorkspace& hworkspace,
        CCDLassoWorkspace& lworkspace,
        CCDLassoSolver& solver
    );

protected:

    //----------------------------------------------------------------------
    // Observation selection
    //----------------------------------------------------------------------

    virtual ProcessingMask select_observations(
        const HarmonicWorkspace& workspace
    ) const = 0;

    //----------------------------------------------------------------------
    // Number of coefficients
    //----------------------------------------------------------------------

    virtual index_t coefficient_count(
        ArrayView<const std::int64_t, 1> dates,
        const HarmonicOptions& options
    ) const = 0;

    //----------------------------------------------------------------------
    // Curve QA assigned to result
    //----------------------------------------------------------------------

    virtual CCDCurveQA curve_qa() const = 0;

};

class TestCCD final : public CCDFitProcedure
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

    CCDCurveQA curve_qa() const override
    {
        return CCDCurveQA::InsufficientClear;
    }

};

} // namespace ccd