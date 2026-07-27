#pragma once

#include <array>
#include <vector>

#include "ccd/types.hpp"
#include "ccd/pmask.hpp"

#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"

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

    std::array<LassoResult, 7> bands;
};

struct FitResult
{
    std::vector<ChangeModel> models;
    ProcessingMask mask;
};

class FitProcedure
{
public:

    virtual ~FitProcedure() = default;

    //----------------------------------------------------------------------
    // Execute complete fitting procedure
    //----------------------------------------------------------------------

    FitResult run(
        HarmonicWorkspace& workspace,
        LassoSolver& solver
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

    virtual CurveQA curve_qa() const = 0;

};

} // namespace ccd