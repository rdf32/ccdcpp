#pragma once

#include <array>
#include <vector>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/pmask.hpp"

#include "ccd/harmonic/harmonic.hpp"

#include "ccd/regression/lasso_solver.hpp"
#include "ccd/regression/robust_solver.hpp"


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

struct Window
{
    index_t start = 0;
    index_t stop  = 0;

    Window() = default;

    Window(index_t start_, index_t stop_)
        : start(start_), stop(stop_){}

    index_t size() const
    {
        return stop - start;
    }

    void grow()
    {
        ++stop;
    }

    void shift()
    {
        ++start;
        ++stop;
    }

    void rewind(index_t amount)
    {
        start -= amount;
    }

    void extend(index_t amount)
    {
        stop += amount;
    }
};

// full timeseries
scalar_t adjust_change_threshold(
    index_t peek,
    index_t defpeek,
    scalar_t defthresh
);

std::vector<scalar_t> calculate_variogram(
    ArrayView<const scalar_t, 2> spectral
);

std::vector<scalar_t> adjusted_variogram(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral
);

bool check_variogram(
    const std::vector<scalar_t>& variogram
);

// timeseries window operations
bool enough_time(
    ArrayView<const std::int64_t, 1> dates_window,
    std::int64_t day_delta
);

bool enough_samples(
    ArrayView<const std::int64_t, 1> dates_window,
    index_t meow_size
);

bool stable(
    ArrayView<const std::int64_t, 1> dates_window,
    std::vector<scalar_t>& variogram,
    std::vector<LassoResult>& results,
    scalar_t change_threshold,
    std::vector<index_t>& detection_bands
);

// window models
bool detect_change(
    const std::vector<scalar_t>& magnitudes,
    scalar_t change_threshold
);

inline bool detect_outlier(
    scalar_t magnitude,
    scalar_t outlier_threshold
)
{
    return magnitude > outlier_threshold;
}

std::vector<scalar_t> change_magnitude(
    const std::vector<std::vector<scalar_t>>& residuals,
    const std::vector<scalar_t>& variogram,
    const std::vector<scalar_t>& rmse
);

ProcessingMask tmask(
    ArrayView<const std::int64_t, 1> dates_window,
    ArrayView<const scalar_t, 2> spect_window,
    std::vector<scalar_t>& variogram,
    const std::vector<index_t>& bands,
    scalar_t t_const
);

} // namespace ccd