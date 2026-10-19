#pragma once

#include "ccd/types.hpp"
#include "ccd/pmask.hpp"

#include "ccd/harmonic/harmonic.hpp"
#include "ccd/harmonic/quality.hpp"

#include "ccd/procedure/fit.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace ccd
{

//==============================================================================
//
// The per-pixel procedure
//
// Before any change detection, each pixel takes one of three procedures from
// its QA statistics. Standard runs the run's change detector (today the
// Standard machine, standard.hpp); PermanentSnow and InsufficientClear fit
// one curve over the whole series instead. The choice and the whole-series
// fit are the same whichever detector a run uses, so they are plain
// functions here and not any detector's: a second detector calls them as the
// Standard one does (standard::Detector::fit()).
//
//==============================================================================

// A pixel's procedure, chosen per pixel by the run's mode (ModeRule,
// strategies/stages.hpp).
enum class PixelProcedure
{
    Standard,
    InsufficientClear,
    PermanentSnow
};

// The whole-series fit of a PermanentSnow or InsufficientClear pixel: one
// segment over every observation `mask` keeps, every band fitted with
// num_coefficients coefficients:
//
//     start_day, end_day  the first and last date of the whole series (not
//                         only the kept observations), as pyccd
//     break_day           end_day
//     observation_count   mask.count()
//     change_probability  0
//     curve_qa            PersistentSnow (54) or InsufficientClear (44)
//     bands[b].score      rmse, magnitude 0, no residuals
//
// The result carries `mask`, also when there are fewer than min_observations
// kept observations and so no segment.
//
// `procedure` is PermanentSnow or InsufficientClear; `mask` is that
// procedure's processing mask over hworkspace's series, built by the
// caller (Run::pixel builds it with the procedure's filter chain).
FitResult fit_whole_series(
    PixelProcedure procedure,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    index_t min_observations,
    index_t num_coefficients,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
);

} // namespace ccd
