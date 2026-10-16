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

// A pixel's procedure, chosen per pixel by select_procedure().
enum class PixelProcedure
{
    Standard,
    InsufficientClear,
    PermanentSnow
};

// pyccd's choice, from the statistics of the observations dated up to
// STAT_ORD: Standard when the clear percentage reaches CLEAR_PCT_THRESHOLD;
// otherwise PermanentSnow when the snow percentage reaches
// SNOW_PCT_THRESHOLD; otherwise InsufficientClear.
PixelProcedure select_procedure(
    const Quality::Statistics& statistics,
    const HarmonicOptions& options
);

// The whole-series fit of a PermanentSnow or InsufficientClear pixel: one
// segment over every observation `mask` keeps, every band fitted with
// COEFFICIENT_MIN coefficients:
//
//     start_day, end_day  the first and last date of the whole series (not
//                         only the kept observations), as pyccd
//     break_day           end_day
//     observation_count   mask.count()
//     change_probability  0
//     curve_qa            PersistentSnow (54) or InsufficientClear (44)
//     bands[b].score      rmse, magnitude 0, no residuals
//
// The result carries `mask`. With fewer than MEOW_SIZE kept observations:
// no segments and an empty mask (quirk 5, kept until 0.8.0).
//
// `procedure` is PermanentSnow or InsufficientClear; `mask` is that
// procedure's processing mask over hworkspace's series, built by the
// caller (ccd::detect builds it with the procedure's filter chain).
FitResult fit_whole_series(
    PixelProcedure procedure,
    const HarmonicWorkspace& hworkspace,
    ProcessingMask mask,
    LassoWorkspace& lworkspace,
    LassoSolver& solver
);

} // namespace ccd
