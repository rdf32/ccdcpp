#pragma once

#include "ccd/core/types.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/input/procedure.hpp"

namespace ccd
{

//==============================================================================
//
// The per-pixel procedure
//
// Before any change detection, each pixel takes one of three procedures from
// its QA statistics. Standard runs the run's change detector (the Standard
// machine); PermanentSnow and InsufficientClear fit one curve over the whole
// series instead. Whichever detector a run uses, the mode stage chooses
// (ModeRule, which the run asks) and the edge fit fits the whole series
// (EdgeFit::whole_series(), which standard::Detector::fit() asks).
//
//==============================================================================

// A pixel's procedure with what the mode asks of its change detection: mask
// steps after the procedure's chain (preprocess::apply_steps(), the run's
// build_mask step), and a cap on the coefficients its segments report (0:
// none; the detection itself is not capped).
struct PixelMode
{
    PixelMode(PixelProcedure procedure = PixelProcedure::Standard) : procedure(procedure) {}

    PixelProcedure procedure;
    preprocess::Chain extra_mask;
    index_t max_coefficients = 0;
};

} // namespace ccd
