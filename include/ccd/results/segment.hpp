#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ccd/core/constants.hpp"
#include "ccd/core/types.hpp"
#include "ccd/input/processing_mask.hpp"

namespace ccd
{

//==============================================================================
//
// Segments
//
// What a run reports of one pixel: its segments (ChangeModel), each with one
// fit per band (BandFit) and its curve QA, and the processing mask they were
// fitted under (FitResult).
//
//==============================================================================

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

//------------------------------------------------------------------------------
// BandFit
//
// One band's fit over a segment, as the segment reports it, whatever fitted
// it:
//
//     intercept, coefficients  the harmonic model, coefficients in
//                              HarmonicBasis column order (t, cos1, sin1,
//                              ...), those the fit did not use 0
//     rmse                     its RMSE over the segment's observations
//     magnitude                the band's change magnitude: the median
//                              absolute peek residual of a segment Extend
//                              ended, 0 for the others
//     iterations               the solver's iteration count, 0 for a fit in
//                              closed form
//------------------------------------------------------------------------------
struct BandFit
{
    scalar_t intercept = 0.0;
    std::array<scalar_t, CCD_MAX_COEFS> coefficients{};
    scalar_t rmse = 0.0;
    scalar_t magnitude = 0.0;
    index_t iterations = 0;
};

struct ChangeModel {

    std::int64_t start_day;
    std::int64_t end_day;
    std::int64_t break_day;

    index_t observation_count;

    scalar_t change_probability;

    CurveQA curve_qa;

    // One fit per row of the input's spectral array, in row order.
    std::vector<BandFit> bands;
};

struct FitResult
{
    std::vector<ChangeModel> models;
    ProcessingMask mask;
};

} // namespace ccd
