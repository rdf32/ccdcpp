#pragma once

#include <cstdint>
#include <vector>

#include "ccd/procedure/pixel_procedure.hpp"
#include "ccd/procedure/standard.hpp"

namespace ccd
{

//==============================================================================
//
// The resume record
//
// What a later resume needs to continue a run on a longer series: a header
// per call and a record per pixel, plain values the caller stores beside the
// segments. A run on the observations up to T and a run on the whole series
// make the same decisions up to the first size guard that failed in the
// shorter one (PrefixStability, tests/ccd/golden/test_prefix.cpp), so the
// record holds the state at that guard and the inputs it rests on.
//
// The statistics window (the QA statistics, the cadence, the variogram and
// the green median) is the one input that is not causal: a resumed run caps
// every statistics cutoff (STAT_ORD and each GreenMedian step's date) at
// statistics_end, the end of the first run's series, so that it computes
// them from the observations the first run saw.
//
//==============================================================================

// The layout of the record. Bumped when a field is added or its meaning
// changes.
inline constexpr std::uint16_t RESUME_FORMAT = 1;

// The outputs. Bumped whenever an output can change for the same input and
// options, so a record of another version describes a run this one would
// not make. 1 is ccdcpp 0.8.0's.
inline constexpr std::uint32_t RESULTS_VERSION = 1;

// One call's part of the record, shared by its pixels.
struct ResumeHeader
{
    std::uint16_t format = RESUME_FORMAT;
    std::uint32_t results_version = RESULTS_VERSION;

    // options_digest() of the options, the plan and the QA table of the run.
    std::uint64_t options_digest = 0;

    // The last date of the series the run saw (the end of its input, fill
    // included).
    std::int64_t series_end = 0;

    // The date every statistics cutoff is capped at: series_end for a first
    // run, the first run's for a run resumed from it.
    std::int64_t statistics_end = 0;
};

// One pixel's part of the record.
struct PixelRecord
{
    // input_fingerprint() of the pixel's observations dated up to the
    // header's series_end, fill left out.
    std::uint64_t input_fingerprint = 0;

    // The procedure the pixel took.
    PixelProcedure procedure = PixelProcedure::Standard;

    // Where the Standard machine stopped (exit None for the other
    // procedures, which a resume refits whole).
    standard::Suspended standard;

    // The days of the observations the machine dropped from the processing
    // mask (Tmask, Lookback and Extend outliers), in date order.
    std::vector<std::int64_t> excluded_days;
};

} // namespace ccd
