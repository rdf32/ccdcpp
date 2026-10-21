#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd/core/array_view.hpp"
#include "ccd/core/types.hpp"
#include "ccd/machine/pixel_procedure.hpp"
#include "ccd/machine/suspended.hpp"
#include "ccd/results/segment.hpp"

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
// every statistics cutoff (the shared statistics_end and each GreenMedian step's date) at
// statistics_end, the end of the first run's series, so that it computes
// them from the observations the first run saw.
//
//==============================================================================

// The layout of the record. Bumped when a field is added or its meaning
// changes. 2: the header's digest covers the input (its QA table included)
// and the resolved configuration (its preprocessing included); 0.10.0's
// header held options_digest in its place.
inline constexpr std::uint16_t RESUME_FORMAT = 2;

// The outputs. Bumped whenever a release can change an output for the same
// input and configuration, or changes the default configuration's outputs,
// so a record of another version describes a run this one would not make.
// 3: the "ccdc" preset has no statistics cutoff, and its snow and
// insufficient-clear procedures test the thermal range in degrees Celsius
// (the snow procedure over its snow observations too).
inline constexpr std::uint32_t RESULTS_VERSION = 3;

// One call's part of the record, shared by its pixels.
struct ResumeHeader
{
    std::uint16_t format = RESUME_FORMAT;
    std::uint32_t results_version = RESULTS_VERSION;

    // The run's input and resolved configuration (config::Resolved::digest:
    // the QA table and the preprocessing included).
    std::uint64_t digest = 0;

    // The last date of the series the run saw (the end of its input, fill
    // included).
    std::int64_t series_end = 0;

    // The date every statistics cutoff is capped at: series_end for a first
    // run, the first run's for a run resumed from it.
    std::int64_t statistics_end = 0;
};

// The refusal of a header of another record format or results version
// (std::invalid_argument): a record of another format is not read, and one
// of another results version describes a run this ccdcpp would not make.
// A resume checks it before anything else of the record.
inline void check_version(const ResumeHeader& header)
{
    if (header.format != RESUME_FORMAT)
        throw std::invalid_argument(
            "the resume record's format is " + std::to_string(header.format) +
            "; this ccdcpp reads " + std::to_string(RESUME_FORMAT));
    if (header.results_version != RESULTS_VERSION)
        throw std::invalid_argument(
            "the resume record's results version is " +
            std::to_string(header.results_version) + "; this ccdcpp's is " +
            std::to_string(RESULTS_VERSION) + ": rerun in full");
}

// A fingerprint of one pixel's input up to `through`: of each observation
// dated on or before it whose QA code is not `qa_fill`, its date, its QA
// code and every band's value (the bits of the value as scalar_t), in date
// order. Fill observations are left out, so a scene that does not cover the
// pixel changes nothing.
std::uint64_t input_fingerprint(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qa_codes,
    std::uint8_t qa_fill,
    std::int64_t through
);

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

// What a resume did with a pixel.
enum class ResumeStatus : std::uint8_t
{
    Resumed,            // continued from the record, or rerun with the same
                        // result: the pixels that take no machine or stopped
                        // early, and a pixel whose stored drop is on a date
                        // the plan keeps twice (it does not say which)
    RerunInputChanged   // the pixel's observations up to the stored series
                        // end differ from the stored run's: rerun in full
};

// A stored call to resume from (detect's and detect_cube's `resume`): its
// header, and each pixel's record and stored segments (of which the first
// record.standard.final_segments are kept as given), in raster order over
// the cube: H x W entries each, one each for detect. A pixel without a
// stored record (one that failed) has a default PixelRecord, whose
// fingerprint does not match: it is rerun.
struct Resume
{
    ResumeHeader header;
    std::vector<PixelRecord> records;
    std::vector<std::vector<ChangeModel>> segments;
};

} // namespace ccd
