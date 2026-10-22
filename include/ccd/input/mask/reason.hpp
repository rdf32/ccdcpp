#pragma once

#include <cstdint>

namespace ccd
{

//==============================================================================
//
// Reason
//
// Why an observation is where a run left it, one byte per observation of a
// pixel (RunSpec::reasons). The first three are kept by the processing mask,
// the others are not: the mask is reasons < FIRST_DROP. A preprocessing
// step's code is the last step that removed the observation (an add_qa can
// take one back); the machine's, the stage that dropped it. The
// preprocessing and the machine leave every observation they keep
// InSegment; the run then gives those no segment holds NoFit or NoSegment.
//
//==============================================================================

enum class Reason : std::uint8_t
{
    // Kept by the processing mask:
    InSegment,        // in a segment
    NoFit,            // after the last segment of a pixel whose model could
                      // not be fitted on its window (machine::Exit::NoFit)
    NoSegment,        // in no segment otherwise: an edge segment not due, the
                      // gap a Lookback stopped by a change leaves, a pixel
                      // with too few observations or a NaN variogram

    // Removed by the procedure's preprocessing (its chain, then the mode's
    // extra steps):
    Qa,               // keep_qa: its QA code is not kept
    Thermal,          // thermal_range
    BandRange,        // band_range
    Green,            // green_median
    DuplicateDate,    // one_per_date: a later observation of its date
    CustomStep,       // a step a registry adds (preprocess::StepKind::Custom)

    // Dropped by the machine:
    WindowScreen,     // Initialize's window screen (Tmask)
    LookbackOutlier,  // Lookback's outlier, before the stable window
    Outlier,          // Extend's outlier rule, the first observation of a peek
    BeforeChange,     // the peek's observations before the change test's
                      // break (ChangeTest::break_at() past the peek's first)
    Stored            // a drop of the run this one resumed: the record keeps
                      // its day, not why
};

// The first code the processing mask does not keep.
inline constexpr Reason FIRST_DROP = Reason::Qa;

// The number of codes: every Reason is below it.
inline constexpr std::uint8_t REASON_COUNT = static_cast<std::uint8_t>(Reason::Stored) + 1;

// "in_segment", "no_fit", "no_segment", "qa", ..., "stored": the code's name, as
// the Python package's REASONS lists them.
const char* name(Reason reason);

} // namespace ccd
