
#pragma once

// ccdcpp's C++ API: the checks every call makes, the resume record's digest
// and fingerprint, and (run/run.hpp) the entry points run_pixel and run_cube.

#include <cstdint>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"

#include "ccd/input/harmonic.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/input/qa_table.hpp"
#include "ccd/results/record.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/run/run.hpp"

namespace ccd
{

//------------------------------------------------------------------------------
//
// The options check every entry point runs once per call, before any work.
//
// The procedures hold one fit per band and read bands by the indices in
// `options`, so every index must name a row of an array with `bands` bands.
// Throws std::invalid_argument when:
//
//   * `bands` is not options.NUM_BANDS, or exceeds CCD_MAX_BANDS;
//   * DETECTION_BANDS or TMASK_BANDS is empty;
//   * DETECTION_BANDS, TMASK_BANDS or SATURATION_BANDS holds an index not
//     below `bands`, or holds one index twice;
//   * GREEN_IDX is NO_BAND or not below `bands`;
//   * THERMAL_IDX is neither NO_BAND nor below `bands`, or is in
//     SATURATION_BANDS.
//
// and the cadence and change-probability options when:
//
//   * PEEK_REFERENCE_DAYS or VARIOGRAM_MIN_LAG_DAYS is not greater than 0;
//   * RETRAIN_MIN_OBS or SEASONAL_RMSE_COUNT is below 1;
//   * RETRAIN_SPAN_FACTOR or SEASONAL_RMSE_DIVISOR is not greater than 0;
//   * CHANGE_PROBABILITY is not in (0, 1);
//
// and the coefficient counts when:
//
//   * COEFFICIENT_MIN is below 1;
//   * COEFFICIENT_MIN, COEFFICIENT_MID, COEFFICIENT_MAX decrease;
//   * COEFFICIENT_MAX is above 8, the harmonic basis's width (the intercept
//     and seven columns).
//
// The message names the field and the counts or the value.
//
//------------------------------------------------------------------------------

void validate(const HarmonicOptions& options, index_t bands);

//------------------------------------------------------------------------------
//
// The date check every entry point runs once per call, before any work.
//
// Throws std::invalid_argument unless `dates` is non-decreasing. Equal dates
// are allowed: the filters keep the first observation of each date. The
// message names the first position out of order and both dates.
//
//------------------------------------------------------------------------------

void validate_dates(ArrayView<const std::int64_t, 1> dates);

//------------------------------------------------------------------------------
//
// The resume record (record.hpp)
//
//------------------------------------------------------------------------------

// A fingerprint of everything besides the input that decides a run's
// outputs: every field of the options, the configuration, every field of the
// plan, and the QA table's codes (none when qa_table is null). Equal ones
// give equal digests on every platform; unequal ones, different digests but
// for a 64-bit collision. A configuration 0.10's options describe
// (config::recover()) is digested as those options were, so the digest is
// 0.10's and its resume records stay valid. STAT_ORD is in it: a resumed run
// is given the first run's options, and caps the cutoffs itself
// (ResumeHeader::statistics_end).
std::uint64_t options_digest(
    const HarmonicOptions& hoptions,
    const config::ConfigTree& config,
    const preprocess::PreprocessPlan& plan,
    const QaTable* qa_table
);

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

} // namespace ccd
