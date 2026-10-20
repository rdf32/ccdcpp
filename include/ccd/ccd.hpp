
#pragma once

// ccdcpp's C++ API: the checks every call makes, the resume record's
// fingerprint, and (run/run.hpp) the entry points run_pixel and run_cube.

#include <cstdint>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"

#include "ccd/config/registry.hpp"
#include "ccd/input/input.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/input/qa_tables.hpp"
#include "ccd/results/record.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/run/run.hpp"

namespace ccd
{

//------------------------------------------------------------------------------
//
// The input check every entry point runs once per call, before any work.
//
// The procedures hold one fit per band and read bands by their rows in
// `input`, so every row must be one of an array with `bands` bands. Throws
// std::invalid_argument when:
//
//   * `bands` is not input.bands() (one role per row), or exceeds
//     CCD_MAX_BANDS;
//   * a role is not one of ROLES, or a role other than "other" is held
//     twice;
//   * the input has no "green" row;
//   * saturation_bands holds a row not below `bands`, holds one twice, or
//     holds the "thermal" row;
//   * reflectance_unit or thermal_unit is not a positive number.
//
// The message names the field and the counts or the value. The algorithm's
// parameters are the configuration's, checked by their schemas and against
// the input when an Algorithm resolves them (RunSpec::algorithm).
//
//------------------------------------------------------------------------------

void validate(const Input& input, index_t bands);

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
