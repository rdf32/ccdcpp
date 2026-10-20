#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ccd/core/band_list.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/types.hpp"
#include "ccd/input/qa_table.hpp"

namespace ccd
{

//==============================================================================
//
// Input
//
// What the caller's data is: its QA codes and the table that decodes its raw
// QA, the role of each row of the spectral array, and its units. It holds no
// algorithm parameter: those are the configuration's (config/registry.hpp),
// written in physical terms and resolved against an Input when a run is
// built (Registry::resolve()).
//
// This struct contains no computation; validate() (ccd.hpp) checks it
// against the input's band count once per call.
//
//==============================================================================

// The roles a row can play. Every role but "other" names at most one row of
// an input; "other" rows are fitted and reported, and no band list can name
// them. "nir" is the Landsat near infrared (TM/ETM+ band 4, OLI band 5,
// 865 nm; Sentinel-2's B8A); Sentinel-2's broad 842 nm B08 is "nir_broad".
extern const std::vector<std::string> ROLES;

// Whether `role` is one of ROLES.
bool is_role(const std::string& role);

struct Input
{
    //--------------------------------------------------------------------------
    // QA codes
    //
    // The codes the procedures compare `qas` against, one per observation:
    // the caller's uint8 codes, or a QaTable's output (qa_table.hpp), which
    // the table builders take from these fields.
    //
    //   qa_fill               not a valid observation in the statistics
    //   qa_clear, qa_water    kept for the fit; the clear percentage
    //   qa_snow               the snow percentage; PermanentSnow's fit
    //   qa_cloud              the statistics' cloud count
    //
    // Every other code keeps the observation out of the fit and counts only
    // toward the total. qa_shadow is not read by the procedures; the table
    // builders write it. qa_cirrus1, qa_cirrus2 and qa_occlusion are
    // informational: nothing in ccdcpp reads them.
    //--------------------------------------------------------------------------
    std::uint8_t qa_fill      = 255;
    std::uint8_t qa_clear     = 0;
    std::uint8_t qa_water     = 1;
    std::uint8_t qa_shadow    = 2;
    std::uint8_t qa_snow      = 3;
    std::uint8_t qa_cloud     = 4;
    std::uint8_t qa_cirrus1   = 8;
    std::uint8_t qa_cirrus2   = 9;
    std::uint8_t qa_occlusion = 10;

    // Decodes the raw QA values of `qas` (uint8 or uint16) into the codes
    // above (qa_table.hpp), built with this input's codes (qa_tables.hpp);
    // null, the default: `qas` are the codes, uint8.
    QaTablePtr qa_table;

    //--------------------------------------------------------------------------
    // Band layout
    //
    // One role (ROLES) per row of the spectral array: the band count every
    // call must pass is roles.size(). Every row is fitted and reported --
    // ChangeModel::bands has one entry per row, whatever its role. "green"
    // is required (the green median filter reads it); without "thermal" the
    // thermal transform and filter are skipped.
    //
    // The configuration names bands by role (its detection bands, Tmask's
    // bands, band_range's), or band_range's as "reflective", every row whose
    // role is not "thermal" or "other", in row order; either resolves to
    // rows through this list, so one configuration fits any band order.
    //
    // The default is the Landsat layout: blue, green, red, nir, swir1,
    // swir2, thermal.
    //--------------------------------------------------------------------------
    std::vector<std::string> roles = {"blue", "green", "red", "nir", "swir1", "swir2", "thermal"};

    //--------------------------------------------------------------------------
    // Units
    //
    // How the data encodes physical quantities, and nothing else: the
    // configuration says what to do with them, in physical units, and a run
    // translates once (Registry::resolve()).
    //
    // Reflectance r is held as r x reflectance_unit + reflectance_offset:
    // 10000 and 0 for Collection-1 DN (the default), 1 and 0 for
    // reflectance. A configuration's reflectance levels (band_range's
    // bounds) resolve the same way; its reflectance differences and
    // magnitudes (the Lasso's alpha and coef_floor, green_median's offset)
    // are multiplied by reflectance_unit only. 1e-4 x 10000, 0.04 x 10000,
    // 0 x 10000 and 1 x 10000 are 1.0, 400.0, 0 and 10000 exactly, so a DN
    // input runs pyccd's DN values bit for bit.
    //
    // The thermal transform (thermal_scale, thermal_offset) takes the thermal
    // row to a multiple of degrees Celsius, and thermal_unit says which: the
    // value 100 degrees Celsius has after the transform. Celsius x 100 on
    // Collection-1 DN (Kelvin x 10 times 10 less 27315, the default; 100
    // degrees is 10000), Celsius / 100 on the physical-unit input (the
    // identity transform; 100 degrees is 1). A configuration's thermal
    // bounds, in degrees Celsius, resolve to that unit: c x 100 x
    // thermal_unit / 10000, so pyccd's -93.2 and 70.7 are -9320 and 7070 on
    // DN and -0.932 and 0.707 on the physical-unit input, exactly.
    //
    // docs/units.md has the physical-unit input (reflectance and degrees
    // Celsius / 100) and its measured equivalence with DN, why degrees
    // Celsius itself is not the unit, Collection 2, the thermal range on the
    // procedures without the transform, and the guards of every unit-bearing
    // value (tests/ccd/machine/test_units.cpp).
    //--------------------------------------------------------------------------
    scalar_t reflectance_unit = 10000.0;

    // The data's value of reflectance 0 (see reflectance_unit): 0 by
    // default. Collection-2 DN, r = DN x 2.75e-5 - 0.2, is reflectance_unit
    // 1 / 2.75e-5 and reflectance_offset 0.2 / 2.75e-5.
    scalar_t reflectance_offset = 0.0;

    // Applied to the thermal row of Run::pixel's per-pixel copy before a
    // procedure whose configuration sets thermal_transform runs:
    //
    //     thermal <- thermal * thermal_scale + thermal_offset
    //
    // The default is pyccd's Kelvin x 10 -> Celsius x 100. The caller's
    // buffer is never written. The identity (1.0, 0.0) skips the pass, and
    // with it the copy when the pixel is already contiguous float64, and so
    // does an input without a "thermal" row.
    scalar_t thermal_scale  = KELVIN_SCALE;
    scalar_t thermal_offset = -KELVIN_OFFSET;

    // The value 100 degrees Celsius has in the thermal row after the
    // transform above: 10000 for the default (Celsius x 100), 1 for Celsius
    // / 100. It and the transform are one fact, written together: a
    // transform to Celsius x 100 goes with 10000, the identity on data
    // already in Celsius / 100 with 1. validate() checks that it is a
    // positive number; the transform cannot say which unit its input is in,
    // so the pair's agreement is the writer's (each shipped profile's is
    // tested).
    scalar_t thermal_unit = 10000.0;

    // The value a saturated observation holds, as the sensor writes it
    // (20000 in Collection-1 DN), or none (the default) when the input does
    // not declare one. A step or strategy that reads it refuses an input
    // without one.
    std::optional<scalar_t> saturated;

    // The band count: one row per role.
    index_t bands() const { return roles.size(); }

    // The row of `role`, or NO_BAND when the input has none.
    index_t index(const std::string& role) const;
};

bool operator==(const Input& a, const Input& b);
inline bool operator!=(const Input& a, const Input& b) { return !(a == b); }

// A digest of every field of `input`, its QA table's codes included: equal
// for equal inputs, on any platform.
std::uint64_t digest(const Input& input);

} // namespace ccd
