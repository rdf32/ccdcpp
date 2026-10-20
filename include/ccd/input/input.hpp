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
    // bands), resolved to rows through this list, so one configuration fits
    // any band order.
    //
    // The default is the Landsat layout: blue, green, red, nir, swir1,
    // swir2, thermal.
    //--------------------------------------------------------------------------
    std::vector<std::string> roles = {"blue", "green", "red", "nir", "swir1", "swir2", "thermal"};

    // Rows the saturation filter tests against refl_min and refl_max, in
    // order. May be empty; must not hold the thermal row.
    BandList saturation_bands = {0, 1, 2, 3, 4, 5};

    //--------------------------------------------------------------------------
    // Units
    //
    // reflectance_unit is the value one unit of reflectance has in the data:
    // 10000 for Collection-1 DN (the default), 1 for reflectance. Every
    // configuration value in reflectance (the Lasso's alpha and coef_floor,
    // the green filter) is multiplied by it when a run resolves its
    // configuration; 1e-4 x 10000, 0.04 x 10000 and 0.002 x 10000 are 1.0,
    // 400.0 and 20.0 exactly, so a DN input runs pyccd's DN values bit for
    // bit.
    //
    // The bounds below are absolute, in the data's own unit. The *_inclusive
    // flags say whether a value equal to a bound is kept, which is the same
    // question in every unit.
    //
    // THE PHYSICAL-UNIT INPUT
    //
    // One divisor, 10000, applied to every band and every bound:
    //
    //     reflectance_unit = 1.0
    //     thermal_scale = 1.0       thermal_offset = 0.0    (identity)
    //     refl_min      = 0.0       refl_max       = 1.0
    //     therm_min     = -0.9320   therm_max      = 0.7070
    //
    // with the caller's array pre-divided by 10000:
    //
    //     optical  DN / 10000                  -> reflectance 0..1
    //     thermal  (K*10 - 27315) / 10000      -> degrees Celsius / 100
    //
    // The coefficients then come out in those same units: no descaling pass
    // on the way out, at any scale, for any band.
    //
    // MEASURED, on the pyccd reference pixel in
    // tests/ccd/machine/test_3657_3610_observations.csv (443 observations,
    // 5 segments): against the DN input, this input reproduces the
    // processing mask bit for bit, the same 5 segments with identical
    // start/end/break days, observation counts and curve_qa, change
    // probabilities equal to 0.0e+00 absolute, and every coefficient,
    // intercept, rmse and magnitude equal after multiplying by 10000 to a
    // worst relative difference of 4.4e-13. It is the same fit.
    //
    // WHY DEGREES CELSIUS ITSELF IS NOT THE PHYSICAL UNIT
    //
    // The Lasso's alpha is ONE value shared by all the band fits, and it is
    // an absolute soft-threshold, so the equivalence holds only if every
    // band shares one scale factor. Optical DN is 10000x reflectance; the DN
    // thermal path lands on Celsius x 100, so plain Celsius would be a 100x
    // factor -- two scales, one alpha. Measured on the same pixel, the
    // Celsius variant leaves the six optical bands exact (1e-13) and breaks
    // thermal alone: 2 extra nonzero coefficients, rmse off by 1.7e-2
    // relative, intercept by 4.0e-2. Celsius/100 is the unit the downstream
    // change-detection archive already stores.
    //
    // Two more things are worth knowing before changing any of these.
    //
    // 1. The two collection variants of the saturation bound are ONE
    //    predicate. Collection-1's (0, 10000) and Collection-2's
    //    (7273, 43636) both mean "reflectance in (0, 1)" -- C2 surface
    //    reflectance is DN * 2.75e-5 - 0.2, which maps 7273 -> 0.0000 and
    //    43636 -> 1.0000. In physical units there is no collection branch.
    //
    // 2. therm_min/therm_max are tested AFTER the transform, and the default
    //    plan transforms for the Standard procedure ONLY. That asymmetry is
    //    pyccd's: kelvin_to_celsius() is called inside standard_procedure()
    //    while snow_procedure_filter() applies filter_thermal_celsius() to
    //    values nothing converted. On a DN input the PermanentSnow and
    //    InsufficientClear procedures therefore test Kelvin x 10 against
    //    Celsius x 100 bounds, and since Kelvin x 10 is ~2200..3300,
    //    comfortably inside [-9320, 7070], the filter rejects nothing a
    //    sensor records there. It is preserved deliberately: ccdcpp is a
    //    recreation of pyccd and the regression references encode it.
    //
    //    On the physical-unit input thermal is already Celsius / 100 for
    //    every procedure, so PermanentSnow and InsufficientClear test
    //    converted values against converted bounds, as Standard does, and
    //    the filter is live there: a clear or water observation outside
    //    (-93.20, 70.70) degrees Celsius is dropped (PermanentSnow then adds
    //    back every snow observation regardless). The DN input keeps that
    //    observation. On those two procedures the two inputs agree only
    //    while no clear or water observation's thermal value is out of
    //    range; the equivalence measured above is on a Standard pixel.
    //
    // tests/ccd/machine/test_units.cpp asserts the equivalence and then, one
    // value at a time, that leaving it behind is DETECTED. The three shapes
    // of detection there are not interchangeable:
    //
    //   reflectance_unit     caught by the equivalence test directly: it
    //                        scales alpha (which moves the segment count)
    //                        and coef_floor (which leaves sparsity identical,
    //                        so only the value comparison catches it).
    //   refl_*, therm_*      NOT caught by it. The physical bounds are
    //                        narrower, so leaving them at the DN value merely
    //                        loosens a screen, and a screen that rejects
    //                        nothing is indistinguishable from a correct one.
    //                        The guard runs them the other way -- DN input,
    //                        physical bound -- where the mask empties and the
    //                        pixel returns 0 segments instead of 5.
    //   the green filter     unreachable through run_pixel() on the
    //                        reference pixel, which takes the Standard
    //                        procedure; the default plan runs the green
    //                        median filter in the InsufficientClear chain
    //                        only. Its guard fits the InsufficientClear
    //                        procedure directly (ccd::fit_whole_series).
    //
    // If you add a unit-bearing field, add its guard in that file and work
    // out which of the three shapes it is.
    //--------------------------------------------------------------------------
    scalar_t reflectance_unit = 10000.0;

    // The value a saturated observation holds, as the sensor writes it:
    // 20000 in Collection-1 DN (the default), or none when the input does not
    // declare one. A step or strategy that reads it refuses an input without
    // one.
    std::optional<scalar_t> saturated = 20000.0;

    // Bound of the saturation_bands. An observation is kept when every one
    // of them lies within (refl_min, refl_max), each end inclusive or
    // exclusive by its flag (default exclusive on both ends, pyccd's
    // 0 < value < 10000). With either end exclusive NaN is rejected; with
    // both ends inclusive the test is !(value < refl_min || value >
    // refl_max), under which NaN passes.
    scalar_t refl_min = 0.0;
    scalar_t refl_max = 10000.0;
    bool refl_min_inclusive = false;
    bool refl_max_inclusive = false;

    // Applied to the thermal row of Run::pixel's per-pixel copy before the
    // Standard procedure runs (the default plan's transform):
    //
    //     thermal <- thermal * thermal_scale + thermal_offset
    //
    // The default is pyccd's Kelvin x 10 -> Celsius x 100. The caller's
    // buffer is never written. The identity (1.0, 0.0) skips the pass, and
    // with it the copy when the pixel is already contiguous float64, and so
    // does an input without a "thermal" row.
    scalar_t thermal_scale  = KELVIN_SCALE;
    scalar_t thermal_offset = -KELVIN_OFFSET;

    // Valid range of the thermal row, tested after the transform, each end
    // inclusive or exclusive by its flag (default exclusive on both ends,
    // pyccd's -9320 < value < 7070, which rejects NaN). Defaults are
    // Celsius x 100; the physical-unit input's are those divided by 10000.
    // NaN passes only with both ends inclusive, as for the saturation bound.
    scalar_t therm_min = MIN_CELSIUS;
    scalar_t therm_max = MAX_CELSIUS;
    bool therm_min_inclusive = false;
    bool therm_max_inclusive = false;

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
