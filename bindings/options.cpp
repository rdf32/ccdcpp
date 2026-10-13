#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "ccd/constants.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace py = pybind11;


void bind_options(py::module_& m)
{
    // THERMAL_IDX for an input without a thermal band.
    m.attr("NO_BAND") = ccd::NO_BAND;

    // The largest band count, and the capacity of each band list.
    m.attr("CCD_MAX_BANDS") = ccd::CCD_MAX_BANDS;

    py::class_<ccd::HarmonicOptions>(
        m,
        "HarmonicOptions"
    )
    .def(py::init<>())

    .def_readwrite(
        "MEOW_SIZE",
        &ccd::HarmonicOptions::MEOW_SIZE
    )
    .def_readwrite(
        "PEEK_SIZE",
        &ccd::HarmonicOptions::PEEK_SIZE
    )
    .def_readwrite(
        "DAY_DELTA",
        &ccd::HarmonicOptions::DAY_DELTA
    )

    .def_readwrite(
        "OUTLIER_THRESHOLD",
        &ccd::HarmonicOptions::OUTLIER_THRESHOLD
    )
    .def_readwrite(
        "CHANGE_THRESHOLD",
        &ccd::HarmonicOptions::CHANGE_THRESHOLD
    )

    .def_readwrite(
        "CLEAR_OBSERVATION_THRESHOLD",
        &ccd::HarmonicOptions::CLEAR_OBSERVATION_THRESHOLD
    )

    .def_readwrite(
        "CLEAR_PCT_THRESHOLD",
        &ccd::HarmonicOptions::CLEAR_PCT_THRESHOLD
    )

    .def_readwrite(
        "COEFFICIENT_MIN",
        &ccd::HarmonicOptions::COEFFICIENT_MIN
    )
    .def_readwrite(
        "COEFFICIENT_MID",
        &ccd::HarmonicOptions::COEFFICIENT_MID
    )
    .def_readwrite(
        "COEFFICIENT_MAX",
        &ccd::HarmonicOptions::COEFFICIENT_MAX
    )

    //--------------------------------------------------------------------------
    // Input unit convention
    //
    // These seven say what unit the caller's `spectral` array is in. They
    // default to Collection-1 DN, so a caller that touches none of them gets
    // pyccd's numbers.
    //
    // The physical-unit preset is one divisor, 10000, on the input and on
    // every absolute threshold:
    //
    //     spectral[:6] = dn[:6] / 1e4             # reflectance 0..1
    //     spectral[6]  = (k10 * 10 - 27315) / 1e4 # degrees Celsius / 100
    //
    //     o = ccdcpp.HarmonicOptions()
    //     o.THERMAL_SCALE, o.THERMAL_OFFSET = 1.0, 0.0
    //     o.REFL_MIN,  o.REFL_MAX  = 0.0, 1.0
    //     o.THERM_MIN, o.THERM_MAX = -0.9320, 0.7070
    //     o.MEDIAN_GREEN_FILTER    = 0.04
    //
    //     lo = ccdcpp.LassoOptions()
    //     lo.alpha = lo.coef_floor = 1e-4
    //
    // Both LassoOptions fields are required, and neither omission raises.
    // Measured on the pyccd reference pixel: alpha alone left at 1.0 zeroes
    // 89 % of the coefficients and returns 4 segments instead of 5;
    // coef_floor alone left at 1.0 returns the right sparsity and the right
    // segments with coefficients wrong by up to 163 %. See harmonic.hpp for
    // the measured equivalence and why plain Celsius is not the preset.
    //--------------------------------------------------------------------------
    .def_readwrite(
        "THERMAL_SCALE",
        &ccd::HarmonicOptions::THERMAL_SCALE
    )
    .def_readwrite(
        "THERMAL_OFFSET",
        &ccd::HarmonicOptions::THERMAL_OFFSET
    )
    .def_readwrite(
        "REFL_MIN",
        &ccd::HarmonicOptions::REFL_MIN
    )
    .def_readwrite(
        "REFL_MAX",
        &ccd::HarmonicOptions::REFL_MAX
    )
    .def_readwrite(
        "THERM_MIN",
        &ccd::HarmonicOptions::THERM_MIN
    )
    .def_readwrite(
        "THERM_MAX",
        &ccd::HarmonicOptions::THERM_MAX
    )
    .def_readwrite(
        "MEDIAN_GREEN_FILTER",
        &ccd::HarmonicOptions::MEDIAN_GREEN_FILTER
    )

    //--------------------------------------------------------------------------
    // Band layout
    //
    // Which row of `spectral` holds which band role. NUM_BANDS is the band
    // count every call must pass; every row is fitted and reported. The
    // lists are read and written as lists of ints and kept in the order
    // given, which is the order the procedures iterate them in. Exposed
    // because a caller assembling the array from a differently-ordered
    // archive would otherwise have to transpose it -- these are cheaper than
    // a copy of the cube. ccdcpp.detect* raise ValueError on a layout that
    // does not fit the input (see ccd::validate in ccd.hpp).
    //--------------------------------------------------------------------------
    .def_readwrite(
        "NUM_BANDS",
        &ccd::HarmonicOptions::NUM_BANDS,
        "Band count of the spectral array every call must pass (default 7)."
    )
    .def_property(
        "DETECTION_BANDS",
        [](const ccd::HarmonicOptions& self)
        {
            return self.DETECTION_BANDS.to_vector();
        },
        [](ccd::HarmonicOptions& self, const std::vector<ccd::index_t>& bands)
        {
            self.DETECTION_BANDS = ccd::BandList(bands);
        },
        "Bands whose residuals make up the change magnitude, in order "
        "(default [1, 2, 3, 4, 5]). The count is the chi-squared degrees of "
        "freedom behind CHANGE_THRESHOLD and OUTLIER_THRESHOLD, whose "
        "defaults are chi2(5) quantiles."
    )
    .def_property(
        "TMASK_BANDS",
        [](const ccd::HarmonicOptions& self)
        {
            return self.TMASK_BANDS.to_vector();
        },
        [](ccd::HarmonicOptions& self, const std::vector<ccd::index_t>& bands)
        {
            self.TMASK_BANDS = ccd::BandList(bands);
        },
        "Bands the Tmask outlier screen fits (default [1, 4])."
    )
    .def_property(
        "SATURATION_BANDS",
        [](const ccd::HarmonicOptions& self)
        {
            return self.SATURATION_BANDS.to_vector();
        },
        [](ccd::HarmonicOptions& self, const std::vector<ccd::index_t>& bands)
        {
            self.SATURATION_BANDS = ccd::BandList(bands);
        },
        "Bands tested against [REFL_MIN, REFL_MAX] (default [0, 1, 2, 3, 4, "
        "5]). May be empty; must not hold THERMAL_IDX."
    )
    .def_readwrite(
        "GREEN_IDX",
        &ccd::HarmonicOptions::GREEN_IDX,
        "Band read by the InsufficientClear green median filter (default 1). "
        "Required."
    )
    .def_readwrite(
        "THERMAL_IDX",
        &ccd::HarmonicOptions::THERMAL_IDX,
        "Band read by the thermal transform and the thermal filter (default "
        "6), or ccdcpp.NO_BAND for an input without one, which skips both."
    )
    .def_readwrite("BLUE_IDX",  &ccd::HarmonicOptions::BLUE_IDX,
        "Informational; not read by ccdcpp (default 0).")
    .def_readwrite("RED_IDX",   &ccd::HarmonicOptions::RED_IDX,
        "Informational; not read by ccdcpp (default 2).")
    .def_readwrite("NIR_IDX",   &ccd::HarmonicOptions::NIR_IDX,
        "Informational; not read by ccdcpp (default 3).")
    .def_readwrite("SWIR1_IDX", &ccd::HarmonicOptions::SWIR1_IDX,
        "Informational; not read by ccdcpp (default 4).")
    .def_readwrite("SWIR2_IDX", &ccd::HarmonicOptions::SWIR2_IDX,
        "Informational; not read by ccdcpp (default 5).")
    .def_readwrite("QA_IDX",    &ccd::HarmonicOptions::QA_IDX,
        "Informational; not read by ccdcpp (default 7).")

    //--------------------------------------------------------------------------
    // QA vocabulary
    //
    // The codes the caller's `qas` array uses. A caller unpacking a different
    // archive's bitmask sets these rather than remapping the array.
    //--------------------------------------------------------------------------
    .def_readwrite("QA_FILL",      &ccd::HarmonicOptions::QA_FILL)
    .def_readwrite("QA_CLEAR",     &ccd::HarmonicOptions::QA_CLEAR)
    .def_readwrite("QA_WATER",     &ccd::HarmonicOptions::QA_WATER)
    .def_readwrite("QA_SHADOW",    &ccd::HarmonicOptions::QA_SHADOW)
    .def_readwrite("QA_SNOW",      &ccd::HarmonicOptions::QA_SNOW)
    .def_readwrite("QA_CLOUD",     &ccd::HarmonicOptions::QA_CLOUD)
    .def_readwrite("QA_CIRRUS1",   &ccd::HarmonicOptions::QA_CIRRUS1)
    .def_readwrite("QA_CIRRUS2",   &ccd::HarmonicOptions::QA_CIRRUS2)
    .def_readwrite("QA_OCCLUSION", &ccd::HarmonicOptions::QA_OCCLUSION)

    //--------------------------------------------------------------------------
    // Remaining thresholds
    //
    // SNOW_PCT_THRESHOLD and T_CONST were the last two HarmonicOptions fields
    // reachable only from C++; STAT_ORD and NUM_OBS_FACTOR likewise. All four
    // are scale-invariant (fractions, rmse-normalised residuals, an ordinal
    // date and a count) so they are not part of the unit block above.
    //--------------------------------------------------------------------------
    .def_readwrite(
        "SNOW_PCT_THRESHOLD",
        &ccd::HarmonicOptions::SNOW_PCT_THRESHOLD
    )
    .def_readwrite(
        "T_CONST",
        &ccd::HarmonicOptions::T_CONST
    )
    .def_readwrite(
        "STAT_ORD",
        &ccd::HarmonicOptions::STAT_ORD
    )
    .def_readwrite(
        "NUM_OBS_FACTOR",
        &ccd::HarmonicOptions::NUM_OBS_FACTOR
    );


    py::class_<ccd::LassoOptions>(
        m,
        "LassoOptions"
    )
    .def(py::init<>())

    .def_readwrite(
        "max_iter",
        &ccd::LassoOptions::max_iter
    )
    // alpha and coef_floor are ABSOLUTE, so both are statements about the
    // unit of y. Defaults are calibrated for Collection-1 DN; reflectance
    // needs 1e-4 for both. Getting this wrong does not raise -- it returns
    // all-zero coefficients, faster. See LassoOptions in lasso_solver.hpp.
    .def_readwrite(
        "alpha",
        &ccd::LassoOptions::alpha
    )
    .def_readwrite(
        "coef_floor",
        &ccd::LassoOptions::coef_floor
    )
    .def_readwrite(
        "tolerance",
        &ccd::LassoOptions::tolerance
    )
    .def_readwrite(
        "fit_intercept",
        &ccd::LassoOptions::fit_intercept
    )
    .def_readwrite(
        "warm_start",
        &ccd::LassoOptions::warm_start
    )
    .def_readwrite(
        "unbiased_rmse",
        &ccd::LassoOptions::unbiased_rmse
    );
}