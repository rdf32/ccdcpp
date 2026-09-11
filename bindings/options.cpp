#include <pybind11/pybind11.h>

#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace py = pybind11;


void bind_options(py::module_& m)
{
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
    // Which row of `spectral` holds which band. Exposed because a caller
    // assembling the array from a differently-ordered archive would otherwise
    // have to transpose it -- these are cheaper than a copy of the cube.
    //--------------------------------------------------------------------------
    .def_readwrite("BLUE_IDX",    &ccd::HarmonicOptions::BLUE_IDX)
    .def_readwrite("GREEN_IDX",   &ccd::HarmonicOptions::GREEN_IDX)
    .def_readwrite("RED_IDX",     &ccd::HarmonicOptions::RED_IDX)
    .def_readwrite("NIR_IDX",     &ccd::HarmonicOptions::NIR_IDX)
    .def_readwrite("SWIR1_IDX",   &ccd::HarmonicOptions::SWIR1_IDX)
    .def_readwrite("SWIR2_IDX",   &ccd::HarmonicOptions::SWIR2_IDX)
    .def_readwrite("THERMAL_IDX", &ccd::HarmonicOptions::THERMAL_IDX)
    .def_readwrite("QA_IDX",      &ccd::HarmonicOptions::QA_IDX)

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