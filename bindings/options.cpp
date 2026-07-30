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
    .def_readwrite(
        "alpha",
        &ccd::LassoOptions::alpha
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