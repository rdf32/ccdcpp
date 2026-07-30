#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ccd/regression/lasso_solver.hpp"

namespace py = pybind11;


void bind_regression(py::module_& m)
{

    py::class_<ccd::LassoScore>(
        m,
        "LassoScore"
    )

    .def_readwrite(
        "rmse",
        &ccd::LassoScore::rmse
    )

    .def_readwrite(
        "magn",
        &ccd::LassoScore::magn
    )

    .def_readwrite(
        "residuals",
        &ccd::LassoScore::residuals
    );


    py::class_<ccd::LassoModel>(
        m,
        "LassoModel"
    )

    .def(
        "iterations",
        &ccd::LassoModel::iterations
    )

    .def(
        "intercept",
        &ccd::LassoModel::intercept
    )

    .def(
        "coefficients",
        &ccd::LassoModel::coefficients
    );


    py::class_<ccd::LassoResult>(
        m,
        "LassoResult"
    )

    .def_readwrite(
        "model",
        &ccd::LassoResult::model
    )

    .def_readwrite(
        "score",
        &ccd::LassoResult::score
    );
}