#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ccd/procedure/fit.hpp"

namespace py = pybind11;


void bind_fit(py::module_& m)
{

    py::enum_<ccd::CurveQA>(
        m,
        "CurveQA"
    )

    .value(
        "MinCoefficients",
        ccd::CurveQA::MinCoefficients
    )
    .value(
        "MidCoefficients",
        ccd::CurveQA::MidCoefficients
    )
    .value(
        "MaxCoefficients",
        ccd::CurveQA::MaxCoefficients
    )
    .value(
        "Start",
        ccd::CurveQA::Start
    )
    .value(
        "End",
        ccd::CurveQA::End
    )
    .value(
        "InsufficientClear",
        ccd::CurveQA::InsufficientClear
    )
    .value(
        "PersistentSnow",
        ccd::CurveQA::PersistentSnow
    )
    .value(
        "Stable",
        ccd::CurveQA::Stable
    )
    .value(
        "Recovery",
        ccd::CurveQA::Recovery
    )
    .value(
        "Disturbance",
        ccd::CurveQA::Disturbance
    );


    py::class_<ccd::ChangeModel>(
        m,
        "ChangeModel"
    )

    .def_readonly(
        "start_day",
        &ccd::ChangeModel::start_day
    )

    .def_readonly(
        "end_day",
        &ccd::ChangeModel::end_day
    )

    .def_readonly(
        "break_day",
        &ccd::ChangeModel::break_day
    )

    .def_readonly(
        "observation_count",
        &ccd::ChangeModel::observation_count
    )

    .def_readonly(
        "change_probability",
        &ccd::ChangeModel::change_probability
    )

    .def_readonly(
        "curve_qa",
        &ccd::ChangeModel::curve_qa
    )

    .def_readonly(
        "bands",
        &ccd::ChangeModel::bands
    );


    py::class_<ccd::FitResult>(
        m,
        "FitResult"
    )

    .def_readonly(
        "models",
        &ccd::FitResult::models
    )

    .def_readonly(
        "mask",
        &ccd::FitResult::mask
    );
}