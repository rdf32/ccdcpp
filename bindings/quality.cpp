#include <pybind11/pybind11.h>

#include "ccd/harmonic/quality.hpp"

namespace py = pybind11;


void bind_quality(py::module_& m)
{

    py::class_<ccd::Quality::Statistics>(
        m,
        "QualityStatistics"
    )

    .def_readonly(
        "total_count",
        &ccd::Quality::Statistics::total_count
    )

    .def_readonly(
        "clear_count",
        &ccd::Quality::Statistics::clear_count
    )

    .def_readonly(
        "cloud_count",
        &ccd::Quality::Statistics::cloud_count
    )

    .def_readonly(
        "snow_count",
        &ccd::Quality::Statistics::snow_count
    )

    .def_readonly(
        "clear_probability",
        &ccd::Quality::Statistics::clear_probability
    );
}