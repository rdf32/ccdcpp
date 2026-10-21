#include <pybind11/pybind11.h>

#include "ccd/input/qa_statistics.hpp"

namespace py = pybind11;


void bind_quality(py::module_& m)
{

    py::class_<ccd::QaStatistics>(
        m,
        "QaStatistics"
    )

    .def_readonly(
        "total_count",
        &ccd::QaStatistics::total_count
    )

    .def_readonly(
        "clear_count",
        &ccd::QaStatistics::clear_count
    )

    .def_readonly(
        "cloud_count",
        &ccd::QaStatistics::cloud_count
    )

    .def_readonly(
        "snow_count",
        &ccd::QaStatistics::snow_count
    )

    .def_readonly(
        "clear_probability",
        &ccd::QaStatistics::clear_probability
    );
}