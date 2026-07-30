#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ccd/pmask.hpp"

namespace py = pybind11;


void bind_processing_mask(py::module_& m)
{
    py::class_<ccd::ProcessingMask>(
        m,
        "ProcessingMask"
    )

    .def(py::init<>())
    .def(
        py::init<ccd::index_t>()
    )

    .def(
        "size",
        &ccd::ProcessingMask::size
    )

    .def(
        "count",
        &ccd::ProcessingMask::count
    )

    .def(
        "any",
        &ccd::ProcessingMask::any
    )

    .def(
        "none",
        &ccd::ProcessingMask::none
    )

    .def(
        "data",
        [](const ccd::ProcessingMask& self)
        {
            return self.data();
        }
    );
}