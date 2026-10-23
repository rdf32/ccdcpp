// bindings/
//     module.cpp          // PYBIND11_MODULE: each bind_* below
//     data_spec.cpp       // bind_data_spec: DataSpec, GreenFilterScope
//     qa_tables.cpp       // bind_qa_tables: QaTable, ccdcpp.qa_tables
//     preprocess.cpp      // bind_preprocess: PreprocessPlan and its parts, plan_of
//     arrays.cpp          // the arrays of detect and detect_cube as the run's views (arrays.hpp)
//     detect.cpp          // bind_detect: detect, detect_cube, resume_schema
//     results.cpp         // the results as Python data and back (results.hpp)
//     config.cpp          // bind_config: Algorithm, the configuration helpers

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_data_spec(py::module_& m);
void bind_qa_tables(py::module_& m);
void bind_preprocess(py::module_& m);
void bind_detect(py::module_& m);
void bind_config(py::module_& m);


PYBIND11_MODULE(ccdcpp, m)
{
    m.doc() = "CCD change detection C++ implementation";
    m.attr("__version__") = CCD_VERSION;

    bind_data_spec(m);
    bind_qa_tables(m);
    bind_config(m);       // before the calls that take an Algorithm
    bind_preprocess(m);
    bind_detect(m);
}
