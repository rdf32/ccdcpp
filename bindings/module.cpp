// bindings/
//     module.cpp          // PYBIND11_MODULE: __version__, build_info and each bind_* below
//     data_spec.cpp       // bind_data_spec: DataSpec, GreenFilterScope
//     qa_tables.cpp       // bind_qa_tables: QaTable, ccdcpp.qa_tables
//     preprocess.cpp      // bind_preprocess: PreprocessPlan and its parts, plan_of
//     arrays.cpp          // the arrays of detect and detect_cube as the run's views (arrays.hpp)
//     detect.cpp          // bind_detect: detect, detect_cube, resume_schema
//     results.cpp         // the results as Python data and back (results.hpp)
//     config.cpp          // bind_config: Algorithm, the configuration helpers

#include <pybind11/pybind11.h>

#include "ccd/core/build_info.hpp"

namespace py = pybind11;

void bind_data_spec(py::module_& m);
void bind_qa_tables(py::module_& m);
void bind_preprocess(py::module_& m);
void bind_detect(py::module_& m);
void bind_config(py::module_& m);


PYBIND11_MODULE(ccdcpp, m)
{
    m.doc() = "CCD change detection C++ implementation";
    // ccd::build_info().version. pyproject.toml carries the same number; the
    // python_bindings test checks that they agree.
    m.attr("__version__") = ccd::build_info().version;
    m.def(
        "build_info",
        []
        {
            const ccd::BuildInfo info = ccd::build_info();
            py::dict d;
            d["version"] = info.version;
            d["commit"] = info.commit;
            d["describe"] = info.describe;
            d["openmp"] = info.openmp;
            d["openmp_version"] = info.openmp_version;
            d["compiler"] = info.compiler;
            d["cxx_standard"] = info.cxx_standard;
            d["observe"] = info.observe;
            d["build_type"] = info.build_type;
            return d;
        },
        "What this build of ccdcpp is: version, commit (the hash at configure time, '-dirty' "
        "appended for changed tracked files, or 'unknown'), describe (git describe --always "
        "--dirty --tags then, or 'unknown'), openmp (False: detect_cube runs on one thread), "
        "openmp_version (_OPENMP, or 0), compiler, cxx_standard (__cplusplus), observe "
        "(CCD_OBSERVE) and build_type.");

    bind_data_spec(m);
    bind_qa_tables(m);
    bind_config(m);       // before the calls that take an Algorithm
    bind_preprocess(m);
    bind_detect(m);
}
