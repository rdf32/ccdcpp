// python/
//     module.cpp          // PYBIND11_MODULE + detect wrapper
//     input.cpp           // Input, GreenFilterScope
//     regression.cpp      // LassoModel, LassoScore, LassoResult
//     fit.cpp             // ChangeModel, FitResult, CurveQA
//     pmask.cpp           // ProcessingMask
//     quality.cpp         // Quality::Statistics
//     qa_tables.cpp       // QaTable, ccdcpp.qa_tables
//     preprocess.cpp      // PreprocessPlan, default_plan, validate_plan
//     config.cpp          // Algorithm, the configuration helpers

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_input(py::module_& m);
// void bind_regression(py::module_& m);
// void bind_fit(py::module_& m);
// void bind_processing_mask(py::module_& m);
// void bind_quality(py::module_& m);
void bind_qa_tables(py::module_& m);
void bind_preprocess(py::module_& m);
void bind_detect(py::module_& m);
void bind_config(py::module_& m);


PYBIND11_MODULE(ccdcpp, m)
{
    m.doc() = "CCD change detection C++ implementation";
    m.attr("__version__") = CCD_VERSION;

    bind_input(m);
    // bind_regression(m);
    // bind_fit(m);
    // bind_processing_mask(m);
    // bind_quality(m);
    bind_qa_tables(m);
    bind_preprocess(m);
    bind_detect(m);
    bind_config(m);
}



