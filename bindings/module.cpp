// python/
//     module.cpp          // PYBIND11_MODULE + detect wrapper
//     options.cpp         // HarmonicOptions, LassoOptions
//     regression.cpp      // LassoModel, LassoScore, LassoResult
//     fit.cpp             // ChangeModel, FitResult, CurveQA
//     pmask.cpp           // ProcessingMask
//     quality.cpp         // Quality::Statistics

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_options(py::module_& m);
void bind_regression(py::module_& m);
void bind_fit(py::module_& m);
void bind_processing_mask(py::module_& m);
void bind_quality(py::module_& m);
void bind_detect(py::module_& m);


PYBIND11_MODULE(ccdcpp, m)
{
    m.doc() = "CCD change detection C++ implementation";

    bind_options(m);
    bind_regression(m);
    bind_fit(m);
    bind_processing_mask(m);
    bind_quality(m);
    bind_detect(m);
}



