#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/ccd.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/procedure/fit.hpp"

namespace py = pybind11;


//------------------------------------------------------------------------------
// detect wrapper
//
// Zero-copy:
//   dates    -> int64 numpy memory
//   spectral -> float64 numpy memory
//   qas      -> uint8 numpy memory
//
// No allocations are performed here.
// ArrayViews simply wrap existing buffers.
//------------------------------------------------------------------------------

ccd::FitResult detect_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::array_t<ccd::scalar_t,
        py::array::c_style> spectral,

    py::array_t<std::uint8_t,
        py::array::c_style> qas,

    ccd::HarmonicOptions& hoptions,
    ccd::LassoOptions& loptions
)
{
    auto dates_buf = dates.request();
    auto spectral_buf = spectral.request();
    auto qas_buf = qas.request();


    if (dates_buf.ndim != 1)
        throw std::runtime_error(
            "dates must have shape (time)"
        );

    if (spectral_buf.ndim != 2)
        throw std::runtime_error(
            "spectral must have shape (bands,time)"
        );

    if (qas_buf.ndim != 1)
        throw std::runtime_error(
            "qas must have shape (time)"
        );


    const auto timesteps =
        static_cast<ccd::index_t>(
            dates_buf.shape[0]
        );


    if (static_cast<ccd::index_t>(qas_buf.shape[0]) != timesteps)
        throw std::runtime_error(
            "qas length must match dates length"
        );


    if (static_cast<ccd::index_t>(spectral_buf.shape[1]) != timesteps)
        throw std::runtime_error(
            "spectral time dimension must match dates"
        );


    //--------------------------------------------------------------------------
    // Create zero-copy ArrayViews
    //--------------------------------------------------------------------------

    auto dates_view =
        ccd::ArrayView<const std::int64_t,1>::contiguous(
            static_cast<std::int64_t*>(
                dates_buf.ptr
            ),
            {
                timesteps
            }
        );


    auto spectral_view =
        ccd::ArrayView<ccd::scalar_t,2>::contiguous(
            static_cast<ccd::scalar_t*>(
                spectral_buf.ptr
            ),
            {
                static_cast<ccd::index_t>(
                    spectral_buf.shape[0]
                ),

                static_cast<ccd::index_t>(
                    spectral_buf.shape[1]
                )
            }
        );


    auto qas_view =
        ccd::ArrayView<const std::uint8_t,1>::contiguous(
            static_cast<std::uint8_t*>(
                qas_buf.ptr
            ),
            {
                timesteps
            }
        );


    return ccd::detect(
        dates_view,
        spectral_view,
        qas_view,
        hoptions,
        loptions
    );
}



//------------------------------------------------------------------------------
// Python registration
//------------------------------------------------------------------------------

void bind_detect(py::module_& m)
{
    m.def(
        "detect",
        &detect_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions")
    );
}