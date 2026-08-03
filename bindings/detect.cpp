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

py::dict detect_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::array_t<ccd::scalar_t,
        py::array::c_style> spectral,

    py::array_t<std::uint8_t,
        py::array::c_style> qas,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions
)
{
    if (!dates.flags() & py::array::c_style)
        throw std::runtime_error("dates must be C contiguous");

    if (!spectral.flags() & py::array::c_style)
        throw std::runtime_error("spectral must be C contiguous");

    if (!qas.flags() & py::array::c_style)
        throw std::runtime_error("qas must be C contiguous");

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
    
    //----------------------------------------------------------------------
    // Run CCD WITHOUT Python GIL
    //----------------------------------------------------------------------

    ccd::FitResult final_result;
    {
        py::gil_scoped_release release;

        final_result =
            ccd::detect(
                dates_view,
                spectral_view,
                qas_view,
                hoptions,
                loptions
            );
    }

    //----------------------------------------------------------------------
    // Python conversion (GIL restored)
    //----------------------------------------------------------------------

    py::dict output;
    py::list models;

    for (const auto& model : final_result.models)
    {
        py::dict m;

        m["start_day"] =
            model.start_day;

        m["end_day"] =
            model.end_day;

        m["break_day"] =
            model.break_day;

        m["observation_count"] =
            model.observation_count;

        m["change_probability"] =
            model.change_probability;

        m["curve_qa"] =
            static_cast<std::uint8_t>(model.curve_qa);

        py::list bands;

        for (const auto& band : model.bands)
        {
            py::dict b;

            b["rmse"] =
                band.score.rmse;

            b["magnitude"] =
                band.score.magn;

            const auto& c = band.model.coefficients();
            py::tuple coef(c.size());

            for (size_t i = 0; i < c.size(); ++i)
            {
                coef[i] = c[i];
            }
            b["coefficients"] = coef;


            b["intercept"] =
                band.model.intercept();


            bands.append(b);
        }

        m["bands"] = bands;
        models.append(m);
    }

    output["models"] = models;

    // processing mask
    py::array_t<std::uint8_t> mask_array(
        final_result.mask.size()
    );

    auto mask_buf = mask_array.mutable_unchecked<1>();
    const auto& mask = final_result.mask.data();
    for (size_t i = 0; i < mask.size(); ++i)
    {
        mask_buf(i) = mask[i];
    }

    output["processing_mask"] = mask_array;

    return output;

}

py::list detect_cube_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::array_t<ccd::scalar_t,
        py::array::c_style> spectral,

    py::array_t<std::uint8_t,
        py::array::c_style> qas,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions
)
{
    auto dates_buf    = dates.request();
    auto spectral_buf = spectral.request();
    auto qas_buf      = qas.request();

    if (dates_buf.ndim != 1)
        throw std::runtime_error(
            "dates must have shape (timesteps)"
        );

    if (spectral_buf.ndim != 4)
        throw std::runtime_error(
            "spectral must have shape (height,width,bands,timesteps)"
        );

    if (qas_buf.ndim != 3)
        throw std::runtime_error(
            "qas must have shape (height,width,timesteps)"
        );

    const ccd::index_t H =
        static_cast<ccd::index_t>(spectral_buf.shape[0]);

    const ccd::index_t W =
        static_cast<ccd::index_t>(spectral_buf.shape[1]);

    const ccd::index_t B =
        static_cast<ccd::index_t>(spectral_buf.shape[2]);

    const ccd::index_t T =
        static_cast<ccd::index_t>(spectral_buf.shape[3]);

    if (static_cast<ccd::index_t>(dates_buf.shape[0]) != T)
        throw std::runtime_error(
            "dates length must equal spectral time dimension"
        );

    if (static_cast<ccd::index_t>(qas_buf.shape[0]) != H ||
        static_cast<ccd::index_t>(qas_buf.shape[1]) != W ||
        static_cast<ccd::index_t>(qas_buf.shape[2]) != T)
    {
        throw std::runtime_error(
            "qas must have shape (height,width,timesteps)"
        );
    }

    //----------------------------------------------------------------------
    // Zero-copy ArrayViews
    //----------------------------------------------------------------------

    auto dates_view =
        ccd::ArrayView<const std::int64_t,1>::contiguous(
            static_cast<std::int64_t*>(dates_buf.ptr),
            {T}
        );

    auto spectral_view =
        ccd::ArrayView<ccd::scalar_t,4>::contiguous(
            static_cast<ccd::scalar_t*>(spectral_buf.ptr),
            {H,W,B,T}
        );

    auto qas_view =
        ccd::ArrayView<const std::uint8_t,3>::contiguous(
            static_cast<std::uint8_t*>(qas_buf.ptr),
            {H,W,T}
        );

    //----------------------------------------------------------------------
    // Run CCD
    //----------------------------------------------------------------------

    std::vector<ccd::PixelResult> results;

    {
        py::gil_scoped_release release;

        results =
            ccd::detect_cube(
                dates_view,
                spectral_view,
                qas_view,
                hoptions,
                loptions
            );
    }
    std::cout << "wrapper results "
        << results.size()
        << std::endl;

    for(auto& p : results)
    {
        std::cout 
        << p.row << ","
        << p.col
        << " models="
        << p.result.models.size()
        << " mask="
        << p.result.mask.size()
        << std::endl;
    }
    //----------------------------------------------------------------------
    // Convert to Python
    //----------------------------------------------------------------------

    py::list output;

    for (const auto& pixel : results)
    {
        py::dict p;

        p["row"] = pixel.row;
        p["col"] = pixel.col;

        py::list models;

        for (const auto& model : pixel.result.models)
        {
            py::dict m;

            m["start_day"] = model.start_day;
            m["end_day"] = model.end_day;
            m["break_day"] = model.break_day;
            m["observation_count"] = model.observation_count;
            m["change_probability"] = model.change_probability;
            m["curve_qa"] =
                static_cast<std::uint8_t>(
                    model.curve_qa
                );

            py::list bands;

            for (const auto& band : model.bands)
            {
                py::dict b;

                b["rmse"] =
                    band.score.rmse;

                b["magnitude"] =
                    band.score.magn;

                b["intercept"] =
                    band.model.intercept();

                const auto& coef =
                    band.model.coefficients();

                py::tuple c(coef.size());

                for (size_t i = 0; i < coef.size(); ++i)
                    c[i] = coef[i];

                b["coefficients"] = c;

                bands.append(b);
            }

            m["bands"] = bands;

            models.append(m);
        }

        p["models"] = models;

        py::array_t<std::uint8_t> mask(
            pixel.result.mask.size()
        );

        auto mask_view =
            mask.mutable_unchecked<1>();

        for (size_t i = 0;
             i < pixel.result.mask.size();
             ++i)
        {
            mask_view(i) =
                pixel.result.mask[i];
        }

        p["processing_mask"] = mask;

        output.append(std::move(p));
    }

    return output;
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
    m.def(
        "detect_cube",
        &detect_cube_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions")
    );
}

