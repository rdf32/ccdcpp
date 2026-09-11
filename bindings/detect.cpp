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

//------------------------------------------------------------------------------
// Shape checking, shared by both cube wrappers.
//
// Extracted rather than duplicated because the two wrappers differ only in how
// they hand results back, and two copies of a shape check is two chances for
// them to disagree about what a valid cube is.
//------------------------------------------------------------------------------

struct CubeViews
{
    ccd::ArrayView<const std::int64_t, 1> dates;
    ccd::ArrayView<ccd::scalar_t, 4> spectral;
    ccd::ArrayView<const std::uint8_t, 3> qas;

    ccd::index_t H = 0;
    ccd::index_t W = 0;
    ccd::index_t B = 0;
    ccd::index_t T = 0;
};

static CubeViews check_cube(
    py::buffer_info& dates_buf,
    py::buffer_info& spectral_buf,
    py::buffer_info& qas_buf
)
{
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

    return CubeViews{
        ccd::ArrayView<const std::int64_t,1>::contiguous(
            static_cast<std::int64_t*>(dates_buf.ptr),
            {T}
        ),
        ccd::ArrayView<ccd::scalar_t,4>::contiguous(
            static_cast<ccd::scalar_t*>(spectral_buf.ptr),
            {H,W,B,T}
        ),
        ccd::ArrayView<const std::uint8_t,3>::contiguous(
            static_cast<std::uint8_t*>(qas_buf.ptr),
            {H,W,T}
        ),
        H, W, B, T
    };
}

//------------------------------------------------------------------------------
// Failures, as Python data.
//
// A list of (row, col, message) tuples -- empty in the normal case. Both cube
// wrappers report the same shape, so a caller handles failures one way.
//------------------------------------------------------------------------------

static py::list failures_to_python(
    const std::vector<ccd::CubeFailure>& failures
)
{
    py::list out;

    for (const auto& f : failures)
    {
        out.append(
            py::make_tuple(
                f.row,
                f.col,
                f.message
            )
        );
    }

    return out;
}

py::list detect_cube_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::array_t<ccd::scalar_t,
        py::array::c_style> spectral,

    py::array_t<std::uint8_t,
        py::array::c_style> qas,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    int threads
)
{
    auto dates_buf    = dates.request();
    auto spectral_buf = spectral.request();
    auto qas_buf      = qas.request();

    const CubeViews views =
        check_cube(dates_buf, spectral_buf, qas_buf);

    //----------------------------------------------------------------------
    // Run CCD
    //----------------------------------------------------------------------

    ccd::CubeResult cube;

    {
        py::gil_scoped_release release;

        cube =
            ccd::detect_cube(
                views.dates,
                views.spectral,
                views.qas,
                hoptions,
                loptions,
                threads
            );
    }

    //----------------------------------------------------------------------
    // Failed pixels
    //
    // This wrapper's return type is a flat list of per-pixel dicts, so there
    // is nowhere in it to put a failure count. Warning is the alternative to
    // dropping them on the floor. detect_cube_columns() returns them as data
    // instead, which is what a pipeline should use.
    //----------------------------------------------------------------------

    if (!cube.failures.empty())
    {
        const std::string message =
            "detect_cube: " +
            std::to_string(cube.failures.size()) +
            " of " +
            std::to_string(
                static_cast<std::int64_t>(views.H) *
                static_cast<std::int64_t>(views.W)
            ) +
            " pixels raised and were skipped; first at (" +
            std::to_string(cube.failures.front().row) + "," +
            std::to_string(cube.failures.front().col) + "): " +
            cube.failures.front().message +
            " -- detect_cube_columns() returns every failure as data";

        PyErr_WarnEx(
            PyExc_RuntimeWarning,
            message.c_str(),
            1
        );
    }

    //----------------------------------------------------------------------
    // Convert to Python
    //----------------------------------------------------------------------

    py::list output;

    for (const auto& pixel : cube.pixels)
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
// detect_cube_columns wrapper
//
// Same computation as detect_cube, different way out: one numpy array per
// field, one ROW PER (PIXEL, SEGMENT), instead of a tree of Python objects.
//
// Why this exists, in numbers. For a 500x500 cell with T=987, detect_cube
// builds ~250,000 py::dicts for pixels, ~350,000 more for segments,
// ~2,500,000 for bands, 2.5 million py::tuples of 7 floats, and -- the
// expensive one -- 250,000 length-987 uint8 numpy arrays for the processing
// masks, about 247 MB of allocations whose only purpose is to be thrown away.
// Feeding a columnar writer (parquet, arrow) from that means walking every one
// of those objects in Python. This wrapper writes into preallocated buffers,
// so the path from C++ to a parquet file contains no per-row Python object at
// all.
//
// The processing mask is deliberately absent. It is the single largest
// allocation and the change-detection schema this feeds has no column for it;
// a caller who needs it uses detect_cube.
//
// Two passes: count segments, allocate, fill. Counting is cheap (it is a
// vector size per pixel) and it is the only way to allocate exactly once.
//
// Returned dict:
//
//   row, col              int32 [n]        cube-relative, 0-based
//   start_day, end_day,
//   break_day             int64 [n]        proleptic-Gregorian ordinals
//   curve_qa              uint8 [n]
//   observation_count     int32 [n]
//   change_probability  float64 [n]
//   intercept           float64 [n, 7]     band-major
//   coefficients        float64 [n, 7, 7]  (band, coefficient)
//   rmse                float64 [n, 7]
//   magnitude           float64 [n, 7]
//   failures              list of (row, col, message)
//   threads               int              what actually ran
//   pixels_total          int              H * W
//   pixels_with_segments  int              pixels that produced >= 1 row
//
// The last two exist because a pixel can produce ZERO segments and that is
// neither an error nor a failure -- it is simply absent from every array. A
// pixel whose observations are all fill, all NaN, all zero, or all cloud
// returns no models and no address, and `failures` stays empty. Measured: of a
// 4x4 cube with three pixels made degenerate three different ways, 13 pixels
// appear in the output and 0 failures are reported. So a caller reconciling
// against an archive that guarantees every pixel is present -- which the
// legacy change-detection parquet does -- must compare these two numbers, and
// nothing else in this dict would have told it.
//
// Band order is ccdcpp's own -- blue, green, red, nir, swir1, swir2, thermal.
// Coefficient order is build_basis()'s -- slope, cos1, sin1, cos2, sin2, cos3,
// sin3 -- and the intercept is NOT among them, which is why it is a separate
// array. Coefficients above the model's own count are zero, not absent; zero
// is also a legitimate fitted value under L1, so a caller that needs to know
// which coefficients were fitted must derive the count from curve_qa and must
// not test values against zero.
//------------------------------------------------------------------------------

py::dict detect_cube_columns_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::array_t<ccd::scalar_t,
        py::array::c_style> spectral,

    py::array_t<std::uint8_t,
        py::array::c_style> qas,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    int threads
)
{
    auto dates_buf    = dates.request();
    auto spectral_buf = spectral.request();
    auto qas_buf      = qas.request();

    const CubeViews views =
        check_cube(dates_buf, spectral_buf, qas_buf);

    //----------------------------------------------------------------------
    // Run CCD
    //----------------------------------------------------------------------

    ccd::CubeResult cube;

    {
        py::gil_scoped_release release;

        cube =
            ccd::detect_cube(
                views.dates,
                views.spectral,
                views.qas,
                hoptions,
                loptions,
                threads
            );
    }

    //----------------------------------------------------------------------
    // Pass 1 -- count segments
    //----------------------------------------------------------------------

    std::size_t n = 0;
    std::size_t with_segments = 0;

    for (const auto& pixel : cube.pixels)
    {
        n += pixel.result.models.size();

        if (!pixel.result.models.empty())
            ++with_segments;
    }

    //----------------------------------------------------------------------
    // Allocate
    //----------------------------------------------------------------------

    constexpr ccd::index_t NB = ccd::CCD_NUM_BANDS;
    constexpr ccd::index_t NC = ccd::CCD_MAX_COEFS;

    const auto rows = static_cast<py::ssize_t>(n);

    py::array_t<std::int32_t> row_arr(rows);
    py::array_t<std::int32_t> col_arr(rows);

    py::array_t<std::int64_t> start_arr(rows);
    py::array_t<std::int64_t> end_arr(rows);
    py::array_t<std::int64_t> break_arr(rows);

    py::array_t<std::uint8_t> qa_arr(rows);
    py::array_t<std::int32_t> obs_arr(rows);
    py::array_t<ccd::scalar_t> prob_arr(rows);

    py::array_t<ccd::scalar_t> int_arr(
        std::vector<py::ssize_t>{rows, NB}
    );
    py::array_t<ccd::scalar_t> rmse_arr(
        std::vector<py::ssize_t>{rows, NB}
    );
    py::array_t<ccd::scalar_t> magn_arr(
        std::vector<py::ssize_t>{rows, NB}
    );
    py::array_t<ccd::scalar_t> coef_arr(
        std::vector<py::ssize_t>{rows, NB, NC}
    );

    auto row_v   = row_arr.mutable_unchecked<1>();
    auto col_v   = col_arr.mutable_unchecked<1>();
    auto start_v = start_arr.mutable_unchecked<1>();
    auto end_v   = end_arr.mutable_unchecked<1>();
    auto break_v = break_arr.mutable_unchecked<1>();
    auto qa_v    = qa_arr.mutable_unchecked<1>();
    auto obs_v   = obs_arr.mutable_unchecked<1>();
    auto prob_v  = prob_arr.mutable_unchecked<1>();
    auto int_v   = int_arr.mutable_unchecked<2>();
    auto rmse_v  = rmse_arr.mutable_unchecked<2>();
    auto magn_v  = magn_arr.mutable_unchecked<2>();
    auto coef_v  = coef_arr.mutable_unchecked<3>();

    //----------------------------------------------------------------------
    // Pass 2 -- fill
    //
    // Segment order is pixel-major and, within a pixel, chronological, which
    // is the order the procedures produced them. Pixel order is whatever the
    // dynamic OpenMP schedule produced -- it is NOT raster order, and a
    // caller that needs raster order must sort.
    //----------------------------------------------------------------------

    py::ssize_t i = 0;

    for (const auto& pixel : cube.pixels)
    {
        for (const auto& model : pixel.result.models)
        {
            row_v(i) = static_cast<std::int32_t>(pixel.row);
            col_v(i) = static_cast<std::int32_t>(pixel.col);

            start_v(i) = model.start_day;
            end_v(i)   = model.end_day;
            break_v(i) = model.break_day;

            qa_v(i) =
                static_cast<std::uint8_t>(model.curve_qa);

            obs_v(i) =
                static_cast<std::int32_t>(model.observation_count);

            prob_v(i) = model.change_probability;

            for (ccd::index_t b = 0; b < NB; ++b)
            {
                const auto& band = model.bands[b];

                int_v(i, b)  = band.model.intercept();
                rmse_v(i, b) = band.score.rmse;
                magn_v(i, b) = band.score.magn;

                const auto& coef = band.model.coefficients();

                for (ccd::index_t c = 0; c < NC; ++c)
                    coef_v(i, b, c) = coef[c];
            }

            ++i;
        }
    }

    //----------------------------------------------------------------------
    // Out
    //----------------------------------------------------------------------

    py::dict output;

    output["row"] = row_arr;
    output["col"] = col_arr;

    output["start_day"] = start_arr;
    output["end_day"]   = end_arr;
    output["break_day"] = break_arr;

    output["curve_qa"]           = qa_arr;
    output["observation_count"]  = obs_arr;
    output["change_probability"] = prob_arr;

    output["intercept"]    = int_arr;
    output["coefficients"] = coef_arr;
    output["rmse"]         = rmse_arr;
    output["magnitude"]    = magn_arr;

    output["failures"] = failures_to_python(cube.failures);
    output["threads"]  = cube.threads;

    output["pixels_total"] =
        static_cast<std::int64_t>(views.H) *
        static_cast<std::int64_t>(views.W);

    output["pixels_with_segments"] =
        static_cast<std::int64_t>(with_segments);

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
        py::arg("loptions"),
        py::arg("threads") = 0
    );
    m.def(
        "detect_cube_columns",
        &detect_cube_columns_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions"),
        py::arg("threads") = 0
    );
}

