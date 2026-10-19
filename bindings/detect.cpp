#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/ccd.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/qa_table.hpp"
#include "ccd/preprocess/preprocess.hpp"

#include "resume.hpp"

namespace py = pybind11;

//------------------------------------------------------------------------------
// Array input, shared by every wrapper below.
//------------------------------------------------------------------------------

static bool usable_strides(const py::array& a)
{
    for (py::ssize_t d = 0; d < a.ndim(); ++d)
    {
        if (a.strides(d) < 0 || a.strides(d) % a.itemsize() != 0)
            return false;
    }
    return true;
}

template<typename T>
static bool takes_as_is(const py::array& a)
{
    return py::isinstance<py::array_t<T>>(a) && usable_strides(a);
}

template<typename T>
static py::array as_c_array(const py::object& obj, const char* name)
{
    auto converted = py::array_t<T, py::array::c_style>::ensure(obj);

    if (!converted)
        throw py::type_error(
            std::string(name) + " cannot be converted to " +
            py::str(py::dtype::of<T>()).cast<std::string>() +
            " without an unsafe cast"
        );

    return std::move(converted);
}

template<typename T, std::size_t Rank>
static ccd::ArrayView<const T, Rank> strided_view(const py::array& a)
{
    std::array<std::size_t, Rank> shape{};
    std::array<std::size_t, Rank> strides{};

    for (std::size_t d = 0; d < Rank; ++d)
    {
        shape[d] = static_cast<std::size_t>(a.shape(d));
        strides[d] = static_cast<std::size_t>(a.strides(d)) / sizeof(T);
    }

    return ccd::ArrayView<const T, Rank>::strided(
        static_cast<const T*>(a.data()),
        shape,
        strides
    );
}

//------------------------------------------------------------------------------
// `obj` as it is if it is an ndarray of one of the types Ts in a layout
// takes_as_is accepts, else converted to a C-contiguous array of the first.
//------------------------------------------------------------------------------
template<typename... Ts>
static py::array accept(const py::object& obj, const char* name)
{
    if (py::isinstance<py::array>(obj))
    {
        auto a = py::reinterpret_borrow<py::array>(obj);

        if ((takes_as_is<Ts>(a) || ...))
            return a;
    }

    using First = std::tuple_element_t<0, std::tuple<Ts...>>;
    return as_c_array<First>(obj, name);
}

//------------------------------------------------------------------------------
// `qas`, as an entry point accepts it.
//
// Without a table `qas` is QA codes: uint8, under the same two rules as
// `spectral` below. A uint16 `qas` holds raw QA values, which have no meaning
// without a table, and is a TypeError. With a table `qas` is raw values,
// uint8 or uint16, anything else converted to uint16 under numpy safe
// casting.
//------------------------------------------------------------------------------

enum class QaType
{
    UInt8,
    UInt16
};

static void refuse_raw_without_table(const py::object& obj, bool with_table)
{
    if (!with_table && py::isinstance<py::array_t<std::uint16_t>>(obj))
        throw py::type_error(
            "qas is uint16, which holds raw QA values; pass qa_table= "
            "(see ccdcpp.qa_tables) to decode them, or pass uint8 QA codes"
        );
}

static py::array accept_qas(const py::object& obj, bool with_table)
{
    refuse_raw_without_table(obj, with_table);

    if (!with_table)
        return accept<std::uint8_t>(obj, "qas");

    return accept<std::uint16_t, std::uint8_t>(obj, "qas");
}

static QaType qa_type(const py::array& qas)
{
    return py::isinstance<py::array_t<std::uint16_t>>(qas)
        ? QaType::UInt16
        : QaType::UInt8;
}

//------------------------------------------------------------------------------
// `preprocess=`, as an entry point accepts it: None for the default plan, or a
// ccdcpp.PreprocessPlan, copied here so that the call never reads an object
// Python can change while the GIL is released, and checked against the band
// count (ValueError) before it is.
//------------------------------------------------------------------------------

static ccd::PreprocessPlanPtr take_plan(
    const ccd::preprocess::PreprocessPlan* plan,
    ccd::index_t bands
)
{
    if (plan == nullptr)
        return nullptr;

    auto copy = std::make_shared<const ccd::preprocess::PreprocessPlan>(*plan);
    ccd::preprocess::validate(*copy, bands);
    return copy;
}

//------------------------------------------------------------------------------
// detect wrapper
//
// Zero-copy:
//   dates    -> int64 numpy memory
//   spectral -> float64 numpy memory
//   qas      -> uint8 numpy memory, or uint16 with a table
//
// No allocations are performed here.
// ArrayViews simply wrap existing buffers. The array_t casters hand over
// C-contiguous arrays of these dtypes, converting when needed, and detect()
// reads `spectral` without writing to it.
//------------------------------------------------------------------------------

// What a single-pixel resume continues from, converted while the GIL is held.
using PixelResumeFrom = ccd::PixelResume;

// detect(), detect(resume_record=True) and resume() on one pixel: one input
// check and one output dict. `from` null: detect; `record` adds the record.
static py::dict pixel_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::array_t<ccd::scalar_t,
        py::array::c_style> spectral,

    py::object qas_obj,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess_plan,
    bool record,
    const PixelResumeFrom* from
)
{
    refuse_raw_without_table(qas_obj, qa_table != nullptr);

    // As accept_qas(): with a table, uint8 stays uint8 and anything else is
    // uint16.
    const bool raw16 =
        qa_table && !py::isinstance<py::array_t<std::uint8_t>>(qas_obj);

    py::array qas = raw16
        ? as_c_array<std::uint16_t>(qas_obj, "qas")
        : as_c_array<std::uint8_t>(qas_obj, "qas");

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

    // ValueError, raised here, before the GIL is released.
    ccd::validate(
        hoptions,
        static_cast<ccd::index_t>(spectral_buf.shape[0])
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

    // ValueError, as above.
    ccd::validate_dates(dates_view);

    const ccd::PreprocessPlanPtr preprocess = take_plan(
        preprocess_plan,
        static_cast<ccd::index_t>(spectral_buf.shape[0])
    );


    auto spectral_view =
        ccd::ArrayView<const ccd::scalar_t,2>::contiguous(
            static_cast<const ccd::scalar_t*>(
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


    //----------------------------------------------------------------------
    // Run CCD WITHOUT Python GIL
    //----------------------------------------------------------------------

    ccd::FitResult final_result;
    ccd::ResumeHeader header;
    ccd::PixelRecord pixel_record;
    ccd::ResumeStatus status = ccd::ResumeStatus::Resumed;

    ccd::RunSpec spec{hoptions, loptions, qa_table, preprocess};
    spec.record = record;

    const auto run = [&](auto qa_view)
    {
        ccd::PixelRun r = ccd::run_pixel(spec, dates_view, spectral_view, qa_view, from);
        final_result = std::move(r.result);
        header = r.header;
        pixel_record = std::move(r.record);
        status = r.status;
    };

    {
        py::gil_scoped_release release;

        if (raw16)
            run(ccd::ArrayView<const std::uint16_t,1>::contiguous(
                static_cast<const std::uint16_t*>(qas_buf.ptr), {timesteps}));
        else
            run(ccd::ArrayView<const std::uint8_t,1>::contiguous(
                static_cast<const std::uint8_t*>(qas_buf.ptr), {timesteps}));
    }

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
                band.rmse;

            b["magnitude"] =
                band.magnitude;

            const auto& c = band.coefficients;
            py::tuple coef(c.size());

            for (size_t i = 0; i < c.size(); ++i)
            {
                coef[i] = c[i];
            }
            b["coefficients"] = coef;


            b["intercept"] =
                band.intercept;


            bands.append(b);
        }

        m["bands"] = bands;
        models.append(m);
    }

    output["models"] = models;

    // The length of every model's band list, also when there are no models.
    output["num_bands"] =
        static_cast<ccd::index_t>(spectral_buf.shape[0]);

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

    if (record || from)
    {
        output["resume"] = record_to_python(pixel_record);
        output["resume_header"] = header_to_python(header);
    }
    if (from)
        output["status"] = status == ccd::ResumeStatus::Resumed ? "resumed" : "rerun_input_changed";

    return output;
}

py::dict detect_wrapper(
    py::array_t<std::int64_t, py::array::c_style> dates,
    py::array_t<ccd::scalar_t, py::array::c_style> spectral,
    py::object qas_obj,
    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess_plan,
    bool resume_record
)
{
    return pixel_wrapper(dates, spectral, qas_obj, hoptions, loptions, std::move(qa_table),
                         preprocess_plan, resume_record, nullptr);
}

py::dict resume_wrapper(
    py::array_t<std::int64_t, py::array::c_style> dates,
    py::array_t<ccd::scalar_t, py::array::c_style> spectral,
    py::object qas_obj,
    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    py::dict resume,
    py::dict resume_header,
    py::list models,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess_plan
)
{
    const PixelResumeFrom from{
        header_from_python(resume_header),
        record_from_python(resume),
        segments_from_python(models)};
    return pixel_wrapper(dates, spectral, qas_obj, hoptions, loptions, std::move(qa_table),
                         preprocess_plan, true, &from);
}

//------------------------------------------------------------------------------
// Cube input, shared by both cube wrappers.
//
// Extracted rather than duplicated because the two wrappers differ only in how
// they hand results back, and two copies of a shape check is two chances for
// them to disagree about what a valid cube is.
//
// `spectral` is taken as it is when it is float64, float32 or uint16 in native
// byte order with non-negative strides -- any layout, C-contiguous or not --
// and read through a strided ArrayView. Each pixel is converted to float64
// inside detect_cube, per thread, so a uint16 cube is never widened as a
// whole. Anything else is converted to a C-contiguous float64 array first,
// under the same rule the float64 argument used to apply (numpy safe casting),
// so the accepted inputs are a superset of before. `qas` is uint8 under the
// same two rules, or uint8 or uint16 with a table; see accept_qas().
//
// Nothing is written to `spectral`: a plan's transform happens in the
// per-pixel copy. `preprocess` is copied and checked by take_plan().
//------------------------------------------------------------------------------

// The element type of CubeInput::spectral, read while the GIL is held:
// testing a py::array's dtype creates and releases Python objects, which
// run_cube cannot do.
enum class SpectralType
{
    Float64,
    Float32,
    UInt16
};

struct CubeInput
{
    py::array_t<std::int64_t, py::array::c_style> dates;
    py::array spectral;
    py::array qas;

    SpectralType spectral_type = SpectralType::Float64;
    QaType qa_type = QaType::UInt8;
    std::shared_ptr<ccd::QaTable> qa_table;
    ccd::PreprocessPlanPtr preprocess;

    ccd::index_t H = 0;
    ccd::index_t W = 0;
    ccd::index_t B = 0;
    ccd::index_t T = 0;
};

static CubeInput check_cube(
    py::array_t<std::int64_t, py::array::c_style> dates,
    const py::object& spectral_obj,
    const py::object& qas_obj,
    const ccd::HarmonicOptions& hoptions,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess_plan
)
{
    py::array spectral =
        accept<double, float, std::uint16_t>(spectral_obj, "spectral");

    py::array qas =
        accept_qas(qas_obj, qa_table != nullptr);

    if (dates.ndim() != 1)
        throw std::runtime_error(
            "dates must have shape (timesteps)"
        );

    if (spectral.ndim() != 4)
        throw std::runtime_error(
            "spectral must have shape (height,width,bands,timesteps)"
        );

    if (qas.ndim() != 3)
        throw std::runtime_error(
            "qas must have shape (height,width,timesteps)"
        );

    const ccd::index_t H =
        static_cast<ccd::index_t>(spectral.shape(0));

    const ccd::index_t W =
        static_cast<ccd::index_t>(spectral.shape(1));

    const ccd::index_t B =
        static_cast<ccd::index_t>(spectral.shape(2));

    const ccd::index_t T =
        static_cast<ccd::index_t>(spectral.shape(3));

    if (static_cast<ccd::index_t>(dates.shape(0)) != T)
        throw std::runtime_error(
            "dates length must equal spectral time dimension"
        );

    if (static_cast<ccd::index_t>(qas.shape(0)) != H ||
        static_cast<ccd::index_t>(qas.shape(1)) != W ||
        static_cast<ccd::index_t>(qas.shape(2)) != T)
    {
        throw std::runtime_error(
            "qas must have shape (height,width,timesteps)"
        );
    }

    // ValueError, raised here, before the GIL is released.
    ccd::validate(hoptions, B);
    ccd::validate_dates(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}));
    ccd::PreprocessPlanPtr preprocess = take_plan(preprocess_plan, B);

    const SpectralType spectral_type =
        takes_as_is<std::uint16_t>(spectral) ? SpectralType::UInt16 :
        takes_as_is<float>(spectral)         ? SpectralType::Float32 :
                                               SpectralType::Float64;

    const QaType qas_type = qa_type(qas);

    return CubeInput{
        std::move(dates),
        std::move(spectral),
        std::move(qas),
        spectral_type,
        qas_type,
        std::move(qa_table),
        std::move(preprocess),
        H, W, B, T
    };
}

//------------------------------------------------------------------------------
// run_cube on a checked input: its spectral and QA dtypes as the views
// ccd::run_cube dispatches on.
//
// Called with the GIL released: it only reads buffers that `in` keeps alive,
// and touches no Python object.
//------------------------------------------------------------------------------

static ccd::CubeResult run_cube(
    const CubeInput& in,
    const ccd::HarmonicOptions& hoptions,
    const ccd::LassoOptions& loptions,
    int threads,
    bool keep_masks,
    bool record = false,
    const ccd::CubeResumeFrom* from = nullptr
)
{
    const auto dates =
        ccd::ArrayView<const std::int64_t, 1>::contiguous(
            in.dates.data(),
            {in.T}
        );

    const ccd::CubeSpectral spectral =
        in.spectral_type == SpectralType::UInt16
            ? ccd::CubeSpectral(strided_view<std::uint16_t, 4>(in.spectral))
        : in.spectral_type == SpectralType::Float32
            ? ccd::CubeSpectral(strided_view<float, 4>(in.spectral))
            : ccd::CubeSpectral(strided_view<double, 4>(in.spectral));

    const ccd::CubeQa qas = in.qa_type == QaType::UInt16
        ? ccd::CubeQa(strided_view<std::uint16_t, 3>(in.qas))
        : ccd::CubeQa(strided_view<std::uint8_t, 3>(in.qas));

    ccd::RunSpec spec{hoptions, loptions, in.qa_table, in.preprocess};
    spec.keep_masks = keep_masks;
    spec.record = record;
    return ccd::run_cube(spec, dates, spectral, qas, threads, from);
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

    py::object spectral,

    py::object qas,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    int threads,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess
)
{
    const CubeInput views =
        check_cube(dates, spectral, qas, hoptions, std::move(qa_table), preprocess);

    //----------------------------------------------------------------------
    // Run CCD
    //----------------------------------------------------------------------

    ccd::CubeResult cube;

    {
        py::gil_scoped_release release;

        cube =
            run_cube(
                views,
                hoptions,
                loptions,
                threads,
                true
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
                    band.rmse;

                b["magnitude"] =
                    band.magnitude;

                b["intercept"] =
                    band.intercept;

                const auto& coef =
                    band.coefficients;

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
//   intercept           float64 [n, B]     band-major
//   coefficients        float64 [n, B, 7]  (band, coefficient)
//   rmse                float64 [n, B]
//   magnitude           float64 [n, B]
//   num_bands             int              B, the input's band count
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
// Band order is the input's row order, B = HarmonicOptions.NUM_BANDS rows;
// under the default layout that is blue, green, red, nir, swir1, swir2,
// thermal. Coefficient order is HarmonicBasis's column order -- slope, cos1,
// sin1, cos2, sin2, cos3, sin3 -- and the intercept is NOT among them, which
// is why it is a separate array. Coefficients above the model's own count are
// zero, not absent; zero is also a legitimate fitted value under L1, so a
// caller that needs to know which coefficients were fitted must derive the
// count from curve_qa and must not test values against zero.
//------------------------------------------------------------------------------

// detect_cube_columns()'s dict of one cube's segments, pixel-major in raster
// order.
static py::dict cube_columns(const ccd::CubeResult& cube, const CubeInput& views)
{
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

    const ccd::index_t NB = views.B;
    const auto nb = static_cast<py::ssize_t>(NB);
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
        std::vector<py::ssize_t>{rows, nb}
    );
    py::array_t<ccd::scalar_t> rmse_arr(
        std::vector<py::ssize_t>{rows, nb}
    );
    py::array_t<ccd::scalar_t> magn_arr(
        std::vector<py::ssize_t>{rows, nb}
    );
    py::array_t<ccd::scalar_t> coef_arr(
        std::vector<py::ssize_t>{rows, nb, NC}
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
    // is the order the procedures produced them. Pixel order is raster order
    // (row-major), whatever the thread count.
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

                int_v(i, b)  = band.intercept;
                rmse_v(i, b) = band.rmse;
                magn_v(i, b) = band.magnitude;

                const auto& coef = band.coefficients;

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
    output["num_bands"]    = NB;

    output["failures"] = failures_to_python(cube.failures);
    output["threads"]  = cube.threads;

    output["pixels_total"] =
        static_cast<std::int64_t>(views.H) *
        static_cast<std::int64_t>(views.W);

    output["pixels_with_segments"] =
        static_cast<std::int64_t>(with_segments);

    return output;
}

py::dict detect_cube_columns_wrapper(
    py::array_t<std::int64_t,
        py::array::c_style> dates,

    py::object spectral,

    py::object qas,

    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    int threads,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess,
    bool resume_record
)
{
    const CubeInput views =
        check_cube(dates, spectral, qas, hoptions, std::move(qa_table), preprocess);

    ccd::CubeResult cube;
    {
        py::gil_scoped_release release;

        cube = run_cube(views, hoptions, loptions, threads,
                        false,   // no column holds the mask
                        resume_record);
    }

    py::dict output = cube_columns(cube, views);
    if (resume_record)
    {
        output["resume"] = record_columns(cube, views.H, views.W, views.B, false);
        output["resume_header"] = header_to_python(cube.header);
    }
    return output;
}

py::dict resume_cube_columns_wrapper(
    py::array_t<std::int64_t, py::array::c_style> dates,
    py::object spectral,
    py::object qas,
    ccd::HarmonicOptions hoptions,
    ccd::LassoOptions loptions,
    int threads,
    py::dict resume,
    py::dict resume_header,
    py::dict segments,
    std::shared_ptr<ccd::QaTable> qa_table,
    const ccd::preprocess::PreprocessPlan* preprocess
)
{
    const CubeInput views =
        check_cube(dates, spectral, qas, hoptions, std::move(qa_table), preprocess);
    const ccd::CubeResumeFrom from =
        cube_from_python(resume_header, resume, segments, views.H, views.W, views.B);

    ccd::CubeResult cube;
    {
        py::gil_scoped_release release;

        cube = run_cube(views, hoptions, loptions, threads, false, true, &from);
    }

    py::dict output = cube_columns(cube, views);
    output["resume"] = record_columns(cube, views.H, views.W, views.B, true);
    output["resume_header"] = header_to_python(cube.header);

    std::int64_t resumed = 0, rerun = 0;
    for (const auto& px : cube.pixels)
        (px.status == ccd::ResumeStatus::Resumed ? resumed : rerun) += 1;
    py::dict counts;
    counts["resumed"] = resumed;
    counts["rerun_input_changed"] = rerun;
    counts["failed"] = static_cast<std::int64_t>(cube.failures.size());
    output["counts"] = counts;
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
        py::arg("loptions"),
        py::kw_only(),
        py::arg("qa_table") = py::none(),
        py::arg("preprocess") = py::none(),
        py::arg("resume_record") = false
    );
    m.def(
        "resume",
        &resume_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions"),
        py::kw_only(),
        py::arg("resume"),
        py::arg("resume_header"),
        py::arg("models"),
        py::arg("qa_table") = py::none(),
        py::arg("preprocess") = py::none()
    );
    m.def(
        "detect_cube",
        &detect_cube_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions"),
        py::arg("threads") = 0,
        py::kw_only(),
        py::arg("qa_table") = py::none(),
        py::arg("preprocess") = py::none()
    );
    m.def(
        "detect_cube_columns",
        &detect_cube_columns_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions"),
        py::arg("threads") = 0,
        py::kw_only(),
        py::arg("qa_table") = py::none(),
        py::arg("preprocess") = py::none(),
        py::arg("resume_record") = false
    );
    m.def(
        "resume_cube_columns",
        &resume_cube_columns_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::arg("hoptions"),
        py::arg("loptions"),
        py::arg("threads") = 0,
        py::kw_only(),
        py::arg("resume"),
        py::arg("resume_header"),
        py::arg("segments"),
        py::arg("qa_table") = py::none(),
        py::arg("preprocess") = py::none()
    );
    m.def("resume_schema", &resume_schema);
    m.attr("RESUME_FORMAT") = ccd::RESUME_FORMAT;
    m.attr("RESULTS_VERSION") = ccd::RESULTS_VERSION;
}

