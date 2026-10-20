#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/ccd.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/input/mask/preprocess.hpp"

#include "ccd/results/diagnostics.hpp"
#include "config.hpp"
#include "resume.hpp"

namespace py = pybind11;

namespace
{

//------------------------------------------------------------------------------
// The events of observed pixels, recorded for Python (trace=, trace_pixels=):
// each a dict, ready for pandas.DataFrame.
//------------------------------------------------------------------------------
class RecordingObserver final : public ccd::Observer
{
public:
    void event(ccd::index_t pixel, const ccd::ObserverEvent& e) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_[pixel].push_back(e);
    }

    py::list rows(ccd::index_t pixel) const
    {
        py::list out;
        const auto it = events_.find(pixel);
        if (it == events_.end())
            return out;
        for (const ccd::ObserverEvent& e : it->second)
        {
            py::dict d;
            d["kind"] = ccd::to_string(e.kind);
            d["machine"] = e.machine;
            d["phase"] = e.phase;
            d["next"] = e.next;
            d["window_start"] = e.window_start;
            d["window_stop"] = e.window_stop;
            d["prev_break"] = e.prev_break;
            d["observations"] = e.observations;
            d["day"] = e.day;
            d["day_to"] = e.day_to;
            d["outcome"] = e.outcome;
            py::list values;
            for (const ccd::scalar_t v : e.values)
                values.append(v);
            d["values"] = values;
            out.append(d);
        }
        return out;
    }

private:
    std::mutex mutex_;
    std::map<ccd::index_t, std::vector<ccd::ObserverEvent>> events_;
};

} // namespace

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
// Without the input's QA table `qas` is QA codes: uint8, under the same two
// rules as `spectral` below. A uint16 `qas` holds raw QA values, which have
// no meaning without a table, and is a TypeError. With a table `qas` is raw
// values, uint8 or uint16, anything else converted to uint16 under numpy
// safe casting.
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
            "qas is uint16, which holds raw QA values; give the input a qa_table "
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
// resume=, as the entry points accept it: the previous call's output, made
// with record=True or by a resume. ValueError for anything else, naming what
// it lacks.
//------------------------------------------------------------------------------

static py::dict previous_output(const py::object& resume, const char* entry, const char* has)
{
    if (!py::isinstance<py::dict>(resume))
        throw py::type_error(std::string(entry) + "(): resume= is the output of an earlier " +
                             entry + "() call, a dict");
    const py::dict out = resume.cast<py::dict>();
    if (!out.contains("resume") || !out.contains("resume_header"))
        throw py::value_error(std::string(entry) + "(): resume= holds no resume record; make "
                              "the earlier call with record=True");
    if (!out.contains(has))
        throw py::value_error(std::string(entry) + "(): resume= has no \"" + has +
                              "\"; it is the output of an earlier " + entry + "() call");
    return out;
}

//------------------------------------------------------------------------------
// One pixel's models, as detect() returns them: a list of dicts with a
// "bands" list each.
//------------------------------------------------------------------------------

static py::list models_to_python(const std::vector<ccd::ChangeModel>& models)
{
    py::list out;
    for (const auto& model : models)
    {
        py::dict m;
        m["start_day"] = model.start_day;
        m["end_day"] = model.end_day;
        m["break_day"] = model.break_day;
        m["observation_count"] = model.observation_count;
        m["change_probability"] = model.change_probability;
        m["curve_qa"] = static_cast<std::uint8_t>(model.curve_qa);

        py::list bands;
        for (const auto& band : model.bands)
        {
            py::dict b;
            b["rmse"] = band.rmse;
            b["magnitude"] = band.magnitude;
            const auto& c = band.coefficients;
            py::tuple coef(c.size());
            for (std::size_t i = 0; i < c.size(); ++i)
                coef[i] = c[i];
            b["coefficients"] = coef;
            b["intercept"] = band.intercept;
            bands.append(b);
        }
        m["bands"] = bands;
        out.append(m);
    }
    return out;
}

//------------------------------------------------------------------------------
// detect
//
// Zero-copy:
//   dates    -> int64 numpy memory
//   spectral -> float64 numpy memory
//   qas      -> uint8 numpy memory, or uint16 with the input's table
//
// ArrayViews wrap the existing buffers. The array_t casters hand over
// C-contiguous arrays of these dtypes, converting when needed, and the run
// reads `spectral` without writing to it.
//------------------------------------------------------------------------------

static py::dict detect_wrapper(
    py::array_t<std::int64_t, py::array::c_style> dates,
    py::array_t<ccd::scalar_t, py::array::c_style> spectral,
    py::object qas_obj,
    const ccd::Input& input,
    py::object config,
    bool record,
    py::object resume,
    bool trace
)
{
    const bool with_table = input.qa_table != nullptr;
    refuse_raw_without_table(qas_obj, with_table);

    // As accept_qas(): with a table, uint8 stays uint8 and anything else is
    // uint16.
    const bool raw16 = with_table && !py::isinstance<py::array_t<std::uint8_t>>(qas_obj);

    py::array qas = raw16
        ? as_c_array<std::uint16_t>(qas_obj, "qas")
        : as_c_array<std::uint8_t>(qas_obj, "qas");

    auto dates_buf = dates.request();
    auto spectral_buf = spectral.request();
    auto qas_buf = qas.request();

    if (dates_buf.ndim != 1)
        throw std::runtime_error("dates must have shape (time)");
    if (spectral_buf.ndim != 2)
        throw std::runtime_error("spectral must have shape (bands,time)");
    if (qas_buf.ndim != 1)
        throw std::runtime_error("qas must have shape (time)");

    const auto timesteps = static_cast<ccd::index_t>(dates_buf.shape[0]);
    const auto bands = static_cast<ccd::index_t>(spectral_buf.shape[0]);

    if (static_cast<ccd::index_t>(qas_buf.shape[0]) != timesteps)
        throw std::runtime_error("qas length must match dates length");
    if (static_cast<ccd::index_t>(spectral_buf.shape[1]) != timesteps)
        throw std::runtime_error("spectral time dimension must match dates");

    // ValueError, raised here, before the GIL is released.
    ccd::validate(input, bands);

    const auto dates_view = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        static_cast<std::int64_t*>(dates_buf.ptr), {timesteps});

    // ValueError, as above.
    ccd::validate_dates(dates_view);

    std::optional<ccd::PixelResume> from;
    if (!resume.is_none())
    {
        const py::dict previous = previous_output(resume, "detect", "models");
        from = ccd::PixelResume{
            header_from_python(previous["resume_header"].cast<py::dict>()),
            record_from_python(previous["resume"].cast<py::dict>()),
            segments_from_python(previous["models"].cast<py::list>())};
    }

    const auto spectral_view = ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
        static_cast<const ccd::scalar_t*>(spectral_buf.ptr), {bands, timesteps});

    ccd::RunSpec spec;
    spec.encoding = input;
    spec.algorithm = algorithm_from_python(config);
    spec.record = record;
    spec.masks = true;   // detect returns the processing mask
    const auto recording = trace ? std::make_shared<RecordingObserver>() : nullptr;
    spec.observer = recording;

    //----------------------------------------------------------------------
    // Run CCD without the GIL
    //----------------------------------------------------------------------

    ccd::PixelOutput run;
    {
        py::gil_scoped_release release;

        const ccd::PixelResume* prior = from ? &*from : nullptr;
        if (raw16)
            run = ccd::detect(spec, dates_view, spectral_view,
                              ccd::ArrayView<const std::uint16_t, 1>::contiguous(
                                  static_cast<const std::uint16_t*>(qas_buf.ptr), {timesteps}),
                              prior);
        else
            run = ccd::detect(spec, dates_view, spectral_view,
                              ccd::ArrayView<const std::uint8_t, 1>::contiguous(
                                  static_cast<const std::uint8_t*>(qas_buf.ptr), {timesteps}),
                              prior);
    }

    py::dict output;
    output["models"] = models_to_python(run.result.models);

    // The length of every model's band list, also when there are no models.
    output["num_bands"] = bands;

    py::array_t<std::uint8_t> mask_array(run.result.mask.size());
    auto mask_buf = mask_array.mutable_unchecked<1>();
    const auto& mask = run.result.mask.data();
    for (std::size_t i = 0; i < mask.size(); ++i)
        mask_buf(i) = mask[i];
    output["processing_mask"] = mask_array;

    if (record || from)
    {
        output["resume"] = record_to_python(run.record);
        output["resume_header"] = header_to_python(run.header);
    }
    if (from)
        output["status"] =
            run.status == ccd::ResumeStatus::Resumed ? "resumed" : "rerun_input_changed";
    if (recording)
        output["trace"] = recording->rows(0);

    return output;
}

//------------------------------------------------------------------------------
// Cube input.
//
// `spectral` is taken as it is when it is float64, float32 or uint16 in native
// byte order with non-negative strides -- any layout, C-contiguous or not --
// and read through a strided ArrayView. Each pixel is converted to float64
// inside detect_cube, per thread, so a uint16 cube is never widened as a whole.
// Anything else is converted to a C-contiguous float64 array first, under
// numpy safe casting. `qas` is uint8 under the same two rules, or uint8 or
// uint16 with the input's table; see accept_qas().
//
// Nothing is written to `spectral`: a plan's transform happens in the
// per-pixel copy.
//------------------------------------------------------------------------------

// The element type of CubeInput::spectral, read while the GIL is held:
// testing a py::array's dtype creates and releases Python objects, which
// detect_cube cannot do.
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

    ccd::index_t H = 0;
    ccd::index_t W = 0;
    ccd::index_t B = 0;
    ccd::index_t T = 0;
};

static CubeInput check_cube(
    py::array_t<std::int64_t, py::array::c_style> dates,
    const py::object& spectral_obj,
    const py::object& qas_obj,
    const ccd::Input& input
)
{
    py::array spectral = accept<double, float, std::uint16_t>(spectral_obj, "spectral");
    py::array qas = accept_qas(qas_obj, input.qa_table != nullptr);

    if (dates.ndim() != 1)
        throw std::runtime_error("dates must have shape (timesteps)");
    if (spectral.ndim() != 4)
        throw std::runtime_error("spectral must have shape (height,width,bands,timesteps)");
    if (qas.ndim() != 3)
        throw std::runtime_error("qas must have shape (height,width,timesteps)");

    const auto H = static_cast<ccd::index_t>(spectral.shape(0));
    const auto W = static_cast<ccd::index_t>(spectral.shape(1));
    const auto B = static_cast<ccd::index_t>(spectral.shape(2));
    const auto T = static_cast<ccd::index_t>(spectral.shape(3));

    if (static_cast<ccd::index_t>(dates.shape(0)) != T)
        throw std::runtime_error("dates length must equal spectral time dimension");

    if (static_cast<ccd::index_t>(qas.shape(0)) != H ||
        static_cast<ccd::index_t>(qas.shape(1)) != W ||
        static_cast<ccd::index_t>(qas.shape(2)) != T)
        throw std::runtime_error("qas must have shape (height,width,timesteps)");

    // ValueError, raised here, before the GIL is released.
    ccd::validate(input, B);
    ccd::validate_dates(ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}));

    const SpectralType spectral_type =
        takes_as_is<std::uint16_t>(spectral) ? SpectralType::UInt16 :
        takes_as_is<float>(spectral)         ? SpectralType::Float32 :
                                               SpectralType::Float64;
    const QaType qas_type = qa_type(qas);

    return CubeInput{std::move(dates), std::move(spectral), std::move(qas), spectral_type,
                     qas_type, H, W, B, T};
}

//------------------------------------------------------------------------------
// ccd::detect_cube on a checked input: its spectral and QA dtypes as the views
// it dispatches on.
//
// Called with the GIL released: it only reads buffers that `in` keeps alive,
// and touches no Python object.
//------------------------------------------------------------------------------

static ccd::CubeOutput detect_views(
    const CubeInput& in,
    const ccd::RunSpec& spec,
    int threads,
    const ccd::CubeResumeFrom* from
)
{
    const auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(in.dates.data(), {in.T});

    const ccd::CubeSpectral spectral =
        in.spectral_type == SpectralType::UInt16
            ? ccd::CubeSpectral(strided_view<std::uint16_t, 4>(in.spectral))
        : in.spectral_type == SpectralType::Float32
            ? ccd::CubeSpectral(strided_view<float, 4>(in.spectral))
            : ccd::CubeSpectral(strided_view<double, 4>(in.spectral));

    const ccd::CubeQa qas = in.qa_type == QaType::UInt16
        ? ccd::CubeQa(strided_view<std::uint16_t, 3>(in.qas))
        : ccd::CubeQa(strided_view<std::uint8_t, 3>(in.qas));

    return ccd::detect_cube(spec, dates, spectral, qas, threads, from);
}

//------------------------------------------------------------------------------
// Failures, as Python data: a list of (row, col, message) tuples, empty in
// the normal case.
//------------------------------------------------------------------------------

static py::list failures_to_python(const std::vector<ccd::CubeFailure>& failures)
{
    py::list out;
    for (const auto& f : failures)
        out.append(py::make_tuple(f.row, f.col, f.message));
    return out;
}

//------------------------------------------------------------------------------
// detect_cube's columns
//
// One numpy array per field, one ROW PER (PIXEL, SEGMENT), instead of a tree
// of Python objects. For a 500x500 cell with T=987, a dict per pixel would be
// ~250,000 py::dicts for pixels, ~350,000 more for segments, ~2,500,000 for
// bands, 2.5 million py::tuples of 7 floats and 250,000 length-987 uint8
// numpy arrays for the processing masks, about 247 MB of allocations whose
// only purpose is to be thrown away. Feeding a columnar writer (parquet,
// arrow) from that means walking every one of those objects in Python. The
// columns are written into preallocated buffers, so the path from C++ to a
// parquet file contains no per-row Python object at all. ccdcpp.to_pixels()
// turns them into per-pixel dicts.
//
// The processing mask is absent unless asked for (masks=True): it is the
// single largest allocation, T bytes per pixel, and the change-detection
// schema this feeds has no column for it. Asked for, it is one more array,
// one contiguous uint8 array over the cube's pixels (not per segment):
//
//   processing_mask       uint8 [H, W, T]  [r, c]: pixel (r, c)'s mask, 1
//                                          where the observation is kept;
//                                          zeros for a failed pixel
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
// Band order is the input's row order, B = Input.bands() rows; under the
// default layout that is blue, green, red, nir, swir1, swir2, thermal.
// Coefficient order is HarmonicBasis's column order -- slope, cos1, sin1,
// cos2, sin2, cos3, sin3 -- and the intercept is NOT among them, which is why it is a separate array. Coefficients above the model's
// own count are zero, not absent; zero is also a legitimate fitted value
// under L1, so a caller that needs to know which coefficients were fitted
// must derive the count from curve_qa and must not test values against zero.
//------------------------------------------------------------------------------

static py::dict cube_columns(const ccd::CubeOutput& cube, const CubeInput& views, bool masks)
{
    std::size_t n = 0;
    std::size_t with_segments = 0;
    for (const auto& pixel : cube.pixels)
    {
        n += pixel.result.models.size();
        if (!pixel.result.models.empty())
            ++with_segments;
    }

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
    py::array_t<ccd::scalar_t> int_arr(std::vector<py::ssize_t>{rows, nb});
    py::array_t<ccd::scalar_t> rmse_arr(std::vector<py::ssize_t>{rows, nb});
    py::array_t<ccd::scalar_t> magn_arr(std::vector<py::ssize_t>{rows, nb});
    py::array_t<ccd::scalar_t> coef_arr(std::vector<py::ssize_t>{rows, nb, NC});

    auto row_v = row_arr.mutable_unchecked<1>();
    auto col_v = col_arr.mutable_unchecked<1>();
    auto start_v = start_arr.mutable_unchecked<1>();
    auto end_v = end_arr.mutable_unchecked<1>();
    auto break_v = break_arr.mutable_unchecked<1>();
    auto qa_v = qa_arr.mutable_unchecked<1>();
    auto obs_v = obs_arr.mutable_unchecked<1>();
    auto prob_v = prob_arr.mutable_unchecked<1>();
    auto int_v = int_arr.mutable_unchecked<2>();
    auto rmse_v = rmse_arr.mutable_unchecked<2>();
    auto magn_v = magn_arr.mutable_unchecked<2>();
    auto coef_v = coef_arr.mutable_unchecked<3>();

    // Pixel-major in raster order (row-major), whatever the thread count, and
    // chronological within a pixel, the order the procedures produced them.
    py::ssize_t i = 0;
    for (const auto& pixel : cube.pixels)
    {
        for (const auto& model : pixel.result.models)
        {
            row_v(i) = static_cast<std::int32_t>(pixel.row);
            col_v(i) = static_cast<std::int32_t>(pixel.col);
            start_v(i) = model.start_day;
            end_v(i) = model.end_day;
            break_v(i) = model.break_day;
            qa_v(i) = static_cast<std::uint8_t>(model.curve_qa);
            obs_v(i) = static_cast<std::int32_t>(model.observation_count);
            prob_v(i) = model.change_probability;
            for (ccd::index_t b = 0; b < NB; ++b)
            {
                const auto& band = model.bands[b];
                int_v(i, b) = band.intercept;
                rmse_v(i, b) = band.rmse;
                magn_v(i, b) = band.magnitude;
                for (ccd::index_t c = 0; c < NC; ++c)
                    coef_v(i, b, c) = band.coefficients[c];
            }
            ++i;
        }
    }

    py::dict output;
    output["row"] = row_arr;
    output["col"] = col_arr;
    output["start_day"] = start_arr;
    output["end_day"] = end_arr;
    output["break_day"] = break_arr;
    output["curve_qa"] = qa_arr;
    output["observation_count"] = obs_arr;
    output["change_probability"] = prob_arr;
    output["intercept"] = int_arr;
    output["coefficients"] = coef_arr;
    output["rmse"] = rmse_arr;
    output["magnitude"] = magn_arr;
    output["num_bands"] = NB;
    output["failures"] = failures_to_python(cube.failures);
    output["threads"] = cube.threads;
    output["pixels_total"] =
        static_cast<std::int64_t>(views.H) * static_cast<std::int64_t>(views.W);
    output["pixels_with_segments"] = static_cast<std::int64_t>(with_segments);

    if (masks)
    {
        const auto T = static_cast<py::ssize_t>(views.T);
        py::array_t<std::uint8_t> mask_arr(std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(views.H), static_cast<py::ssize_t>(views.W), T});
        std::uint8_t* const mask_v = mask_arr.mutable_data();
        std::fill(mask_v, mask_v + mask_arr.size(), std::uint8_t{0});
        for (const auto& pixel : cube.pixels)
        {
            const auto& mask = pixel.result.mask.data();
            std::copy(mask.begin(), mask.end(),
                      mask_v + (pixel.row * views.W + pixel.col) * views.T);
        }
        output["processing_mask"] = mask_arr;
    }
    return output;
}

static py::dict detect_cube_wrapper(
    py::array_t<std::int64_t, py::array::c_style> dates,
    py::object spectral,
    py::object qas,
    const ccd::Input& input,
    py::object config,
    int threads,
    bool record,
    py::object resume,
    py::object trace_pixels,
    bool masks
)
{
    const CubeInput views = check_cube(dates, spectral, qas, input);

    ccd::RunSpec spec;
    spec.encoding = input;
    spec.algorithm = algorithm_from_python(config);
    spec.masks = masks;
    spec.record = record;

    std::optional<ccd::CubeResumeFrom> from;
    if (!resume.is_none())
    {
        const py::dict previous = previous_output(resume, "detect_cube", "row");
        from = cube_from_python(previous["resume_header"].cast<py::dict>(),
                                previous["resume"].cast<py::dict>(), previous, views.H,
                                views.W, views.B);
    }

    std::vector<std::pair<ccd::index_t, ccd::index_t>> traced;
    std::shared_ptr<RecordingObserver> recording;
    if (!trace_pixels.is_none())
    {
        for (const py::handle& rc : trace_pixels)
        {
            const auto pair = rc.cast<std::pair<std::int64_t, std::int64_t>>();
            if (pair.first < 0 || pair.second < 0 ||
                static_cast<ccd::index_t>(pair.first) >= views.H ||
                static_cast<ccd::index_t>(pair.second) >= views.W)
                throw py::value_error("trace_pixels: (" + std::to_string(pair.first) + ", " +
                                      std::to_string(pair.second) +
                                      ") is not a pixel of the cube");
            traced.emplace_back(pair.first, pair.second);
            spec.observe.push_back(static_cast<ccd::index_t>(pair.first) * views.W +
                                   static_cast<ccd::index_t>(pair.second));
        }
        if (!traced.empty())
        {
            recording = std::make_shared<RecordingObserver>();
            spec.observer = recording;
        }
    }

    ccd::CubeOutput cube;
    {
        py::gil_scoped_release release;
        cube = detect_views(views, spec, threads, from ? &*from : nullptr);
    }

    py::dict output = cube_columns(cube, views, masks);
    if (record || from)
    {
        output["resume"] = record_columns(cube, views.H, views.W, views.B, from.has_value());
        output["resume_header"] = header_to_python(cube.header);
    }
    if (from)
    {
        std::int64_t resumed = 0, rerun = 0;
        for (const auto& px : cube.pixels)
            (px.status == ccd::ResumeStatus::Resumed ? resumed : rerun) += 1;
        py::dict counts;
        counts["resumed"] = resumed;
        counts["rerun_input_changed"] = rerun;
        counts["failed"] = static_cast<std::int64_t>(cube.failures.size());
        output["counts"] = counts;
    }
    if (recording)
    {
        py::dict traces;
        for (const auto& [row, col] : traced)
            traces[py::make_tuple(row, col)] = recording->rows(row * views.W + col);
        output["traces"] = traces;
    }
    return output;
}

//------------------------------------------------------------------------------
// Python registration: the extension's two calls. The package's ccdcpp.detect
// and ccdcpp.detect_cube take a SensorProfile or an Input and a preset name
// too, and call these.
//------------------------------------------------------------------------------

void bind_detect(py::module_& m)
{
    m.def(
        "detect",
        &detect_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::kw_only(),
        py::arg("encoding"),
        py::arg("config") = py::none(),
        py::arg("record") = false,
        py::arg("resume") = py::none(),
        py::arg("trace") = false,
        "CCD on one pixel (see ccdcpp.detect)."
    );
    m.def(
        "detect_cube",
        &detect_cube_wrapper,
        py::arg("dates"),
        py::arg("spectral"),
        py::arg("qas"),
        py::kw_only(),
        py::arg("encoding"),
        py::arg("config") = py::none(),
        py::arg("threads") = 0,
        py::arg("record") = false,
        py::arg("resume") = py::none(),
        py::arg("trace_pixels") = py::none(),
        py::arg("masks") = false,
        "CCD on every pixel of a cube, as columns (see ccdcpp.detect_cube)."
    );
    m.def("resume_schema", &resume_schema);
    m.attr("RESUME_FORMAT") = ccd::RESUME_FORMAT;
    m.attr("RESULTS_VERSION") = ccd::RESULTS_VERSION;
}
