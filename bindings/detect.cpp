#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/ccd.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/results/segment.hpp"
#include "ccd/input/mask/chain.hpp"

#include "ccd/results/diagnostics.hpp"
#include "arrays.hpp"
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
// detect: one pixel's arrays (arrays.hpp), read where they are.
//------------------------------------------------------------------------------

static py::dict detect_wrapper(
    py::object dates,
    py::object spectral,
    py::object qas,
    const ccd::Input& input,
    std::shared_ptr<ccd::Algorithm> config,
    bool record,
    py::object resume,
    bool trace
)
{
    const PixelArrays arrays = pixel_arrays(dates, spectral, qas, input.qa_table != nullptr);

    std::optional<ccd::Resume> from;
    if (!resume.is_none())
    {
        const py::dict previous = previous_output(resume, "detect", "models");
        from = ccd::Resume{
            header_from_python(previous["resume_header"].cast<py::dict>()),
            {record_from_python(previous["resume"].cast<py::dict>())},
            {segments_from_python(previous["models"].cast<py::list>())}};
    }

    ccd::RunSpec spec;
    spec.input = input;
    spec.algorithm = algorithm_or_ccdc(std::move(config));
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

        run = ccd::detect(spec, arrays.dates, arrays.spectral, arrays.qas,
                          from ? &*from : nullptr);
    }

    py::dict output;
    output["models"] = models_to_python(run.result.models);

    // The length of every model's band list, also when there are no models.
    output["num_bands"] = arrays.B;

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
// one contiguous uint8 array over the cube's pixels (not per segment): the
// run's CubeOutput::masks, which each pixel wrote its mask into, handed over
// without a copy:
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

static py::dict cube_columns(ccd::CubeOutput& cube, const CubeArrays& views, bool masks)
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
        // The cube's (H, W, T) masks become the array's memory, without a copy.
        auto* held = new std::vector<std::uint8_t>(std::move(cube.masks));
        const py::capsule owner(
            held, [](void* p) { delete static_cast<std::vector<std::uint8_t>*>(p); });
        output["processing_mask"] = py::array_t<std::uint8_t>(
            std::vector<py::ssize_t>{static_cast<py::ssize_t>(views.H),
                                     static_cast<py::ssize_t>(views.W),
                                     static_cast<py::ssize_t>(views.T)},
            held->data(), owner);
    }
    return output;
}

static py::dict detect_cube_wrapper(
    py::object dates,
    py::object spectral,
    py::object qas,
    const ccd::Input& input,
    std::shared_ptr<ccd::Algorithm> config,
    int threads,
    bool record,
    py::object resume,
    py::object trace_pixels,
    bool masks
)
{
    const CubeArrays views = cube_arrays(dates, spectral, qas, input.qa_table != nullptr);

    ccd::RunSpec spec;
    spec.input = input;
    spec.algorithm = algorithm_or_ccdc(std::move(config));
    spec.masks = masks;
    spec.record = record;

    std::optional<ccd::Resume> from;
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
        cube = ccd::detect_cube(spec, views.dates, views.spectral, views.qas, threads,
                                from ? &*from : nullptr);
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
        py::arg("input"),
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
        py::arg("input"),
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
