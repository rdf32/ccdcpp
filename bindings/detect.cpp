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
#include "results.hpp"

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
            {models_from_python(previous["models"].cast<py::list>())}};
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
        from = ccd::Resume{
            header_from_python(previous["resume_header"].cast<py::dict>()),
            records_from_columns(previous["resume"].cast<py::dict>(), views.H, views.W, views.B),
            segments_from_columns(previous, views.H, views.W, views.B)};
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

    py::dict output = segment_columns(cube, views.H, views.W, views.B, views.T, masks);
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
