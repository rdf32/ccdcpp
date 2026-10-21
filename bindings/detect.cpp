#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ccd/ccd.hpp"
#include "ccd/results/diagnostics.hpp"

#include "arrays.hpp"
#include "config.hpp"
#include "results.hpp"

namespace py = pybind11;

namespace
{

//------------------------------------------------------------------------------
// The events of observed pixels, recorded for Python (trace=, trace_pixels=).
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
        const auto it = events_.find(pixel);
        return it == events_.end() ? py::list() : trace_rows(it->second);
    }

private:
    std::mutex mutex_;
    std::map<ccd::index_t, std::vector<ccd::ObserverEvent>> events_;
};

// What a call runs: the input and config= (the "ccdc" preset's for None).
ccd::RunSpec run_spec(
    const ccd::Input& input,
    std::shared_ptr<ccd::Algorithm> config,
    bool record,
    bool masks
)
{
    ccd::RunSpec spec;
    spec.input = input;
    spec.algorithm = algorithm_or_ccdc(std::move(config));
    spec.record = record;
    spec.masks = masks;
    return spec;
}

// trace_pixels=: the (row, col) pairs of the cube to observe, each refused
// unless it is a pixel of the H x W cube; with any, `spec` observes them into
// `recording`.
struct Traced
{
    std::vector<std::pair<ccd::index_t, ccd::index_t>> pixels;
    std::shared_ptr<RecordingObserver> recording;
};

Traced observe(ccd::RunSpec& spec, const py::object& trace_pixels, ccd::index_t H, ccd::index_t W)
{
    Traced traced;
    if (trace_pixels.is_none())
        return traced;
    for (const py::handle& rc : trace_pixels)
    {
        const auto pair = rc.cast<std::pair<std::int64_t, std::int64_t>>();
        if (pair.first < 0 || pair.second < 0 || static_cast<ccd::index_t>(pair.first) >= H ||
            static_cast<ccd::index_t>(pair.second) >= W)
            throw py::value_error("trace_pixels: (" + std::to_string(pair.first) + ", " +
                                  std::to_string(pair.second) + ") is not a pixel of the cube");
        traced.pixels.emplace_back(pair.first, pair.second);
        spec.observe.push_back(static_cast<ccd::index_t>(pair.first) * W +
                               static_cast<ccd::index_t>(pair.second));
    }
    if (!traced.pixels.empty())
    {
        traced.recording = std::make_shared<RecordingObserver>();
        spec.observer = traced.recording;
    }
    return traced;
}

// detect()'s output: the models, num_bands (the length of every model's band
// list, also when there are none) and the processing mask; the record with
// record= or a resume, the status with a resume, the trace with trace=.
py::dict pixel_to_python(
    const ccd::PixelOutput& run,
    ccd::index_t B,
    bool record,
    bool resumed,
    const RecordingObserver* recording
)
{
    py::dict output;
    output["models"] = models_to_python(run.result.models);
    output["num_bands"] = B;

    const std::vector<std::uint8_t>& mask = run.result.mask.data();
    py::array_t<std::uint8_t> mask_array(static_cast<py::ssize_t>(mask.size()));
    std::copy(mask.begin(), mask.end(), mask_array.mutable_data());
    output["processing_mask"] = mask_array;

    if (record || resumed)
    {
        output["resume"] = record_to_python(run.record);
        output["resume_header"] = header_to_python(run.header);
    }
    if (resumed)
        output["status"] =
            run.status == ccd::ResumeStatus::Resumed ? "resumed" : "rerun_input_changed";
    if (recording)
        output["trace"] = recording->rows(0);
    return output;
}

// detect_cube()'s output: the segment columns; the record columns with
// record= or a resume, the counts with a resume, the traces with
// trace_pixels=.
py::dict cube_to_python(
    ccd::CubeOutput& cube,
    const CubeArrays& arrays,
    bool record,
    bool resumed,
    bool masks,
    const Traced& traced
)
{
    py::dict output = segment_columns(cube, arrays.H, arrays.W, arrays.B, arrays.T, masks);
    if (record || resumed)
    {
        output["resume"] = record_columns(cube, arrays.H, arrays.W, arrays.B, resumed);
        output["resume_header"] = header_to_python(cube.header);
    }
    if (resumed)
    {
        std::int64_t done = 0, rerun = 0;
        for (const auto& px : cube.pixels)
            (px.status == ccd::ResumeStatus::Resumed ? done : rerun) += 1;
        py::dict counts;
        counts["resumed"] = done;
        counts["rerun_input_changed"] = rerun;
        counts["failed"] = static_cast<std::int64_t>(cube.failures.size());
        output["counts"] = counts;
    }
    if (traced.recording)
    {
        py::dict traces;
        for (const auto& [row, col] : traced.pixels)
            traces[py::make_tuple(row, col)] = traced.recording->rows(row * arrays.W + col);
        output["traces"] = traces;
    }
    return output;
}

} // namespace

//------------------------------------------------------------------------------
// The two calls: the arrays read where they are (arrays.hpp), the run without
// the GIL, the output as Python data (results.hpp).
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
    const std::optional<ccd::Resume> from = pixel_resume(resume);
    ccd::RunSpec spec = run_spec(input, std::move(config), record, true);   // with its mask
    const auto recording = trace ? std::make_shared<RecordingObserver>() : nullptr;
    spec.observer = recording;

    ccd::PixelOutput run;
    {
        py::gil_scoped_release release;
        run = ccd::detect(spec, arrays.dates, arrays.spectral, arrays.qas,
                          from ? &*from : nullptr);
    }
    return pixel_to_python(run, arrays.B, record, from.has_value(), recording.get());
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
    const CubeArrays arrays = cube_arrays(dates, spectral, qas, input.qa_table != nullptr);
    ccd::RunSpec spec = run_spec(input, std::move(config), record, masks);
    const std::optional<ccd::Resume> from = cube_resume(resume, arrays.H, arrays.W, arrays.B);
    const Traced traced = observe(spec, trace_pixels, arrays.H, arrays.W);

    ccd::CubeOutput cube;
    {
        py::gil_scoped_release release;
        cube = ccd::detect_cube(spec, arrays.dates, arrays.spectral, arrays.qas, threads,
                                from ? &*from : nullptr);
    }
    return cube_to_python(cube, arrays, record, from.has_value(), masks, traced);
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
