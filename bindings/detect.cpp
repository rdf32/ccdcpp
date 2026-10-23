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
    const ccd::DataSpec& spec,
    std::shared_ptr<ccd::Algorithm> config,
    bool record,
    bool masks,
    bool reasons,
    bool diagnostics
)
{
    ccd::RunSpec out;
    out.spec = spec;
    out.algorithm = algorithm_or_ccdc(std::move(config));
    out.record = record;
    out.masks = masks;
    out.reasons = reasons;
    out.diagnostics = diagnostics;
    return out;
}

// T bytes as a new (T,) uint8 array.
py::array_t<std::uint8_t> bytes_array(const std::vector<std::uint8_t>& bytes)
{
    py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(bytes.size()));
    std::copy(bytes.begin(), bytes.end(), out.mutable_data());
    return out;
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
// list, also when there are none) and the processing mask; the reason codes
// with reasons=, the diagnostics with diagnostics=, the record with record=
// or a resume, the status with a resume, the trace with trace=.
py::dict pixel_to_python(
    const ccd::PixelOutput& run,
    ccd::index_t B,
    bool reasons,
    bool diagnostics,
    bool record,
    bool resumed,
    const RecordingObserver* recording
)
{
    py::dict output;
    output["models"] = models_to_python(run.result.models);
    output["num_bands"] = B;
    output["processing_mask"] = bytes_array(run.result.mask.data());
    if (reasons)
        output["mask_reasons"] = bytes_array(run.reasons);
    if (diagnostics)
    {
        output["pixel_diagnostics"] = pixel_diagnostics_to_python(run.diagnostics);
        output["segment_diagnostics"] =
            segment_diagnostics_columns(run.diagnostics.segments);
    }

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

// detect_cube()'s output: the segment columns (with the masks and reason
// codes asked for); the diagnostics with diagnostics=, the record columns
// with record= or a resume, the counts with a resume, the traces with
// trace_pixels=.
py::dict cube_to_python(
    ccd::CubeOutput& cube,
    const CubeArrays& arrays,
    bool record,
    bool resumed,
    bool masks,
    bool reasons,
    bool diagnostics,
    const Traced& traced
)
{
    py::dict output =
        segment_columns(cube, arrays.H, arrays.W, arrays.B, arrays.T, masks, reasons);
    if (diagnostics)
        add_cube_diagnostics(output, cube, arrays.H, arrays.W);
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
    const ccd::DataSpec& spec,
    std::shared_ptr<ccd::Algorithm> config,
    bool record,
    py::object resume,
    bool freeze_statistics,
    bool trace,
    bool reasons,
    bool diagnostics
)
{
    const PixelArrays arrays = pixel_arrays(dates, spectral, qas, spec.qa_table != nullptr);
    std::optional<ccd::Resume> from = pixel_resume(resume);
    if (from)
        from->freeze_statistics = freeze_statistics;
    ccd::RunSpec call = run_spec(spec, std::move(config), record, true, reasons,
                                 diagnostics);   // with its mask
    const auto recording = trace ? std::make_shared<RecordingObserver>() : nullptr;
    call.observer = recording;

    ccd::PixelOutput run;
    {
        py::gil_scoped_release release;
        run = ccd::detect(call, arrays.dates, arrays.spectral, arrays.qas,
                          from ? &*from : nullptr);
    }
    return pixel_to_python(run, arrays.B, reasons, diagnostics, record, from.has_value(),
                           recording.get());
}

static py::dict detect_cube_wrapper(
    py::object dates,
    py::object spectral,
    py::object qas,
    const ccd::DataSpec& spec,
    std::shared_ptr<ccd::Algorithm> config,
    int threads,
    bool record,
    py::object resume,
    bool freeze_statistics,
    py::object trace_pixels,
    bool masks,
    bool reasons,
    bool diagnostics
)
{
    const CubeArrays arrays = cube_arrays(dates, spectral, qas, spec.qa_table != nullptr);
    ccd::RunSpec call = run_spec(spec, std::move(config), record, masks, reasons, diagnostics);
    std::optional<ccd::Resume> from = cube_resume(resume, arrays.H, arrays.W, arrays.B);
    if (from)
        from->freeze_statistics = freeze_statistics;
    const Traced traced = observe(call, trace_pixels, arrays.H, arrays.W);

    ccd::CubeOutput cube;
    {
        py::gil_scoped_release release;
        cube = ccd::detect_cube(call, arrays.dates, arrays.spectral, arrays.qas, threads,
                                from ? &*from : nullptr);
    }
    return cube_to_python(cube, arrays, record, from.has_value(), masks, reasons, diagnostics,
                          traced);
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
        py::arg("freeze_statistics") = false,
        py::arg("trace") = false,
        py::arg("reasons") = false,
        py::arg("diagnostics") = false,
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
        py::arg("freeze_statistics") = false,
        py::arg("trace_pixels") = py::none(),
        py::arg("masks") = false,
        py::arg("reasons") = false,
        py::arg("diagnostics") = false,
        "CCD on every pixel of a cube, as columns (see ccdcpp.detect_cube)."
    );
    py::tuple reasons(ccd::REASON_COUNT);
    for (std::uint8_t code = 0; code < ccd::REASON_COUNT; ++code)
        reasons[code] = ccd::name(static_cast<ccd::Reason>(code));
    m.attr("REASONS") = reasons;
    m.attr("FIRST_DROP") = static_cast<std::uint8_t>(ccd::FIRST_DROP);
    py::tuple exits(ccd::machine::EXIT_COUNT);
    for (std::uint8_t code = 0; code < ccd::machine::EXIT_COUNT; ++code)
        exits[code] = ccd::machine::name(static_cast<ccd::machine::Exit>(code));
    m.attr("EXITS") = exits;
    m.def("resume_schema", &resume_schema);
    m.attr("RESUME_FORMAT") = ccd::RESUME_FORMAT;
    m.attr("RESULTS_VERSION") = ccd::RESULTS_VERSION;
}
