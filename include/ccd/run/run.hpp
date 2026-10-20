#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "ccd/core/array_view.hpp"
#include "ccd/strategies/register_core.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/input/harmonic.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/machine/standard.hpp"
#include "ccd/input/qa_table.hpp"
#include "ccd/results/record.hpp"
#include "ccd/results/diagnostics.hpp"
#include "ccd/regression/lasso_solver.hpp"

namespace ccd
{

//==============================================================================
//
// Running CCD
//
// Every call goes through three objects:
//
//   Run      built once per call from a RunSpec, the dates and the input's
//            layout: everything checked (the options against the band count,
//            the dates, the plan, a QA table for raw QA, a resume header
//            against this run), and everything every pixel shares (the plan,
//            the resume header and its capped options). Immutable: one Run
//            serves every thread.
//   Worker   one per thread: the buffers and the change detector a pixel
//            needs, reused from pixel to pixel.
//   Run::pixel(input, worker)
//            one pixel through the steps below, in order.
//
// run_pixel() and run_cube() do this for one pixel and for a cube; they are
// the whole entry point.
//
// The steps of one pixel:
//
//   decode_qa        raw QA through the QA table (or the codes as given)
//   choose_procedure the QA statistics and the pixel's procedure
//   values           the pixel's (B, T) as scalar_t, with its procedure's
//                    transform (read in place when nothing needs changing)
//   mask             its procedure's chain: the processing mask
//   fit              the change detector, or a resume of the stored run
//   finish           the record for the next resume
//
//==============================================================================

// One pixel's values, (B, T), and QA, (T), in any supported element type.
using PixelSpectral = std::variant<
    ArrayView<const double, 2>,
    ArrayView<const float, 2>,
    ArrayView<const std::uint16_t, 2>>;

using PixelQa = std::variant<
    ArrayView<const std::uint8_t, 1>,
    ArrayView<const std::uint16_t, 1>>;

// A cube's values, (H, W, B, T), and QA, (H, W, T).
using CubeSpectral = std::variant<
    ArrayView<const double, 4>,
    ArrayView<const float, 4>,
    ArrayView<const std::uint16_t, 4>>;

using CubeQa = std::variant<
    ArrayView<const std::uint8_t, 3>,
    ArrayView<const std::uint16_t, 3>>;

//------------------------------------------------------------------------------
//
// What a call runs.
//
//   hoptions            the input's description (validate() checks it)
//   config              the algorithm: one strategy per stage by name, with
//                       its parameters (config/registry.hpp), in the
//                       library's registry with its extensions; the "ccdc"
//                       preset by default (Registry::preset()). It is checked
//                       when the run is built (names, parameters,
//                       capabilities, band lists against the input's bands)
//                       and is part of the resume digest
//   qa_table            decodes raw QA; null: `qas` are the codes. Required
//                       for uint16 QA.
//   plan                the preprocessing plan; null: default_plan(hoptions)
//   record              also return each pixel's resume record and the
//                       call's header
//   keep_masks          false drops each pixel's processing mask (T bytes
//                       per pixel) once it is done; FitResult::mask is empty
//   observer, observe   where the observed pixels report each step
//                       (results/diagnostics.hpp): run_pixel's pixel, or
//                       run_cube's pixels at the flat indices `observe`
//                       (row * W + col; empty: every pixel). With
//                       CCD_OBSERVE=OFF nothing is reported
//
// A resume is an input of the call (run_pixel's and run_cube's `resume`),
// not part of what it runs.
//
//------------------------------------------------------------------------------
struct RunSpec
{
    HarmonicOptions hoptions;
    config::ConfigTree config = strategies::library_registry().preset("ccdc");
    QaTablePtr qa_table = nullptr;
    PreprocessPlanPtr plan = nullptr;
    bool record = false;
    bool keep_masks = true;
    std::shared_ptr<Observer> observer = nullptr;
    std::vector<index_t> observe;
};

// One pixel's stored run, for a resume: its record and its stored segments
// (of which the first record.standard.final_segments are kept as given).
struct PixelPrior
{
    const PixelRecord* record = nullptr;
    const std::vector<ChangeModel>* segments = nullptr;
};

// A stored pixel run to continue: the stored call's header and the pixel's
// record and segments.
struct PixelResume
{
    ResumeHeader header;
    PixelRecord record;
    std::vector<ChangeModel> segments;
};

// One pixel's outcome: its result, and with RunSpec::record or a resume its
// record for the next resume and what was done.
struct PixelOutcome
{
    FitResult result;
    PixelRecord record;
    ResumeStatus status = ResumeStatus::Resumed;
};

struct PixelResult
{
    index_t row;
    index_t col;
    FitResult result;

    // The pixel's resume record and what a resume did with it: filled with
    // RunSpec::record and for a resume.
    PixelRecord record;
    ResumeStatus status = ResumeStatus::Resumed;
};

//==============================================================================
//
// One pixel that threw.
//
// An exception cannot propagate out of an OpenMP parallel region -- the
// runtime terminates the process. So a single bad time series would take
// down a whole cube, losing every other pixel's result and printing nothing
// useful. run_cube catches per pixel instead and records the failure here.
//
// The message is std::exception::what() copied at catch time. Copying a string
// inside the region is safe; letting the exception escape is not.
//
//==============================================================================

struct CubeFailure
{
    index_t row;
    index_t col;
    std::string message;
};

//==============================================================================
//
// Everything run_cube learned about one cube.
//
// `failures` is deliberately not a bare count: a caller reporting "12,000
// pixels failed" wants to say WHY, and the addresses are what make a failure
// reproducible on a single pixel. It is empty in the normal case, so the extra
// member costs nothing.
//
// `threads` is the thread count actually used, which is not necessarily the
// one requested -- the OpenMP runtime may cap it. Recording what ran, rather
// than what was asked for, is what makes a timing comparable later.
//
// `pixels` and `failures` are in raster order (row-major over H x W),
// independent of the thread count and schedule.
//
//==============================================================================

struct CubeResult
{
    std::vector<PixelResult> pixels;
    std::vector<CubeFailure> failures;
    int threads = 0;

    // The call's resume header (RunSpec::record, or a resume).
    ResumeHeader header;
};

//------------------------------------------------------------------------------
//
// A pixel's buffers and change detector, reused from one pixel to the next.
//
// Every buffer in here is rewritten before it is read for each pixel, so a
// thread that keeps one across pixels gets the same results as a fresh one
// per pixel, without reallocating.
//
//------------------------------------------------------------------------------
struct Worker
{
    // Where the pixel being run reports (the run sets it per pixel).
    Observe observe;

    // The run's change detector, with its per-pixel record.
    standard::Detector standard;

    // The pixel's (B, T) as scalar_t, when it has to be converted or its
    // thermal band transformed.
    std::vector<scalar_t> spectral;

    // The pixel's (T) QA codes, when its raw QA is decoded through a QaTable.
    std::vector<std::uint8_t> qas;
};

//------------------------------------------------------------------------------
//
// One call's run.
//
// The constructor checks, before any pixel, and throws std::invalid_argument,
// in this order, on: options that fail validate() for `bands`; `dates` or the
// QA without `timesteps` entries; dates that fail validate_dates(); raw
// (uint16) QA without a QA table; a plan that fails preprocess::validate();
// and for a resume (`resume` not null), a header of another record format or
// results version, made with other options, plan or QA table (by
// options_digest()), or for a series that ends before the stored one.
//
// pixel() throws std::invalid_argument for a resume on a prior with fewer
// stored segments than its record's final ones.
//
//------------------------------------------------------------------------------
class Run
{
public:

    Run(const RunSpec& spec,
        ArrayView<const std::int64_t, 1> dates,
        index_t bands,
        index_t timesteps,
        index_t qa_timesteps,
        bool raw_qa,
        const ResumeHeader* resume = nullptr);

    // The call's header: options digest, series end, statistics cutoff (for
    // a resume the stored run's), with RunSpec::record or a resume.
    const ResumeHeader& header() const { return header_; }

    bool resuming() const { return resume_.has_value(); }

    // Where the pixel at flat index `at` of a cube reports: the spec's
    // observer when it observes that pixel.
    Observe observed(index_t at) const;

    const RunSpec& spec() const { return spec_; }

    // One pixel through the steps of the file comment. Without a resume
    // `prior` is ignored; with one it is required (std::invalid_argument
    // without it). The input must have the run's band and timestep counts,
    // and raw (uint16) QA only when the Run was built for it (run_pixel and
    // run_cube see to both; std::invalid_argument otherwise).
    //
    // A Run reads the `dates` it was built with on every pixel: they must
    // outlive it.
    //
    // A resume continues the pixel from its record when its input up to the
    // stored series end is the stored run's and the machine stopped where it
    // can continue; otherwise the pixel is rerun under the stored statistics
    // cutoff, and the status says why. Either way the result equals the
    // pixel's run under that cutoff, whole output.
    PixelOutcome pixel(const PixelSpectral& spectral, const PixelQa& qa,
                       const PixelPrior& prior, Worker& worker) const;

private:

    // pixel() for a known element type, as the cube loop calls it (instantiated
    // in run.cpp for every supported pair).
    template<typename Value, typename Qa>
    PixelOutcome pixel_typed(ArrayView<const Value, 2> spectral, ArrayView<const Qa, 1> qa,
                             const PixelPrior& prior, Worker& worker) const;

    template<typename Value, typename Qa>
    friend CubeResult cube_loop(const Run&, ArrayView<const Value, 4>, ArrayView<const Qa, 3>,
                                int, const CubeResumeFrom*);


    RunSpec spec_;
    ArrayView<const std::int64_t, 1> dates_;
    index_t bands_;
    index_t timesteps_;
    bool raw_qa_;

    // The plan the call runs: spec.plan, or default_plan(hoptions).
    preprocess::PreprocessPlan plan_;

    // What the pixels run: the options and plan, with every statistics
    // cutoff capped at header_.statistics_end for a resume.
    HarmonicOptions options_;
    preprocess::PreprocessPlan run_plan_;

    ResumeHeader header_;

    // The stored call's header, for a resume.
    std::optional<ResumeHeader> resume_;

    // One strategy per stage, shared by every thread.
    Strategies strategies_;
};

//------------------------------------------------------------------------------
//
// CCD on one pixel: `spectral` (B, T), `qas` (T), `dates` (T).
//
// `spectral` is read, never written: the plan's transform (by default the
// Standard procedure's thermal transform, HarmonicOptions::THERMAL_SCALE/
// THERMAL_OFFSET) is applied to a private copy. Any strides. The QA is the
// codes (uint8) or raw values decoded through RunSpec::qa_table (uint8 or
// uint16). Throws what Run and Run::pixel throw.
//
// With `resume`, continues that stored run on the whole of a longer series
// (Run::pixel): the stored run's input with the observations after its series
// end.
//
// The returned header is the run's (RunSpec::record or a resume).
//
//------------------------------------------------------------------------------
struct PixelRun
{
    FitResult result;
    PixelRecord record;
    ResumeStatus status = ResumeStatus::Resumed;
    ResumeHeader header;
};

PixelRun run_pixel(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const PixelSpectral& spectral,
    const PixelQa& qas,
    const PixelResume* resume = nullptr
);

inline PixelRun run_pixel(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const PixelSpectral& spectral,
    const PixelQa& qas,
    const PixelResume& resume
)
{
    return run_pixel(spec, dates, spectral, qas, &resume);
}

//------------------------------------------------------------------------------
//
// CCD on every pixel of a cube, in parallel over H*W.
//
// `threads` is the OpenMP thread count. 0 means "whatever the runtime would
// choose", omp_get_max_threads(). Passing it explicitly matters because
// omp_get_max_threads() is fixed from OMP_NUM_THREADS at library load, so a
// caller embedding this in a scheduler cannot otherwise honour its own
// allocation without setting an environment variable before the shared
// library is imported. A build without OpenMP runs one thread whatever the
// count, and CubeResult::threads says 1.
//
// This never throws on account of a pixel; see CubeResult::failures. It
// throws, before any pixel runs, std::runtime_error on a QA cube that is not
// (H, W, T), and std::invalid_argument on what Run throws on and for a resume
// on records or segment lists that are not H x W.
//
// The cube is read, never written, in any strides, as float64, float32 or
// uint16. Each pixel is converted to scalar_t in a per-thread buffer --
// static_cast, which is exact for all three -- so a uint16 or float32 cube
// gives the results of the same values widened to float64 beforehand, without
// a float64 copy of the cube. Raw QA is decoded per pixel the same way. The
// element types are dispatched once per call.
//
// `resume`, the stored call: its header and each pixel's record and stored
// segments in raster order. Every pixel's result is then the complete one,
// with its record and status; CubeResult::header is this call's. A pixel with
// fewer stored segments than its record's final ones is a CubeFailure.
//
//------------------------------------------------------------------------------
CubeResult run_cube(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const CubeSpectral& spectral,
    const CubeQa& qas,
    int threads = 0,
    const CubeResumeFrom* resume = nullptr
);

} // namespace ccd
