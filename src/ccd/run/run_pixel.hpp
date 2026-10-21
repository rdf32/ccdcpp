#pragma once

// Internal: the run behind detect() and detect_cube() (run/run.hpp).

#include <cstdint>
#include <memory>
#include <vector>

#include "ccd/core/array_view.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/processing_mask.hpp"
#include "../machine/detector.hpp"
#include "ccd/results/diagnostics.hpp"
#include "ccd/results/record.hpp"
#include "ccd/run/run.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

//==============================================================================
//
// Running CCD
//
// Every call goes through three objects:
//
//   Run      built once per call from a RunSpec, the dates and the input's
//            layout: everything checked (the input against the band count,
//            the dates, the input's QA table for raw QA, a resume header
//            against this run), and everything every pixel shares (the
//            algorithm resolved for the input with its plan, the resume header
//            and its capped shared values). Immutable: one Run serves every
//            thread.
//   Worker   one per thread: the buffers and the change detector a pixel
//            needs, reused from pixel to pixel.
//   Run::pixel<Value, Qa>(spectral, qa, at, worker)
//            one pixel through the steps below, in order.
//
// detect() and detect_cube() do this for one pixel and for a cube.
//
// The steps of one pixel:
//
//   decode_qa        raw QA through the QA table (or the codes as given)
//   mode             the QA statistics and the mode stage's choice: the
//                    pixel's procedure
//   values           the pixel's (B, T) as scalar_t, with its procedure's
//                    transform (read in place when nothing needs changing)
//   mask             its procedure's chain: the processing mask (and with
//                    RunSpec::reasons the preprocessing's reason codes)
//   fit              the change detector, or a resume of the stored run
//   finish           the record for the next resume, the reason codes of
//                    the kept observations no segment holds, the diagnostics
//
// Run::prepare() makes the first four, into a PreparedPixel; Run::fresh()
// (no resume) or Run::resumed() the last two.
//
//==============================================================================

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
    machine::Detector detector;

    // The pixel's (B, T) as scalar_t, when it has to be converted or its
    // thermal band transformed.
    std::vector<scalar_t> spectral;

    // The pixel's (T) QA codes, when its raw QA is decoded through a QaTable.
    std::vector<std::uint8_t> qas;
};

//------------------------------------------------------------------------------
//
// One pixel after the first four steps: its mode, its series (the dates, its
// values and its QA codes, in place or in the Worker's buffers) and its
// processing mask, which the fit takes, with its reason codes
// (RunSpec::reasons; empty otherwise).
//
//------------------------------------------------------------------------------
struct PreparedPixel
{
    PixelMode mode;
    PixelSeries series;
    ProcessingMask mask;
    std::vector<std::uint8_t> reasons;
};

//------------------------------------------------------------------------------
//
// One call's run.
//
// The constructor makes the checks detect() lists, before any pixel, in that
// order (std::invalid_argument).
//
// A Run reads the `spec`, the `dates` and the `resume` it was built with,
// without a copy: they must outlive it (a temporary spec is refused at
// compile time).
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
        const Resume* resume = nullptr);

    Run(RunSpec&& spec,
        ArrayView<const std::int64_t, 1> dates,
        index_t bands,
        index_t timesteps,
        index_t qa_timesteps,
        bool raw_qa,
        const Resume* resume = nullptr) = delete;

    // The call's header: digest, series end, statistics cutoff (for
    // a resume the stored run's), with RunSpec::record or a resume.
    const ResumeHeader& header() const { return header_; }

    bool resuming() const { return resume_ != nullptr; }

    // Where the pixel at flat index `at` of a cube reports: the spec's
    // observer when it observes that pixel.
    Observe observed(index_t at) const;

    const RunSpec& spec() const { return spec_; }

    // One pixel through the steps of the file comment, for one pair of
    // element types (instantiated in run.cpp for every supported pair): the
    // pixel at flat index `at` of the resume's records and segments, which
    // must hold it (detect and detect_cube check their sizes); without a
    // resume `at` is not read. The input must have the run's band and
    // timestep counts, and raw (uint16) QA only when the Run was built for it
    // (detect and detect_cube see to both; std::invalid_argument otherwise),
    // and a resumed pixel at least its record's final segments
    // (std::invalid_argument). The output's header is left to the caller
    // (header()).
    //
    // A resume continues the pixel from its record when its input up to the
    // stored series end is the stored run's and the machine stopped where it
    // can continue; otherwise the pixel is rerun under the stored statistics
    // cutoff, and the status says why. Either way the result equals the
    // pixel's run under that cutoff, whole output.
    template<typename Value, typename Qa>
    PixelOutput pixel(ArrayView<const Value, 2> spectral, ArrayView<const Qa, 1> qa,
                      index_t at, Worker& worker) const;

private:

    // The first four steps on one pixel, into the worker's buffers.
    template<typename Value, typename Qa>
    PreparedPixel prepare(ArrayView<const Value, 2> spectral, ArrayView<const Qa, 1> qa,
                          Worker& worker) const;

    // The fit of a run without a resume and, with RunSpec::record, the
    // record's suspension and drops.
    PixelOutput fresh(PreparedPixel& pixel, Worker& worker) const;

    // The fit of the resume's pixel at flat index `at`, whose input up to the
    // stored series end has `fingerprint`: the stored run continued, or the
    // pixel rerun. The status, the result and the record's suspension and
    // drops (run/resume.cpp).
    PixelOutput resumed(PreparedPixel& pixel, Worker& worker, index_t at,
                        std::uint64_t fingerprint) const;

    const RunSpec& spec_;
    ArrayView<const std::int64_t, 1> dates_;
    index_t bands_;
    index_t timesteps_;
    bool raw_qa_;

    // The algorithm resolved for the input: its stages, shared values and
    // strategies, shared by every thread.
    std::shared_ptr<const config::Resolved> resolved_;

    // What the pixels run: the shared values and plan, with every
    // statistics cutoff capped at header_.statistics_end for a resume.
    config::Shared shared_;
    preprocess::PreprocessPlan run_plan_;

    ResumeHeader header_;

    // The stored call, for a resume.
    const Resume* resume_;
};

} // namespace ccd
