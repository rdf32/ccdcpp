#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "ccd/core/array_view.hpp"
#include "ccd/input/input.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/results/diagnostics.hpp"
#include "ccd/results/record.hpp"
#include "ccd/run/algorithm.hpp"

namespace ccd
{

//==============================================================================
//
// Running CCD: the two entry points, detect() for one pixel and
// detect_cube() for a cube, with what a call runs (RunSpec) and what it
// returns and resumes from.
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
//   input               how to read the data, its QA table included
//                       (validate() checks it)
//   algorithm           the algorithm (run/algorithm.hpp): a configuration
//                       checked when the Algorithm was built, resolved
//                       against `input` once per Algorithm and input; the
//                       "ccdc" preset's by default. Part of the resume digest
//   record              also return each pixel's resume record and the
//                       call's header
//   masks               also return each pixel's processing mask (T bytes
//                       per pixel): detect's FitResult::mask, detect_cube's
//                       CubeOutput::masks; without it the mask is dropped
//                       once the pixel is done
//   observer, observe   where the observed pixels report each step
//                       (results/diagnostics.hpp): detect's pixel, or
//                       detect_cube's pixels at the flat indices `observe`
//                       (row * W + col; empty: every pixel). With
//                       CCD_OBSERVE=OFF nothing is reported
//
// A resume is an input of the call (detect's and detect_cube's `resume`),
// not part of what it runs.
//
//------------------------------------------------------------------------------
struct RunSpec
{
    Input input;
    AlgorithmPtr algorithm = Algorithm::preset("ccdc");
    bool record = false;
    bool masks = false;
    std::shared_ptr<Observer> observer = nullptr;
    std::vector<index_t> observe;
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
// useful. detect_cube catches per pixel instead and records the failure here.
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
// Everything detect_cube learned about one cube.
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
// With RunSpec::masks, `masks` holds every pixel's processing mask as one
// (H, W, T) array, row-major: each pixel's T bytes are written into it as the
// pixel finishes, and its PixelResult::result.mask stays empty, so the masks
// take T bytes per pixel once. A failed pixel's are zeros. Without
// RunSpec::masks `masks` is empty.
//
//==============================================================================

struct CubeOutput
{
    std::vector<PixelResult> pixels;
    std::vector<CubeFailure> failures;
    int threads = 0;

    // The cube's shape: H, W and T.
    index_t height = 0;
    index_t width = 0;
    index_t timesteps = 0;

    // Every pixel's processing mask, (H, W, T), with RunSpec::masks.
    std::vector<std::uint8_t> masks;

    // The processing mask of pixel (row, col): T values of `masks`.
    ArrayView<const std::uint8_t, 1> mask(index_t row, index_t col) const
    {
        return ArrayView<const std::uint8_t, 1>::contiguous(
            masks.data() + (row * width + col) * timesteps, {timesteps});
    }

    // The call's resume header (RunSpec::record, or a resume).
    ResumeHeader header;
};

//------------------------------------------------------------------------------
//
// CCD on one pixel: `spectral` (B, T), `qas` (T), `dates` (T).
//
// `spectral` is read, never written: the plan's transform (in "ccdc" the
// Standard procedure's thermal transform, Input::thermal_scale/
// thermal_offset) is applied to a private copy. Any strides. The QA is the
// codes (uint8) or raw values decoded through the input's qa_table (uint8 or
// uint16).
//
// Before the pixel runs, it throws std::invalid_argument, in this order, on:
// an input that fails validate() for the spectral's band count; `dates` or
// the QA without the spectral's timestep count; dates that fail
// validate_dates(); raw (uint16) QA without the input's QA table; a
// configuration the input does not fit (Algorithm::for_input(), the plan's
// validate() included); and for a resume, a header of another record format
// or results version, made with another input (its QA table included) or
// configuration (by Resolved::digest), or for a series that ends before the
// stored one. Then on a resume of other than one record and one segment
// list, and on one with fewer stored segments than its record's final ones.
//
// With `resume`, continues that stored pixel run on the whole of a longer
// series: the stored run's input with the observations after its series
// end. The
// pixel continues from its record when its input up to the stored series end
// is the stored run's and the machine stopped where it can continue;
// otherwise it is rerun under the stored statistics cutoff, and the status
// says why. Either way the result equals the pixel's run under that cutoff,
// whole output.
//
// The returned header is the run's (RunSpec::record or a resume).
//
//------------------------------------------------------------------------------
struct PixelOutput
{
    FitResult result;
    PixelRecord record;
    ResumeStatus status = ResumeStatus::Resumed;
    ResumeHeader header;
};

PixelOutput detect(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const PixelSpectral& spectral,
    const PixelQa& qas,
    const Resume* resume = nullptr
);

inline PixelOutput detect(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const PixelSpectral& spectral,
    const PixelQa& qas,
    const Resume& resume
)
{
    return detect(spec, dates, spectral, qas, &resume);
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
// count, and CubeOutput::threads says 1.
//
// This never throws on account of a pixel; see CubeOutput::failures. It
// throws, before any pixel runs, std::runtime_error on a QA cube that is not
// (H, W, T), and std::invalid_argument on what detect() throws on before its
// pixel runs and for a resume on records or segment lists that are not H x W.
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
// with its record and status; CubeOutput::header is this call's. A pixel with
// fewer stored segments than its record's final ones is a CubeFailure.
//
//------------------------------------------------------------------------------
CubeOutput detect_cube(
    const RunSpec& spec,
    ArrayView<const std::int64_t, 1> dates,
    const CubeSpectral& spectral,
    const CubeQa& qas,
    int threads = 0,
    const Resume* resume = nullptr
);

} // namespace ccd
