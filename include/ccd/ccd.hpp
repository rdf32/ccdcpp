
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/ccd.hpp"

#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/standard.hpp"
#include "ccd/qa_table.hpp"
#include "ccd/record.hpp"
#include "ccd/preprocess/preprocess.hpp"

namespace ccd
{

struct PixelResult
{
    index_t row;
    index_t col;
    FitResult result;

    // The pixel's resume record and what a resume did with it: filled by
    // detect_cube(record = true) and resume_cube().
    PixelRecord record;
    ResumeStatus status = ResumeStatus::Resumed;
};

//==============================================================================
//
// One pixel that threw.
//
// An exception cannot propagate out of an OpenMP parallel region -- the
// runtime terminates the process. So a single bad time series used to take
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
//==============================================================================

// `pixels` and `failures` are in raster order (row-major over H x W),
// independent of the thread count and schedule.
struct CubeResult
{
    std::vector<PixelResult> pixels;
    std::vector<CubeFailure> failures;
    int threads = 0;

    // The call's resume header, with the pixels' records (detect_cube(record
    // = true), resume_cube()).
    ResumeHeader header;
};

using Clock = std::chrono::high_resolution_clock;

inline auto elapsed_ms(
    Clock::time_point start,
    Clock::time_point end
)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

inline void print_change_model(const ChangeModel& model)
{
    std::cout << "\n========================================\n";
    std::cout << "ChangeModel\n";
    std::cout << "========================================\n";

    std::cout << "Start Day          : " << model.start_day << '\n';
    std::cout << "End Day            : " << model.end_day << '\n';
    std::cout << "Break Day          : " << model.break_day << '\n';
    std::cout << "Observation Count  : " << model.observation_count << '\n';
    std::cout << "Change Probability : " << model.change_probability << '\n';
    std::cout << "Curve QA           : " << static_cast<int>(model.curve_qa) << '\n';

    std::cout << "\nSpectral Models\n";
    std::cout << "----------------------------------------\n";

    for (std::size_t band = 0; band < model.bands.size(); ++band)
    {
        const BandFit& fit = model.bands[band];
        const auto& coef = fit.coefficients;

        std::cout << "Band " << band << '\n';
        std::cout << "  Iterations : " << fit.iterations << '\n';
        std::cout << "  Intercept  : " << fit.intercept << '\n';
        std::cout << "  RMSE       : " << fit.rmse << '\n';
        std::cout << "  Magnitude  : " << fit.magnitude << '\n';

        std::cout << "  Coefficients (" << coef.size() << "): ";

        for (std::size_t i = 0; i < coef.size(); ++i)
        {
            std::cout << coef[i];

            if (i + 1 != coef.size())
                std::cout << ", ";
        }
        std::cout << "\n";

        std::cout << "\n\n";
    }
    std::cout << "========================================\n";
}

inline void print_change_models(
    const std::vector<ccd::ChangeModel>& models
)
{
    for(std::size_t i = 0;
        i < models.size();
        ++i)
    {
        std::cout << "\n######## Change Model "
                  << i + 1
                  << " ########\n";

        print_change_model(
            models[i]
        );
    }
}

//------------------------------------------------------------------------------
//
// Scratch detect() reuses from one pixel to the next.
//
// Every buffer in here is rewritten before it is read for each pixel, so a
// thread that keeps one of these across pixels gets the same results as a
// fresh one per pixel, without reallocating the buffers.
//
//------------------------------------------------------------------------------

struct DetectScratch
{
    LassoWorkspace lasso{0};

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
// The options check every entry point runs once per call, before any work.
//
// The procedures hold one fit per band and read bands by the indices in
// `options`, so every index must name a row of an array with `bands` bands.
// Throws std::invalid_argument when:
//
//   * `bands` is not options.NUM_BANDS, or exceeds CCD_MAX_BANDS;
//   * DETECTION_BANDS or TMASK_BANDS is empty;
//   * DETECTION_BANDS, TMASK_BANDS or SATURATION_BANDS holds an index not
//     below `bands`, or holds one index twice;
//   * GREEN_IDX is NO_BAND or not below `bands`;
//   * THERMAL_IDX is neither NO_BAND nor below `bands`, or is in
//     SATURATION_BANDS.
//
// and the cadence and change-probability options when:
//
//   * PEEK_REFERENCE_DAYS or VARIOGRAM_MIN_LAG_DAYS is not greater than 0;
//   * RETRAIN_MIN_OBS or SEASONAL_RMSE_COUNT is below 1;
//   * RETRAIN_SPAN_FACTOR or SEASONAL_RMSE_DIVISOR is not greater than 0;
//   * CHANGE_PROBABILITY is not in (0, 1);
//
// and the coefficient counts when:
//
//   * COEFFICIENT_MIN is below 1;
//   * COEFFICIENT_MIN, COEFFICIENT_MID, COEFFICIENT_MAX decrease;
//   * COEFFICIENT_MAX is above 8, the harmonic basis's width (the intercept
//     and seven columns).
//
// The message names the field and the counts or the value.
//
//------------------------------------------------------------------------------

void validate(const HarmonicOptions& options, index_t bands);

//------------------------------------------------------------------------------
//
// The date check every entry point runs once per call, before any work.
//
// Throws std::invalid_argument unless `dates` is non-decreasing. Equal dates
// are allowed: the filters keep the first observation of each date. The
// message names the first position out of order and both dates.
//
//------------------------------------------------------------------------------

void validate_dates(ArrayView<const std::int64_t, 1> dates);

//------------------------------------------------------------------------------
//
// Run CCD on one pixel.
//
// `spectral` is read, never written: a plan's transform (by default the
// Standard procedure's thermal transform, HarmonicOptions::THERMAL_SCALE/
// THERMAL_OFFSET) is applied to a private copy, so calling detect() twice on the same buffer gives the same
// result twice. `spectral` may be strided.
//
// `qas` is either the QA codes themselves (uint8, read as they are), or raw
// QA values decoded through `qa_table` (uint8 or uint16; see qa_table.hpp).
// A uint8 `qas` with a table is decoded too. The decoded codes go to private
// scratch; `qas` is never written.
//
// `preprocess` is the preprocessing plan (preprocess/preprocess.hpp): per
// procedure, the transform applied to the private copy and the filter chain
// that builds the processing mask. Null runs
// preprocess::default_plan(hoptions), built once per call; either plan is
// checked by preprocess::validate() once per call, never per pixel. One
// plan serves every pixel and thread of a call, as a QaTable does.
//
// Throws std::invalid_argument unless `spectral`'s band count passes
// validate(), `dates`, `qas` and `spectral` agree on the number of
// timesteps and `dates` passes validate_dates(), when `qas` is uint16
// and `qa_table` is null, and when the plan fails preprocess::validate().
//
//------------------------------------------------------------------------------

FitResult detect(
    ArrayView<const std::int64_t, 1> dates, // shape -> (timesteps)
    ArrayView<const scalar_t, 2> spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1> qas,   // shape -> (timesteps)
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

// detect() with caller-held scratch.
FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

//------------------------------------------------------------------------------
//
// The resume record (record.hpp)
//
//------------------------------------------------------------------------------

// A fingerprint of everything besides the input that decides a run's
// outputs: every field of the options and of the plan, and the QA table's
// codes (none when qa_table is null). Equal options give equal digests on
// every platform; unequal ones, different digests but for a 64-bit
// collision. STAT_ORD is in it: a resumed run is given the first run's
// options, and caps the cutoffs itself (ResumeHeader::statistics_end).
std::uint64_t options_digest(
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const preprocess::PreprocessPlan& plan,
    const QaTable* qa_table
);

// A fingerprint of one pixel's input up to `through`: of each observation
// dated on or before it whose QA code is not `qa_fill`, its date, its QA
// code and every band's value (the bits of the value as scalar_t), in date
// order. Fill observations are left out, so a scene that does not cover the
// pixel changes nothing.
std::uint64_t input_fingerprint(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qa_codes,
    std::uint8_t qa_fill,
    std::int64_t through
);

// detect()'s result and the record of the run that made it.
struct Recorded
{
    FitResult result;
    ResumeHeader header;
    PixelRecord record;
};

// detect(), and the run's record: the header (the options digest, the series
// end, statistics_end = the series end) and the pixel's record. The result is
// detect()'s, bit for bit.
Recorded detect_recorded(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

Recorded detect_recorded(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

// resume()'s result: the complete result for the whole series, the record of
// this run (for the next resume) and what was done.
struct Resumed
{
    FitResult result;
    ResumeHeader header;
    PixelRecord record;
    ResumeStatus status = ResumeStatus::Resumed;
};

// Continues the run `from_header` and `from` record on the whole of a longer
// series: `dates`, `spectral` and `qas` are the stored run's input with the
// observations after its series end; `stored_segments` are the stored run's
// segments, of which the first from.standard.final_segments are kept as they
// are given.
//
// The result equals, bit for bit, detect() on the same input with every
// statistics cutoff capped at from_header.statistics_end, whole output (the
// segments and the processing mask). A pixel whose observations up to the
// stored series end differ from the stored run's (reprocessed values, a
// back-filled or removed scene that covers it) is rerun in full the same
// way, and says so in the status; so are the pixels with nothing to resume
// (PermanentSnow, InsufficientClear, the early exits) and a pixel whose
// stored drop is on a date the plan keeps twice (a plan without Dedupe),
// whose result is the same either way.
//
// Throws std::invalid_argument, before any work, on a record of another
// format or results version, on other options, plan or QA table (by
// options_digest()), on a series that ends before the stored one, on fewer
// stored segments than the record's final ones, and on what detect() throws
// for.
Resumed resume(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const ResumeHeader& from_header,
    const PixelRecord& from,
    const std::vector<ChangeModel>& stored_segments,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

Resumed resume(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const ResumeHeader& from_header,
    const PixelRecord& from,
    const std::vector<ChangeModel>& stored_segments,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

//------------------------------------------------------------------------------
//
// Run detect() over every pixel of a cube, in parallel over H*W.
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
// throws std::invalid_argument, before any pixel runs, on the checks detect()
// makes, and std::runtime_error when `qas` is not (H, W, T).
//
// `keep_masks` false drops each pixel's processing mask (T bytes per pixel)
// as soon as the pixel is done, for callers that do not read it; the
// returned FitResult::mask is then empty.
//
// The cube is read, never written (see detect()), in any strides, as
// float64, float32 or uint16. Each pixel is converted to scalar_t in a
// per-thread buffer -- static_cast, which is exact for all three -- so a
// uint16 or float32 cube gives the results of the same values widened to
// float64 beforehand, without a float64 copy of the cube.
//
// `qas` is uint8 codes, or uint8 or uint16 raw values with `qa_table`, and
// `preprocess` is a plan or null, as for detect(). Each pixel is decoded in a per-thread buffer, so the cube gives
// the results of its decoded codes passed as uint8, without a decoded copy of
// the cube. One table serves every thread. A uint16 `qas` with a null
// `qa_table` throws std::invalid_argument before any pixel runs.
//
// `record` true also fills each PixelResult::record and the call's
// CubeResult::header, as detect_recorded() does for one pixel.
//
//------------------------------------------------------------------------------

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,      // (H,W,B,T)
    ArrayView<const std::uint8_t, 3> qas,     // (H,W,T)
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr,
    bool record = false
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr,
    bool record = false
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr,
    bool record = false
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr,
    bool record = false
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr,
    bool record = false
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr,
    bool record = false
);

//------------------------------------------------------------------------------
//
// resume() over every pixel of a cube, in parallel as detect_cube(): `from`
// holds the stored call's header and, in raster order, each pixel's record
// and stored segments (H x W entries each). Every pixel's result is the
// complete one, with its record and status; CubeResult::header is this
// call's. The call refuses, before any pixel, what resume() refuses for the
// whole call, and `from` without H x W records and segments; a pixel with
// fewer stored segments than its record's final ones is a CubeFailure.
//
//------------------------------------------------------------------------------

CubeResult resume_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const CubeResumeFrom& from,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

CubeResult resume_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const CubeResumeFrom& from,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

CubeResult resume_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const CubeResumeFrom& from,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

CubeResult resume_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const CubeResumeFrom& from,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

CubeResult resume_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const CubeResumeFrom& from,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

CubeResult resume_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const CubeResumeFrom& from,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr,
    const PreprocessPlanPtr& preprocess = nullptr
);

} // namespace ccd