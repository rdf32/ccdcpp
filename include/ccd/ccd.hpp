
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

namespace ccd
{

    
enum class FitProcedureType
{
    Standard,
    InsufficientClear,
    PermanentSnow
};

struct PixelResult
{
    index_t row;
    index_t col;
    FitResult result;
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
        const auto& r = model.bands[band];
        const auto& m = r.model;
        const auto& s = r.score;
        const auto& coef = m.coefficients();
        // const auto& resi = s.residuals;

        std::cout << "Band " << band << '\n';
        std::cout << "  Iterations : " << m.iterations() << '\n';
        std::cout << "  Intercept  : " << m.intercept() << '\n';
        std::cout << "  RMSE       : " << s.rmse << '\n';
        std::cout << "  Magnitude  : " << s.magn << '\n';

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
    StandardProcedure standard;

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
//   * CHANGE_PROBABILITY is not in (0, 1).
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
// `spectral` is read, never written: the thermal transform of the Standard
// procedure (HarmonicOptions::THERMAL_SCALE/THERMAL_OFFSET) is applied to a
// private copy, so calling detect() twice on the same buffer gives the same
// result twice. `spectral` may be strided.
//
// `qas` is either the QA codes themselves (uint8, read as they are), or raw
// QA values decoded through `qa_table` (uint8 or uint16; see qa_table.hpp).
// A uint8 `qas` with a table is decoded too. The decoded codes go to private
// scratch; `qas` is never written.
//
// Throws std::invalid_argument unless `spectral`'s band count passes
// validate(), `dates`, `qas` and `spectral` agree on the number of
// timesteps and `dates` passes validate_dates(), and when `qas` is uint16
// and `qa_table` is null.
//
//------------------------------------------------------------------------------

FitResult detect(
    ArrayView<const std::int64_t, 1> dates, // shape -> (timesteps)
    ArrayView<const scalar_t, 2> spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1> qas,   // shape -> (timesteps)
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table = nullptr
);

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table = nullptr
);

// detect() with caller-held scratch.
FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table = nullptr
);

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table = nullptr
);

//------------------------------------------------------------------------------
//
// Run detect() over every pixel of a cube, in parallel over H*W.
//
// `threads` is the OpenMP thread count. 0 means "whatever the runtime would
// have chosen", i.e. omp_get_max_threads(), which is what this function used
// unconditionally before -- so 0 reproduces the old behaviour exactly. Passing
// it explicitly matters because omp_get_max_threads() is fixed from
// OMP_NUM_THREADS at library load, which means a caller embedding this in a
// scheduler cannot honour its own allocation without setting an environment
// variable before the shared library is imported.
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
// `qas` is uint8 codes, or uint8 or uint16 raw values with `qa_table`, as for
// detect(). Each pixel is decoded in a per-thread buffer, so the cube gives
// the results of its decoded codes passed as uint8, without a decoded copy of
// the cube. One table serves every thread. A uint16 `qas` with a null
// `qa_table` throws std::invalid_argument before any pixel runs.
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
    const QaTablePtr& qa_table = nullptr
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true,
    const QaTablePtr& qa_table = nullptr
);

} // namespace ccd