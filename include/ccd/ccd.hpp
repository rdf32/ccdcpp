
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
};

//------------------------------------------------------------------------------
//
// Run CCD on one pixel.
//
// `spectral` is read, never written: the thermal transform of the Standard
// procedure (HarmonicOptions::THERMAL_SCALE/THERMAL_OFFSET) is applied to a
// private copy, so calling detect() twice on the same buffer gives the same
// result twice. `spectral` may be strided.
//
//------------------------------------------------------------------------------

FitResult detect(
    ArrayView<const std::int64_t, 1> dates, // shape -> (timesteps)
    ArrayView<const scalar_t, 2> spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1> qas,   // shape -> (timesteps)
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions
);

// detect() with caller-held scratch.
FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch
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
// This never throws on account of a pixel; see CubeResult::failures.
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
//------------------------------------------------------------------------------

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,      // (H,W,B,T)
    ArrayView<const std::uint8_t, 3> qas,     // (H,W,T)
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true
);

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads = 0,
    bool keep_masks = true
);

} // namespace ccd