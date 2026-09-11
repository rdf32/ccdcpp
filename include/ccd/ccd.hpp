
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/ccd.hpp"

#include "ccd/procedure/fit.hpp"

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

FitResult detect(
    ArrayView<const std::int64_t, 1> dates, // shape -> (timesteps)
    ArrayView<scalar_t, 2> spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1> qas,   // shape -> (timesteps)
    HarmonicOptions hoptions,
    LassoOptions loptions
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
//------------------------------------------------------------------------------

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<scalar_t, 4> spectral,          // (H,W,B,T)
    ArrayView<const std::uint8_t, 3> qas,     // (H,W,T)
    HarmonicOptions hoptions,
    LassoOptions loptions,
    int threads = 0
);

} // namespace ccd