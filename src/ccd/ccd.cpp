#include "ccd/ccd.hpp"

#include <omp.h>
#include <iostream>

#include "ccd/array_view.hpp"
#include "ccd/regression/lasso_solver.hpp"

#include "ccd/harmonic/quality.hpp"

#include "ccd/procedure/standard.hpp"
#include "ccd/procedure/permanent_snow.hpp"
#include "ccd/procedure/insufficient_clear.hpp"
#include "ccd/logger.hpp"


namespace ccd 
{

// single pixel timeseries detection
FitResult detect(
    ArrayView<const std::int64_t, 1> dates, // shape -> (timesteps)
    ArrayView<scalar_t, 2> spectral,  // shape -> (bands, timesteps)
    ArrayView<const std::uint8_t, 1> qas,   // shape -> (timesteps)
    HarmonicOptions hoptions,
    LassoOptions loptions
) {

    Quality quality;
    Quality::Statistics quality_stat = quality.compute_until(
        qas,
        dates,
        hoptions
    );

    FitProcedureType fit_type;
    if (!quality_stat.enough_clear(hoptions.CLEAR_PCT_THRESHOLD)) {
        if (quality_stat.enough_snow(hoptions.SNOW_PCT_THRESHOLD)) {
            fit_type = FitProcedureType::PermanentSnow;
        } else {
            fit_type = FitProcedureType::InsufficientClear;
        }
    } else {
        fit_type = FitProcedureType::Standard;
        // convert thermal from kelvin to celsius for standard_procedure
        const index_t T = dates.size();
        const index_t thermal_idx = hoptions.THERMAL_IDX;
        for (index_t t = 0; t < T; ++t) {
            spectral(thermal_idx, t) = 
                spectral(thermal_idx, t) * 0.00341802 + 149.0 - 273.15; // collection-2 transform
                // spectral(thermal_idx, t) * 10.0 - 27315.0; // collection-1 transform from pyccd
        }
    }

    HarmonicWorkspace hworkspace(
        dates,
        spectral,
        qas,
        hoptions
    );
    LOG_DEBUG("initial spectral: " << hworkspace.spectral());
    LassoWorkspace lworkspace(dates.size());
    LassoSolver solver(
        loptions
    );

    FitResult final_result;
    switch(fit_type)
    {
        case FitProcedureType::Standard:
        {   
            std::cout << "Standard" << std::endl;
            StandardProcedure fit_procedure;
            final_result = fit_procedure.run(
                hworkspace,
                lworkspace,
                solver
            );
            break;
        }

        case FitProcedureType::InsufficientClear:
        {   
            std::cout << "Clear" << std::endl;
            InsufficientClear fit_procedure;
            final_result = fit_procedure.run(
                hworkspace,
                lworkspace,
                solver
            );
            break;
        }

        case FitProcedureType::PermanentSnow:
        {   
            std::cout << "Snow" << std::endl;
            PermanentSnow fit_procedure;
            final_result = fit_procedure.run(
                hworkspace,
                lworkspace,
                solver
            );
            break;
        }
  
    }

    return final_result;
}

std::vector<PixelResult> detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<scalar_t, 4> spectral,          // (H,W,B,T)
    ArrayView<const std::uint8_t, 3> qas,     // (H,W,T)
    HarmonicOptions hoptions,
    LassoOptions loptions
)
{
    const index_t H = spectral.extent(0);
    const index_t W = spectral.extent(1);
    const index_t B = spectral.extent(2);
    const index_t T = spectral.extent(3);

    if (qas.extent(0) != H ||
        qas.extent(1) != W ||
        qas.extent(2) != T)
    {
        throw std::runtime_error(
            "qas shape must be (height,width,timesteps)"
        );
    }

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<PixelResult>> thread_results(nthreads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        auto& output = thread_results[tid];

        const std::int64_t npixels =
            static_cast<std::int64_t>(H) *
            static_cast<std::int64_t>(W);

        #pragma omp for schedule(dynamic)
        for (std::int64_t pixel = 0; pixel < npixels; ++pixel)
        {
            const index_t row =
                static_cast<index_t>(pixel) / W;

            const index_t col =
                static_cast<index_t>(pixel) % W;

            //------------------------------------------------------------------
            // Spectral view
            //------------------------------------------------------------------
            auto spect_view = 
                spectral.slice(fixed(row), fixed(col), all(), all()); // B, T

            std::cout 
                << spect_view.extent(0)
                << " "
                << spect_view.extent(1)
                << std::endl;

            //------------------------------------------------------------------
            // QA view
            //------------------------------------------------------------------
            auto qa_view =
                qas.slice(fixed(row), fixed(col), all()); // T

            // std::cout << qa_view.extent(0) << std::endl;

            // std::cout 
            //     << "thread " 
            //     << omp_get_thread_num()
            //     << " pixel "
            //     << row << "," << col
            //     << std::endl;
            //------------------------------------------------------------------
            // Run CCD
            //------------------------------------------------------------------
            FitResult result =
                detect(
                    dates,
                    spect_view,
                    qa_view,
                    hoptions,
                    loptions
                );
            // std::cout 
            //     << "before move models="
            //     << result.models.size()
            //     << " mask="
            //     << result.mask.size()
            //     << std::endl;
            output.emplace_back(
                PixelResult{
                    row,
                    col,
                    std::move(result)
                }
            );
        }
    }

    //----------------------------------------------------------------------
    // Merge thread-local vectors
    //----------------------------------------------------------------------
    std::size_t total = 0;
    for (const auto& v : thread_results)
        total += v.size();

    std::vector<PixelResult> output;
    output.reserve(total);

    for (auto& v : thread_results)
    {
        std::move(
            v.begin(),
            v.end(),
            std::back_inserter(output)
        );
    }

    return output;
}

} // namespace ccd