#include "ccd/ccd.hpp"

#include <omp.h>

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

        //------------------------------------------------------------------
        // Put the thermal band into the unit apply_thermal_filter tests.
        //
        //     thermal <- thermal * THERMAL_SCALE + THERMAL_OFFSET
        //
        // The defaults are pyccd's collection-1 transform, Kelvin x 10 ->
        // Celsius x 100 (the collection-2 form this replaced was
        // `* 0.00341802 + 149.0 - 273.15`, which yields plain Celsius -- one
        // of the two units these options exist to name explicitly).
        //
        // Two things this loop is careful about:
        //
        //  * `spectral` is a non-const view onto the CALLER's buffer, so this
        //    is a destructive write. Under the identity transform we skip it
        //    entirely, which makes detect() idempotent on the same array --
        //    it was not, before.
        //  * It runs for Standard ONLY, matching pyccd, where
        //    kelvin_to_celsius() is called inside standard_procedure(). See
        //    harmonic.hpp's "Input unit convention" block for the
        //    consequences for the other two procedures.
        //------------------------------------------------------------------
        const bool identity =
            hoptions.THERMAL_SCALE  == scalar_t(1.0) &&
            hoptions.THERMAL_OFFSET == scalar_t(0.0);

        if (!identity) {
            const index_t T = dates.size();
            const index_t thermal_idx = hoptions.THERMAL_IDX;
            for (index_t t = 0; t < T; ++t) {
                spectral(thermal_idx, t) =
                    spectral(thermal_idx, t) * hoptions.THERMAL_SCALE
                    + hoptions.THERMAL_OFFSET;
            }
        }
    }

    HarmonicWorkspace hworkspace(
        dates,
        spectral,
        qas,
        hoptions
    );

    LOG_DEBUG(
        "initial spectral: " 
        << hworkspace.spectral()
    );

    LassoWorkspace lworkspace(dates.size());
    LassoSolver solver(
        loptions
    );

    FitResult final_result;
    switch(fit_type)
    {
        case FitProcedureType::Standard:
        {   
            LOG_DEBUG("Standard");
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
            LOG_DEBUG("Clear");
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
            LOG_DEBUG("Snow");
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

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<scalar_t, 4> spectral,          // (H,W,B,T)
    ArrayView<const std::uint8_t, 3> qas,     // (H,W,T)
    HarmonicOptions hoptions,
    LassoOptions loptions,
    int threads
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
            "qas shape must be (height, width, timesteps)"
        );
    }

    // threads == 0 keeps the historical behaviour: let the OpenMP runtime
    // decide, which is OMP_NUM_THREADS or the core count. A positive value is
    // a request, not a guarantee -- the runtime may still hand back fewer, so
    // the count reported below is read from inside the region.
    const int requested =
        threads > 0 ? threads : omp_get_max_threads();

    std::vector<std::vector<PixelResult>> thread_results(requested);
    std::vector<std::vector<CubeFailure>> thread_failures(requested);

    int used = requested;

    #pragma omp parallel num_threads(requested)
    {
        const int tid = omp_get_thread_num();

        #pragma omp single
        used = omp_get_num_threads();

        auto& output   = thread_results[tid];
        auto& failures = thread_failures[tid];

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

            //------------------------------------------------------------------
            // QA view
            //------------------------------------------------------------------
            auto qa_view =
                qas.slice(fixed(row), fixed(col), all()); // T

            LOG_DEBUG( 
                << "thread " 
                << omp_get_thread_num()
                << " pixel "
                << row << "," << col
            );

            //------------------------------------------------------------------
            // Run CCD
            //
            // The try/catch is load-bearing, not defensive habit: an
            // exception that escapes an OpenMP region terminates the process,
            // so without this one pathological time series destroys every
            // other pixel's result in the cube and reports nothing about
            // which pixel did it. Catching per pixel turns that into a
            // recorded address plus a message, and the other 249,999 pixels
            // still come back.
            //
            // Nothing is swallowed -- every catch lands in
            // CubeResult::failures, and it is the caller's job to surface the
            // count.
            //------------------------------------------------------------------
            try
            {
                FitResult result =
                    detect(
                        dates,
                        spect_view,
                        qa_view,
                        hoptions,
                        loptions
                    );

                output.emplace_back(
                    PixelResult{
                        row,
                        col,
                        std::move(result)
                    }
                );
            }
            catch (const std::exception& e)
            {
                failures.emplace_back(
                    CubeFailure{
                        row,
                        col,
                        std::string(e.what())
                    }
                );
            }
            catch (...)
            {
                failures.emplace_back(
                    CubeFailure{
                        row,
                        col,
                        std::string("unknown non-std exception")
                    }
                );
            }
        }
    }

    //----------------------------------------------------------------------
    // Merge thread-local vectors
    //----------------------------------------------------------------------
    CubeResult merged;
    merged.threads = used;

    std::size_t total = 0;
    for (const auto& v : thread_results)
        total += v.size();

    merged.pixels.reserve(total);

    for (auto& v : thread_results)
    {
        std::move(
            v.begin(),
            v.end(),
            std::back_inserter(merged.pixels)
        );
    }

    std::size_t total_failed = 0;
    for (const auto& v : thread_failures)
        total_failed += v.size();

    merged.failures.reserve(total_failed);

    for (auto& v : thread_failures)
    {
        std::move(
            v.begin(),
            v.end(),
            std::back_inserter(merged.failures)
        );
    }

    return merged;
}

} // namespace ccd