#include "ccd/ccd.hpp"

#include <omp.h>

#include <algorithm>
#include <type_traits>

#include "ccd/array_view.hpp"
#include "ccd/regression/lasso_solver.hpp"

#include "ccd/harmonic/quality.hpp"

#include "ccd/procedure/standard.hpp"
#include "ccd/procedure/permanent_snow.hpp"
#include "ccd/procedure/insufficient_clear.hpp"
#include "ccd/logger.hpp"


namespace ccd 
{

namespace
{

//------------------------------------------------------------------------------
// One pixel, from a (B, T) view of any supported element type.
//
// The procedures read a contiguous scalar_t (B, T) block. When the input is
// already that and no thermal transform applies, it is read in place;
// otherwise it is converted into scratch.spectral, and the thermal transform
// is applied there. The caller's buffer is never written.
//------------------------------------------------------------------------------
template<typename Value>
FitResult detect_pixel(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch
) {

    Quality quality;
    Quality::Statistics quality_stat = quality.compute_until(
        qas,
        dates,
        hoptions
    );

    bool transform_thermal = false;

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
        // It is applied to the per-pixel copy below, never to the caller's
        // buffer, and it runs for Standard ONLY, matching pyccd, where
        // kelvin_to_celsius() is called inside standard_procedure(). See
        // harmonic.hpp's "Input unit convention" block for the consequences
        // for the other two procedures. The identity transform is skipped.
        //------------------------------------------------------------------
        const bool identity =
            hoptions.THERMAL_SCALE  == scalar_t(1.0) &&
            hoptions.THERMAL_OFFSET == scalar_t(0.0);

        transform_thermal = !identity;
    }

    const index_t B = spectral.extent(0);
    const index_t T = spectral.extent(1);

    ArrayView<const scalar_t, 2> pixel =
        ArrayView<const scalar_t, 2>::contiguous(
            scratch.spectral.data(),
            {B, T}
        );

    bool in_place = false;
    if constexpr (std::is_same_v<Value, scalar_t>)
    {
        in_place = !transform_thermal && spectral.is_contiguous();
        if (in_place)
            pixel = spectral;
    }

    if (!in_place)
    {
        auto& block = scratch.spectral;
        block.resize(B * T);

        for (index_t b = 0; b < B; ++b)
        {
            for (index_t t = 0; t < T; ++t)
            {
                block[b * T + t] =
                    static_cast<scalar_t>(spectral(b, t));
            }
        }

        if (transform_thermal)
        {
            const index_t thermal_idx = hoptions.THERMAL_IDX;
            for (index_t t = 0; t < T; ++t) {
                block[thermal_idx * T + t] =
                    block[thermal_idx * T + t] * hoptions.THERMAL_SCALE
                    + hoptions.THERMAL_OFFSET;
            }
        }

        pixel = ArrayView<const scalar_t, 2>::contiguous(
            block.data(),
            {B, T}
        );
    }

    HarmonicWorkspace hworkspace(
        dates,
        pixel,
        qas,
        hoptions
    );

    LOG_DEBUG(
        "initial spectral: " 
        << hworkspace.spectral()
    );

    LassoWorkspace& lworkspace = scratch.lasso;
    LassoSolver solver(
        loptions
    );

    FitResult final_result;
    switch(fit_type)
    {
        case FitProcedureType::Standard:
        {   
            LOG_DEBUG("Standard");
            StandardProcedure& fit_procedure = scratch.standard;
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

template<typename Value>
CubeResult detect_cube_impl(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 4> spectral,       // (H,W,B,T)
    ArrayView<const std::uint8_t, 3> qas,     // (H,W,T)
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks
)
{
    const index_t H = spectral.extent(0);
    const index_t W = spectral.extent(1);
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

    //----------------------------------------------------------------------
    // One slot per pixel, written by whichever thread runs it, so the
    // result comes back in raster order whatever the schedule did.
    //----------------------------------------------------------------------
    const std::int64_t npixels =
        static_cast<std::int64_t>(H) *
        static_cast<std::int64_t>(W);

    std::vector<PixelResult> by_pixel(static_cast<std::size_t>(npixels));
    std::vector<std::uint8_t> succeeded(static_cast<std::size_t>(npixels), 0);
    std::vector<std::vector<CubeFailure>> thread_failures(requested);

    int used = requested;

    #pragma omp parallel num_threads(requested)
    {
        const int tid = omp_get_thread_num();

        #pragma omp single
        used = omp_get_num_threads();

        auto& failures = thread_failures[tid];

        DetectScratch scratch;

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
                    detect_pixel<Value>(
                        dates,
                        spect_view,
                        qa_view,
                        hoptions,
                        loptions,
                        scratch
                    );

                if (!keep_masks)
                    result.mask = ProcessingMask();

                by_pixel[static_cast<std::size_t>(pixel)] =
                    PixelResult{
                        row,
                        col,
                        std::move(result)
                    };

                succeeded[static_cast<std::size_t>(pixel)] = 1;
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
    // Collect, in raster order
    //----------------------------------------------------------------------
    CubeResult merged;
    merged.threads = used;

    std::size_t total = 0;
    for (const auto ok : succeeded)
        total += ok;

    merged.pixels.reserve(total);

    for (std::size_t pixel = 0; pixel < by_pixel.size(); ++pixel)
    {
        if (succeeded[pixel])
            merged.pixels.push_back(std::move(by_pixel[pixel]));
    }

    by_pixel = std::vector<PixelResult>();

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

    std::sort(
        merged.failures.begin(),
        merged.failures.end(),
        [](const CubeFailure& a, const CubeFailure& b)
        {
            return a.row != b.row ? a.row < b.row : a.col < b.col;
        });

    return merged;
}

} // namespace

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions
) {
    DetectScratch scratch;
    return detect_pixel<scalar_t>(dates, spectral, qas, hoptions, loptions, scratch);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch
) {
    return detect_pixel<scalar_t>(dates, spectral, qas, hoptions, loptions, scratch);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks
) {
    return detect_cube_impl<double>(dates, spectral, qas, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks
) {
    return detect_cube_impl<float>(dates, spectral, qas, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks
) {
    return detect_cube_impl<std::uint16_t>(dates, spectral, qas, hoptions, loptions, threads, keep_masks);
}

} // namespace ccd