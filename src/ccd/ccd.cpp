#include "ccd/ccd.hpp"

#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <string>
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

std::string count_text(index_t bands)
{
    return "spectral has " + std::to_string(bands) + " bands";
}

// `name` holds only indices below `bands`, each once.
void validate_list(const char* name, const BandList& list, index_t bands)
{
    for (index_t i = 0; i < list.size(); ++i)
    {
        if (list[i] >= bands)
        {
            throw std::invalid_argument(
                std::string(name) + " holds band " + std::to_string(list[i]) +
                "; " + count_text(bands)
            );
        }

        for (index_t j = 0; j < i; ++j)
        {
            if (list[j] == list[i])
            {
                throw std::invalid_argument(
                    std::string(name) + " holds band " +
                    std::to_string(list[i]) + " more than once"
                );
            }
        }
    }
}

void validate_nonempty(const char* name, const BandList& list)
{
    if (list.empty())
    {
        throw std::invalid_argument(
            std::string(name) + " is empty"
        );
    }
}

} // namespace

void validate(const HarmonicOptions& options, index_t bands)
{
    if (bands != options.NUM_BANDS)
    {
        throw std::invalid_argument(
            count_text(bands) + "; HarmonicOptions.NUM_BANDS expects " +
            std::to_string(options.NUM_BANDS)
        );
    }

    if (bands > CCD_MAX_BANDS)
    {
        throw std::invalid_argument(
            count_text(bands) + "; ccdcpp accepts at most " +
            std::to_string(CCD_MAX_BANDS)
        );
    }

    validate_nonempty("DETECTION_BANDS", options.DETECTION_BANDS);
    validate_nonempty("TMASK_BANDS", options.TMASK_BANDS);

    validate_list("DETECTION_BANDS", options.DETECTION_BANDS, bands);
    validate_list("TMASK_BANDS", options.TMASK_BANDS, bands);
    validate_list("SATURATION_BANDS", options.SATURATION_BANDS, bands);

    if (options.GREEN_IDX == NO_BAND)
    {
        throw std::invalid_argument(
            "GREEN_IDX is NO_BAND; the green median filter needs a green band"
        );
    }

    if (options.GREEN_IDX >= bands)
    {
        throw std::invalid_argument(
            "GREEN_IDX is " + std::to_string(options.GREEN_IDX) + "; " +
            count_text(bands)
        );
    }

    if (options.THERMAL_IDX != NO_BAND)
    {
        if (options.THERMAL_IDX >= bands)
        {
            throw std::invalid_argument(
                "THERMAL_IDX is " + std::to_string(options.THERMAL_IDX) +
                "; " + count_text(bands) + " (NO_BAND for none)"
            );
        }

        if (options.SATURATION_BANDS.contains(options.THERMAL_IDX))
        {
            throw std::invalid_argument(
                "SATURATION_BANDS holds THERMAL_IDX (" +
                std::to_string(options.THERMAL_IDX) +
                "); the thermal band has its own bounds"
            );
        }
    }
}

namespace
{

//------------------------------------------------------------------------------
// The checks detect() and detect_cube() make before any work: the band
// layout, and one timestep count shared by dates, spectral and qas.
//------------------------------------------------------------------------------
void validate_shapes(
    const HarmonicOptions& options,
    index_t bands,
    index_t spectral_timesteps,
    index_t date_count,
    index_t qa_count
)
{
    validate(options, bands);

    if (date_count != spectral_timesteps)
    {
        throw std::invalid_argument(
            "dates has " + std::to_string(date_count) +
            " timesteps; spectral has " + std::to_string(spectral_timesteps)
        );
    }

    if (qa_count != spectral_timesteps)
    {
        throw std::invalid_argument(
            "qas has " + std::to_string(qa_count) +
            " timesteps; spectral has " + std::to_string(spectral_timesteps)
        );
    }
}

//------------------------------------------------------------------------------
// A uint16 `qas` holds raw QA values and has no meaning without a table.
//------------------------------------------------------------------------------
template<typename Qa>
void validate_qa(const QaTable* qa_table)
{
    if (std::is_same_v<Qa, std::uint16_t> && qa_table == nullptr)
    {
        throw std::invalid_argument(
            "qas is uint16, which holds raw QA values; pass a QaTable "
            "(qa_table) to decode them, or pass uint8 QA codes"
        );
    }
}

//------------------------------------------------------------------------------
// The pixel's QA codes.
//
// Without a table a uint8 `qas` is the codes, and is returned as it is. With
// one, every value is decoded into `scratch`, which is returned. A uint16
// `qas` always has a table here; validate_qa() saw to that.
//------------------------------------------------------------------------------
template<typename Qa>
ArrayView<const std::uint8_t, 1> qa_codes(
    ArrayView<const Qa, 1> qas,
    const QaTable* qa_table,
    std::vector<std::uint8_t>& scratch
)
{
    if constexpr (std::is_same_v<Qa, std::uint8_t>)
    {
        if (qa_table == nullptr)
            return qas;
    }

    const index_t T = qas.size();
    scratch.resize(T);

    const auto& code = qa_table->code;
    for (index_t t = 0; t < T; ++t)
        scratch[t] = code[qas(t)];

    return ArrayView<const std::uint8_t, 1>::contiguous(scratch.data(), {T});
}

//------------------------------------------------------------------------------
// One pixel, from a (B, T) view of any supported element type.
//
// The procedures read a contiguous scalar_t (B, T) block. When the input is
// already that and no thermal transform applies, it is read in place;
// otherwise it is converted into scratch.spectral, and the thermal transform
// is applied there. The caller's buffer is never written.
//------------------------------------------------------------------------------
template<typename Value, typename Qa>
FitResult detect_pixel(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 2> spectral,
    ArrayView<const Qa, 1> raw_qas,
    const QaTable* qa_table,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch
) {
    const ArrayView<const std::uint8_t, 1> qas =
        qa_codes<Qa>(raw_qas, qa_table, scratch.qas);

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
        // for the other two procedures. The identity transform is skipped,
        // and so is an input without a thermal band.
        //------------------------------------------------------------------
        const bool identity =
            hoptions.THERMAL_SCALE  == scalar_t(1.0) &&
            hoptions.THERMAL_OFFSET == scalar_t(0.0);

        transform_thermal =
            !identity && hoptions.THERMAL_IDX != NO_BAND;
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

template<typename Value, typename Qa>
CubeResult detect_cube_impl(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 4> spectral,       // (H,W,B,T)
    ArrayView<const Qa, 3> qas,               // (H,W,T)
    const QaTable* qa_table,
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

    validate_shapes(hoptions, spectral.extent(2), T, dates.size(), qas.extent(2));
    validate_qa<Qa>(qa_table);

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
                    detect_pixel<Value, Qa>(
                        dates,
                        spect_view,
                        qa_view,
                        qa_table,
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
    const LassoOptions& loptions,
    const QaTablePtr& qa_table
) {
    DetectScratch scratch;
    return detect(dates, spectral, qas, hoptions, loptions, scratch, qa_table);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table
) {
    DetectScratch scratch;
    return detect(dates, spectral, qas, hoptions, loptions, scratch, qa_table);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table
) {
    validate_shapes(hoptions, spectral.extent(0), spectral.extent(1), dates.size(), qas.size());

    return detect_pixel<scalar_t, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, scratch);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table
) {
    validate_shapes(hoptions, spectral.extent(0), spectral.extent(1), dates.size(), qas.size());
    validate_qa<std::uint16_t>(qa_table.get());

    return detect_pixel<scalar_t, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, scratch);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table
) {
    return detect_cube_impl<double, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table
) {
    return detect_cube_impl<double, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table
) {
    return detect_cube_impl<float, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table
) {
    return detect_cube_impl<float, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table
) {
    return detect_cube_impl<std::uint16_t, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table
) {
    return detect_cube_impl<std::uint16_t, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), hoptions, loptions, threads, keep_masks);
}

} // namespace ccd