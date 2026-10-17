#include "ccd/ccd.hpp"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "ccd/array_view.hpp"
#include "ccd/regression/lasso_solver.hpp"

#include "ccd/harmonic/quality.hpp"

#include "ccd/procedure/pixel_procedure.hpp"
#include "ccd/procedure/standard.hpp"
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

// `value` > 0; written so that NaN fails too.
void validate_positive(const char* name, scalar_t value, const char* unit)
{
    if (!(value > 0.0))
    {
        throw std::invalid_argument(
            std::string(name) + " is " + std::to_string(value) +
            "; it must be greater than 0" + unit
        );
    }
}

void validate_count(const char* name, index_t value)
{
    if (value < 1)
    {
        throw std::invalid_argument(
            std::string(name) + " is " + std::to_string(value) +
            "; it must be at least 1"
        );
    }
}

// The cadence and change-probability options.
void validate_cadence(const HarmonicOptions& options)
{
    validate_positive("PEEK_REFERENCE_DAYS", options.PEEK_REFERENCE_DAYS, " days");

    if (options.VARIOGRAM_MIN_LAG_DAYS <= 0)
    {
        throw std::invalid_argument(
            "VARIOGRAM_MIN_LAG_DAYS is " +
            std::to_string(options.VARIOGRAM_MIN_LAG_DAYS) +
            "; it must be greater than 0 days"
        );
    }

    validate_count("RETRAIN_MIN_OBS", options.RETRAIN_MIN_OBS);
    validate_positive("RETRAIN_SPAN_FACTOR", options.RETRAIN_SPAN_FACTOR, "");
    validate_count("SEASONAL_RMSE_COUNT", options.SEASONAL_RMSE_COUNT);
    validate_positive("SEASONAL_RMSE_DIVISOR", options.SEASONAL_RMSE_DIVISOR, "");

    if (!(options.CHANGE_PROBABILITY > 0.0 && options.CHANGE_PROBABILITY < 1.0))
    {
        throw std::invalid_argument(
            "CHANGE_PROBABILITY is " +
            std::to_string(options.CHANGE_PROBABILITY) +
            "; it must be in (0, 1)"
        );
    }
}

// Initialize fits COEFFICIENT_MIN coefficients, Extend up to
// COEFFICIENT_MAX, and both peek bases are COEFFICIENT_MAX wide, which must
// hold every model's coefficients and fit the harmonic basis (the intercept
// and CCD_MAX_COEFS columns).
void validate_coefficients(const HarmonicOptions& options)
{
    const index_t widest = CCD_MAX_COEFS + 1;

    if (options.COEFFICIENT_MIN < 1)
    {
        throw std::invalid_argument(
            "COEFFICIENT_MIN is " + std::to_string(options.COEFFICIENT_MIN) +
            "; it must be at least 1"
        );
    }

    if (!(options.COEFFICIENT_MIN <= options.COEFFICIENT_MID &&
          options.COEFFICIENT_MID <= options.COEFFICIENT_MAX))
    {
        throw std::invalid_argument(
            "COEFFICIENT_MIN, COEFFICIENT_MID, COEFFICIENT_MAX are " +
            std::to_string(options.COEFFICIENT_MIN) + ", " +
            std::to_string(options.COEFFICIENT_MID) + ", " +
            std::to_string(options.COEFFICIENT_MAX) +
            "; they must not decrease"
        );
    }

    if (options.COEFFICIENT_MAX > widest)
    {
        throw std::invalid_argument(
            "COEFFICIENT_MAX is " + std::to_string(options.COEFFICIENT_MAX) +
            "; the harmonic basis has at most " + std::to_string(widest)
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
    // A change is PEEK_SIZE consecutive observations that all exceed the
    // threshold, and both peeks read the magnitude of their first: with a
    // PEEK_SIZE of 1 a single outlier would be a change, and with 0 a peek
    // would be empty.
    if (options.PEEK_SIZE < 2)
    {
        throw std::invalid_argument(
            "PEEK_SIZE is " + std::to_string(options.PEEK_SIZE) +
            "; it must be at least 2"
        );
    }

    validate_cadence(options);
    validate_coefficients(options);
}

void validate_dates(ArrayView<const std::int64_t, 1> dates)
{
    for (index_t i = 1; i < dates.size(); ++i)
    {
        if (dates(i) < dates(i - 1))
        {
            throw std::invalid_argument(
                "dates must be non-decreasing; dates[" + std::to_string(i) +
                "] = " + std::to_string(dates(i)) + " follows dates[" +
                std::to_string(i - 1) + "] = " + std::to_string(dates(i - 1))
            );
        }
    }
}

namespace
{

//------------------------------------------------------------------------------
// The checks detect() and detect_cube() make before any work: the band
// layout, one timestep count shared by dates, spectral and qas, and dates in
// non-decreasing order.
//------------------------------------------------------------------------------
void validate_shapes(
    const HarmonicOptions& options,
    index_t bands,
    index_t spectral_timesteps,
    ArrayView<const std::int64_t, 1> dates,
    index_t qa_count
)
{
    validate(options, bands);

    const index_t date_count = dates.size();

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

    validate_dates(dates);
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
// The plan of procedure `type`.
//------------------------------------------------------------------------------
const preprocess::ProcedurePlan& procedure_plan(
    PixelProcedure type,
    const preprocess::PreprocessPlan& plan
)
{
    switch (type)
    {
        case PixelProcedure::InsufficientClear:
            return plan.insufficient_clear;

        case PixelProcedure::PermanentSnow:
            return plan.snow;

        case PixelProcedure::Standard:
            break;
    }

    return plan.standard;
}

//------------------------------------------------------------------------------
// FNV-1a over a canonical encoding: integers as little-endian 64-bit,
// floating point as the bits of its scalar_t, booleans as one byte, so the
// digest of equal values is the same whatever the platform's layout.
//------------------------------------------------------------------------------
class Fnv
{
public:

    void byte(std::uint8_t b)
    {
        hash_ = (hash_ ^ b) * 1099511628211ull;
    }

    void integer(std::int64_t v)
    {
        auto u = static_cast<std::uint64_t>(v);
        for (int i = 0; i < 8; ++i, u >>= 8)
            byte(static_cast<std::uint8_t>(u & 0xff));
    }

    void real(scalar_t v)
    {
        std::uint64_t bits;
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        integer(static_cast<std::int64_t>(bits));
    }

    void flag(bool v)
    {
        byte(v ? 1 : 0);
    }

    void bands(const BandList& list)
    {
        integer(static_cast<std::int64_t>(list.size()));
        for (const index_t b : list)
            integer(static_cast<std::int64_t>(b));
    }

    std::uint64_t value() const
    {
        return hash_;
    }

private:

    std::uint64_t hash_ = 1469598103934665603ull;
};

void digest(Fnv& h, const HarmonicOptions& o)
{
    for (const index_t v : {o.MEOW_SIZE, o.PEEK_SIZE})
        h.integer(static_cast<std::int64_t>(v));
    h.integer(o.DAY_DELTA);
    for (const std::uint8_t v : {o.QA_FILL, o.QA_CLEAR, o.QA_WATER, o.QA_SHADOW, o.QA_SNOW,
                                 o.QA_CLOUD, o.QA_CIRRUS1, o.QA_CIRRUS2, o.QA_OCCLUSION})
        h.integer(v);
    h.integer(static_cast<std::int64_t>(o.CLEAR_OBSERVATION_THRESHOLD));
    for (const scalar_t v : {o.CLEAR_PCT_THRESHOLD, o.SNOW_PCT_THRESHOLD, o.OUTLIER_THRESHOLD,
                             o.CHANGE_THRESHOLD, o.T_CONST, o.PEEK_REFERENCE_DAYS})
        h.real(v);
    h.integer(o.VARIOGRAM_MIN_LAG_DAYS);
    h.integer(static_cast<std::int64_t>(o.RETRAIN_MIN_OBS));
    h.real(o.RETRAIN_SPAN_FACTOR);
    h.integer(static_cast<std::int64_t>(o.SEASONAL_RMSE_COUNT));
    for (const scalar_t v : {o.SEASONAL_RMSE_DIVISOR, o.CHANGE_PROBABILITY, o.THERMAL_SCALE,
                             o.THERMAL_OFFSET, o.REFL_MIN, o.REFL_MAX})
        h.real(v);
    h.flag(o.REFL_MIN_INCLUSIVE);
    h.flag(o.REFL_MAX_INCLUSIVE);
    h.real(o.THERM_MIN);
    h.real(o.THERM_MAX);
    h.flag(o.THERM_MIN_INCLUSIVE);
    h.flag(o.THERM_MAX_INCLUSIVE);
    h.real(o.MEDIAN_GREEN_FILTER);
    h.integer(static_cast<std::int64_t>(o.GREEN_FILTER_SCOPE));
    h.integer(o.STAT_ORD);
    for (const index_t v : {o.COEFFICIENT_MIN, o.COEFFICIENT_MID, o.COEFFICIENT_MAX,
                            o.NUM_OBS_FACTOR, o.NUM_BANDS})
        h.integer(static_cast<std::int64_t>(v));
    h.bands(o.DETECTION_BANDS);
    h.bands(o.TMASK_BANDS);
    h.bands(o.SATURATION_BANDS);
    for (const index_t v : {o.BLUE_IDX, o.GREEN_IDX, o.RED_IDX, o.NIR_IDX, o.SWIR1_IDX,
                            o.SWIR2_IDX, o.THERMAL_IDX, o.QA_IDX})
        h.integer(static_cast<std::int64_t>(v));
}

void digest(Fnv& h, const LassoOptions& o)
{
    h.integer(static_cast<std::int64_t>(o.max_iter));
    h.real(o.alpha);
    h.real(o.tolerance);
    h.real(o.coef_floor);
    h.flag(o.fit_intercept);
    h.flag(o.warm_start);
    h.flag(o.unbiased_rmse);
}

void digest(Fnv& h, const preprocess::ProcedurePlan& plan)
{
    h.flag(plan.transform.has_value());
    if (plan.transform)
    {
        h.integer(static_cast<std::int64_t>(plan.transform->band));
        h.real(plan.transform->scale);
        h.real(plan.transform->offset);
    }

    h.integer(static_cast<std::int64_t>(plan.chain.size()));
    for (index_t i = 0; i < plan.chain.size(); ++i)
    {
        const preprocess::Step& step = plan.chain.steps[i];
        h.integer(static_cast<std::int64_t>(step.kind));
        for (std::size_t c = 0; c < step.codes.size(); ++c)
            h.flag(step.codes.test(c));
        h.bands(step.bands);
        h.integer(static_cast<std::int64_t>(step.band));
        h.real(step.range.lo.value);
        h.flag(step.range.lo.inclusive);
        h.real(step.range.hi.value);
        h.flag(step.range.hi.inclusive);
        h.real(step.green_offset);
        h.integer(step.max_date);
        h.integer(static_cast<std::int64_t>(step.scope));
    }
}

//------------------------------------------------------------------------------
// The plan a call runs: `given`, or else default_plan(options) built into
// `local`. Either is validated against `bands` here, once per call.
//------------------------------------------------------------------------------
const preprocess::PreprocessPlan& resolve_plan(
    const PreprocessPlanPtr& given,
    const HarmonicOptions& options,
    index_t bands,
    preprocess::PreprocessPlan& local
)
{
    if (given)
    {
        preprocess::validate(*given, bands);
        return *given;
    }

    local = preprocess::default_plan(options);
    preprocess::validate(local, bands);
    return local;
}

//------------------------------------------------------------------------------
// One pixel, from a (B, T) view of any supported element type.
//
// The steps run in order: decode the QA, compute the QA statistics, select
// the pixel's procedure (select_procedure()), apply its plan's transform,
// run its plan's chain to build the processing mask, and fit that mask with
// the run's change detector.
//
// The procedures read a contiguous scalar_t (B, T) block. When the input is
// already that and the plan has no transform, it is read in place; otherwise
// it is converted into scratch.spectral, and the transform is applied there.
// The caller's buffer is never written.
//
// With `record`, also fills the pixel's resume record (record.hpp): the
// fingerprint of its input up to `through`, its procedure, where the
// Standard machine stopped, and the days the machine dropped.
//------------------------------------------------------------------------------
template<typename Value>
std::uint64_t fingerprint_of(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 2> spectral,
    ArrayView<const std::uint8_t, 1> qa_codes,
    std::uint8_t qa_fill,
    std::int64_t through
)
{
    Fnv h;
    for (index_t t = 0; t < dates.size() && dates(t) <= through; ++t)
    {
        if (qa_codes(t) == qa_fill)
            continue;
        h.integer(dates(t));
        h.integer(qa_codes(t));
        for (index_t b = 0; b < spectral.extent(0); ++b)
            h.real(static_cast<scalar_t>(spectral(b, t)));
    }
    return h.value();
}

template<typename Value, typename Qa>
FitResult detect_pixel(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 2> spectral,
    ArrayView<const Qa, 1> raw_qas,
    const QaTable* qa_table,
    const preprocess::PreprocessPlan& plan,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    PixelRecord* record = nullptr,
    std::int64_t through = 0
) {
    const ArrayView<const std::uint8_t, 1> qas =
        qa_codes<Qa>(raw_qas, qa_table, scratch.qas);

    Quality quality;
    Quality::Statistics quality_stat = quality.compute_until(
        qas,
        dates,
        hoptions
    );

    const PixelProcedure pixel_procedure =
        select_procedure(quality_stat, hoptions);

    if (record)
    {
        record->input_fingerprint =
            fingerprint_of<Value>(dates, spectral, qas, hoptions.QA_FILL, through);
        record->procedure = pixel_procedure;
    }

    const preprocess::ProcedurePlan& procedure =
        procedure_plan(pixel_procedure, plan);

    const bool transform = procedure.transform.has_value();

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
        in_place = !transform && spectral.is_contiguous();
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

        if (transform)
        {
            const preprocess::Affine& affine = *procedure.transform;
            const index_t band = affine.band;
            for (index_t t = 0; t < T; ++t) {
                block[band * T + t] =
                    block[band * T + t] * affine.scale
                    + affine.offset;
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

    ProcessingMask mask;
    preprocess::apply(
        procedure.chain,
        hworkspace.dates(),
        hworkspace.spectral(),
        hworkspace.qas(),
        mask
    );

    LassoWorkspace& lworkspace = scratch.lasso;
    LassoSolver solver(
        loptions
    );

    // The run's change detector fits the pixel's procedure. Standard is the
    // only detector so far; a second one is picked per run, here, and makes
    // the same call.
    standard::Detector& detector = scratch.standard;

    if (!record)
        return detector.fit(pixel_procedure, hworkspace, std::move(mask), lworkspace, solver);

    const ProcessingMask filtered = mask;
    FitResult result =
        detector.fit(pixel_procedure, hworkspace, std::move(mask), lworkspace, solver);

    record->standard = standard::Suspended{};
    record->excluded_days.clear();
    if (pixel_procedure == PixelProcedure::Standard)
    {
        record->standard = detector.suspended();
        for (index_t t = 0; t < filtered.size(); ++t)
            if (filtered[t] && !result.mask[t])
                record->excluded_days.push_back(dates(t));
    }
    return result;
}

template<typename Value, typename Qa>
CubeResult detect_cube_impl(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const Value, 4> spectral,       // (H,W,B,T)
    ArrayView<const Qa, 3> qas,               // (H,W,T)
    const QaTable* qa_table,
    const PreprocessPlanPtr& given_plan,
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

    validate_shapes(hoptions, spectral.extent(2), T, dates, qas.extent(2));
    validate_qa<Qa>(qa_table);

    preprocess::PreprocessPlan local_plan;
    const preprocess::PreprocessPlan& plan =
        resolve_plan(given_plan, hoptions, spectral.extent(2), local_plan);

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
                "thread "
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
                        plan,
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

std::uint64_t options_digest(
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const preprocess::PreprocessPlan& plan,
    const QaTable* qa_table
)
{
    Fnv h;
    digest(h, hoptions);
    digest(h, loptions);
    for (const preprocess::ProcedurePlan* procedure :
         {&plan.standard, &plan.snow, &plan.insufficient_clear})
        digest(h, *procedure);
    h.flag(qa_table != nullptr);
    if (qa_table)
        for (const std::uint8_t code : qa_table->code)
            h.byte(code);
    return h.value();
}

std::uint64_t input_fingerprint(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qa_codes,
    std::uint8_t qa_fill,
    std::int64_t through
)
{
    return fingerprint_of<scalar_t>(dates, spectral, qa_codes, qa_fill, through);
}

namespace
{

// detect_recorded() for either QA type.
template<typename Qa>
Recorded detect_recorded_impl(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const Qa, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
)
{
    validate_shapes(hoptions, spectral.extent(0), spectral.extent(1), dates, qas.size());
    if constexpr (std::is_same_v<Qa, std::uint16_t>)
        validate_qa<std::uint16_t>(qa_table.get());

    preprocess::PreprocessPlan local_plan;
    const preprocess::PreprocessPlan& plan =
        resolve_plan(given_plan, hoptions, spectral.extent(0), local_plan);

    Recorded out;
    out.header.options_digest = options_digest(hoptions, loptions, plan, qa_table.get());
    out.header.series_end = dates.size() > 0
        ? dates(dates.size() - 1)
        : std::numeric_limits<std::int64_t>::min();
    out.header.statistics_end = out.header.series_end;

    DetectScratch scratch;
    out.result = detect_pixel<scalar_t, Qa>(
        dates, spectral, qas, qa_table.get(), plan, hoptions, loptions, scratch,
        &out.record, out.header.series_end);
    return out;
}

} // namespace

Recorded detect_recorded(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
)
{
    return detect_recorded_impl<std::uint8_t>(
        dates, spectral, qas, hoptions, loptions, qa_table, given_plan);
}

Recorded detect_recorded(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
)
{
    return detect_recorded_impl<std::uint16_t>(
        dates, spectral, qas, hoptions, loptions, qa_table, given_plan);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    DetectScratch scratch;
    return detect(dates, spectral, qas, hoptions, loptions, scratch, qa_table, given_plan);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    DetectScratch scratch;
    return detect(dates, spectral, qas, hoptions, loptions, scratch, qa_table, given_plan);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    validate_shapes(hoptions, spectral.extent(0), spectral.extent(1), dates, qas.size());

    preprocess::PreprocessPlan local_plan;
    const preprocess::PreprocessPlan& plan =
        resolve_plan(given_plan, hoptions, spectral.extent(0), local_plan);

    return detect_pixel<scalar_t, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), plan, hoptions, loptions, scratch);
}

FitResult detect(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint16_t, 1> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    DetectScratch& scratch,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    validate_shapes(hoptions, spectral.extent(0), spectral.extent(1), dates, qas.size());
    validate_qa<std::uint16_t>(qa_table.get());

    preprocess::PreprocessPlan local_plan;
    const preprocess::PreprocessPlan& plan =
        resolve_plan(given_plan, hoptions, spectral.extent(0), local_plan);

    return detect_pixel<scalar_t, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), plan, hoptions, loptions, scratch);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    return detect_cube_impl<double, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), given_plan, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const double, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    return detect_cube_impl<double, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), given_plan, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    return detect_cube_impl<float, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), given_plan, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const float, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    return detect_cube_impl<float, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), given_plan, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint8_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    return detect_cube_impl<std::uint16_t, std::uint8_t>(
        dates, spectral, qas, qa_table.get(), given_plan, hoptions, loptions, threads, keep_masks);
}

CubeResult detect_cube(
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const std::uint16_t, 4> spectral,
    ArrayView<const std::uint16_t, 3> qas,
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    int threads,
    bool keep_masks,
    const QaTablePtr& qa_table,
    const PreprocessPlanPtr& given_plan
) {
    return detect_cube_impl<std::uint16_t, std::uint16_t>(
        dates, spectral, qas, qa_table.get(), given_plan, hoptions, loptions, threads, keep_masks);
}

} // namespace ccd