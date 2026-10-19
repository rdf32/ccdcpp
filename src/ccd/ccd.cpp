#include "ccd/ccd.hpp"

#include <stdexcept>
#include <string>

#include "fnv.hpp"

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

void digest(detail::Fnv& h, const HarmonicOptions& o)
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
    // ccdc adds nothing, so its digest is the one it has always had; another
    // method adds its value (and its parameters).
    if (o.METHOD != ChangeMethod::Ccdc)
        h.integer(static_cast<std::int64_t>(o.METHOD));
}

void digest(detail::Fnv& h, const LassoOptions& o)
{
    h.integer(static_cast<std::int64_t>(o.max_iter));
    h.real(o.alpha);
    h.real(o.tolerance);
    h.real(o.coef_floor);
    h.flag(o.fit_intercept);
    h.flag(o.warm_start);
    h.flag(o.unbiased_rmse);
}

void digest(detail::Fnv& h, const preprocess::ProcedurePlan& plan)
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

} // namespace

std::uint64_t options_digest(
    const HarmonicOptions& hoptions,
    const LassoOptions& loptions,
    const preprocess::PreprocessPlan& plan,
    const QaTable* qa_table
)
{
    detail::Fnv h;
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
    return detail::fingerprint_of<scalar_t>(dates, spectral, qa_codes, qa_fill, through);
}

} // namespace ccd
