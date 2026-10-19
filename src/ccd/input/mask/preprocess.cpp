#include "ccd/input/mask/preprocess.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

#include "ccd/core/logger.hpp"

namespace ccd
{

namespace preprocess
{

const char* name(StepKind kind)
{
    switch (kind)
    {
        case StepKind::KeepQa:       return "KeepQa";
        case StepKind::AddQa:        return "AddQa";
        case StepKind::ThermalRange: return "ThermalRange";
        case StepKind::BandRange:    return "BandRange";
        case StepKind::GreenMedian:  return "GreenMedian";
        case StepKind::Dedupe:       return "Dedupe";
    }
    return "unknown";
}

void Chain::push_back(const Step& step)
{
    if (count >= CHAIN_CAPACITY)
    {
        throw std::invalid_argument(
            "a preprocessing chain holds at most " +
            std::to_string(CHAIN_CAPACITY) + " steps"
        );
    }
    steps[count++] = step;
}

std::optional<Affine> thermal_transform(const HarmonicOptions& options)
{
    const bool identity =
        options.THERMAL_SCALE  == scalar_t(1.0) &&
        options.THERMAL_OFFSET == scalar_t(0.0);

    if (identity || options.THERMAL_IDX == NO_BAND)
        return std::nullopt;

    return Affine{
        options.THERMAL_IDX,
        options.THERMAL_SCALE,
        options.THERMAL_OFFSET
    };
}

PreprocessPlan default_plan(const HarmonicOptions& options)
{
    Chain common;

    common.push_back(Step::keep_qa(
        qa_set({options.QA_CLEAR, options.QA_WATER})
    ));

    if (options.THERMAL_IDX != NO_BAND)
    {
        common.push_back(Step::thermal_range(
            options.THERMAL_IDX,
            filter::thermal_range(options)
        ));
    }

    common.push_back(Step::band_range(
        options.SATURATION_BANDS,
        filter::reflectance_range(options)
    ));

    common.push_back(Step::dedupe());

    PreprocessPlan plan;

    plan.standard.transform = thermal_transform(options);
    plan.standard.chain = common;

    plan.snow.chain = common;
    plan.snow.chain.push_back(Step::add_qa(qa_set({options.QA_SNOW})));
    plan.snow.chain.push_back(Step::dedupe());

    plan.insufficient_clear.chain = common;
    plan.insufficient_clear.chain.push_back(Step::green_median(
        options.GREEN_IDX,
        options.MEDIAN_GREEN_FILTER,
        options.STAT_ORD,
        options.GREEN_FILTER_SCOPE
    ));
    plan.insufficient_clear.chain.push_back(Step::dedupe());

    return plan;
}

namespace
{

[[noreturn]] void reject(
    const char* procedure,
    const std::string& where,
    const std::string& why
)
{
    throw std::invalid_argument(
        std::string("preprocess plan ") + procedure + " " + where + ": " + why
    );
}

std::string step_text(index_t i, StepKind kind)
{
    return "step " + std::to_string(i) + " (" + name(kind) + ")";
}

std::string band_text(index_t band, index_t bands)
{
    return "band " + std::to_string(band) + " is not below the band count " +
           std::to_string(bands);
}

void validate_procedure(
    const char* procedure,
    const ProcedurePlan& plan,
    index_t bands
)
{
    const Chain& chain = plan.chain;

    if (chain.count > CHAIN_CAPACITY)
    {
        reject(procedure, "chain",
            "holds " + std::to_string(chain.count) + " steps; at most " +
            std::to_string(CHAIN_CAPACITY));
    }

    for (index_t i = 0; i < chain.size(); ++i)
    {
        const Step& step = chain[i];
        const std::string where = step_text(i, step.kind);

        switch (step.kind)
        {
            case StepKind::KeepQa:
            case StepKind::Dedupe:
                break;

            case StepKind::AddQa:
                if (i + 1 >= chain.size() ||
                    chain[i + 1].kind != StepKind::Dedupe)
                {
                    reject(procedure, where,
                        "must be followed by a Dedupe step: an added "
                        "observation can share a date with a kept one");
                }
                break;

            case StepKind::ThermalRange:
                if (step.band >= bands)
                    reject(procedure, where, band_text(step.band, bands));
                break;

            case StepKind::BandRange:
                for (const index_t band : step.bands)
                {
                    if (band >= bands)
                        reject(procedure, where, band_text(band, bands));
                }
                break;

            case StepKind::GreenMedian:
                if (step.band == NO_BAND)
                    reject(procedure, where, "needs a band; it is NO_BAND");
                if (step.band >= bands)
                    reject(procedure, where, band_text(step.band, bands));
                break;

            default:
                reject(procedure, where, "is not a step kind");
        }
    }

    if (plan.transform)
    {
        const index_t band = plan.transform->band;

        if (band >= bands)
            reject(procedure, "transform", band_text(band, bands));

        for (index_t i = 0; i < chain.size(); ++i)
        {
            const Step& step = chain[i];
            if (step.kind == StepKind::BandRange && step.bands.contains(band))
            {
                reject(procedure, "transform",
                    "band " + std::to_string(band) + " is also tested by " +
                    step_text(i, step.kind) +
                    "; a transformed band has its own range");
            }
        }
    }
}

} // namespace

void validate(const PreprocessPlan& plan, index_t bands)
{
    validate_procedure("standard", plan.standard, bands);
    validate_procedure("snow", plan.snow, bands);
    validate_procedure("insufficient_clear", plan.insufficient_clear, bands);
}

void apply(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask
)
{
    assert(chain.count <= CHAIN_CAPACITY);

    const index_t T = dates.size();
    mask.data().assign(T, static_cast<std::uint8_t>(1));

    for (const Step& step : chain)
    {
        switch (step.kind)
        {
            case StepKind::KeepQa:
            {
                // An Intersection (AND): clear where the code is not kept.
                for (index_t i = 0; i < T; ++i)
                {
                    if (!step.codes[qas(i)])
                        mask.set(i, false);
                }
                break;
            }

            case StepKind::AddQa:
            {
                // A Union (OR): set where the code is added.
                for (index_t i = 0; i < T; ++i)
                {
                    if (step.codes[qas(i)])
                        mask.set(i, true);
                }
                break;
            }

            case StepKind::ThermalRange:
                filter::apply_thermal_filter(
                    spectral, step.band, step.range, mask);
                break;

            case StepKind::BandRange:
                filter::apply_saturation_filter(
                    spectral, step.bands, step.range, mask);
                break;

            case StepKind::GreenMedian:
                filter::apply_green_median_filter(
                    dates,
                    spectral,
                    step.band,
                    step.max_date,
                    step.green_offset,
                    step.scope,
                    mask
                );
                break;

            case StepKind::Dedupe:
                filter::remove_duplicate_dates(dates, mask);
                break;
        }

        LOG_DEBUG(
            "Processing Mask after " << name(step.kind) << ": "
            << mask.as_ints()
        );
    }
}

} // namespace preprocess

} // namespace ccd
