#include "ccd/input/mask/preprocess.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "../../core/logger.hpp"

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
        case StepKind::Custom:       return "Custom";
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

namespace
{

// The doubles in order as integers, NaN aside: key(a) < key(b) exactly when
// a < b, -0 and +0 both 0, -inf and +inf the ends.
std::int64_t key_of(scalar_t value)
{
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    const auto magnitude = static_cast<std::int64_t>(bits & ~(std::uint64_t(1) << 63));
    return (bits >> 63) ? -magnitude : magnitude;
}

scalar_t value_of(std::int64_t key)
{
    std::uint64_t bits = key < 0 ? static_cast<std::uint64_t>(-key) | (std::uint64_t(1) << 63)
                                 : static_cast<std::uint64_t>(key);
    scalar_t value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

// The first double (-inf to +inf) at which `holds`, false and then true
// along the doubles in order, is true; none if it is nowhere true.
template<typename Holds>
std::optional<std::int64_t> first_key(const Holds& holds)
{
    std::int64_t lo = key_of(-std::numeric_limits<scalar_t>::infinity());
    std::int64_t hi = key_of(std::numeric_limits<scalar_t>::infinity());
    if (!holds(value_of(hi)))
        return std::nullopt;
    while (lo < hi)   // holds(hi); not holds below lo
    {
        const std::int64_t mid = lo + static_cast<std::int64_t>(
            (static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo)) / 2);
        if (holds(value_of(mid)))
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

} // namespace

Range stored_range(const Range& range, const Affine& affine)
{
    if (!(affine.scale > 0) || !std::isfinite(affine.scale) || !std::isfinite(affine.offset))
        throw std::invalid_argument(
            "a thermal range is taken back through a thermal transform of a positive "
            "finite thermal_scale and a finite thermal_offset only");
    const auto f = [&](scalar_t v) { return apply(affine, v); };
    const scalar_t lo = range.lo.value;
    const scalar_t hi = range.hi.value;

    // Each end as in_range() tests it: an exclusive low end keeps f(v) > lo,
    // an inclusive one !(f(v) < lo); an exclusive high end keeps f(v) < hi,
    // an inclusive one !(f(v) > hi). Each kept side begins (or the other side
    // ends) at the first double where the comparison turns.
    const std::int64_t lowest = key_of(-std::numeric_limits<scalar_t>::infinity());
    Range out;
    if (!range.lo.inclusive)
    {
        // f(v) > lo from the first such v on: v > the double before it.
        const auto first = first_key([&](scalar_t v) { return f(v) > lo; });
        out.lo = !first ? Bound{std::numeric_limits<scalar_t>::infinity(), false}
                 : *first == lowest ? Bound{value_of(lowest), true}
                 : Bound{value_of(*first - 1), false};
    }
    else
    {
        // !(f(v) < lo) from the first such v on (+inf always is one).
        const auto first = first_key([&](scalar_t v) { return !(f(v) < lo); });
        out.lo = Bound{value_of(*first), true};
    }
    if (!range.hi.inclusive)
    {
        // f(v) < hi below the first v where it is not (+inf always is one).
        const auto first = first_key([&](scalar_t v) { return !(f(v) < hi); });
        out.hi = Bound{value_of(*first), false};
    }
    else
    {
        // !(f(v) > hi) up to the double before the first v where f(v) > hi.
        const auto first = first_key([&](scalar_t v) { return f(v) > hi; });
        out.hi = !first ? Bound{std::numeric_limits<scalar_t>::infinity(), true}
                 : *first == lowest ? Bound{value_of(lowest), false}
                 : Bound{value_of(*first - 1), true};
    }
    return out;
}

std::optional<Affine> thermal_transform(const Input& input)
{
    const bool identity =
        input.thermal_scale  == scalar_t(1.0) &&
        input.thermal_offset == scalar_t(0.0);
    const index_t thermal = input.index("thermal");

    if (identity || thermal == NO_BAND)
        return std::nullopt;

    return Affine{
        thermal,
        input.thermal_scale,
        input.thermal_offset
    };
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

// Step i of `chain` names bands below `bands` and has what its kind needs.
void check_step(
    const char* procedure,
    const Chain& chain,
    index_t i,
    index_t bands
)
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

        case StepKind::Custom:
            if (!step.custom)
                reject(procedure, where, "has no rule");
            try
            {
                step.custom->validate(bands);
            }
            catch (const std::invalid_argument& e)
            {
                reject(procedure, where, e.what());
            }
            break;

        default:
            reject(procedure, where, "is not a step kind");
    }
}

// The plan's transform, if any, is of a band below `bands` that no BandRange
// step of its chain tests.
void check_transform(
    const char* procedure,
    const ProcedurePlan& plan,
    index_t bands
)
{
    if (!plan.transform)
        return;

    const Chain& chain = plan.chain;
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
        check_step(procedure, chain, i, bands);

    check_transform(procedure, plan, bands);
}

} // namespace

void validate(const PreprocessPlan& plan, index_t bands)
{
    for (PixelProcedure procedure : PROCEDURES)
        validate_procedure(name(procedure), plan.of(procedure), bands);
}

void apply(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask
)
{
    mask.data().assign(dates.size(), static_cast<std::uint8_t>(1));
    apply_steps(chain, dates, spectral, qas, mask);
}

void apply_steps(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask
)
{
    assert(chain.count <= CHAIN_CAPACITY);
    assert(mask.size() == dates.size());

    const index_t T = dates.size();

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

            case StepKind::Custom:
                step.custom->apply(dates, spectral, qas, mask);
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
