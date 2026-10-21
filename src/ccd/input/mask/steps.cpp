#include "ccd/input/mask/steps.hpp"

#include <stdexcept>

#include "ccd/config/shared.hpp"

namespace ccd
{
namespace preprocess
{

using config::Field;
using config::Source;
using config::Unit;
using config::Value;

namespace
{

QaSet set_of(const std::vector<std::int64_t>& codes)
{
    QaSet set;
    for (const std::int64_t code : codes)
        set.set(static_cast<std::size_t>(code));
    return set;
}

Range range_of(const config::Interval& ends, const config::Inclusive& inclusive)
{
    return Range{Bound{ends[0], inclusive[0]}, Bound{ends[1], inclusive[1]}};
}

} // namespace

const std::vector<Field<KeepQa::Params>>& KeepQa::schema()
{
    static const std::vector<Field<Params>> s = {
        {"codes", &Params::codes, Unit::QaCodes, config::between(0, 255),
         "the QA codes kept, by name or code", Source::Own, nullptr,
         config::names({"clear", "water"})},
    };
    return s;
}

std::optional<Step> KeepQa::make(const Params& params, const Input&, const config::StepContext&)
{
    return Step::keep_qa(set_of(params.codes));
}

const std::vector<Field<AddQa::Params>>& AddQa::schema()
{
    static const std::vector<Field<Params>> s = {
        {"codes", &Params::codes, Unit::QaCodes, config::between(0, 255),
         "the QA codes added back, by name or code", Source::Own, nullptr,
         config::names({"snow"})},
    };
    return s;
}

std::optional<Step> AddQa::make(const Params& params, const Input&, const config::StepContext&)
{
    return Step::add_qa(set_of(params.codes));
}

const std::vector<Field<ThermalRange::Params>>& ThermalRange::schema()
{
    static const std::vector<Field<Params>> s = {
        {"celsius", &Params::celsius, Unit::Celsius, config::any(),
         "the range of the thermal row, [min, max]", Source::Own, nullptr,
         config::reals({-93.2, 70.7})},
        {"inclusive", &Params::inclusive, Unit::None, config::any(),
         "whether a value equal to each end is kept, [min, max]"},
    };
    return s;
}

std::optional<Step> ThermalRange::make(const Params& params, const Input& input,
                                       const config::StepContext& context)
{
    const index_t thermal = input.index("thermal");
    if (thermal == NO_BAND)
        return std::nullopt;
    const Range range = range_of(params.celsius, params.inclusive);
    // A procedure that does not transform its thermal row tests the stored row
    // against the bounds taken back through the transform.
    const std::optional<Affine> transform = thermal_transform(input);
    if (context.thermal == config::Thermal::Celsius || !transform)
        return Step::thermal_range(thermal, range);
    return Step::thermal_range(thermal, stored_range(range, *transform));
}

const std::vector<Field<BandRange::Params>>& BandRange::schema()
{
    static const std::vector<Field<Params>> s = {
        {"bands", &Params::bands, Unit::Bands, config::any(),
         "the bands tested, in order: roles, rows, or \"reflective\" (every row but "
         "thermal and other)", Source::Own, nullptr, Value(std::string(config::REFLECTIVE))},
        {"reflectance", &Params::reflectance, Unit::Reflectance, config::any(),
         "the range of every band, [min, max]", Source::Own, nullptr, config::reals({0.0, 1.0})},
        {"inclusive", &Params::inclusive, Unit::None, config::any(),
         "whether a value equal to each end is kept, [min, max]"},
    };
    return s;
}

void BandRange::validate(const Params& params)
{
    if (params.bands.empty())
        throw std::invalid_argument("parameter 'bands' is empty; it names the bands tested, "
                                    "or \"reflective\"");
}

std::optional<Step> BandRange::make(const Params& params, const Input&, const config::StepContext&)
{
    return Step::band_range(
        params.bands, range_of(params.reflectance, params.inclusive));
}

const std::vector<Field<GreenMedian::Params>>& GreenMedian::schema()
{
    static const std::vector<Field<Params>> s = {
        {"offset", &Params::offset, Unit::ReflectanceDelta, config::any(),
         "the green above the median an observation is dropped at", Source::Own, nullptr,
         config::real(0.04)},
        {"scope", &Params::scope, Unit::None, config::any(),
         "which observations it drops: all, stat_window or off"},
        {"max_date", &Params::max_date, Unit::None, config::any(),
         "the last date of the observations the median is taken over", Source::Shared,
         "statistics_end", std::nullopt, config::Keyword{"none", config::NO_STATISTICS_END}},
    };
    return s;
}

void GreenMedian::validate(const Params& params)
{
    if (!green_filter_scope(params.scope))
        throw std::invalid_argument("parameter 'scope' is '" + params.scope +
                                    "'; it is all, stat_window or off");
}

std::optional<Step> GreenMedian::make(const Params& params, const Input& input, const config::StepContext&)
{
    return Step::green_median(input.index("green"), params.offset, params.max_date,
                              *green_filter_scope(params.scope));
}

const std::vector<Field<OnePerDate::Params>>& OnePerDate::schema()
{
    static const std::vector<Field<Params>> s = {};
    return s;
}

std::optional<Step> OnePerDate::make(const Params&, const Input&, const config::StepContext&)
{
    return Step::one_per_date();
}

} // namespace preprocess
} // namespace ccd
