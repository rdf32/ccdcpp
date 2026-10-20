#include "ccd/input/mask/steps.hpp"

#include <stdexcept>

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

Range range_of(double min, bool min_inclusive, double max, bool max_inclusive)
{
    return Range{Bound{min, min_inclusive}, Bound{max, max_inclusive}};
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

std::optional<Step> KeepQa::make(const Params& params, const Input&)
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

std::optional<Step> AddQa::make(const Params& params, const Input&)
{
    return Step::add_qa(set_of(params.codes));
}

const std::vector<Field<ThermalRange::Params>>& ThermalRange::schema()
{
    static const std::vector<Field<Params>> s = {
        {"min", &Params::min, Unit::Celsius, config::any(),
         "the lower bound of the thermal row", Source::Own, nullptr, config::real(-93.2)},
        {"max", &Params::max, Unit::Celsius, config::any(),
         "the upper bound of the thermal row", Source::Own, nullptr, config::real(70.7)},
        {"min_inclusive", &Params::min_inclusive, Unit::None, config::any(),
         "whether a value equal to min is kept"},
        {"max_inclusive", &Params::max_inclusive, Unit::None, config::any(),
         "whether a value equal to max is kept"},
    };
    return s;
}

std::optional<Step> ThermalRange::make(const Params& params, const Input& input)
{
    const index_t thermal = input.index("thermal");
    if (thermal == NO_BAND)
        return std::nullopt;
    return Step::thermal_range(
        thermal, range_of(params.min, params.min_inclusive, params.max, params.max_inclusive));
}

const std::vector<Field<BandRange::Params>>& BandRange::schema()
{
    static const std::vector<Field<Params>> s = {
        {"bands", &Params::bands, Unit::Bands, config::any(),
         "the bands tested, in order: roles, rows, or \"reflective\" (every row but "
         "thermal and other)", Source::Own, nullptr, Value(std::string(config::REFLECTIVE))},
        {"min", &Params::min, Unit::Reflectance, config::any(),
         "the lower bound of every band", Source::Own, nullptr, config::real(0.0)},
        {"max", &Params::max, Unit::Reflectance, config::any(),
         "the upper bound of every band", Source::Own, nullptr, config::real(1.0)},
        {"min_inclusive", &Params::min_inclusive, Unit::None, config::any(),
         "whether a value equal to min is kept"},
        {"max_inclusive", &Params::max_inclusive, Unit::None, config::any(),
         "whether a value equal to max is kept"},
    };
    return s;
}

void BandRange::validate(const Params& params)
{
    if (params.bands.empty())
        throw std::invalid_argument("parameter 'bands' is empty; it names the bands tested, "
                                    "or \"reflective\"");
}

std::optional<Step> BandRange::make(const Params& params, const Input&)
{
    return Step::band_range(
        params.bands, range_of(params.min, params.min_inclusive, params.max, params.max_inclusive));
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
         "stat_ord"},
    };
    return s;
}

void GreenMedian::validate(const Params& params)
{
    if (!green_filter_scope(params.scope))
        throw std::invalid_argument("parameter 'scope' is '" + params.scope +
                                    "'; it is all, stat_window or off");
}

std::optional<Step> GreenMedian::make(const Params& params, const Input& input)
{
    return Step::green_median(input.index("green"), params.offset, params.max_date,
                              *green_filter_scope(params.scope));
}

const std::vector<Field<Dedupe::Params>>& Dedupe::schema()
{
    static const std::vector<Field<Params>> s = {};
    return s;
}

std::optional<Step> Dedupe::make(const Params&, const Input&)
{
    return Step::dedupe();
}

} // namespace preprocess
} // namespace ccd
