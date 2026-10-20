#include "ccd/config/registry.hpp"

#include <set>
#include <string>
#include <type_traits>

#include "ccd/machine/fit.hpp"

#include "../core/fnv.hpp"

namespace ccd
{
namespace config
{

namespace
{

std::string listed(const std::vector<std::string>& names)
{
    std::string out;
    for (const std::string& n : names)
        out += (out.empty() ? "" : ", ") + n;
    return out;
}

// `value` of a field in `unit`, as the input holds it: a reflectance times
// input->reflectance_unit, a band list by role as the input's rows. Without
// an input (a check), the value as written, and none for a band list by
// role once its roles are known ones. `at` names the field in messages.
std::optional<Value> convert(Unit unit, const Value& value, const std::string& at,
                             const Input* input)
{
    if (unit == Unit::Reflectance && input)
    {
        if (const auto* v = std::get_if<double>(&value))
            return real(*v * input->reflectance_unit);
        if (const auto* v = std::get_if<std::int64_t>(&value))
            return real(static_cast<double>(*v) * input->reflectance_unit);
        return value;
    }
    if (unit == Unit::Bands)
    {
        const auto* roles = std::get_if<std::vector<std::string>>(&value);
        if (!roles)
            return value;
        std::vector<std::int64_t> rows;
        for (const std::string& role : *roles)
        {
            if (!is_role(role))
                throw std::invalid_argument(at + " holds '" + role +
                                            "', which is not a role; the roles are " +
                                            listed(ROLES));
            if (role == "other")
                throw std::invalid_argument(at + " holds 'other', which no band list names");
            if (!input)
                continue;
            const index_t row = input->index(role);
            if (row == NO_BAND)
                throw std::invalid_argument(at + " holds '" + role +
                                            "', which the input does not have; its rows are " +
                                            listed(input->roles));
            rows.push_back(static_cast<std::int64_t>(row));
        }
        if (!input)
            return std::nullopt;
        return Value(std::move(rows));
    }
    return value;
}

// The number of bands a band-list value names, by role or by row.
index_t list_size(const Value& value)
{
    if (const auto* roles = std::get_if<std::vector<std::string>>(&value))
        return roles->size();
    if (const auto* rows = std::get_if<std::vector<std::int64_t>>(&value))
        return rows->size();
    return 0;
}

// The shared values of `given` resolved against `input` (null: as written),
// into `shared`; returns the number of detection bands written.
index_t resolve_shared(const Values& given, const Input* input, Shared& shared)
{
    const auto& schema = shared_schema();
    Values merged = shared_defaults();
    for (const auto& [name, value] : given)
    {
        if (!merged.count(name))
        {
            std::vector<std::string> names;
            for (const auto& f : schema)
                names.push_back(f.name);
            throw std::invalid_argument("shared has no value '" + name + "'; it has " +
                                        listed(names));
        }
        merged[name] = value;
    }

    Values values;
    for (const auto& field : schema)
    {
        const std::string at = std::string("shared value '") + field.name + "'";
        if (auto v = convert(field.unit, merged.at(field.name), at, input))
            values[field.name] = std::move(*v);
    }
    apply(schema, values, shared, "shared");

    const index_t detection = list_size(merged.at("detection_bands"));
    if (detection == 0)
        throw std::invalid_argument("shared value 'detection_bands' is empty");
    if (input)
        check_band_list(shared.detection_bands, input->bands(), "shared value 'detection_bands'");
    if (!green_filter_scope(shared.green_filter_scope))
        throw std::invalid_argument("shared value 'green_filter_scope' is '" +
                                    shared.green_filter_scope +
                                    "'; it is all, stat_window or off");
    return detection;
}

void hash(detail::Fnv& h, const Values& values)
{
    h.integer(static_cast<std::int64_t>(values.size()));
    for (const auto& [name, value] : values)
    {
        h.text(name);
        h.byte(static_cast<std::uint8_t>(value.index()));
        std::visit(
            [&](const auto& v)
            {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bool>)
                    h.flag(v);
                else if constexpr (std::is_same_v<T, std::int64_t>)
                    h.integer(v);
                else if constexpr (std::is_same_v<T, double>)
                    h.real(v);
                else if constexpr (std::is_same_v<T, std::string>)
                    h.text(v);
                else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
                {
                    h.integer(static_cast<std::int64_t>(v.size()));
                    for (const std::int64_t b : v)
                        h.integer(b);
                }
                else
                {
                    h.integer(static_cast<std::int64_t>(v.size()));
                    for (const std::string& r : v)
                        h.text(r);
                }
            },
            value);
    }
}

void hash(detail::Fnv& h, const ConfigTree& stages)
{
    h.integer(static_cast<std::int64_t>(stages.size()));
    for (const auto& [stage, chosen] : stages)
    {
        h.text(stage);
        h.text(chosen.strategy);
        hash(h, chosen.params);
    }
}

const StageConfig& stage_of(const ConfigTree& config, const char* stage)
{
    const auto it = config.find(stage);
    if (it == config.end())
        throw std::invalid_argument(std::string("the configuration has no '") + stage +
                                    "' stage");
    return it->second;
}

} // namespace

const Registry::Factory* Registry::find(const std::string& stage, const std::string& strategy) const
{
    const auto s = factories_.find(stage);
    if (s != factories_.end())
    {
        const auto f = s->second.find(strategy);
        if (f != s->second.end())
            return &f->second;
    }
    std::string known;
    if (s != factories_.end())
        for (const auto& [name, factory] : s->second)
            known += std::string(known.empty() ? "" : ", ") + name;
    throw std::invalid_argument(stage + " has no strategy '" + strategy + "'" +
                                (known.empty() ? "" : "; it has " + known));
}

std::map<std::string, std::vector<std::string>> Registry::names() const
{
    std::map<std::string, std::vector<std::string>> out;
    for (const auto& [stage, strategies] : factories_)
        for (const auto& [name, factory] : strategies)
            out[stage].push_back(name);
    return out;
}

void check_band_list(const BandList& list, index_t bands, const std::string& where)
{
    std::set<index_t> seen;
    for (const index_t band : list)
    {
        if (band >= bands)
            throw std::invalid_argument(where + " holds band " + std::to_string(band) +
                                        "; the input has " + std::to_string(bands) + " bands");
        if (!seen.insert(band).second)
            throw std::invalid_argument(where + " holds band " + std::to_string(band) +
                                        " twice");
    }
}

std::uint64_t digest(const Values& values)
{
    detail::Fnv h;
    hash(h, values);
    return h.value();
}

std::uint64_t digest(const ConfigTree& stages)
{
    detail::Fnv h;
    hash(h, stages);
    return h.value();
}

std::uint64_t digest(const Config& config)
{
    detail::Fnv h;
    hash(h, config.shared);
    hash(h, config.stages);
    return h.value();
}

Values Registry::defaults(const std::string& stage, const std::string& strategy) const
{
    find(stage, strategy);   // the same refusal as make()'s
    Values out;
    for (const ParameterInfo& p : schemas_.at(stage).at(strategy))
        if (p.source == Source::Own)
            out[p.name] = *p.default_value;
    return out;
}

Values Registry::Resolving::stage(const std::vector<ParameterInfo>& info, const Values& given,
                                  const std::string& where) const
{
    for (const auto& [name, value] : given)
    {
        const ParameterInfo* field = nullptr;
        for (const ParameterInfo& p : info)
            if (p.name == name)
                field = &p;
        if (!field)
        {
            std::vector<std::string> own;
            for (const ParameterInfo& p : info)
                if (p.source == Source::Own)
                    own.push_back(p.name);
            throw std::invalid_argument(where + " has no parameter '" + name + "'" +
                                        (own.empty() ? "" : "; it has " + listed(own)));
        }
        if (field->source != Source::Own)
            throw std::invalid_argument(
                where + " parameter '" + name + "' is " +
                (field->source == Source::Shared ? "the shared value '" + field->from + "'"
                                                 : "derived from the shared value '" +
                                                       field->from + "'") +
                "; set shared '" + field->from + "' instead");
    }

    Values out;
    for (const ParameterInfo& p : info)
    {
        switch (p.source)
        {
            case Source::Own:
            {
                const auto it = given.find(p.name);
                const Value& value = it != given.end() ? it->second : *p.default_value;
                if (auto v = convert(p.unit, value, where + " parameter '" + p.name + "'", input))
                    out[p.name] = std::move(*v);
                break;
            }
            case Source::Shared:
                out[p.name] = shared.at(p.from);
                break;
            case Source::Chi2:
                out[p.name] = real(chi2_quantile(std::get<double>(shared.at(p.from)),
                                                 detection_bands));
                break;
        }
    }
    return out;
}

ConfigTree Registry::resolve_stages(const Config& config, const Resolving& with) const
{
    ConfigTree out;
    for (const auto& [stage, chosen] : config.stages)
    {
        find(stage, chosen.strategy);   // the same refusal as make()'s
        out[stage] = {chosen.strategy,
                      resolvers_.at(stage).at(chosen.strategy)(chosen.params, with)};
    }
    return out;
}

void Registry::check(const Config& config) const
{
    Resolving with;
    Shared shared;
    with.detection_bands = resolve_shared(config.shared, nullptr, shared);
    with.shared = to_values(shared_schema(), shared);
    build(resolve_stages(config, with));
}

Resolved Registry::resolve(const Config& config, const Input& input) const
{
    Resolved out;
    Resolving with;
    with.input = &input;
    with.detection_bands = resolve_shared(config.shared, &input, out.shared);
    with.shared = to_values(shared_schema(), out.shared);
    out.stages = resolve_stages(config, with);
    out.strategies = build(out.stages);

    detail::Fnv h;
    h.integer(static_cast<std::int64_t>(digest(input)));
    hash(h, with.shared);
    hash(h, out.stages);
    out.digest = h.value();
    return out;
}

Strategies Registry::build(const ConfigTree& config) const
{
    Strategies s;
    s.mode = make<ModeRule>(stage_of(config, ModeRule::stage));
    s.noise_floor = make<NoiseFloor>(stage_of(config, NoiseFloor::stage));
    s.start_search = make<StartSearch>(stage_of(config, StartSearch::stage));
    s.window_screen = make<WindowScreen>(stage_of(config, WindowScreen::stage));
    s.stability_fit = make<StabilityFit>(stage_of(config, StabilityFit::stage));
    s.stability_rule = make<StabilityRule>(stage_of(config, StabilityRule::stage));
    s.lookback = make<Lookback>(stage_of(config, Lookback::stage));
    s.model = make<MonitoringModel>(stage_of(config, MonitoringModel::stage));
    s.refit = make<RefitSchedule>(stage_of(config, RefitSchedule::stage));
    s.change_test = make<ChangeTest>(stage_of(config, ChangeTest::stage));
    s.outlier = make<OutlierRule>(stage_of(config, OutlierRule::stage));
    s.segment_fit = make<SegmentFit>(stage_of(config, SegmentFit::stage));
    s.edge_fit = make<EdgeFit>(stage_of(config, EdgeFit::stage));

    // Every strategy's needs, from the stages before it.
    const auto named = [&](const char* stage)
    { return std::string(stage) + " '" + config.at(stage).strategy + "'"; };
    const auto require = [](Capabilities needs, Capabilities provided, const std::string& who,
                            const std::string& providers)
    {
        const Capabilities missing = needs.missing_from(provided);
        if (!missing.empty())
            throw std::invalid_argument(who + " needs " + missing.names() + ", which " +
                                        providers + " does not provide");
    };
    require(s.stability_rule->needs(), s.stability_fit->provides(), named(StabilityRule::stage),
            named(StabilityFit::stage));
    require(s.lookback->needs(), s.stability_fit->provides(), named(Lookback::stage),
            named(StabilityFit::stage));
    const Capabilities model = s.model->provides();
    require(s.change_test->needs(), model, named(ChangeTest::stage), named(MonitoringModel::stage));
    require(s.outlier->needs(), model | s.change_test->provides(), named(OutlierRule::stage),
            named(MonitoringModel::stage) + " with " + named(ChangeTest::stage));
    return s;
}

void Registry::add_preset(const std::string& name, Config config)
{
    if (!presets_.emplace(name, std::move(config)).second)
        throw std::logic_error("preset '" + name + "' registered twice");
}

const Config& Registry::preset(const std::string& name) const
{
    const auto p = presets_.find(name);
    if (p == presets_.end())
    {
        std::string names;
        for (const auto& [n, _] : presets_)
            names += std::string(names.empty() ? "" : ", ") + "\"" + n + "\"";
        throw std::invalid_argument("no preset '" + name + "'; there are " + names);
    }
    return p->second;
}

std::vector<std::string> Registry::presets() const
{
    std::vector<std::string> out;
    for (const auto& [n, _] : presets_)
        out.push_back(n);
    return out;
}

} // namespace config
} // namespace ccd
