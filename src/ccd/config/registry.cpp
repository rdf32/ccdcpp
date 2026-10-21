#include "ccd/config/registry.hpp"

#include <string>

#include "convert.hpp"

namespace ccd
{
namespace config
{

const StrategyType& Registry::strategy(const std::string& stage, const std::string& name) const
{
    const auto s = strategies_.find(stage);
    if (s != strategies_.end())
    {
        const auto t = s->second.find(name);
        if (t != s->second.end())
            return t->second;
    }
    std::string known;
    if (s != strategies_.end())
        for (const auto& [n, _] : s->second)
            known += std::string(known.empty() ? "" : ", ") + n;
    throw std::invalid_argument(stage + " has no strategy '" + name + "'" +
                                (known.empty() ? "" : "; it has " + known));
}

const StepType& Registry::step(const std::string& name, const std::string& where) const
{
    const auto it = steps_.find(name);
    if (it != steps_.end())
        return it->second;
    std::vector<std::string> known;
    for (const auto& [n, _] : steps_)
        known.push_back(n);
    throw std::invalid_argument(where + " is '" + name + "', which is not a preprocessing step; "
                                "the steps are " + listed(known));
}

std::map<std::string, std::vector<std::string>> Registry::names() const
{
    std::map<std::string, std::vector<std::string>> out;
    for (const auto& [stage, types] : strategies_)
        for (const auto& [name, _] : types)
            out[stage].push_back(name);
    return out;
}

std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>> Registry::schemas() const
{
    std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>> out;
    for (const auto& [stage, types] : strategies_)
        for (const auto& [name, type] : types)
            out[stage][name] = type.info;
    return out;
}

Values Registry::defaults(const std::string& stage, const std::string& strategy) const
{
    Values out;
    for (const ParameterInfo& p : this->strategy(stage, strategy).info)
        if (p.source == Source::Own)
            out[p.name] = *p.default_value;
    return out;
}

std::map<std::string, std::vector<ParameterInfo>> Registry::step_schemas() const
{
    std::map<std::string, std::vector<ParameterInfo>> out;
    for (const auto& [name, type] : steps_)
        out[name] = type.info;
    return out;
}

Values Registry::step_defaults(const std::string& step) const
{
    Values out;
    for (const ParameterInfo& p : this->step(step, "the step").info)
        if (p.source == Source::Own)
            out[p.name] = *p.default_value;
    return out;
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
