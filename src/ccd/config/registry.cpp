#include "ccd/config/registry.hpp"

#include <set>
#include <string>
#include <type_traits>

#include "../core/fnv.hpp"

namespace ccd
{
namespace config
{

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

void Registry::check_bands(const ConfigTree& config, index_t bands) const
{
    for (const auto& [stage, chosen] : config)
    {
        const auto s = band_checks_.find(stage);
        if (s == band_checks_.end())
            continue;
        const auto f = s->second.find(chosen.strategy);
        if (f == s->second.end())
            continue;
        f->second(chosen.params, bands, stage + " '" + chosen.strategy + "'");
    }
}

std::uint64_t digest(const ConfigTree& config)
{
    detail::Fnv h;
    const auto text = [&h](const std::string& s)
    {
        h.integer(static_cast<std::int64_t>(s.size()));
        for (const char c : s)
            h.byte(static_cast<std::uint8_t>(c));
    };
    h.integer(static_cast<std::int64_t>(config.size()));
    for (const auto& [stage, chosen] : config)
    {
        text(stage);
        text(chosen.strategy);
        h.integer(static_cast<std::int64_t>(chosen.params.size()));
        for (const auto& [name, value] : chosen.params)
        {
            text(name);
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
                        text(v);
                    else
                    {
                        h.integer(static_cast<std::int64_t>(v.size()));
                        for (const std::int64_t b : v)
                            h.integer(b);
                    }
                },
                value);
        }
    }
    return h.value();
}

Values Registry::defaults(const std::string& stage, const std::string& strategy) const
{
    find(stage, strategy);   // the same refusal as make()'s
    Values out;
    for (const ParameterInfo& p : schemas_.at(stage).at(strategy))
        out[p.name] = p.default_value;
    return out;
}

void Registry::add_preset(const std::string& name, ConfigTree config)
{
    presets_[name] = std::move(config);
}

const ConfigTree& Registry::preset(const std::string& name) const
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
