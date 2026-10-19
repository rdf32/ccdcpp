#include "ccd/config/registry.hpp"

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

} // namespace config
} // namespace ccd
