#include "ccd/run/algorithm.hpp"

namespace ccd
{

Algorithm::Algorithm(config::Config config, const config::Registry& registry)
    : config_(std::move(config)), registry_(&registry)
{
    registry_->check(config_);
}

std::shared_ptr<const Algorithm> Algorithm::preset(const std::string& name)
{
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const Algorithm>> built;

    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = built.find(name);
    if (it != built.end())
        return it->second;
    const config::Registry& registry = strategies::library_registry();
    auto algorithm = std::make_shared<const Algorithm>(registry.preset(name), registry);
    built.emplace(name, algorithm);
    return algorithm;
}

std::shared_ptr<const config::Resolved> Algorithm::for_input(const Input& input) const
{
    const std::uint64_t key = digest(input);

    const std::lock_guard<std::mutex> lock(mutex_);
    const auto [first, last] = resolved_.equal_range(key);
    for (auto it = first; it != last; ++it)
        if (it->second.first == input)
            return it->second.second;

    auto resolved =
        std::make_shared<const config::Resolved>(registry_->resolve(config_, input));
    resolved_.emplace(key, std::make_pair(input, resolved));
    return resolved;
}

preprocess::PreprocessPlan plan_of(
    const config::Config& config,
    const Input& input,
    const config::Registry& registry)
{
    return registry.resolve(config, input).plan;
}

} // namespace ccd
