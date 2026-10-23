#include "ccd/run/algorithm.hpp"

namespace ccd
{

Algorithm::Algorithm(config::Config config, const config::Registry& registry)
    : config_(config::canonical(registry, config)), registry_(&registry)
{
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

std::shared_ptr<const config::Resolved> Algorithm::for_spec(const DataSpec& spec) const
{
    const std::uint64_t key = digest(spec);

    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = resolved_.begin(); it != resolved_.end(); ++it)
        if (it->key == key && it->spec == spec)
        {
            resolved_.splice(resolved_.begin(), resolved_, it);
            return it->resolved;
        }

    auto resolved =
        std::make_shared<const config::Resolved>(config::resolve(*registry_, config_, spec));
    resolved_.push_front(Cached{key, spec, resolved});
    if (resolved_.size() > CACHE_CAPACITY)
        resolved_.pop_back();
    return resolved;
}

preprocess::PreprocessPlan plan_of(
    const config::Config& config,
    const DataSpec& spec,
    const config::Registry& registry)
{
    return config::resolve(registry, config, spec).plan;
}

} // namespace ccd
