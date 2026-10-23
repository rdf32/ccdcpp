#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ccd/config/registry.hpp"
#include "ccd/config/resolve.hpp"
#include "ccd/input/data_spec.hpp"
#include "ccd/strategies/library.hpp"

namespace ccd
{

//==============================================================================
//
// Algorithm
//
// A compiled configuration: a config::Config checked once, at construction,
// by its schemas (config::check(): the strategies, every parameter and
// shared value as written, what each strategy needs), held in full
// (config::canonical()), and resolved against each data spec it runs on once.
// Immutable, so one Algorithm serves every call and thread.
//
// for_spec() resolves the configuration against a data spec and builds its
// strategies the first time it sees that spec, and returns the same
// Resolved for every later call on an equal spec (by its digest, then
// field by field), behind a mutex. A run takes its Resolved from it, so a
// call resolves and builds nothing it has done before. The cache holds the
// CACHE_CAPACITY specs used last: one more evicts the least recently used,
// which a later call on it resolves again (a run that holds its Resolved
// keeps it alive).
//
// Algorithm::preset(name) gives the shared, built-once Algorithm of a preset
// of the library's registry.
//
//==============================================================================

class Algorithm
{
public:

    // The data specs whose resolutions an Algorithm keeps.
    static constexpr std::size_t CACHE_CAPACITY = 16;

    // Throws std::invalid_argument for what config::check() refuses.
    // `registry` must outlive the Algorithm (the library's does).
    explicit Algorithm(config::Config config,
                       const config::Registry& registry = strategies::library_registry());

    // The preset `name` of the library's registry, built on first use and
    // shared after; std::invalid_argument for a name it does not hold.
    static std::shared_ptr<const Algorithm> preset(const std::string& name);

    // The configuration, canonical() (config/resolve.hpp): every value
    // spelled out, in physical terms.
    const config::Config& config() const { return config_; }

    const config::Registry& registry() const { return *registry_; }

    // The configuration resolved against `spec` (config::resolve()),
    // cached per spec, the CACHE_CAPACITY used last. Throws what
    // config::resolve() throws; `spec` must pass validate() (ccd.hpp).
    std::shared_ptr<const config::Resolved> for_spec(const DataSpec& spec) const;

private:

    config::Config config_;
    const config::Registry* registry_;

    // One cached resolution: the spec's digest, the spec, and what it
    // resolved to.
    struct Cached
    {
        std::uint64_t key;
        DataSpec spec;
        std::shared_ptr<const config::Resolved> resolved;
    };

    mutable std::mutex mutex_;
    // Most recently used first; at most CACHE_CAPACITY.
    mutable std::list<Cached> resolved_;
};

using AlgorithmPtr = std::shared_ptr<const Algorithm>;

// The preprocessing plan `config` runs on `spec`: its preprocess section
// resolved against the spec (config::resolve()), for inspection. Throws
// what config::resolve() throws; `spec` must pass validate() (ccd.hpp).
preprocess::PreprocessPlan plan_of(
    const config::Config& config,
    const DataSpec& spec,
    const config::Registry& registry = strategies::library_registry());

} // namespace ccd
