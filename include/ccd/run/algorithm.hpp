#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ccd/config/registry.hpp"
#include "ccd/input/input.hpp"
#include "ccd/strategies/register_core.hpp"

namespace ccd
{

//==============================================================================
//
// Algorithm
//
// A compiled configuration: a config::Config checked once, at construction,
// by its schemas (Registry::check(): the strategies, every parameter and
// shared value as written, what each strategy needs), and resolved against
// each input it runs on once. Immutable, so one Algorithm serves every call
// and thread.
//
// for_input() resolves the configuration against an input and builds its
// strategies the first time it sees that input, and returns the same
// Resolved for every later call on an equal input (by its digest, then
// field by field), behind a mutex. A run takes its Resolved from it, so a
// call resolves and builds nothing it has done before.
//
// Algorithm::preset(name) gives the shared, built-once Algorithm of a preset
// of the library's registry.
//
//==============================================================================

class Algorithm
{
public:

    // Throws std::invalid_argument for what Registry::check() refuses.
    // `registry` must outlive the Algorithm (the library's does).
    explicit Algorithm(config::Config config,
                       const config::Registry& registry = strategies::library_registry());

    // The preset `name` of the library's registry, built on first use and
    // shared after; std::invalid_argument for a name it does not hold.
    static std::shared_ptr<const Algorithm> preset(const std::string& name);

    const config::Config& config() const { return config_; }

    const config::Registry& registry() const { return *registry_; }

    // The configuration resolved against `input` (Registry::resolve()),
    // cached per input. Throws what Registry::resolve() throws; `input` must
    // pass validate() (ccd.hpp).
    std::shared_ptr<const config::Resolved> for_input(const Input& input) const;

private:

    config::Config config_;
    const config::Registry* registry_;

    mutable std::mutex mutex_;
    mutable std::multimap<std::uint64_t, std::pair<Input, std::shared_ptr<const config::Resolved>>>
        resolved_;
};

using AlgorithmPtr = std::shared_ptr<const Algorithm>;

// The preprocessing plan `config` runs on `input`: its preprocess section
// resolved against the input (Registry::resolve()), for inspection. Throws
// what Registry::resolve() throws; `input` must pass validate() (ccd.hpp).
preprocess::PreprocessPlan plan_of(
    const config::Config& config,
    const Input& input,
    const config::Registry& registry = strategies::library_registry());

} // namespace ccd
