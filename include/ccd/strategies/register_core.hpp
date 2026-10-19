#pragma once

#include "ccd/config/registry.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

// Registers every strategy of the public library.
void register_core_strategies(config::Registry& registry);

// The registry with the library's strategies, built once.
const config::Registry& core_registry();

// Registers the strategies of the extensions the library was built with
// (CMake's CCD_EXTENSIONS; none by default).
void register_extension_strategies(config::Registry& registry);

// The registry with the library's strategies and its extensions', built
// once: the one a run's configuration names its strategies in.
const config::Registry& library_registry();

// One call's strategies from its configuration. Throws std::invalid_argument
// on a stage the configuration lacks, a strategy its stage does not have, or
// parameters the strategy refuses.
Strategies build_strategies(const config::Registry& registry, const config::ConfigTree& config);

} // namespace strategies
} // namespace ccd
