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

// One call's strategies from its configuration. Throws std::invalid_argument
// on a stage the configuration lacks, a strategy its stage does not have, or
// parameters the strategy refuses.
Strategies build_strategies(const config::Registry& registry, const config::ConfigTree& config);

} // namespace strategies
} // namespace ccd
