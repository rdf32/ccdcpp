#pragma once

#include "ccd/config/registry.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

// Registers every strategy of the public library.
void register_core_strategies(config::Registry& registry);

// Registers the strategies of the extensions the library was built with
// (CMake's CCD_EXTENSIONS; none by default).
void register_extension_strategies(config::Registry& registry);

// The registry with the library's strategies and its extensions', built
// once: the one an Algorithm (run/algorithm.hpp) names its strategies in.
const config::Registry& library_registry();

} // namespace strategies
} // namespace ccd
