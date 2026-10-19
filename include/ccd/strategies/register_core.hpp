#pragma once

#include "ccd/config/registry.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"
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

// The configuration the options describe, stage by stage: every run's
// configuration is built from its options here.
config::ConfigTree config_from_options(const HarmonicOptions& hoptions,
                                       const LassoOptions& loptions);

// build_strategies(core_registry(), config_from_options(hoptions, loptions)):
// the strategies a run under these options uses.
Strategies strategies_for(const HarmonicOptions& hoptions, const LassoOptions& loptions);

} // namespace strategies
} // namespace ccd
