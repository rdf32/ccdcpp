#pragma once

#include "ccd/config/registry.hpp"
#include "ccd/input/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The configuration the options describe
//
// Every run's configuration is built from its HarmonicOptions and
// LassoOptions here, stage by stage: each strategy parameter is the option
// it replaces (docs/ccdc-state-machine.md lists them).
//
//==============================================================================

ConfigTree from_options(const HarmonicOptions& hoptions, const LassoOptions& loptions);

// build_strategies(core_registry(), from_options(hoptions, loptions)): the
// strategies a run under these options uses.
Strategies strategies_for(const HarmonicOptions& hoptions, const LassoOptions& loptions);

} // namespace config
} // namespace ccd
