#pragma once

#include <cstdint>

#include "ccd/config/config.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// A configuration resolved against an input
//
// resolve() translates a configuration (config/config.hpp) to an input once:
// reflectance values times Input::reflectance_scale, degrees Celsius to the
// thermal row's unit, roles to rows, code names to codes, shared values into
// the stages and steps that read them, chi-squared thresholds from
// probabilities for the input's detection-band count; then it builds the
// strategy objects and the preprocessing plan, by name in `registry`,
// checking every parameter against its strategy's or step's schema:
//
//     const config::Registry& r = strategies::library_registry();
//     config::Resolved run = config::resolve(r, r.preset("ccdc"), input);
//
// check() refuses what resolve() refuses on any input, without one.
//
//==============================================================================

// What a run runs.
struct Resolved
{
    // Every stage's every parameter spelled out, in the input's units and
    // rows, shared values and derived thresholds included.
    ConfigTree stages;
    // The shared values, in the input's units and rows.
    Shared shared;
    // Every procedure's every step's every parameter spelled out, in the
    // input's units, rows and codes.
    PreprocessConfig preprocess;
    // One strategy per stage, built from `stages`.
    Strategies strategies;
    // The preprocessing plan, built from `preprocess`: a step that does not
    // apply to the input (thermal_range without a thermal row) is left out.
    preprocess::PreprocessPlan plan;
    // Over the input and the resolved configuration (stages, shared and
    // preprocess): equal for equal runs, on any platform.
    std::uint64_t digest = 0;
};

// Throws std::invalid_argument for what resolve() refuses on any input: a
// stage the algorithm does not have or a missing one, an unknown strategy,
// shared value or parameter, a stage setting a shared or derived parameter,
// a value of the wrong type or outside its bounds as written, an unknown role
// or QA code name, what a strategy needs that the stages before it do not
// provide; a procedure the preprocessing does not have or a missing one, an
// unknown step type, a chain of more than preprocess::CHAIN_CAPACITY steps,
// and a step's parameters as a stage's.
void check(const Registry& registry, const Config& config);

// `config` against `input`: every stage's and step's parameters resolved
// (its defaults filling what `config` leaves out, reflectance and Celsius
// values in the input's units, roles as input's rows, code names as its
// codes, the shared values and the derived thresholds filled in), the
// strategies and the plan built, and the digest. Refuses what check()
// refuses, a role the input lacks or a band list naming a row it does not
// have, or repeating one, and a plan preprocess::validate() refuses.
// `input` must pass validate() (input/input.hpp).
Resolved resolve(const Registry& registry, const Config& config, const Input& input);

// The strategies of a resolved tree. Throws std::invalid_argument on a stage
// the tree lacks, a strategy its stage does not have, parameters the
// strategy refuses, or a strategy needing what the stages before it do not
// provide.
Strategies build_strategies(const Registry& registry, const ConfigTree& resolved);

} // namespace config
} // namespace ccd
