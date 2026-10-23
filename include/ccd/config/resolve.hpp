#pragma once

#include <cstdint>

#include "ccd/config/config.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/input/data_spec.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// A configuration resolved against a data spec
//
// resolve() translates a configuration (config/config.hpp) to a data spec once:
// reflectance values times DataSpec::reflectance_scale, degrees Celsius to the
// thermal row's unit, roles to rows, code names to codes, shared values into
// the stages and steps that read them, chi-squared thresholds from
// probabilities for the spec's detection-band count; then it builds the
// strategy objects and the preprocessing plan, by name in `registry`,
// checking every parameter against its strategy's or step's schema:
//
//     const config::Registry& r = strategies::library_registry();
//     config::Resolved run = config::resolve(r, r.preset("ccdc"), spec);
//
// check() refuses what resolve() refuses on any data spec, without one, and
// canonical() writes a configuration out in full, in its own terms.
//
//==============================================================================

// What a run runs.
struct Resolved
{
    // Every stage's every parameter spelled out, in the spec's units and
    // rows, shared values and derived thresholds included.
    ConfigTree stages;
    // The shared values, in the spec's units and rows.
    Shared shared;
    // Every procedure's every step's every parameter spelled out, in the
    // spec's units, rows and codes.
    PreprocessConfig preprocess;
    // One strategy per stage, built from `stages`.
    Strategies strategies;
    // The preprocessing plan, built from `preprocess`: a step that does not
    // apply to the spec (thermal_range without a thermal row) is left out.
    preprocess::PreprocessPlan plan;
    // Over the data spec and the resolved configuration (stages, shared and
    // preprocess): equal for equal runs, on any platform.
    std::uint64_t digest = 0;
};

// Throws std::invalid_argument for what resolve() refuses on any data spec: a
// stage the algorithm does not have or a missing one, an unknown strategy,
// shared value or parameter, a stage setting a shared or derived parameter,
// a value of the wrong type or outside its bounds as written, an unknown role
// or QA code name, what a strategy needs that the stages before it do not
// provide; a procedure the preprocessing does not have or a missing one, an
// unknown step type, a chain of more than preprocess::CHAIN_CAPACITY steps,
// and a step's parameters as a stage's.
void check(const Registry& registry, const Config& config);

// `config` in its own terms, as one form for every way of writing it: every
// shared value and every stage's and step's own parameters spelled out (the
// defaults filling what it leaves out), and a whole number given for a real
// value as that real (each of a range's). Refuses what check() refuses. Two configurations that
// differ only in the defaults they leave out, or in a real written as a
// whole number, are equal here, and so are their digests.
Config canonical(const Registry& registry, const Config& config);

// `config` against `spec`: every stage's and step's parameters resolved
// (its defaults filling what `config` leaves out, reflectance and Celsius
// values in the spec's units, roles as its rows, code names as its
// codes, the shared values and the derived thresholds filled in), the
// strategies and the plan built, and the digest. Refuses what check()
// refuses, a role the spec lacks or a band list naming a row it does not
// have, or repeating one, and a plan preprocess::validate() refuses.
// `spec` must pass validate() (input/data_spec.hpp).
Resolved resolve(const Registry& registry, const Config& config, const DataSpec& spec);

// The strategies of a resolved tree. Throws std::invalid_argument on a stage
// the tree lacks, a strategy its stage does not have, parameters the
// strategy refuses, or a strategy needing what the stages before it do not
// provide.
Strategies build_strategies(const Registry& registry, const ConfigTree& resolved);

} // namespace config
} // namespace ccd
