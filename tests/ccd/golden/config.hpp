#pragma once

// The configurations the tests run: the library's "ccdc" preset, edits of it
// by stage and parameter or by shared value (the C++ counterpart of Python's
// Config.replace), and their resolution against an input. Not part of the
// library.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "ccd/config/registry.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/input/input.hpp"
#include "ccd/run/algorithm.hpp"
#include "ccd/strategies/register_core.hpp"

namespace golden
{

// The registry with the library's own strategies, without its extensions',
// built once.
inline const ccd::config::Registry& core_registry()
{
    static const ccd::config::Registry registry = []
    {
        ccd::config::Registry r;
        ccd::strategies::register_core_strategies(r);
        return r;
    }();
    return registry;
}

// The default configuration: pyccd's algorithm, in physical terms, with no
// statistics cutoff.
inline ccd::config::Config ccdc()
{
    return ccd::strategies::library_registry().preset("ccdc");
}

// `config` with parameter `name` of stage `stage` set to `value`. The stage and
// the parameter must be in `config` (std::out_of_range otherwise), so a
// misspelt name fails the test instead of configuring nothing.
inline ccd::config::Config with(ccd::config::Config config, const std::string& stage,
                                const std::string& name, ccd::config::Value value)
{
    config.stages.at(stage).params.at(name) = std::move(value);
    return config;
}

// `config` with its shared value `name` set to `value` (std::out_of_range for
// a name the preset does not hold).
inline ccd::config::Config with_shared(ccd::config::Config config, const std::string& name,
                                       ccd::config::Value value)
{
    config.shared.at(name) = std::move(value);
    return config;
}

// `config` with parameter `name` of step `index` of procedure `procedure`
// set to `value`; std::out_of_range for a procedure, step or parameter it
// does not hold.
inline ccd::config::Config with_step(ccd::config::Config config, const std::string& procedure,
                                     std::size_t index, const std::string& name,
                                     ccd::config::Value value)
{
    config.preprocess.at(procedure).steps.at(index).params.at(name) = std::move(value);
    return config;
}

// `config` with parameter `name` of every step of type `step`, in every
// procedure, set to `value` (the "ccdc" chains share their first four
// steps); std::out_of_range when no step of that type holds it.
inline ccd::config::Config with_steps(ccd::config::Config config, const std::string& step,
                                      const std::string& name, const ccd::config::Value& value)
{
    std::size_t set = 0;
    for (auto& [procedure, chain] : config.preprocess)
        for (auto& s : chain.steps)
            if (s.step == step)
            {
                s.params.at(name) = value;
                ++set;
            }
    if (set == 0)
        throw std::out_of_range("no step '" + step + "'");
    return config;
}

// `config` with procedure `procedure`'s chain replaced by `chain`.
inline ccd::config::Config with_chain(ccd::config::Config config, const std::string& procedure,
                                      ccd::config::ProcedureConfig chain)
{
    config.preprocess.at(procedure) = std::move(chain);
    return config;
}

// pyccd's statistics cutoff, 2017-12-31: the pyccd references and the golden
// fixtures ran with it.
constexpr std::int64_t PYCCD_STATISTICS_END = 736694;

// The "ccdc" preset with pyccd's statistics cutoff: the configuration the
// pyccd references and the golden fixtures ran.
inline ccd::config::Config pyccd()
{
    return with_shared(ccdc(), "statistics_end", ccd::config::integer(PYCCD_STATISTICS_END));
}

// `config` resolved against `input`, in the library's registry.
inline ccd::config::Resolved resolved(const ccd::config::Config& config = pyccd(),
                                      const ccd::Input& input = ccd::Input{})
{
    return ccd::strategies::library_registry().resolve(config, input);
}

// The strategies of `config` on `input`.
inline ccd::Strategies strategies_of(const ccd::config::Config& config = pyccd(),
                                     const ccd::Input& input = ccd::Input{})
{
    return resolved(config, input).strategies;
}

// The shared values of `config` on `input`.
inline ccd::config::Shared shared_of(const ccd::config::Config& config = pyccd(),
                                     const ccd::Input& input = ccd::Input{})
{
    return resolved(config, input).shared;
}

// An Algorithm of `config`.
inline ccd::AlgorithmPtr algorithm(ccd::config::Config config)
{
    return std::make_shared<const ccd::Algorithm>(std::move(config));
}

// The Algorithm of pyccd(), built once.
inline ccd::AlgorithmPtr pyccd_algorithm()
{
    static const ccd::AlgorithmPtr built = algorithm(pyccd());
    return built;
}

// The value of parameter `name` of stage `stage` in a resolved tree, as a T
// (std::int64_t, double, bool, std::string or std::vector<std::int64_t>).
template<class T>
T parameter(const ccd::config::ConfigTree& tree, const std::string& stage, const std::string& name)
{
    return std::get<T>(tree.at(stage).params.at(name));
}

// The preprocessing plan of `config` on `input`.
inline ccd::preprocess::PreprocessPlan plan_of(const ccd::config::Config& config = pyccd(),
                                               const ccd::Input& input = ccd::Input{})
{
    return ccd::plan_of(config, input);
}

// The plan of procedure `procedure` ("standard", "snow", "insufficient_clear").
inline const ccd::preprocess::ProcedurePlan& procedure_of(
    const ccd::preprocess::PreprocessPlan& plan, const std::string& procedure)
{
    if (procedure == "standard")
        return plan.standard;
    if (procedure == "snow")
        return plan.snow;
    return plan.insufficient_clear;
}

// The processing mask of procedure `procedure`'s chain in `config`, resolved
// against the workspace's input, over its spectral block as it is (no
// transform applied).
inline ccd::ProcessingMask chain_mask(const ccd::PixelSeries& workspace,
                                      const std::string& procedure,
                                      const ccd::config::Config& config = pyccd())
{
    const ccd::preprocess::PreprocessPlan plan = golden::plan_of(config, workspace.input());
    ccd::ProcessingMask mask;
    ccd::preprocess::apply(procedure_of(plan, procedure).chain, workspace.dates(),
                           workspace.spectral(), workspace.qas(), mask);
    return mask;
}

} // namespace golden
