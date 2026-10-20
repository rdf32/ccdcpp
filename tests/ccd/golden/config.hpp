#pragma once

// The configurations the tests run: the library's "ccdc" preset, edits of it
// by stage and parameter or by shared value (the C++ counterpart of Python's
// Config.replace), and their resolution against an input. Not part of the
// library.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "ccd/config/registry.hpp"
#include "ccd/input/input.hpp"
#include "ccd/run/algorithm.hpp"
#include "ccd/strategies/register_core.hpp"

namespace golden
{

// The default configuration: pyccd's algorithm, in physical terms.
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

// `config` resolved against `input`, in the library's registry.
inline ccd::config::Resolved resolved(const ccd::config::Config& config = ccdc(),
                                      const ccd::Input& input = ccd::Input{})
{
    return ccd::strategies::library_registry().resolve(config, input);
}

// The strategies of `config` on `input`.
inline ccd::Strategies strategies_of(const ccd::config::Config& config = ccdc(),
                                     const ccd::Input& input = ccd::Input{})
{
    return resolved(config, input).strategies;
}

// The shared values of `config` on `input`.
inline ccd::config::Shared shared_of(const ccd::config::Config& config = ccdc(),
                                     const ccd::Input& input = ccd::Input{})
{
    return resolved(config, input).shared;
}

// An Algorithm of `config`.
inline ccd::AlgorithmPtr algorithm(ccd::config::Config config = ccdc())
{
    return std::make_shared<const ccd::Algorithm>(std::move(config));
}

// The value of parameter `name` of stage `stage` in a resolved tree, as a T
// (std::int64_t, double, bool, std::string or std::vector<std::int64_t>).
template<class T>
T parameter(const ccd::config::ConfigTree& tree, const std::string& stage, const std::string& name)
{
    return std::get<T>(tree.at(stage).params.at(name));
}

} // namespace golden
