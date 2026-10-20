#pragma once

// The configurations the tests run: the library's "ccdc" preset, and edits
// of it by stage and parameter (the C++ counterpart of Python's
// Config.replace). Not part of the library.

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "ccd/config/registry.hpp"
#include "ccd/strategies/register_core.hpp"

namespace golden
{

// The default configuration: pyccd's algorithm, each strategy at its
// defaults.
inline ccd::config::ConfigTree ccdc()
{
    return ccd::strategies::library_registry().preset("ccdc");
}

// `tree` with parameter `name` of stage `stage` set to `value`. The stage and
// the parameter must be in `tree` (std::out_of_range otherwise), so a
// misspelt name fails the test instead of configuring nothing.
inline ccd::config::ConfigTree with(ccd::config::ConfigTree tree, const std::string& stage,
                                    const std::string& name, ccd::config::Value value)
{
    tree.at(stage).params.at(name) = std::move(value);
    return tree;
}

// The value of parameter `name` of stage `stage` in `tree`, as a T
// (std::int64_t, double, bool, std::string or std::vector<std::int64_t>).
template<class T>
T parameter(const ccd::config::ConfigTree& tree, const std::string& stage, const std::string& name)
{
    return std::get<T>(tree.at(stage).params.at(name));
}

} // namespace golden
