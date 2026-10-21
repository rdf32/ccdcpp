#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "ccd/config/params.hpp"
#include "ccd/config/preprocess.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The algorithm's configuration
//
// A configuration (Config) holds the shared values (config/shared.hpp); for
// each stage of the algorithm, the strategy that decides it and that
// strategy's parameters (StageConfig); and the preprocessing of each
// procedure, its steps by name and their parameters (config/preprocess.hpp).
// It is written in physical terms -- reflectance, degrees Celsius, days,
// observations, probabilities, bands by role, QA codes by name -- so it means
// the same on every input. config::resolve() (config/resolve.hpp) translates
// it to an input once; a registry (config/registry.hpp) holds the strategies
// and steps it names, and its presets:
//
//     config::Config c = strategies::library_registry().preset("ccdc");
//     c.stages.at("refit").params["factor"] = config::real(2.0);
//
//==============================================================================

struct StageConfig
{
    std::string strategy;
    Values params;
};

inline bool operator==(const StageConfig& a, const StageConfig& b)
{
    return a.strategy == b.strategy && a.params == b.params;
}

inline bool operator!=(const StageConfig& a, const StageConfig& b)
{
    return !(a == b);
}

// The stages, by name.
using ConfigTree = std::map<std::string, StageConfig>;

// A configuration: the shared values, the stages and the preprocessing. A
// shared value it leaves out is shared_schema()'s default; a stage's or a
// step's parameter, its strategy's or step's default. Every stage and every
// procedure is required.
struct Config
{
    Values shared;
    ConfigTree stages;
    PreprocessConfig preprocess;
};

inline bool operator==(const Config& a, const Config& b)
{
    return a.shared == b.shared && a.stages == b.stages && a.preprocess == b.preprocess;
}

inline bool operator!=(const Config& a, const Config& b)
{
    return !(a == b);
}

// A digest of a set of values, of the stages, of the preprocessing and of a
// configuration: equal for equal ones (every value by name, every stage,
// strategy name and parameter, every procedure, step name and parameter), on
// any platform.
std::uint64_t digest(const Values& values);
std::uint64_t digest(const ConfigTree& stages);
std::uint64_t digest(const PreprocessConfig& preprocess);
std::uint64_t digest(const Config& config);

} // namespace config
} // namespace ccd
