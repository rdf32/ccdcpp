#pragma once

// Internal to the library: the digest of a resolution (config::Resolved),
// which config::Registry::resolve() records.

#include <cstdint>

#include "ccd/config/registry.hpp"
#include "ccd/input/input.hpp"

namespace ccd
{
namespace config
{

// Over `input` (its digest) and a resolved configuration: its shared values
// by name, its stages and its preprocessing, in that order. Equal for equal
// runs, on any platform.
std::uint64_t resolution_digest(const Input& input, const Values& shared,
                                const ConfigTree& stages, const PreprocessConfig& preprocess);

} // namespace config
} // namespace ccd
