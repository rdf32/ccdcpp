#pragma once

// Internal to the library: how config::resolve() and config::check() fill a
// section's values from a configuration (convert.cpp).

#include <string>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/input/input.hpp"

namespace ccd
{
namespace config
{

// Where a section's values are resolved: against an input, or (check()) as
// written; with the shared values already resolved there.
struct SectionContext
{
    const Input* input = nullptr;   // null: as written
    Values shared;                  // the resolved shared values, by name
    index_t detection_bands = 0;
};

// The values of a section -- a stage's or a step's parameters, or the shared
// values -- for its schema `info`, from `given`: each Own one given or at its
// default, converted to the input's units, rows and codes (as written
// without an input, and left out while a list is by name); each Shared one
// the shared value it names; each Chi2 one chi2_quantile() of it for the
// detection bands. Throws std::invalid_argument naming `where` and the
// `noun` ("refit 'span-factor' has no parameter 'x'", "shared value 'x'")
// for a name `info` does not hold, a name a Shared or Chi2 source fills, an
// unknown role or QA code name, a role the input lacks.
Values section_values(const std::vector<ParameterInfo>& info, const Values& given,
                      const std::string& where, const SectionContext& with,
                      const char* noun = "parameter");

} // namespace config
} // namespace ccd
