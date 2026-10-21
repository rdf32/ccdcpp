#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "ccd/config/params.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The configuration's preprocess section
//
// For each procedure ("standard", "snow", "insufficient_clear"): whether it
// reads the thermal row through the input's thermal transform, and its chain,
// the steps in order, each a step type by name (Registry::add_step()) with its
// parameters in physical terms -- QA codes by name, bands by role, bounds in
// reflectance and degrees Celsius. A run resolves it with the rest of the
// configuration (config::resolve()) into a preprocess::PreprocessPlan in the
// input's codes, rows and units.
//
// The library's step types are keep_qa, add_qa, thermal_range, band_range,
// green_median and dedupe (input/mask/steps.hpp).
//
//==============================================================================

struct StepConfig
{
    std::string step;
    Values params;
};

inline bool operator==(const StepConfig& a, const StepConfig& b)
{
    return a.step == b.step && a.params == b.params;
}

inline bool operator!=(const StepConfig& a, const StepConfig& b)
{
    return !(a == b);
}

struct ProcedureConfig
{
    bool thermal_transform = false;
    std::vector<StepConfig> steps;
};

inline bool operator==(const ProcedureConfig& a, const ProcedureConfig& b)
{
    return a.thermal_transform == b.thermal_transform && a.steps == b.steps;
}

inline bool operator!=(const ProcedureConfig& a, const ProcedureConfig& b)
{
    return !(a == b);
}

// The procedures, by name.
using PreprocessConfig = std::map<std::string, ProcedureConfig>;

// Where a step of a configuration's chain is, as its type's make() is handed
// it (Registry::add_step()): its procedure ("standard", "snow",
// "insufficient_clear"), whether that procedure reads the thermal row
// through the input's transform, and the step's position in the chain.
struct StepContext
{
    std::string procedure;
    bool thermal_transform = false;
    std::size_t position = 0;

    // "preprocess standard step 2", as a refusal names the step.
    std::string where() const
    {
        return "preprocess " + procedure + " step " + std::to_string(position);
    }
};

// The procedures a preprocess section holds, each once: "standard", "snow",
// "insufficient_clear".
const std::vector<std::string>& procedures();

} // namespace config
} // namespace ccd
