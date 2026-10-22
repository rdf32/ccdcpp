#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
// For each procedure ("standard", "insufficient_clear", "snow"): how it reads
// the thermal row (Thermal), and its chain,
// the steps in order, each a step type by name (Registry::add_step()) with its
// parameters in physical terms -- QA codes by name, bands by role, bounds in
// reflectance and degrees Celsius. A run resolves it with the rest of the
// configuration (config::resolve()) into a preprocess::PreprocessPlan in the
// input's codes, rows and units.
//
// The library's step types are keep_qa, add_qa, thermal_range, band_range,
// green_median and one_per_date (input/mask/steps.hpp).
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

// How a procedure reads the thermal row, as a configuration names it:
//
//   Stored   "stored": as the input holds it;
//   Celsius  "celsius": through the input's thermal transform, in degrees
//            Celsius (in the unit thermal_at_100c says: pyccd's Celsius x 100
//            on Collection-1 DN).
//
// A thermal_range step tests degrees Celsius either way (ThermalRange,
// input/mask/steps.hpp). "ccdc" reads it in Celsius for Standard only, as
// pyccd does.
enum class Thermal : std::uint8_t
{
    Stored,
    Celsius
};

// "stored", "celsius".
const char* name(Thermal thermal);

// The Thermal a configuration names, none for another name.
std::optional<Thermal> thermal_named(const std::string& name);

struct ProcedureConfig
{
    Thermal thermal = Thermal::Stored;
    std::vector<StepConfig> steps;
};

inline bool operator==(const ProcedureConfig& a, const ProcedureConfig& b)
{
    return a.thermal == b.thermal && a.steps == b.steps;
}

inline bool operator!=(const ProcedureConfig& a, const ProcedureConfig& b)
{
    return !(a == b);
}

// The procedures, by name.
using PreprocessConfig = std::map<std::string, ProcedureConfig>;

// Where a step of a configuration's chain is, as its type's make() is handed
// it (Registry::add_step()): its procedure ("standard", "snow",
// "insufficient_clear"), how that procedure reads the thermal row, and the
// step's position in the chain.
struct StepContext
{
    std::string procedure;
    Thermal thermal = Thermal::Stored;
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
