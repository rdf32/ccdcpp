#pragma once

#include <array>
#include <optional>
#include <string>

namespace ccd
{

//==============================================================================
//
// The per-pixel procedures
//
// Before any change detection, each pixel takes one of three procedures from
// its QA statistics, chosen by the run's mode (ModeRule,
// strategies/stages.hpp). Standard runs the change detector; PermanentSnow
// and InsufficientClear fit one curve over the whole series instead (the
// edge fit's EdgeFit::whole_series()). Each has its own preprocessing
// (preprocess::PreprocessPlan::of()), named as in the configuration's
// preprocess section:
//
//     for (PixelProcedure procedure : PROCEDURES)
//         std::cout << name(procedure) << '\n';   // standard, snow,
//                                                 // insufficient_clear
//
//==============================================================================

enum class PixelProcedure
{
    Standard,
    InsufficientClear,
    PermanentSnow
};

// Every procedure, in the order a preprocess section lists them.
constexpr std::array<PixelProcedure, 3> PROCEDURES = {
    PixelProcedure::Standard,
    PixelProcedure::PermanentSnow,
    PixelProcedure::InsufficientClear,
};

// The procedure's name in a preprocess section: "standard", "snow",
// "insufficient_clear".
const char* name(PixelProcedure procedure);

// The procedure of a preprocess section's name, none for another:
// procedure_named("snow") is PixelProcedure::PermanentSnow.
std::optional<PixelProcedure> procedure_named(const std::string& name);

} // namespace ccd
