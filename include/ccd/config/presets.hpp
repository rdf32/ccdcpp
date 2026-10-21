#pragma once

#include "ccd/config/config.hpp"
#include "ccd/config/registry.hpp"

namespace ccd
{
namespace config
{

// "ccdc": pyccd's algorithm, in physical terms, from `registry`'s defaults:
// the shared values' (alpha 1e-4, the probabilities, the detection bands by
// role, no statistics cutoff), each pyccd strategy's, and pyccd's
// preprocessing, each step at its defaults (clear and water kept, the
// thermal range -93.2 to 70.7 degrees Celsius, reflectance in (0, 1), the
// green median 0.04 above). register_core_strategies() registers it:
//
//     registry.add_preset("ccdc", config::ccdc_preset(registry));
//
// Every procedure's chain begins with the same four steps, keep_qa,
// thermal_range, band_range and dedupe; Standard then ends, reading thermal
// through the input's transform; PermanentSnow adds snow back, dedupes and
// tests the thermal range again, over the snow observations too (the first
// test keeps a clear observation with an implausible thermal value from
// taking its date in the dedupe); InsufficientClear runs the green median
// and dedupes. Every procedure's thermal range covers every observation its
// fit uses, in degrees Celsius: Standard tests the transformed row, the other
// two the stored row against the bounds taken back through the transform.
Config ccdc_preset(const Registry& registry);

} // namespace config
} // namespace ccd
