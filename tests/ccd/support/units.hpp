#pragma once

//==============================================================================
//
// The units the corpus runs in: the physical-unit input and the conversion of
// DN to it, saturation values on the DN reflectance bounds, and the raw-number
// configuration of the Collection-2 reference pixel.
//
//==============================================================================

#include <cstdint>
#include <limits>

#include "ccd/ccd.hpp"

#include "config.hpp"

namespace golden
{

// The physical-unit input of docs/units.md: data pre-divided by 10000, and
// reflectance its unit, so the configuration's reflectance values (the
// Lasso's alpha and coefficient floor, the bounds, the green filter) are its
// own, and its thermal bounds are degrees Celsius / 100.
constexpr double SCALE = 10000.0;

inline ccd::Input physical_input()
{
    ccd::Input h;
    h.reflectance_scale = 1.0;
    h.thermal_scale = 1.0;
    h.thermal_offset = 0.0;
    h.thermal_at_100c = 1.0;
    return h;
}

// DN (7, T) block -> the preset's units, in place.
inline void to_physical(double* block, std::size_t T)
{
    for (std::size_t b = 0; b < 6; ++b)
        for (std::size_t t = 0; t < T; ++t)
            block[b * T + t] = block[b * T + t] / SCALE;

    for (std::size_t t = 0; t < T; ++t)
        block[6 * T + t] =
            (block[6 * T + t] * ccd::KELVIN_SCALE - ccd::KELVIN_OFFSET) / SCALE;
}

// Puts saturation-band values on the DN reflectance bounds of a (7, T) block,
// at clear observations (QA 0) numbered 10, 60, 110, 160 and 210 in date
// order: blue 0, nir 10000, red NaN, green 10000, swir2 0.
inline void put_bound_values(double* block, const std::uint8_t* qas, std::size_t T)
{
    struct Put { std::size_t clear_index; std::size_t band; double value; };
    const Put puts[] = {
        {10, 0, 0.0},
        {60, 3, 10000.0},
        {110, 2, std::numeric_limits<double>::quiet_NaN()},
        {160, 1, 10000.0},
        {210, 5, 0.0},
    };

    std::size_t clear = 0;
    for (std::size_t t = 0; t < T; ++t)
    {
        if (qas[t] != 0)
            continue;
        for (const auto& put : puts)
            if (put.clear_index == clear)
                block[put.band * T + t] = put.value;
        ++clear;
    }
}

// A RAW-NUMBER REFERENCE CONFIGURATION, not a physical one: the two C2
// reference files' run, which applies pyccd's Collection-1 DN thresholds to
// Collection-2 DN numbers as they are. c2_input(), the physical-unit input,
// says the data's numbers are its units (reflectance_scale 1, offset 0,
// thermal_at_100c 1, the identity transform), so c2_config()'s values are the
// data's own numbers: the Lasso's alpha and coefficient floor 1 and the green
// median's offset 400 (pyccd's DN values), the band range (7273, 43636) and
// the thermal range +-1e6 (no thermal screen). The reference files pin those
// numbers. The physical description of Collection-2 DN (reflectance_scale
// 1 / 2.75e-5, reflectance_offset 0.2 / 2.75e-5) resolves "ccdc" to other
// values (test_units.cpp, Collection2DnResolvesReflectanceZeroToOne).
inline ccd::Input c2_input()
{
    return physical_input();
}

inline ccd::config::Config c2_config()
{
    using ccd::config::real;
    ccd::config::Config c = with_shared(pyccd(), "lasso.alpha", real(1.0));
    c = with_shared(c, "lasso.coef_floor", real(1.0));
    c = with_steps(c, "band_range", "reflectance", ccd::config::reals({7273.0, 43636.0}));
    // Degrees Celsius: 1e8 x 100 x thermal_at_100c 1 / 10000 is 1e6.
    c = with_steps(c, "thermal_range", "celsius", ccd::config::reals({-1.0e8, 1.0e8}));
    return with_steps(c, "green_median", "offset", real(400.0));
}

} // namespace golden
