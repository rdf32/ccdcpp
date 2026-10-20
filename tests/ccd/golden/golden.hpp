#pragma once

//==============================================================================
//
// Canonical text form of run_pixel() output, for bit-exact comparison.
//
// Every floating point field is written with "%a" (hexfloat), so two dumps are
// byte-equal exactly when every value is bit-equal. The processing mask is
// written as hex nibbles, four observations per character.
//
// The fixture these dumps are compared against is a property of the platform
// that produced it: libm, the compiler's floating point contraction and Eigen's
// kernels all reach the low bits. platform() names that platform so a
// mismatch on a different one can be reported as such instead of as a
// regression.
//
//==============================================================================

#include <cstdarg>
#include <cstdio>
#include <limits>
#include <string>

#include <Eigen/Core>

#include "ccd/ccd.hpp"

#include "config.hpp"

namespace golden
{

inline void append(std::string& out, const char* fmt, ...)
{
    char buf[256];
    va_list args;
    va_start(args, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    out.append(buf, static_cast<std::size_t>(n));
}

inline void dump_models(std::string& out, const ccd::FitResult& result)
{
    for (const auto& m : result.models)
    {
        append(out, "M %lld %lld %lld %zu %a %d\n",
            static_cast<long long>(m.start_day),
            static_cast<long long>(m.end_day),
            static_cast<long long>(m.break_day),
            static_cast<std::size_t>(m.observation_count),
            m.change_probability,
            static_cast<int>(m.curve_qa));

        for (const auto& b : m.bands)
        {
            append(out, " %zu %a %a %a",
                static_cast<std::size_t>(b.iterations),
                b.intercept,
                b.rmse,
                b.magnitude);

            for (const auto c : b.coefficients)
                append(out, " %a", c);

            out.push_back('\n');
        }
    }
}

inline void dump_mask(std::string& out, const ccd::ProcessingMask& mask)
{
    static const char* HEX = "0123456789abcdef";

    append(out, "K %zu ", static_cast<std::size_t>(mask.size()));

    const auto& bits = mask.data();
    for (std::size_t i = 0; i < bits.size(); i += 4)
    {
        unsigned nibble = 0;
        for (std::size_t j = 0; j < 4 && i + j < bits.size(); ++j)
            nibble |= (bits[i + j] ? 1u : 0u) << j;
        out.push_back(HEX[nibble]);
    }
    out.push_back('\n');
}

inline void dump_pixel(
    std::string& out,
    const std::string& label,
    const ccd::FitResult& result,
    bool with_mask = true
)
{
    append(out, "P %s %zu\n", label.c_str(), result.models.size());
    dump_models(out, result);
    if (with_mask)
        dump_mask(out, result.mask);
}

//------------------------------------------------------------------------------
// Option sets the corpus is run under.
//------------------------------------------------------------------------------

// The physical-unit input of input.hpp: data pre-divided by 10000, and
// reflectance its unit, so the configuration's reflectance values (the
// Lasso's alpha and coefficient floor, the bounds, the green filter) are its
// own, and its thermal bounds are degrees Celsius / 100.
constexpr double SCALE = 10000.0;

inline ccd::Input physical_input()
{
    ccd::Input h;
    h.reflectance_unit = 1.0;
    h.thermal_scale = 1.0;
    h.thermal_offset = 0.0;
    h.thermal_unit = 1.0;
    h.saturated.reset();
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

// The two C2 reference files' run: Collection-2 DN with pyccd's Collection-1
// DN algorithm values and Collection-2 bounds. The input's reflectance_unit
// is 1, so c2_config()'s values are the data's own numbers: the Lasso's
// alpha and coefficient floor 1 and the green median's offset 400 (pyccd's
// DN values), the band range (7273, 43636) and the thermal range +-1e6 (no
// thermal screen). Those cannot be written in reflectance times 10000:
// 0.7273 x 10000 is 7272.999999999999.
inline ccd::Input c2_input()
{
    ccd::Input h;
    h.reflectance_unit = 1.0;
    h.thermal_scale = 1.0;
    h.thermal_offset = 0.0;
    h.thermal_unit = 1.0;
    h.saturated.reset();
    return h;
}

inline ccd::config::Config c2_config()
{
    using ccd::config::real;
    ccd::config::Config c = with_shared(pyccd(), "alpha", real(1.0));
    c = with_shared(c, "coef_floor", real(1.0));
    c = with_steps(c, "band_range", "min", real(7273.0));
    c = with_steps(c, "band_range", "max", real(43636.0));
    // Degrees Celsius: 1e8 x 100 x thermal_unit 1 / 10000 is 1e6.
    c = with_steps(c, "thermal_range", "min", real(-1.0e8));
    c = with_steps(c, "thermal_range", "max", real(1.0e8));
    return with_steps(c, "green_median", "offset", real(400.0));
}

//------------------------------------------------------------------------------
// The platform a fixture belongs to.
//------------------------------------------------------------------------------

inline std::string platform()
{
    std::string p;

#if defined(__clang__)
    p += "clang " __clang_version__;
#elif defined(_MSC_VER)
    p += "msvc " + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
    p += "gcc " __VERSION__;
#else
    p += "unknown-compiler";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    p += " / arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    p += " / x86_64";
#else
    p += " / other-arch";
#endif

#if defined(__APPLE__)
    p += " / apple";
#elif defined(_WIN32)
    p += " / windows";
#elif defined(__linux__)
    p += " / linux";
#endif

    p += " / eigen " + std::to_string(EIGEN_WORLD_VERSION) + "."
        + std::to_string(EIGEN_MAJOR_VERSION) + "."
        + std::to_string(EIGEN_MINOR_VERSION);

    return p;
}

} // namespace golden
