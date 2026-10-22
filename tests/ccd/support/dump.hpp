#pragma once

//==============================================================================
//
// Canonical text form of detect() output, for bit-exact comparison.
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

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <string>

#include <Eigen/Core>

#include "ccd/ccd.hpp"

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

inline void dump_mask(std::string& out, ccd::ArrayView<const std::uint8_t, 1> bits)
{
    static const char* HEX = "0123456789abcdef";

    append(out, "K %zu ", static_cast<std::size_t>(bits.size()));

    for (std::size_t i = 0; i < bits.size(); i += 4)
    {
        unsigned nibble = 0;
        for (std::size_t j = 0; j < 4 && i + j < bits.size(); ++j)
            nibble |= (bits(i + j) ? 1u : 0u) << j;
        out.push_back(HEX[nibble]);
    }
    out.push_back('\n');
}

inline void dump_mask(std::string& out, const ccd::ProcessingMask& mask)
{
    const auto& bits = mask.data();
    dump_mask(out, ccd::ArrayView<const std::uint8_t, 1>::contiguous(bits.data(), {bits.size()}));
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

// A cube pixel's dump, as detect()'s: its models and, as the cube holds it,
// its processing mask (CubeOutput::masks, with RunSpec::masks).
inline void dump_pixel(
    std::string& out,
    const std::string& label,
    const ccd::CubeOutput& cube,
    const ccd::CubePixel& pixel
)
{
    append(out, "P %s %zu\n", label.c_str(), pixel.result.models.size());
    dump_models(out, pixel.result);
    if (cube.masks.empty())
        dump_mask(out, pixel.result.mask);
    else
        dump_mask(out, cube.mask_of(pixel.row, pixel.col));
}

// One result's dump, for comparing two runs.
inline std::string dump(const ccd::FitResult& result)
{
    std::string out;
    dump_pixel(out, "pixel", result);
    return out;
}

// One cube pixel's dump, its mask as the cube holds it.
inline std::string dump(const ccd::CubeOutput& cube, const ccd::CubePixel& pixel)
{
    std::string out;
    dump_pixel(out, "pixel", cube, pixel);
    return out;
}

// Every pixel of `cube` in raster order, each labelled "row,col".
inline std::string dump_cube(ccd::CubeOutput cube)
{
    std::sort(
        cube.pixels.begin(),
        cube.pixels.end(),
        [](const ccd::CubePixel& a, const ccd::CubePixel& b)
        {
            return a.row != b.row ? a.row < b.row : a.col < b.col;
        });

    std::string out;
    for (const auto& px : cube.pixels)
        dump_pixel(out, std::to_string(px.row) + "," + std::to_string(px.col), cube, px);
    return out;
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
