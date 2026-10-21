#pragma once

//==============================================================================
//
// The tests' pixels as arrays:
//
//   * Series: one pixel's dates, (B, T) spectral block and QA codes, with the
//     input and configuration it runs under; read_csv() reads one from a
//     test CSV (tests/ccd/data);
//   * FlatCube: pixels on shared dates, (N, B, T) spectral and (N, T) QA,
//     from a synthetic cube or a Series;
//   * PixelView: the three views of one pixel, which detect() takes.
//
//==============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd/ccd.hpp"

#include "config.hpp"
#include "synthetic.hpp"
#include "synthetic_bands.hpp"
#include "units.hpp"

namespace golden
{

struct PixelView
{
    ccd::ArrayView<const std::int64_t, 1> dates;
    ccd::ArrayView<const double, 2> spectral;
    ccd::ArrayView<const std::uint8_t, 1> qas;
};

// The views of a pixel's T dates, (B, T) block and T QA codes.
inline PixelView view_of(const std::int64_t* dates, const double* spectral, std::size_t B,
                         const std::uint8_t* qas, std::size_t T)
{
    return {
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates, {T}),
        ccd::ArrayView<const double, 2>::contiguous(spectral, {B, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas, {T}),
    };
}

// detect() under `spec` on one pixel.
inline ccd::FitResult detect(const ccd::RunSpec& spec, const PixelView& px)
{
    return ccd::detect(spec, px.dates, px.spectral, px.qas).result;
}

struct Series
{
    std::string name;
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (B, T), band-major
    std::vector<std::uint8_t> qas;
    ccd::Input h;
    ccd::config::Config config = golden::pyccd();

    std::size_t size() const
    {
        return dates.size();
    }

    std::size_t bands() const
    {
        return dates.empty() ? 0 : spectral.size() / dates.size();
    }

    // The configuration's statistics cutoff.
    std::int64_t statistics_end() const
    {
        return std::get<std::int64_t>(config.shared.at("statistics_end"));
    }

    // The first `count` observations.
    Series prefix(std::size_t count) const
    {
        const std::size_t T = size();
        const std::size_t B = bands();
        Series out = *this;
        out.dates.resize(count);
        out.qas.resize(count);
        out.spectral.resize(B * count);
        for (std::size_t b = 0; b < B; ++b)
            std::copy_n(spectral.begin() + b * T, count, out.spectral.begin() + b * count);
        return out;
    }

    PixelView view() const
    {
        return view_of(dates.data(), spectral.data(), bands(), qas.data(), size());
    }

    ccd::ArrayView<const std::int64_t, 1> date_view() const
    {
        return view().dates;
    }

    ccd::ArrayView<const double, 2> spectral_view() const
    {
        return view().spectral;
    }

    ccd::ArrayView<const std::uint8_t, 1> qa_view() const
    {
        return view().qas;
    }
};

// detect() on `s`, under its input and configuration, with the processing mask.
inline ccd::FitResult detect(const Series& s)
{
    return detect(spec_of(s.h, s.config), s.view());
}

// The pixel `name` of `all`.
inline const Series& named(const std::vector<Series>& all, const std::string& name)
{
    for (const auto& s : all)
        if (s.name == name)
            return s;
    throw std::runtime_error("no pixel " + name);
}

// A test CSV at `path`: date, seven bands, QA per row, under the DN defaults.
// `skip_first_line` skips line 1, the header of the Collection-2 files. The
// pyccd reference pixel's file has none, but the golden fixture,
// test_preprocess and test_kernels read it skipping its first observation
// (442 of 443); their fixtures and pinned values are of those 442.
inline Series read_csv(const std::string& path, bool skip_first_line)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Unable to open " + path);

    Series s;
    std::array<std::vector<double>, 7> bands;

    std::string line;
    if (skip_first_line)
        std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        s.dates.push_back(static_cast<std::int64_t>(std::stod(value)));
        for (std::size_t b = 0; b < 7; ++b)
        {
            std::getline(ss, value, ',');
            bands[b].push_back(std::stod(value));
        }
        std::getline(ss, value, ',');
        s.qas.push_back(static_cast<std::uint8_t>(std::stod(value)));
    }

    const std::size_t T = s.dates.size();
    s.spectral.resize(7 * T);
    for (std::size_t b = 0; b < 7; ++b)
        std::copy(bands[b].begin(), bands[b].end(), s.spectral.begin() + b * T);
    return s;
}

// `s` with its DN (7, T) block in the physical-unit input's units
// (to_physical()); its input and configuration are as they were.
inline Series to_physical(Series s)
{
    to_physical(s.spectral.data(), s.size());
    return s;
}

struct FlatCube
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (N, B, T)
    std::vector<std::uint8_t> qas;  // (N, T)
    std::size_t N = 0;
    std::size_t B = 0;
    std::size_t T = 0;

    const double* pixel(std::size_t p) const { return spectral.data() + p * B * T; }
    const std::uint8_t* qa(std::size_t p) const { return qas.data() + p * T; }

    PixelView view(std::size_t p) const
    {
        return view_of(dates.data(), pixel(p), B, qa(p), T);
    }
};

inline FlatCube flat(const Cube& c)
{
    return {c.dates, c.spectral, c.qas, c.H * c.W, 7, c.T};
}

inline FlatCube flat(const BandCube& c)
{
    return {c.dates, c.spectral, c.qas, c.H * c.W, c.B, c.T};
}

inline FlatCube flat(const Series& s)
{
    return {s.dates, s.spectral, s.qas, 1, s.bands(), s.size()};
}

// `c` with every pixel's DN (7, T) block in the physical-unit input's units.
inline FlatCube to_physical(FlatCube c)
{
    for (std::size_t p = 0; p < c.N; ++p)
        to_physical(c.spectral.data() + p * c.B * c.T, c.T);
    return c;
}

} // namespace golden
