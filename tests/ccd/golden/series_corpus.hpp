#pragma once

//==============================================================================
//
// Whole pixel series for the tests that cut them at a date: prefix stability
// (test_prefix.cpp) and resume (test_resume.cpp). Each Series is one pixel's
// dates, (7, T) spectral and QA codes with the options it runs under:
//
//   * characterization_series(): the pixels of characterisation.hpp;
//   * csv_series(dir): the pyccd reference pixel, the Atlanta pixel with a
//     half-day median gap, and the Collection-2 pixel under c2_harmonic();
//   * synthetic_series(): every pixel of the golden cubes (synthetic.hpp)
//     under the DN defaults and the physical-unit preset.
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

#include "characterisation.hpp"
#include "golden.hpp"
#include "synthetic.hpp"

namespace golden
{

struct Series
{
    std::string name;
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, T), band-major
    std::vector<std::uint8_t> qas;
    ccd::HarmonicOptions h;
    ccd::LassoOptions l;

    std::size_t size() const
    {
        return dates.size();
    }

    // The first `count` observations.
    Series prefix(std::size_t count) const
    {
        const std::size_t T = size();
        Series out = *this;
        out.dates.resize(count);
        out.qas.resize(count);
        out.spectral.resize(7 * count);
        for (std::size_t b = 0; b < 7; ++b)
            std::copy_n(spectral.begin() + b * T, count, out.spectral.begin() + b * count);
        return out;
    }

    ccd::ArrayView<const std::int64_t, 1> date_view() const
    {
        return ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {size()});
    }

    ccd::ArrayView<const double, 2> spectral_view() const
    {
        return ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, size()});
    }

    ccd::ArrayView<const std::uint8_t, 1> qa_view() const
    {
        return ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {size()});
    }
};

// detect() on `s` with STAT_ORD at `stat_ord`.
inline ccd::FitResult detect_at(const Series& s, std::int64_t stat_ord)
{
    ccd::HarmonicOptions h = s.h;
    h.STAT_ORD = stat_ord;
    return ccd::detect(s.date_view(), s.spectral_view(), s.qa_view(), h, s.l);
}

// detect_recorded() on `s` with STAT_ORD at `stat_ord`.
inline ccd::Recorded recorded_at(const Series& s, std::int64_t stat_ord)
{
    ccd::HarmonicOptions h = s.h;
    h.STAT_ORD = stat_ord;
    return ccd::detect_recorded(s.date_view(), s.spectral_view(), s.qa_view(), h, s.l);
}

// A test CSV: date, seven bands, QA per row; `header` skips the first line.
inline Series read_csv(const std::string& name, const std::string& path, bool header)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Unable to open " + path);

    Series s;
    s.name = name;
    std::array<std::vector<double>, 7> bands;

    std::string line;
    if (header)
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

inline std::vector<Series> characterization_series()
{
    std::vector<Series> out;
    for (const auto& p : characterisation_pixels())
        out.push_back(Series{p.name, p.dates, p.spectral, p.qas, p.h, p.l});
    return out;
}

inline std::vector<Series> csv_series(const std::string& dir)
{
    std::vector<Series> out;
    out.push_back(read_csv("pyccd", dir + "/test_3657_3610_observations.csv", false));
    out.push_back(read_csv("atlanta_481_393", dir + "/test_atlanta_024013_00030008_481_393.csv", false));
    Series c2 = read_csv("c2", dir + "/test002_c2.csv", true);
    c2.h = c2_harmonic();
    out.push_back(std::move(c2));
    return out;
}

inline std::vector<Series> synthetic_series()
{
    struct Spec { const char* name; std::uint64_t seed; Regime regime; std::size_t H, W; };
    const Spec cubes[] = {
        {"c16",    11, Regime::Cadence16,  10, 16},
        {"c8",     12, Regime::Cadence8,    6,  8},
        {"dup",    13, Regime::Duplicates,  5,  8},
        {"short",  14, Regime::Short,       4,  5},
        {"medium", 15, Regime::Medium,      5,  8},
    };

    std::vector<Series> out;
    for (const auto& spec : cubes)
    {
        const Cube c = make_cube(spec.seed, spec.regime, spec.H, spec.W);
        const std::size_t T = c.T;
        for (const bool physical : {false, true})
        {
            for (std::size_t p = 0; p < c.H * c.W; ++p)
            {
                Series s;
                s.name = std::string(spec.name) + (physical ? ":phys:" : ":dn:") + std::to_string(p);
                s.dates = c.dates;
                s.spectral.assign(c.spectral.begin() + p * 7 * T, c.spectral.begin() + (p + 1) * 7 * T);
                s.qas.assign(c.qas.begin() + p * T, c.qas.begin() + (p + 1) * T);
                if (physical)
                {
                    to_physical(s.spectral.data(), T);
                    s.h = physical_harmonic();
                    s.l = physical_lasso();
                }
                out.push_back(std::move(s));
            }
        }
    }
    return out;
}

// Every observation index; `count` indexes spread evenly over [0, T).
inline std::vector<std::size_t> every(std::size_t T)
{
    std::vector<std::size_t> out(T);
    for (std::size_t i = 0; i < T; ++i)
        out[i] = i;
    return out;
}

inline std::vector<std::size_t> spread(std::size_t T, std::size_t count)
{
    std::vector<std::size_t> out;
    for (std::size_t k = 1; k <= count; ++k)
        out.push_back(k * T / (count + 1));
    return out;
}

// Whether a split after index `last` keeps a date whole and leaves at least
// one observation after it.
inline bool splits_cleanly(const Series& s, std::size_t last)
{
    return last + 1 < s.size() && s.dates[last + 1] != s.dates[last];
}

} // namespace golden
