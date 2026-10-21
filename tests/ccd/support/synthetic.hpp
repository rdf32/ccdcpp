#pragma once

//==============================================================================
//
// Deterministic synthetic Landsat-like time series for the golden test.
//
// Everything here is driven by a splitmix64 stream, so a (cube seed, regime)
// pair always produces the same cube on a given platform. The values are only
// as portable as std::sin/std::cos/std::log of the C library, which is why the
// golden fixture records the platform it was produced on (see dump.hpp).
//
// The generator aims at the paths of the algorithm, not at realism for its own
// sake. Every pixel is one of three kinds:
//
//   Standard   clear-dominated, 0-4 abrupt breaks, harmonic seasonality,
//              noise, clouds/shadow/cirrus/water/fill QA, occasional winter
//              snow, unflagged cloud-like spikes (tmask / Extend
//              outliers), saturated and out-of-range thermal values, and
//              sometimes long fill gaps that leave too few observations.
//   Snow       snow-dominated QA, so detect() takes PermanentSnow.
//   Cloudy     cloud-dominated QA, so detect() takes InsufficientClear.
//
// Values are Collection-1 DN: six optical bands as reflectance x 10000 and the
// thermal band as Kelvin x 10. Fill observations carry -9999 in every band.
//
// WARNING: characterization.hpp builds its pixels from Rng and detail:: here;
// any change to them changes those pixels and their transitions_fixture.txt.
//
//==============================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace golden
{

class Rng
{
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next()
    {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // [0, 1)
    double uniform()
    {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }

    double uniform(double lo, double hi)
    {
        return lo + (hi - lo) * uniform();
    }

    // [0, n)
    std::uint64_t below(std::uint64_t n)
    {
        return next() % n;
    }

    double normal()
    {
        const double u1 = 1.0 - uniform();
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

private:
    std::uint64_t state_;
};

enum class Regime
{
    Cadence16,   // ~16-day, 1984-2024, random drops and long gaps
    Cadence8,    // ~8-day (two sensors), 1984-2024
    Duplicates,  // ~16-day with repeated dates
    Short,       // ~16-day over 2 years
    Medium       // ~16-day over 6 years
};

struct Cube
{
    std::vector<std::int64_t> dates;   // (T)
    std::vector<double> spectral;      // (H, W, 7, T)
    std::vector<std::uint8_t> qas;     // (H, W, T)
    std::size_t H = 0;
    std::size_t W = 0;
    std::size_t T = 0;
};

constexpr std::int64_t ORDINAL_1984 = 724276; // date(1984, 1, 1).toordinal()
constexpr std::int64_t ORDINAL_2025 = 739252; // date(2025, 1, 1).toordinal()

inline std::vector<std::int64_t> make_dates(Rng& rng, Regime regime)
{
    std::int64_t start = ORDINAL_1984 + static_cast<std::int64_t>(rng.below(16));
    std::int64_t stop = ORDINAL_2025;
    std::int64_t step = 16;
    double drop = 0.12;

    switch (regime)
    {
        case Regime::Cadence8:   step = 8;  drop = 0.10; break;
        case Regime::Short:      stop = start + 2 * 365; drop = 0.05; break;
        case Regime::Medium:     start = 734000; stop = start + 6 * 365; break;
        default: break;
    }

    // Long gaps: two spans of missing acquisitions.
    std::array<std::int64_t, 2> gap_start{};
    std::array<std::int64_t, 2> gap_len{};
    for (std::size_t g = 0; g < gap_start.size(); ++g)
    {
        gap_start[g] = start + static_cast<std::int64_t>(rng.uniform() * (stop - start));
        gap_len[g] = (regime == Regime::Short || regime == Regime::Medium)
            ? 0
            : 150 + static_cast<std::int64_t>(rng.below(450));
    }

    std::vector<std::int64_t> dates;
    for (std::int64_t d = start; d < stop; d += step)
    {
        bool in_gap = false;
        for (std::size_t g = 0; g < gap_start.size(); ++g)
            in_gap = in_gap || (d >= gap_start[g] && d < gap_start[g] + gap_len[g]);

        if (in_gap || rng.uniform() < drop)
            continue;

        dates.push_back(d);

        if (regime == Regime::Duplicates && rng.uniform() < 0.06)
            dates.push_back(d);
    }
    return dates;
}

namespace detail
{

constexpr double PI2 = 6.283185307179586;
constexpr double YEAR = 365.2425;

constexpr std::array<double, 7> BASE  = {450, 750, 650, 2800, 1900, 1100, 2950};
constexpr std::array<double, 7> AMP1  = {120, 180, 220, 900, 450, 300, 120};
constexpr std::array<double, 7> NOISE = {25, 30, 35, 90, 60, 45, 15};

inline double clamp_round(double v)
{
    return std::floor(v + 0.5);
}

} // namespace detail

// Fill one pixel's (7, T) block and its (T) QA row.
inline void make_pixel(
    Rng& rng,
    const std::vector<std::int64_t>& dates,
    double* spectral,      // 7 * T, band-major
    std::uint8_t* qas      // T
)
{
    using namespace detail;

    const std::size_t T = dates.size();

    const double kind_u = rng.uniform();
    const int kind = kind_u < 0.08 ? 1 : (kind_u < 0.16 ? 2 : 0); // 0 std, 1 snow, 2 cloudy

    // Per-pixel character.
    std::array<double, 7> level{};
    std::array<double, 7> amp1{};
    std::array<double, 7> amp2{};
    std::array<double, 7> phase{};
    std::array<double, 7> slope{};
    for (std::size_t b = 0; b < 7; ++b)
    {
        level[b] = BASE[b] * rng.uniform(0.6, 1.4);
        amp1[b]  = AMP1[b] * rng.uniform(0.2, 1.5);
        amp2[b]  = amp1[b] * rng.uniform(0.0, 0.4);
        phase[b] = rng.uniform(0.0, PI2);
        slope[b] = rng.normal() * BASE[b] * 0.004 / YEAR;
    }
    const double noise_scale = rng.uniform(0.5, 2.5);

    // Breaks.
    const std::size_t nbreaks = kind == 0 ? static_cast<std::size_t>(rng.below(5)) : 0;
    std::array<std::int64_t, 4> break_day{};
    std::array<std::array<double, 7>, 4> shift{};
    for (std::size_t k = 0; k < nbreaks; ++k)
    {
        break_day[k] = dates.front()
            + static_cast<std::int64_t>(rng.uniform(0.1, 0.9) * (dates.back() - dates.front()));
        const double severity = rng.uniform(0.3, 1.6);
        for (std::size_t b = 0; b < 7; ++b)
            shift[k][b] = rng.normal() * BASE[b] * 0.5 * severity;
    }

    // QA character.
    double p_cloud = rng.uniform(0.08, 0.5);
    double p_snow = 0.0;
    double p_water = rng.uniform() < 0.1 ? rng.uniform(0.2, 0.9) : 0.0;
    double p_fill = rng.uniform(0.0, 0.05);
    const double p_spike = rng.uniform(0.0, 0.03);
    const double p_sat = rng.uniform(0.0, 0.01);
    if (kind == 1) { p_snow = rng.uniform(0.75, 0.95); p_cloud = 0.04; p_water = 0.0; }
    if (kind == 2) { p_cloud = rng.uniform(0.78, 0.95); p_snow = rng.uniform(0.0, 0.05); p_water = 0.0; }

    // A long fill run; sometimes long enough to starve the pixel.
    const bool starve = rng.uniform() < 0.04;
    const std::size_t run_start = static_cast<std::size_t>(rng.below(T > 0 ? T : 1));
    const std::size_t run_len = starve ? T : static_cast<std::size_t>(rng.below(T / 6 + 1));

    for (std::size_t t = 0; t < T; ++t)
    {
        const double d = static_cast<double>(dates[t]);
        const double wt = PI2 * d / YEAR;
        const double season = std::cos(wt);  // +1 near new year: winter

        std::array<double, 7> v{};
        for (std::size_t b = 0; b < 7; ++b)
        {
            double x = level[b]
                + slope[b] * (d - static_cast<double>(dates.front()))
                + amp1[b] * std::cos(wt + phase[b])
                + amp2[b] * std::cos(2.0 * wt + phase[b])
                + NOISE[b] * noise_scale * rng.normal();

            for (std::size_t k = 0; k < nbreaks; ++k)
                if (dates[t] >= break_day[k])
                    x += shift[k][b];

            v[b] = x;
        }

        std::uint8_t qa = 0;
        const double u = rng.uniform();
        const bool in_run = t >= run_start && t < run_start + run_len;
        const double winter_snow = (kind == 0 && season > 0.7) ? 0.15 : 0.0;

        if (in_run || u < p_fill)
        {
            qa = 255;
        }
        else if (u < p_fill + p_snow + winter_snow)
        {
            qa = 3;
            for (std::size_t b = 0; b < 6; ++b)
                v[b] = 5000 + 2500 * rng.uniform() - 400 * static_cast<double>(b);
            v[6] = 2650 + 50 * rng.normal();
        }
        else if (u < p_fill + p_snow + winter_snow + p_cloud)
        {
            const std::uint64_t c = rng.below(4);
            qa = c == 0 ? 2 : (c == 1 ? 8 : 4);
            for (std::size_t b = 0; b < 6; ++b)
                v[b] += 2500 + 1500 * rng.uniform();
            v[6] -= 300;
        }
        else if (rng.uniform() < p_water)
        {
            qa = 1;
        }

        if (qa == 0 || qa == 1)
        {
            if (rng.uniform() < p_spike)
            {
                // Unflagged cloud: bright optical, cold thermal.
                for (std::size_t b = 0; b < 6; ++b)
                    v[b] += 1500 + 2500 * rng.uniform();
                v[6] -= 250;
            }
            if (rng.uniform() < p_sat)
            {
                const std::uint64_t which = rng.below(3);
                if (which == 0)      v[rng.below(6)] = 16000;   // saturated
                else if (which == 1) v[rng.below(6)] = -50;     // negative
                else                 v[6] = 0;                  // thermal out of range
            }
        }

        for (std::size_t b = 0; b < 7; ++b)
            spectral[b * T + t] = qa == 255 ? -9999.0 : clamp_round(v[b]);

        qas[t] = qa;
    }
}

inline Cube make_cube(
    std::uint64_t seed,
    Regime regime,
    std::size_t H,
    std::size_t W
)
{
    Rng rng(seed);

    Cube cube;
    cube.dates = make_dates(rng, regime);
    cube.H = H;
    cube.W = W;
    cube.T = cube.dates.size();

    cube.spectral.resize(H * W * 7 * cube.T);
    cube.qas.resize(H * W * cube.T);

    for (std::size_t p = 0; p < H * W; ++p)
    {
        Rng pixel_rng(seed * 1000003ull + p);
        make_pixel(
            pixel_rng,
            cube.dates,
            cube.spectral.data() + p * 7 * cube.T,
            cube.qas.data() + p * cube.T
        );
    }
    return cube;
}

// A cube of a corpus: its name, its make_cube() seed and regime, its size.
struct CubeSpec
{
    const char* name;
    std::uint64_t seed;
    Regime regime;
    std::size_t H;
    std::size_t W;
};

// The golden fixture's cubes (test_golden.cpp), which every test that runs
// the Landsat corpus runs.
inline const CubeSpec GOLDEN_CUBES[] = {
    {"c16",    11, Regime::Cadence16,  10, 16},
    {"c8",     12, Regime::Cadence8,    6,  8},
    {"dup",    13, Regime::Duplicates,  5,  8},
    {"short",  14, Regime::Short,       4,  5},
    {"medium", 15, Regime::Medium,      5,  8},
};

} // namespace golden
