#pragma once

//==============================================================================
//
// Deterministic synthetic time series for any band layout.
//
// synthetic.hpp generates the Landsat corpus of the golden fixture, and its
// random stream defines that fixture, so it stays seven bands. This generator
// takes the layout as data -- one BandSpec per row, any count, at most one
// thermal row anywhere -- and produces the same three pixel kinds with the
// same QA vocabulary:
//
//   Standard   clear-dominated, 0-4 abrupt breaks, harmonic seasonality,
//              noise, clouds/shadow/cirrus/water/fill QA, occasional winter
//              snow, unflagged cloud-like spikes, saturated or negative
//              optical values and (with a thermal row) out-of-range thermal
//              values, and sometimes long fill gaps that starve the pixel.
//   Snow       snow-dominated QA, so detect() takes PermanentSnow.
//   Cloudy     cloud-dominated QA, so detect() takes InsufficientClear.
//
// Optical rows are reflectance x 10000 and a thermal row is Kelvin x 10, the
// units of the default Input. Fill observations carry -9999 in
// every row. Dates come from synthetic.hpp's make_dates().
//
// The values are only as portable as the C library's sin/cos/log; the
// fixture records its platform (see golden.hpp).
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <vector>

#include "synthetic.hpp"

namespace golden
{

// One row of the layout: its level, annual amplitude and noise in DN.
struct BandSpec
{
    double base;
    double amp;
    double noise;
    bool thermal = false;
};

// (H, W, B, T) spectral, (H, W, T) QA.
struct BandCube
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;
    std::vector<std::uint8_t> qas;
    std::size_t H = 0;
    std::size_t W = 0;
    std::size_t B = 0;
    std::size_t T = 0;
};

// Fill one pixel's (B, T) block and its (T) QA row.
inline void make_band_pixel(
    Rng& rng,
    const std::vector<std::int64_t>& dates,
    const std::vector<BandSpec>& specs,
    double* spectral,      // B * T, band-major
    std::uint8_t* qas      // T
)
{
    using detail::PI2;
    using detail::YEAR;

    const std::size_t T = dates.size();
    const std::size_t B = specs.size();

    std::vector<std::size_t> optical;
    for (std::size_t b = 0; b < B; ++b)
        if (!specs[b].thermal)
            optical.push_back(b);

    const double kind_u = rng.uniform();
    const int kind = kind_u < 0.08 ? 1 : (kind_u < 0.16 ? 2 : 0); // 0 std, 1 snow, 2 cloudy

    // Per-pixel character.
    std::vector<double> level(B), amp1(B), amp2(B), phase(B), slope(B);
    for (std::size_t b = 0; b < B; ++b)
    {
        level[b] = specs[b].base * rng.uniform(0.6, 1.4);
        amp1[b]  = specs[b].amp * rng.uniform(0.2, 1.5);
        amp2[b]  = amp1[b] * rng.uniform(0.0, 0.4);
        phase[b] = rng.uniform(0.0, PI2);
        slope[b] = rng.normal() * specs[b].base * 0.004 / YEAR;
    }
    const double noise_scale = rng.uniform(0.5, 2.5);

    // Breaks.
    const std::size_t nbreaks = kind == 0 ? static_cast<std::size_t>(rng.below(5)) : 0;
    std::vector<std::int64_t> break_day(nbreaks);
    std::vector<std::vector<double>> shift(nbreaks, std::vector<double>(B));
    for (std::size_t k = 0; k < nbreaks; ++k)
    {
        break_day[k] = dates.front()
            + static_cast<std::int64_t>(rng.uniform(0.1, 0.9) * (dates.back() - dates.front()));
        const double severity = rng.uniform(0.3, 1.6);
        for (std::size_t b = 0; b < B; ++b)
            shift[k][b] = rng.normal() * specs[b].base * 0.5 * severity;
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

    std::vector<double> v(B);
    for (std::size_t t = 0; t < T; ++t)
    {
        const double d = static_cast<double>(dates[t]);
        const double wt = PI2 * d / YEAR;
        const double season = std::cos(wt);  // +1 near new year: winter

        for (std::size_t b = 0; b < B; ++b)
        {
            double x = level[b]
                + slope[b] * (d - static_cast<double>(dates.front()))
                + amp1[b] * std::cos(wt + phase[b])
                + amp2[b] * std::cos(2.0 * wt + phase[b])
                + specs[b].noise * noise_scale * rng.normal();

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
            for (std::size_t k = 0; k < optical.size(); ++k)
                v[optical[k]] = 5000 + 2500 * rng.uniform() - 300 * static_cast<double>(k);
            for (std::size_t b = 0; b < B; ++b)
                if (specs[b].thermal)
                    v[b] = 2650 + 50 * rng.normal();
        }
        else if (u < p_fill + p_snow + winter_snow + p_cloud)
        {
            const std::uint64_t c = rng.below(4);
            qa = c == 0 ? 2 : (c == 1 ? 8 : 4);
            for (std::size_t b = 0; b < B; ++b)
                v[b] += specs[b].thermal ? -300.0 : 2500 + 1500 * rng.uniform();
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
                for (std::size_t b = 0; b < B; ++b)
                    v[b] += specs[b].thermal ? -250.0 : 1500 + 2500 * rng.uniform();
            }
            if (rng.uniform() < p_sat)
            {
                const std::uint64_t which = rng.below(3);
                const std::size_t band = optical[rng.below(optical.size())];
                if (which == 0)
                    v[band] = 16000;                          // saturated
                else if (which == 1)
                    v[band] = -50;                            // negative
                else
                    for (std::size_t b = 0; b < B; ++b)       // thermal out of range
                        if (specs[b].thermal)
                            v[b] = 0;
            }
        }

        for (std::size_t b = 0; b < B; ++b)
            spectral[b * T + t] = qa == 255 ? -9999.0 : detail::clamp_round(v[b]);

        qas[t] = qa;
    }
}

inline BandCube make_band_cube(
    std::uint64_t seed,
    Regime regime,
    std::size_t H,
    std::size_t W,
    const std::vector<BandSpec>& specs
)
{
    Rng rng(seed);

    BandCube cube;
    cube.dates = make_dates(rng, regime);
    cube.H = H;
    cube.W = W;
    cube.B = specs.size();
    cube.T = cube.dates.size();

    cube.spectral.resize(H * W * cube.B * cube.T);
    cube.qas.resize(H * W * cube.T);

    for (std::size_t p = 0; p < H * W; ++p)
    {
        Rng pixel_rng(seed * 1000003ull + p);
        make_band_pixel(
            pixel_rng,
            cube.dates,
            specs,
            cube.spectral.data() + p * cube.B * cube.T,
            cube.qas.data() + p * cube.T
        );
    }
    return cube;
}

} // namespace golden
