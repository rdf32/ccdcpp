#pragma once

//==============================================================================
//
// Characterization pixels for the Standard procedure's state machine.
//
// Each pixel is built so that detect() takes one transition, guard or rule
// of the Standard procedure that the synthetic golden corpus does not reach
// (or reaches only in Python): see test_transitions.cpp for which pixel takes
// which path and why. The pixels are a deterministic function of their
// parameters (a splitmix64 stream from synthetic.hpp), and their outputs are
// pinned, byte for byte, in transitions_fixture.txt.
//
// Values are Collection-1 DN like synthetic.hpp: six optical bands as
// reflectance x 10000, the thermal band as Kelvin x 10, every observation
// clear (QA 0) unless a pixel says otherwise.
//
//==============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ccd/ccd.hpp"

#include "config.hpp"
#include "series.hpp"

#include "synthetic.hpp"

namespace golden
{

// What a pixel's spectral series is made of.
struct SeriesSpec
{
    std::uint64_t seed = 1;
    double noise = 1.0;                // times detail::NOISE
    double season = 1.0;               // times detail::AMP1
    // (date, severity): from that date on every band shifts by
    // severity * BASE * 0.5, up or (halved, so values stay in range) down,
    // the direction drawn per band from the stream.
    std::vector<std::pair<std::int64_t, double>> breaks;
    // (observation index, DN added to the six optical bands).
    std::vector<std::pair<std::size_t, double>> spikes;
};

// Dates from `start`, `count` observations `gap` days apart.
inline void append_dates(
    std::vector<std::int64_t>& dates,
    std::int64_t start,
    std::size_t count,
    std::int64_t gap
)
{
    for (std::size_t i = 0; i < count; ++i)
        dates.push_back(start + static_cast<std::int64_t>(i) * gap);
}

inline std::vector<double> make_series(
    const std::vector<std::int64_t>& dates,
    const SeriesSpec& spec
)
{
    using namespace detail;

    Rng rng(spec.seed);
    const std::size_t T = dates.size();

    std::array<double, 7> phase{};
    for (std::size_t b = 0; b < 7; ++b)
        phase[b] = rng.uniform(0.0, PI2);

    std::vector<std::array<double, 7>> shift(spec.breaks.size());
    for (std::size_t k = 0; k < spec.breaks.size(); ++k)
        for (std::size_t b = 0; b < 7; ++b)
            shift[k][b] = (rng.uniform() < 0.5 ? -0.5 : 1.0)
                * BASE[b] * 0.5 * spec.breaks[k].second;

    std::vector<double> out(7 * T);
    for (std::size_t t = 0; t < T; ++t)
    {
        const double wt = PI2 * static_cast<double>(dates[t]) / YEAR;
        for (std::size_t b = 0; b < 7; ++b)
        {
            double x = BASE[b]
                + AMP1[b] * spec.season * std::cos(wt + phase[b])
                + NOISE[b] * spec.noise * rng.normal();

            for (std::size_t k = 0; k < spec.breaks.size(); ++k)
                if (dates[t] >= spec.breaks[k].first)
                    x += shift[k][b];

            out[b * T + t] = x;
        }
    }

    for (const auto& spike : spec.spikes)
        for (std::size_t b = 0; b < 6; ++b)
            out[b * T + spike.first] += spike.second;

    for (auto& v : out)
        v = detail::clamp_round(v);

    return out;
}

inline Series make_char_pixel(
    std::string name,
    std::vector<std::int64_t> dates,
    const SeriesSpec& spec
)
{
    Series p;
    p.name = std::move(name);
    p.spectral = make_series(dates, spec);
    p.qas.assign(dates.size(), 0);
    p.dates = std::move(dates);
    return p;
}

//------------------------------------------------------------------------------
// Pixel families. The seeds characterization_pixels() passes were found by
// a scan with a trace-instrumented build: for a family built for a path, an
// early seed that takes it; for the random families, a seed whose output a
// rule decides (see test_transitions.cpp).
//
// Every draw from the stream is its own statement or an element of a braced
// list, both evaluated in order. Two draws in the arguments of one call are
// evaluated in an order the compiler picks (here clang went left to right
// and GCC right to left), and the pixel would differ by platform.
//------------------------------------------------------------------------------

// 13 observations 34 days apart, then 500 `gap` days apart, all before
// statistics_end: the median gap is `gap`, so the peek is adjusted to 13-25 for
// gaps of 4-6 days. A break at observation 13 with a decaying transient after
// it, so the second Initialize shifts and its Lookback starts at or below
// peek_size with prev_break = 13.
inline Series lookback_near_break(std::uint64_t seed, std::int64_t gap)
{
    Rng r(seed * 7919 + 17);
    std::vector<std::int64_t> d;
    append_dates(d, 730000, 13, 34);
    append_dates(d, d.back() + gap, 500, gap);

    SeriesSpec s;
    s.seed = seed;
    s.noise = r.uniform(0.5, 1.5);
    s.season = r.uniform(0.3, 1.2);
    s.breaks.push_back({d[13], r.uniform(0.8, 1.6)});
    const int k = 1 + static_cast<int>(r.below(8));
    for (int i = 0; i < k; ++i)
        s.spikes.push_back({13 + static_cast<std::size_t>(i), r.uniform(100, 1500) * (k - i) / k});

    return make_char_pixel("lookback_near_break", d, s);
}

// 243 observations 3 days apart (2010-2012), then 45 30 days apart: the
// median gap is 3 days, so the peek is 32 (>= 26). No break; spikes among the
// last 31 observations, which the forward search never reaches.
inline Series dense_series_end(std::uint64_t seed)
{
    Rng r(seed * 104729 + 5);
    std::vector<std::int64_t> d;
    append_dates(d, 734138, 243, 3);
    append_dates(d, d.back() + 30, 45, 30);
    const std::size_t n = d.size();

    SeriesSpec s;
    s.seed = seed;
    s.noise = r.uniform(0.5, 1.2);
    s.season = r.uniform(0.3, 1.0);
    const int k = 1 + static_cast<int>(r.below(4));
    for (int i = 0; i < k; ++i)
        s.spikes.push_back({n - 31 + static_cast<std::size_t>(r.below(31)), r.uniform(300, 3000)});

    return make_char_pixel("dense_series_end", d, s);
}

// 400 observations 5 days apart (peek 19), a break 80-99 observations before
// the end: the Initialize after it ends within peek_size of the series end.
inline Series peek_guard_tail(std::uint64_t seed)
{
    Rng r(seed * 15485863 + 3);
    std::vector<std::int64_t> d;
    append_dates(d, 732000, 400, 5);
    const std::size_t n = d.size();

    SeriesSpec s;
    s.seed = seed;
    s.noise = r.uniform(0.5, 1.2);
    s.season = r.uniform(0.3, 1.0);
    s.breaks.push_back({d[n - 80 - r.below(20)], r.uniform(0.8, 1.6)});

    return make_char_pixel("peek_guard_tail", d, s);
}

// 16-day cadence, 2000-2010, with a spike on observation n - 6 - back: the
// forward search drops it as an outlier on its last step.
inline Series series_end_after_outlier(std::uint64_t seed, std::size_t back)
{
    std::vector<std::int64_t> d;
    append_dates(d, 730120, 228, 16);
    const std::size_t n = d.size();

    SeriesSpec s;
    s.seed = seed;
    s.noise = 1.0;
    s.season = 0.8;
    s.spikes.push_back({n - 6 - back, 2500});

    return make_char_pixel("series_end_after_outlier", d, s);
}

// One clear observation dated up to statistics_end (2017-12-31), 60 after it: the
// QA statistics see 1 of 1 clear, so Standard runs, and the variogram over the
// statistics window of one observation is NaN.
inline Series variogram_nan()
{
    std::vector<std::int64_t> d;
    d.push_back(736600);
    append_dates(d, 736700, 60, 16);

    SeriesSpec s;
    s.seed = 41;
    return make_char_pixel("variogram_nan", d, s);
}

// Red (a detection band, not a Tmask band) constant: its variogram is 0, and
// the initialization model fits it exactly.
inline Series constant_detection_band()
{
    std::vector<std::int64_t> d;
    append_dates(d, 730120, 228, 16);

    SeriesSpec s;
    s.seed = 43;
    Series p = make_char_pixel("constant_detection_band", d, s);
    const std::size_t T = p.dates.size();
    for (std::size_t t = 0; t < T; ++t)
        p.spectral[2 * T + t] = 650.0;
    return p;
}

// `n` observations 16 days apart before statistics_end; `clear` of them clear
// (QA 0) spread evenly, then of the others the first `snow` snow (QA 3) and
// the rest cloud (QA 4). The clear percentage is clear / n and the snow
// percentage snow / (clear + snow + 0.01).
inline Series qa_pattern(const char* name, std::size_t n, std::size_t clear, std::size_t snow)
{
    std::vector<std::int64_t> d;
    append_dates(d, 731000, n, 16);

    SeriesSpec s;
    s.seed = 47;
    Series p = make_char_pixel(name, d, s);

    std::size_t placed_clear = 0, other = 0;
    for (std::size_t t = 0; t < n; ++t)
    {
        if ((t + 1) * clear / n > placed_clear)
        {
            p.qas[t] = 0;
            ++placed_clear;
        }
        else
        {
            p.qas[t] = other < snow ? 3 : 4;
            ++other;
        }
    }
    return p;
}

// 16-day cadence, 2000-2010, with a level shift 8-17 observations in: the
// first stable window starts after it, and Lookback stops on change.
inline Series start_catch(std::uint64_t seed)
{
    Rng r(seed * 2654435761u + 11);
    std::vector<std::int64_t> d;
    append_dates(d, 730120, 228, 16);

    SeriesSpec s;
    s.seed = seed;
    s.noise = r.uniform(0.5, 1.2);
    s.season = r.uniform(0.3, 1.0);
    s.breaks.push_back({d[8 + r.below(10)], r.uniform(0.5, 1.6)});

    return make_char_pixel("start_catch", d, s);
}

// Random dense pixels: 12-17 observations 30-39 days apart, then 300-599
// 3-5 days apart (peek 19-32); a break near the cadence change, up to two
// more, up to two transients (a run of 1-30 observations offset together)
// and up to five spikes.
inline Series dense_random(const char* name, std::uint64_t seed)
{
    Rng r(seed * 6364136223846793005ull + 1442695040888963407ull);
    std::vector<std::int64_t> d;
    const std::size_t k1 = 12 + r.below(6);
    append_dates(d, 732000, k1, 30 + static_cast<std::int64_t>(r.below(10)));
    const std::int64_t gap = 3 + static_cast<std::int64_t>(r.below(3));
    append_dates(d, d.back() + gap, 300 + r.below(300), gap);
    const std::size_t n = d.size();

    SeriesSpec s;
    s.seed = seed;
    s.noise = r.uniform(0.5, 1.5);
    s.season = r.uniform(0.2, 1.2);
    if (r.uniform() < 0.8)
        s.breaks.push_back({d[k1 + r.below(4)], r.uniform(0.5, 1.6)});
    const int nb = static_cast<int>(r.below(3));
    for (int i = 0; i < nb; ++i)
        s.breaks.push_back({d[k1 + r.below(n - k1)], r.uniform(0.5, 1.6)});
    const int nt = static_cast<int>(r.below(3));
    for (int i = 0; i < nt; ++i)
    {
        const std::size_t a = k1 + r.below(60);
        const std::size_t len = 1 + r.below(30);
        const double m = r.uniform(-300, 1500);
        for (std::size_t j = a; j < a + len && j < n; ++j)
            s.spikes.push_back({j, m});
    }
    const int ns = static_cast<int>(r.below(6));
    for (int i = 0; i < ns; ++i)
        s.spikes.push_back({r.below(n), r.uniform(300, 3000)});

    return make_char_pixel(name, d, s);
}

// Random 16-day pixels: 300-599 observations from 1991; up to three breaks,
// one to three transients of 3-8 observations, up to five spikes.
inline Series sparse_random(const char* name, std::uint64_t seed)
{
    Rng r(seed * 2862933555777941757ull + 3037000493ull);
    std::vector<std::int64_t> d;
    const std::int64_t first = 727000 + static_cast<std::int64_t>(r.below(16));
    const std::size_t count = 300 + r.below(300);
    append_dates(d, first, count, 16);
    const std::size_t n = d.size();

    SeriesSpec s;
    s.seed = seed;
    s.noise = r.uniform(0.5, 1.5);
    s.season = r.uniform(0.2, 1.2);
    const int nb = static_cast<int>(r.below(4));
    for (int i = 0; i < nb; ++i)
        s.breaks.push_back({d[r.below(n)], r.uniform(0.5, 1.6)});
    const int nt = 1 + static_cast<int>(r.below(3));
    for (int i = 0; i < nt; ++i)
    {
        const std::size_t a = r.below(n);
        const std::size_t len = 3 + r.below(6);
        const double m = r.uniform(-300, 1500);
        for (std::size_t j = a; j < a + len && j < n; ++j)
            s.spikes.push_back({j, m});
    }
    const int ns = static_cast<int>(r.below(6));
    for (int i = 0; i < ns; ++i)
        s.spikes.push_back({r.below(n), r.uniform(300, 3000)});

    return make_char_pixel(name, d, s);
}

//------------------------------------------------------------------------------
// The pixels test_transitions.cpp pins, in fixture order.
//------------------------------------------------------------------------------
inline std::vector<Series> characterization_pixels()
{
    std::vector<Series> out;
    out.push_back(qa_pattern("select_clear_at_threshold", 100, 25, 0));
    out.push_back(qa_pattern("select_clear_below_threshold", 100, 24, 0));
    out.push_back(qa_pattern("select_snow_at_threshold", 100, 20, 61));
    out.push_back(qa_pattern("select_snow_below_threshold", 100, 20, 60));
    out.push_back(qa_pattern("snow_too_few", 20, 2, 9));
    out.push_back(qa_pattern("standard_too_few", 30, 12, 0));
    out.push_back(qa_pattern("standard_no_room", 20, 20, 0));
    out.push_back(variogram_nan());
    out.push_back(constant_detection_band());
    out.push_back(start_catch(2));
    {
        // The fewest coefficients raised to 6: Initialize, Extend's
        // coefficient schedule and the Start/End fits all read it.
        Series p = sparse_random("coefficient_min_6", 3);
        p.config.shared["coefficients.min"] = ccd::config::integer(6);

        out.push_back(std::move(p));
    }
    out.push_back(lookback_near_break(2, 5));
    out.push_back(dense_random("lookback_near_break_decides", 732));
    out.push_back(sparse_random("lookback_far_peek_decides", 172));
    out.push_back(peek_guard_tail(46));
    out.push_back(series_end_after_outlier(53, 0));
    out.push_back(dense_series_end(1995));
    return out;
}

} // namespace golden
