#pragma once

//==============================================================================
//
// Whole pixel series for the tests that cut them at a date: prefix stability
// (test_prefix.cpp) and resume (test_resume.cpp, test_run_spec.cpp), each a
// Series (series.hpp) with the input and configuration it runs under:
//
//   * characterization_pixels() (characterization.hpp);
//   * csv_series(dir): the pyccd reference pixel, the Atlanta pixel with a
//     half-day median gap, and the Collection-2 pixel under c2_input();
//   * synthetic_series(): every pixel of the golden cubes (synthetic.hpp)
//     under the DN defaults and the physical-unit preset.
//
//==============================================================================

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

#include "characterization.hpp"
#include "series.hpp"
#include "units.hpp"
#include "synthetic.hpp"

namespace golden
{

// detect() on `s` with the shared statistics_end at `statistics_end`.
inline ccd::FitResult detect_at(const Series& s, std::int64_t statistics_end)
{
    return ccd::detect(
        golden::spec_of(s.h, with_shared(s.config, "statistics_end", ccd::config::integer(statistics_end))),
        s.date_view(),
        s.spectral_view(),
        s.qa_view()).result;
}

// detect() with RunSpec::record on `s` with the shared statistics_end at
// `statistics_end`.
inline ccd::PixelOutput recorded_at(const Series& s, std::int64_t statistics_end)
{
    return ccd::detect(
        golden::spec_of(s.h, with_shared(s.config, "statistics_end", ccd::config::integer(statistics_end)),
                        true),
        s.date_view(),
        s.spectral_view(),
        s.qa_view());
}

// The stored run continued on the whole series: detect() on `whole` with the
// stored record and segments as its resume, its statistics frozen at the
// stored end (Resume::freeze_statistics): the run capped there.
inline ccd::PixelOutput resume_on(const Series& whole, const ccd::PixelOutput& stored)
{
    return ccd::detect(
        golden::spec_of(whole.h, whole.config),
        whole.date_view(),
        whole.spectral_view(),
        whole.qa_view(),
        ccd::Resume{stored.header, {stored.record}, {stored.result.models}, true});
}

// Two records say the same, field by field.
inline void expect_same_record(const ccd::PixelRecord& a, const ccd::PixelRecord& b, const std::string& at)
{
    EXPECT_EQ(a.input_fingerprint, b.input_fingerprint) << at;
    EXPECT_EQ(a.procedure, b.procedure) << at;
    EXPECT_EQ(a.excluded_days, b.excluded_days) << at;
    const auto& x = a.standard;
    const auto& y = b.standard;
    EXPECT_EQ(x.exit, y.exit) << at;
    EXPECT_EQ(x.window.start, y.window.start) << at;
    EXPECT_EQ(x.window.stop, y.window.stop) << at;
    EXPECT_EQ(x.prev_break, y.prev_break) << at;
    EXPECT_EQ(x.fit_stop, y.fit_stop) << at;
    EXPECT_EQ(x.grow, y.grow) << at;
    EXPECT_EQ(x.num_coef, y.num_coef) << at;
    EXPECT_EQ(x.magnitude, y.magnitude) << at;
    EXPECT_EQ(x.final_segments, y.final_segments) << at;
}

inline std::vector<Series> csv_series(const std::string& dir)
{
    std::vector<Series> out;
    out.push_back(read_csv(dir + "/test_3657_3610_observations.csv", false));
    out.back().name = "pyccd";
    out.push_back(read_csv(dir + "/test_atlanta_024013_00030008_481_393.csv", false));
    out.back().name = "atlanta_481_393";
    Series c2 = read_csv(dir + "/test002_c2.csv", true);
    c2.name = "c2";
    c2.h = c2_input();
    c2.config = c2_config();
    out.push_back(std::move(c2));
    return out;
}

inline std::vector<Series> synthetic_series()
{
    std::vector<Series> out;
    for (const auto& spec : GOLDEN_CUBES)
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
                    s.h = physical_input();

                }
                out.push_back(std::move(s));
            }
        }
    }
    return out;
}

// The number of leading segments of `r` that no longer series can rewrite:
// those up to its last break, or its first when it is a Start segment and
// none breaks.
inline std::size_t final_count(const ccd::FitResult& r)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < r.models.size(); ++i)
        if (r.models[i].change_probability == 1.0)
            n = i + 1;

    if (n == 0 && !r.models.empty() && r.models[0].curve_qa == ccd::CurveQA::Start)
        n = 1;

    return n;
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

// The sweeps over cuts run as SHARDS tests each, so ctest runs them in
// parallel. Shard 0 is the sample, the quick tier's part: every eighth cut,
// a series' last clean cut, and every cut from two observations before a
// break of the full run to NEAR_BREAK after it (where the machine finds the
// break); shards 1 to SHARDS - 1 hold the rest, interleaved. Together they
// take each cut once. The sample follows the full run's breaks, so a change
// to the procedure can move cuts between shards, never out of them.
inline constexpr std::size_t SHARDS = 8;
inline constexpr std::size_t NEAR_BREAK = 10;

// The cuts of `cuts` (indexes into `s`, ascending) that split `s` cleanly
// and fall in `shard`; `ordinal` staggers the interleave from one series to
// the next.
inline std::vector<std::size_t> shard_of(const Series& s, std::size_t ordinal,
                                         const std::vector<std::size_t>& cuts, std::size_t shard)
{
    std::vector<bool> near(s.size(), false);
    for (const auto& m : detect_at(s, s.statistics_end()).models)
    {
        if (m.change_probability != 1.0)
            continue;
        const std::size_t at = static_cast<std::size_t>(
            std::lower_bound(s.dates.begin(), s.dates.end(), m.break_day) - s.dates.begin());
        const std::size_t from = at < 2 ? 0 : at - 2;
        for (std::size_t t = from; t < std::min(s.size(), at + NEAR_BREAK + 1); ++t)
            near[t] = true;
    }

    std::vector<std::size_t> clean;
    for (const std::size_t last : cuts)
        if (splits_cleanly(s, last))
            clean.push_back(last);

    std::vector<std::size_t> out;
    std::size_t rest = ordinal;
    for (std::size_t k = 0; k < clean.size(); ++k)
    {
        const std::size_t last = clean[k];
        const bool sample = k % SHARDS == 0 || k + 1 == clean.size() || near[last];
        const std::size_t in = sample ? 0 : 1 + rest++ % (SHARDS - 1);
        if (in == shard)
            out.push_back(last);
    }
    return out;
}

// The test name of a shard: "sample" for shard 0, "restN" for the others.
inline std::string shard_name(std::size_t shard)
{
    return shard == 0 ? "sample" : "rest" + std::to_string(shard);
}

} // namespace golden
