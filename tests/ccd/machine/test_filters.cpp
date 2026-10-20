//==============================================================================
//
// The observation filters, on hand-built series.
//
//   Bounds.*   in_range() and the two range filters, at the bound values
//              themselves and at NaN, with each end inclusive or exclusive.
//              The saturation bound is tested in DN (0, 10000) and in
//              reflectance (0.0, 1.0): the flags mean the same in both units.
//   GreenScope.*
//              the InsufficientClear chain's green median step under each
//              GreenFilterScope, on a series with green spikes before and
//              after statistics_end, and on one dated entirely after it.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/core/array_view.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/pmask.hpp"

#include "../golden/config.hpp"

namespace
{

constexpr ccd::scalar_t NaN = std::numeric_limits<ccd::scalar_t>::quiet_NaN();

ccd::filter::Range range(
    ccd::scalar_t lo, bool lo_inclusive,
    ccd::scalar_t hi, bool hi_inclusive
)
{
    return {{lo, lo_inclusive}, {hi, hi_inclusive}};
}

// The mask a filter leaves on an all-true mask, as 0/1 per observation.
std::vector<int> kept(const ccd::ProcessingMask& mask)
{
    std::vector<int> out;
    for (ccd::index_t i = 0; i < mask.size(); ++i)
        out.push_back(mask.test(i) ? 1 : 0);
    return out;
}

ccd::ProcessingMask all_true(ccd::index_t T)
{
    ccd::ProcessingMask mask(T);
    for (ccd::index_t i = 0; i < T; ++i)
        mask.set(i, true);
    return mask;
}

// Values at, inside and outside [lo, hi], and NaN, in this order:
//   lo, hi, midpoint, below lo, above hi, NaN
std::vector<ccd::scalar_t> probe(ccd::scalar_t lo, ccd::scalar_t hi)
{
    const ccd::scalar_t step = (hi - lo) / 1000.0;
    return {lo, hi, (lo + hi) / 2.0, lo - step, hi + step, NaN};
}

struct Mode
{
    bool lo_inclusive;
    bool hi_inclusive;
    std::vector<int> expected;   // per probe() value
};

// What each combination of flags keeps of probe().
const std::vector<Mode> MODES = {
    {true,  true,  {1, 1, 1, 0, 0, 1}},   // NaN passes
    {false, false, {0, 0, 1, 0, 0, 0}},   // NaN rejected
    {true,  false, {1, 0, 1, 0, 0, 0}},
    {false, true,  {0, 1, 1, 0, 0, 0}},
};

// (B, T): B rows, each the same series.
std::vector<ccd::scalar_t> spectral_of(
    const std::vector<ccd::scalar_t>& series,
    ccd::index_t B
)
{
    std::vector<ccd::scalar_t> out;
    for (ccd::index_t b = 0; b < B; ++b)
        out.insert(out.end(), series.begin(), series.end());
    return out;
}

} // namespace


TEST(Bounds, InRange)
{
    for (const auto& mode : MODES)
    {
        const auto values = probe(0.0, 10000.0);
        const auto r = range(0.0, mode.lo_inclusive, 10000.0, mode.hi_inclusive);

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            EXPECT_EQ(ccd::filter::in_range(values[i], r), mode.expected[i] == 1)
                << "value " << values[i] << " lo_inclusive " << mode.lo_inclusive
                << " hi_inclusive " << mode.hi_inclusive;
        }
    }
}

// Both ends inclusive is the expression !(v < lo || v > hi), value for value,
// NaN included; both exclusive is v > lo && v < hi.
TEST(Bounds, InclusiveAndExclusiveAreTheLiteralForms)
{
    const std::vector<ccd::scalar_t> values = {
        -1.0, 0.0, 1e-300, 0.5, 1.0 - 1e-16, 1.0, 1.0 + 1e-15, 2.0,
        NaN, std::numeric_limits<ccd::scalar_t>::infinity(),
        -std::numeric_limits<ccd::scalar_t>::infinity()
    };

    for (const auto v : values)
    {
        EXPECT_EQ(ccd::filter::in_range(v, range(0.0, true, 1.0, true)),
                  !(v < 0.0 || v > 1.0)) << v;
        EXPECT_EQ(ccd::filter::in_range(v, range(0.0, false, 1.0, false)),
                  v > 0.0 && v < 1.0) << v;
    }
}

// The "ccdc" bounds on the default input: pyccd's exclusive (0, 10000) and
// (-9320, 7070), from reflectance (0, 1) and (-93.2, 70.7) degrees Celsius.
TEST(Bounds, ConfigurationDefaults)
{
    const ccd::preprocess::PreprocessPlan plan = golden::plan_of(golden::ccdc());
    const auto& thermal = plan.standard.chain[1];
    const auto& band = plan.standard.chain[2];
    ASSERT_EQ(thermal.kind, ccd::preprocess::StepKind::ThermalRange);
    ASSERT_EQ(band.kind, ccd::preprocess::StepKind::BandRange);

    EXPECT_EQ(band.range.lo.value, 0.0);
    EXPECT_EQ(band.range.hi.value, 10000.0);
    EXPECT_FALSE(band.range.lo.inclusive);
    EXPECT_FALSE(band.range.hi.inclusive);

    EXPECT_EQ(thermal.range.lo.value, ccd::MIN_CELSIUS);
    EXPECT_EQ(thermal.range.hi.value, ccd::MAX_CELSIUS);
    EXPECT_FALSE(thermal.range.lo.inclusive);
    EXPECT_FALSE(thermal.range.hi.inclusive);
}

// The saturation filter, in DN and in reflectance, every mode. One band out
// of range is enough to reject the observation.
TEST(Bounds, SaturationFilterAtTheBounds)
{
    struct Unit { ccd::scalar_t lo, hi; };

    for (const Unit unit : {Unit{0.0, 10000.0}, Unit{0.0, 1.0}})
    {
        const auto series = probe(unit.lo, unit.hi);
        const auto T = static_cast<ccd::index_t>(series.size());
        constexpr ccd::index_t B = 3;

        for (const auto& mode : MODES)
        {
            // Every band carries the probe series.
            {
                const auto spectral = spectral_of(series, B);
                auto mask = all_true(T);
                ccd::filter::apply_saturation_filter(
                    ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                        spectral.data(), {B, T}),
                    ccd::BandList{0, 1, 2},
                    range(unit.lo, mode.lo_inclusive, unit.hi, mode.hi_inclusive),
                    mask);
                EXPECT_EQ(kept(mask), mode.expected)
                    << "unit hi " << unit.hi << " lo_inclusive "
                    << mode.lo_inclusive << " hi_inclusive " << mode.hi_inclusive;
            }

            // Only the last tested band carries it; the others are mid-range.
            {
                std::vector<ccd::scalar_t> spectral(
                    B * T, (unit.lo + unit.hi) / 2.0);
                for (ccd::index_t t = 0; t < T; ++t)
                    spectral[2 * T + t] = series[t];
                auto mask = all_true(T);
                ccd::filter::apply_saturation_filter(
                    ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                        spectral.data(), {B, T}),
                    ccd::BandList{0, 1, 2},
                    range(unit.lo, mode.lo_inclusive, unit.hi, mode.hi_inclusive),
                    mask);
                EXPECT_EQ(kept(mask), mode.expected);
            }
        }
    }
}

// The thermal filter, in Celsius x 100 and Celsius / 100, every mode.
TEST(Bounds, ThermalFilterAtTheBounds)
{
    struct Unit { ccd::scalar_t lo, hi; };

    for (const Unit unit : {
             Unit{ccd::MIN_CELSIUS, ccd::MAX_CELSIUS},
             Unit{ccd::MIN_CELSIUS / 10000.0, ccd::MAX_CELSIUS / 10000.0}})
    {
        const auto series = probe(unit.lo, unit.hi);
        const auto T = static_cast<ccd::index_t>(series.size());
        constexpr ccd::index_t B = 2;

        // Band 0 is out of range everywhere: the thermal filter does not
        // read it.
        std::vector<ccd::scalar_t> spectral(B * T, -1e9);
        for (ccd::index_t t = 0; t < T; ++t)
            spectral[1 * T + t] = series[t];

        for (const auto& mode : MODES)
        {
            auto mask = all_true(T);
            ccd::filter::apply_thermal_filter(
                ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                    spectral.data(), {B, T}),
                1,
                range(unit.lo, mode.lo_inclusive, unit.hi, mode.hi_inclusive),
                mask);
            EXPECT_EQ(kept(mask), mode.expected)
                << "lo_inclusive " << mode.lo_inclusive
                << " hi_inclusive " << mode.hi_inclusive;
        }
    }
}

// The Standard chain reads the flags from the configuration's steps.
TEST(Bounds, TheStandardChainReadsTheFlags)
{
    const auto series = probe(0.0, 10000.0);
    const auto T = static_cast<ccd::index_t>(series.size());
    constexpr ccd::index_t B = 7;

    std::vector<std::int64_t> dates(T);
    for (ccd::index_t t = 0; t < T; ++t)
        dates[t] = 730000 + 16 * t;

    // Optical bands carry the probe; thermal is mid-range.
    std::vector<ccd::scalar_t> spectral(B * T, 0.0);
    for (ccd::index_t b = 0; b < 6; ++b)
        for (ccd::index_t t = 0; t < T; ++t)
            spectral[b * T + t] = series[t];
    for (ccd::index_t t = 0; t < T; ++t)
        spectral[6 * T + t] = 2000.0;

    const std::vector<std::uint8_t> qas(T, 0);

    for (const auto& mode : MODES)
    {
        const ccd::Input h;
        const ccd::config::Config config = golden::with_steps(
            golden::with_steps(golden::pyccd(), "band_range", "min_inclusive",
                               ccd::config::flag(mode.lo_inclusive)),
            "band_range", "max_inclusive", ccd::config::flag(mode.hi_inclusive));

        const ccd::config::Shared shared = golden::shared_of(config);
        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                spectral.data(), {B, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}),
            h, shared);

        EXPECT_EQ(kept(golden::chain_mask(workspace, "standard", config)), mode.expected);
    }

    // The thermal flags, with the optical bands mid-range.
    const auto therm = probe(ccd::MIN_CELSIUS, ccd::MAX_CELSIUS);
    for (ccd::index_t b = 0; b < 6; ++b)
        for (ccd::index_t t = 0; t < T; ++t)
            spectral[b * T + t] = 5000.0;
    for (ccd::index_t t = 0; t < T; ++t)
        spectral[6 * T + t] = therm[t];

    for (const auto& mode : MODES)
    {
        const ccd::Input h;
        const ccd::config::Config config = golden::with_steps(
            golden::with_steps(golden::pyccd(), "thermal_range", "min_inclusive",
                               ccd::config::flag(mode.lo_inclusive)),
            "thermal_range", "max_inclusive", ccd::config::flag(mode.hi_inclusive));

        const ccd::config::Shared shared = golden::shared_of(config);
        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                spectral.data(), {B, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}),
            h, shared);

        EXPECT_EQ(kept(golden::chain_mask(workspace, "standard", config)), mode.expected);
    }
}


namespace
{

constexpr std::int64_t STATISTICS_END = 731000;

// The configuration with the cutoff at STATISTICS_END and the green median's scope
// `scope`.
ccd::config::Config config_with(ccd::GreenFilterScope scope)
{
    return golden::with_steps(
        golden::with_shared(golden::pyccd(), "statistics_end", ccd::config::integer(STATISTICS_END)),
        "green_median", "scope", ccd::config::Value(std::string(ccd::name(scope))));
}

// A 7-band DN series of `T` observations 16 days apart from `first`, all
// clear, green 1000 except where `green` says otherwise.
struct Series
{
    std::vector<std::int64_t> dates;
    std::vector<ccd::scalar_t> spectral;
    std::vector<std::uint8_t> qas;
    ccd::index_t T;

    Series(std::int64_t first, ccd::index_t T_)
        : dates(T_), spectral(7 * T_, 1000.0), qas(T_, 0), T(T_)
    {
        for (ccd::index_t t = 0; t < T; ++t)
        {
            dates[t] = first + 16 * t;
            spectral[6 * T + t] = 2900.0;
        }
    }

    void green(ccd::index_t t, ccd::scalar_t value)
    {
        spectral[1 * T + t] = value;
    }

    std::vector<int> insufficient_clear(ccd::GreenFilterScope scope) const
    {
        const ccd::Input h;
        const ccd::config::Config config = config_with(scope);
        const ccd::config::Shared shared = golden::shared_of(config);

        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}),
            h, shared);

        return kept(golden::chain_mask(workspace, "insufficient_clear", config));
    }
};

} // namespace

TEST(GreenScope, DefaultIsStatWindow)
{
    EXPECT_EQ(golden::plan_of(golden::ccdc()).insufficient_clear.chain[3].scope,
              ccd::GreenFilterScope::StatWindow);
}

// Green median before statistics_end is 1000, so the threshold is 1400. Spikes of
// 2000 sit at t = 2 (before statistics_end) and t = 17 (after); t = 4 is exactly
// the threshold, which the filter drops (green >= threshold), and t = 6 is
// just below it. t = 9 is a cloud: out of the mask whatever its green.
TEST(GreenScope, SpikesBeforeAndAfterStatOrd)
{
    // 20 observations; the first 10 are dated <= statistics_end.
    Series s(STATISTICS_END - 16 * 9, 20);
    ASSERT_LE(s.dates[9], STATISTICS_END);
    ASSERT_GT(s.dates[10], STATISTICS_END);

    s.green(2, 2000.0);
    s.green(4, 1400.0);
    s.green(6, 1399.0);
    s.green(17, 2000.0);
    s.green(14, 1400.0);
    s.qas[9] = 4;

    std::vector<int> off(20, 1);
    off[9] = 0;

    std::vector<int> window = off;
    window[2] = window[4] = 0;

    std::vector<int> all = window;
    all[17] = all[14] = 0;

    EXPECT_EQ(s.insufficient_clear(ccd::GreenFilterScope::All), all);
    EXPECT_EQ(s.insufficient_clear(ccd::GreenFilterScope::StatWindow), window);
    EXPECT_EQ(s.insufficient_clear(ccd::GreenFilterScope::Off), off);
}

// Every observation after statistics_end: no median, so no scope drops anything.
TEST(GreenScope, NothingBeforeStatOrd)
{
    Series s(STATISTICS_END + 1, 12);
    s.green(3, 5000.0);
    s.green(8, 9000.0);

    const std::vector<int> every(12, 1);

    for (const auto scope : {ccd::GreenFilterScope::All,
                             ccd::GreenFilterScope::StatWindow,
                             ccd::GreenFilterScope::Off})
    {
        EXPECT_EQ(s.insufficient_clear(scope), every)
            << "scope " << static_cast<int>(scope);
    }
}

// The scope is read only by InsufficientClear: the Standard filter mask is
// the same under each.
TEST(GreenScope, StandardFilterIgnoresIt)
{
    Series s(STATISTICS_END - 16 * 9, 20);
    s.green(2, 2000.0);
    s.green(17, 2000.0);

    std::vector<std::vector<int>> masks;
    for (const auto scope : {ccd::GreenFilterScope::All,
                             ccd::GreenFilterScope::StatWindow,
                             ccd::GreenFilterScope::Off})
    {
        const ccd::Input h;
        const ccd::config::Config config = config_with(scope);
        const ccd::config::Shared shared = golden::shared_of(config);
        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(s.dates.data(), {s.T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                s.spectral.data(), {7, s.T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(s.qas.data(), {s.T}),
            h, shared);
        masks.push_back(kept(golden::chain_mask(workspace, "standard", config)));
    }

    EXPECT_EQ(masks[0], std::vector<int>(20, 1));
    EXPECT_EQ(masks[1], masks[0]);
    EXPECT_EQ(masks[2], masks[0]);
}
