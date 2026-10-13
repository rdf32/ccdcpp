//==============================================================================
//
// The observation filters, on hand-built series.
//
//   Bounds.*   in_range() and the two range filters, at the bound values
//              themselves and at NaN, with each end inclusive or exclusive.
//              The saturation bound is tested in DN (0, 10000) and in
//              reflectance (0.0, 1.0): the flags mean the same in both units.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/array_view.hpp"
#include "ccd/constants.hpp"
#include "ccd/harmonic/filter.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/pmask.hpp"

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

TEST(Bounds, OptionDefaults)
{
    const ccd::HarmonicOptions h;

    EXPECT_TRUE(h.REFL_MIN_INCLUSIVE);
    EXPECT_TRUE(h.REFL_MAX_INCLUSIVE);
    EXPECT_FALSE(h.THERM_MIN_INCLUSIVE);
    EXPECT_FALSE(h.THERM_MAX_INCLUSIVE);

    const auto refl = ccd::filter::reflectance_range(h);
    EXPECT_EQ(refl.lo.value, h.REFL_MIN);
    EXPECT_EQ(refl.hi.value, h.REFL_MAX);
    EXPECT_TRUE(refl.lo.inclusive);
    EXPECT_TRUE(refl.hi.inclusive);

    const auto therm = ccd::filter::thermal_range(h);
    EXPECT_EQ(therm.lo.value, ccd::MIN_CELSIUS);
    EXPECT_EQ(therm.hi.value, ccd::MAX_CELSIUS);
    EXPECT_FALSE(therm.lo.inclusive);
    EXPECT_FALSE(therm.hi.inclusive);
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

// filter::standard reads the flags from the options.
TEST(Bounds, StandardFilterReadsTheFlags)
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
        ccd::HarmonicOptions h;
        h.REFL_MIN_INCLUSIVE = mode.lo_inclusive;
        h.REFL_MAX_INCLUSIVE = mode.hi_inclusive;

        const ccd::HarmonicWorkspace workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                spectral.data(), {B, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}),
            h);

        EXPECT_EQ(kept(ccd::filter::standard(workspace)), mode.expected);
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
        ccd::HarmonicOptions h;
        h.THERM_MIN_INCLUSIVE = mode.lo_inclusive;
        h.THERM_MAX_INCLUSIVE = mode.hi_inclusive;

        const ccd::HarmonicWorkspace workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
                spectral.data(), {B, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}),
            h);

        EXPECT_EQ(kept(ccd::filter::standard(workspace)), mode.expected);
    }
}
