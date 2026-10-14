//==============================================================================
//
// The cadence and change-probability options of HarmonicOptions.
//
// PEEK_REFERENCE_DAYS, VARIOGRAM_MIN_LAG_DAYS, RETRAIN_MIN_OBS,
// RETRAIN_SPAN_FACTOR, SEASONAL_RMSE_COUNT, SEASONAL_RMSE_DIVISOR and
// CHANGE_PROBABILITY. These tests hold two claims:
//
//   * Cadence.DefaultsSpelledOutEqualDefault: each option set explicitly to
//     its default gives output bit for bit equal to HarmonicOptions{}, on the
//     Landsat corpus of test_golden.cpp in DN and in physical units.
//
//   * One test per option: moving it moves the result in the direction the
//     option's meaning predicts, on a synthetic pixel built so that the
//     option decides the outcome. The step sizes and drift rates were chosen
//     by scanning; each case sits on the side of a boundary the option moves.
//
// The PeekThreshold tests hold the threshold that goes with an adjusted peek
// window: a series denser than PEEK_REFERENCE_DAYS gets a longer peek window
// and, with it, the lower change threshold adjust_change_threshold() takes
// from the caller's PEEK_SIZE and CHANGE_THRESHOLD.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/procedure/fit.hpp"

#include "golden.hpp"
#include "synthetic.hpp"

namespace
{

//------------------------------------------------------------------------------
// A synthetic 7-band DN pixel at a fixed cadence, every observation clear.
//
// Optical band b is 1000 + 100 b, plus `amplitude` sin(2 pi t) and
// `drift` t^2 in years since the first date, plus N(0, 30) noise; thermal is
// 2900 + 50 sin(2 pi t), noise-free. Each Step adds `offset` to the five
// detection bands over observations [first, last).
//------------------------------------------------------------------------------

struct Step
{
    std::size_t first;
    std::size_t last;
    double offset;
};

struct Pixel
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, T)
    std::vector<std::uint8_t> qas;  // (T)
};

constexpr std::size_t T = 500;

Pixel make_pixel(
    double amplitude,
    double drift,
    const std::vector<Step>& steps,
    std::int64_t cadence = 16
)
{
    constexpr double TWO_PI = 6.283185307179586;

    golden::Rng rng(1);
    Pixel p;
    p.dates.resize(T);
    p.spectral.resize(7 * T);
    p.qas.assign(T, 0);

    for (std::size_t t = 0; t < T; ++t)
        p.dates[t] = golden::ORDINAL_1984 + cadence * static_cast<std::int64_t>(t);

    for (std::size_t b = 0; b < 7; ++b)
    {
        for (std::size_t t = 0; t < T; ++t)
        {
            const double years = static_cast<double>(p.dates[t] - p.dates[0]) / 365.25;
            const double season = std::sin(TWO_PI * years);
            p.spectral[b * T + t] = b == 6
                ? 2900.0 + 50.0 * season
                : 1000.0 + 100.0 * b + amplitude * season + drift * years * years
                      + 30.0 * rng.normal();
        }
    }

    for (const auto& s : steps)
        for (std::size_t b = 1; b <= 5; ++b)
            for (std::size_t t = s.first; t < s.last; ++t)
                p.spectral[b * T + t] += s.offset;

    return p;
}

ccd::FitResult run(const Pixel& p, const ccd::HarmonicOptions& h)
{
    return ccd::detect(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
        h,
        ccd::LassoOptions{}
    );
}

// Segments that end in a detected change.
int breaks(const ccd::FitResult& r)
{
    int n = 0;
    for (const auto& m : r.models)
        n += m.change_probability == 1.0;
    return n;
}

// The day the first segment ends: its break, or the last date without one.
std::int64_t first_break(const ccd::FitResult& r)
{
    return r.models.front().break_day;
}

// Every option of this file set to its default value by hand.
ccd::HarmonicOptions spelled_out(ccd::HarmonicOptions h)
{
    h.PEEK_REFERENCE_DAYS = 16.0;
    h.VARIOGRAM_MIN_LAG_DAYS = 30;
    h.RETRAIN_MIN_OBS = 24;
    h.RETRAIN_SPAN_FACTOR = 1.33;
    h.SEASONAL_RMSE_COUNT = 24;
    h.SEASONAL_RMSE_DIVISOR = 4.0;
    h.CHANGE_PROBABILITY = 0.99;
    return h;
}

std::string dump_cube(
    golden::Cube c,
    const ccd::HarmonicOptions& h,
    const ccd::LassoOptions& l,
    bool physical
)
{
    std::string out;
    for (std::size_t p = 0; p < c.H * c.W; ++p)
    {
        double* block = c.spectral.data() + p * 7 * c.T;
        if (physical)
            golden::to_physical(block, c.T);

        golden::dump_pixel(out, std::to_string(p), ccd::detect(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {c.T}),
            ccd::ArrayView<const double, 2>::contiguous(block, {7, c.T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data() + p * c.T, {c.T}),
            h, l));
    }
    return out;
}

} // namespace

//==============================================================================
// Defaults
//==============================================================================

TEST(Cadence, Defaults)
{
    const ccd::HarmonicOptions h;
    EXPECT_EQ(h.PEEK_REFERENCE_DAYS, 16.0);
    EXPECT_EQ(h.VARIOGRAM_MIN_LAG_DAYS, 30);
    EXPECT_EQ(h.RETRAIN_MIN_OBS, 24u);
    EXPECT_EQ(h.RETRAIN_SPAN_FACTOR, 1.33);
    EXPECT_EQ(h.SEASONAL_RMSE_COUNT, 24u);
    EXPECT_EQ(h.SEASONAL_RMSE_DIVISOR, 4.0);
    EXPECT_EQ(h.CHANGE_PROBABILITY, 0.99);

    // CHANGE_THRESHOLD is the chi2(5) quantile at CHANGE_PROBABILITY, to the
    // last bit.
    EXPECT_EQ(ccd::chi2_quantile(h.CHANGE_PROBABILITY, 5), h.CHANGE_THRESHOLD);
}

TEST(Cadence, DefaultsSpelledOutEqualDefault)
{
    const struct { std::uint64_t seed; golden::Regime regime; std::size_t H, W; } corpus[] = {
        {11, golden::Regime::Cadence16, 10, 16},
        {12, golden::Regime::Cadence8,   6,  8},
        {13, golden::Regime::Duplicates, 5,  8},
        {14, golden::Regime::Short,      4,  5},
        {15, golden::Regime::Medium,     5,  8},
    };

    for (const auto& c : corpus)
    {
        const auto cube = golden::make_cube(c.seed, c.regime, c.H, c.W);

        EXPECT_EQ(
            dump_cube(cube, spelled_out(ccd::HarmonicOptions{}), ccd::LassoOptions{}, false),
            dump_cube(cube, ccd::HarmonicOptions{}, ccd::LassoOptions{}, false));

        EXPECT_EQ(
            dump_cube(cube, spelled_out(golden::physical_harmonic()), golden::physical_lasso(), true),
            dump_cube(cube, golden::physical_harmonic(), golden::physical_lasso(), true));
    }
}

//==============================================================================
// PEEK_REFERENCE_DAYS: a longer reference interval makes a longer peek
// window, which a short-lived disturbance does not fill.
//==============================================================================

TEST(Cadence, PeekReferenceDays)
{
    std::vector<std::int64_t> dates16(100), dates8(100);
    for (std::size_t i = 0; i < 100; ++i)
    {
        dates16[i] = golden::ORDINAL_1984 + 16 * static_cast<std::int64_t>(i);
        dates8[i] = golden::ORDINAL_1984 + 8 * static_cast<std::int64_t>(i);
    }
    const auto d16 = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates16.data(), {100});
    const auto d8 = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates8.data(), {100});

    EXPECT_EQ(ccd::adjust_peek(d16, 6, 16.0), 6u);
    EXPECT_EQ(ccd::adjust_peek(d16, 6, 32.0), 12u);
    EXPECT_EQ(ccd::adjust_peek(d16, 6, 8.0), 6u);    // never below PEEK_SIZE
    EXPECT_EQ(ccd::adjust_peek(d8, 6, 16.0), 12u);
    EXPECT_EQ(ccd::adjust_peek(d8, 6, 4.0), 6u);

    // Gaps alternating 8 and 9 days: an even count whose middle two differ,
    // so the median is 8.5 (np.median), 96 / 8.501 rounds to 11. A median
    // truncated to 8 would give 12.
    std::vector<std::int64_t> dates85(101);
    dates85[0] = golden::ORDINAL_1984;
    for (std::size_t i = 1; i < dates85.size(); ++i)
        dates85[i] = dates85[i - 1] + (i % 2 ? 8 : 9);
    const auto d85 = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates85.data(), {101});
    EXPECT_EQ(ccd::adjust_peek(d85, 6, 16.0), 11u);

    ccd::HarmonicOptions longer;
    longer.PEEK_REFERENCE_DAYS = 32.0;

    // Eight disturbed observations fill a peek window of 6, not one of 12.
    const auto transient = make_pixel(300.0, 0.0, {{250, 258, 600.0}});
    EXPECT_EQ(breaks(run(transient, ccd::HarmonicOptions{})), 1);
    EXPECT_EQ(breaks(run(transient, longer)), 0);

    // A lasting step is found by both, on the same day.
    const auto step = make_pixel(300.0, 0.0, {{250, T, 600.0}});
    const auto at6 = run(step, ccd::HarmonicOptions{});
    const auto at12 = run(step, longer);
    ASSERT_EQ(breaks(at6), 1);
    ASSERT_EQ(breaks(at12), 1);
    EXPECT_EQ(first_break(at6), step.dates[250]);
    EXPECT_EQ(first_break(at12), step.dates[250]);
}

//==============================================================================
// VARIOGRAM_MIN_LAG_DAYS: a longer minimum lag measures the variogram over
// pairs further apart, which on a seasonal signal is larger, so a step must
// be larger to count as change.
//==============================================================================

TEST(Cadence, VariogramMinLagDays)
{
    // A ramp of 10 per observation at a 16-day cadence: the variogram is 10
    // times the lag chosen.
    constexpr std::size_t N = 50;
    std::vector<std::int64_t> dates(N);
    std::vector<double> ramp(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        dates[i] = golden::ORDINAL_1984 + 16 * static_cast<std::int64_t>(i);
        ramp[i] = 10.0 * static_cast<double>(i);
    }
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {N});
    const auto s = ccd::ArrayView<const double, 2>::contiguous(ramp.data(), {1, N});

    EXPECT_EQ(ccd::adjusted_variogram(d, s, 10)[0], 10.0);  // lag 1, 16 days
    EXPECT_EQ(ccd::adjusted_variogram(d, s, 30)[0], 20.0);  // lag 2, 32 days
    EXPECT_EQ(ccd::adjusted_variogram(d, s, 40)[0], 30.0);  // lag 3, 48 days
    EXPECT_EQ(ccd::adjusted_variogram(d, s, 60)[0], 40.0);  // lag 4, 64 days

    ccd::HarmonicOptions shorter, longer;
    shorter.VARIOGRAM_MIN_LAG_DAYS = 10;
    longer.VARIOGRAM_MIN_LAG_DAYS = 60;

    const auto small = make_pixel(300.0, 0.0, {{250, T, 150.0}});
    EXPECT_EQ(breaks(run(small, shorter)), 1);
    EXPECT_EQ(breaks(run(small, ccd::HarmonicOptions{})), 0);

    const auto large = make_pixel(300.0, 0.0, {{250, T, 300.0}});
    EXPECT_EQ(breaks(run(large, ccd::HarmonicOptions{})), 1);
    EXPECT_EQ(breaks(run(large, longer)), 0);
}

//==============================================================================
// RETRAIN_SPAN_FACTOR and RETRAIN_MIN_OBS: on a pixel with accelerating
// drift, a model refitted less often departs from the data sooner, so the
// first break comes earlier. A larger span factor refits less often; a
// larger minimum count refits more often.
//==============================================================================

TEST(Cadence, RetrainSpanFactor)
{
    const auto drifting = make_pixel(0.0, 0.5, {});

    ccd::HarmonicOptions often;
    often.RETRAIN_SPAN_FACTOR = 1.2;
    EXPECT_EQ(breaks(run(drifting, often)), 0);

    std::int64_t previous = 0;
    for (const double factor : {1.33, 1.6, 2.0, 4.0})
    {
        ccd::HarmonicOptions h;
        h.RETRAIN_SPAN_FACTOR = factor;
        const auto r = run(drifting, h);
        ASSERT_GE(breaks(r), 1) << "factor " << factor;
        if (previous != 0)
            EXPECT_LT(first_break(r), previous) << "factor " << factor;
        previous = first_break(r);
    }
}

// The span rule disabled, so that only RETRAIN_MIN_OBS refits: the models
// are refitted while the window holds fewer observations than it.
TEST(Cadence, RetrainMinObs)
{
    const auto drifting = make_pixel(0.0, 0.5, {});

    std::int64_t previous = 0;
    std::size_t segments = 0;
    for (const ccd::index_t count : {24, 48, 96, 200})
    {
        ccd::HarmonicOptions h;
        h.RETRAIN_SPAN_FACTOR = 1e6;
        h.RETRAIN_MIN_OBS = count;
        const auto r = run(drifting, h);
        ASSERT_GE(breaks(r), 1) << "count " << count;
        if (previous != 0)
        {
            EXPECT_GT(first_break(r), previous) << "count " << count;
            EXPECT_LE(r.models.size(), segments) << "count " << count;
        }
        previous = first_break(r);
        segments = r.models.size();
    }
}

//==============================================================================
// SEASONAL_RMSE_DIVISOR and SEASONAL_RMSE_COUNT: the seasonal RMSE is the
// norm of SEASONAL_RMSE_COUNT residuals over SEASONAL_RMSE_DIVISOR, and it
// normalises the change magnitude. On a pixel without seasonality, where it
// outweighs the variogram, a larger divisor or a smaller count makes a
// smaller step count as change.
//==============================================================================

TEST(Cadence, SeasonalRmseDivisor)
{
    ccd::HarmonicOptions larger, smaller;
    larger.SEASONAL_RMSE_DIVISOR = 8.0;
    smaller.SEASONAL_RMSE_DIVISOR = 2.0;

    const auto small = make_pixel(0.0, 0.0, {{250, T, 50.0}});
    EXPECT_EQ(breaks(run(small, ccd::HarmonicOptions{})), 0);
    EXPECT_EQ(breaks(run(small, larger)), 1);

    const auto medium = make_pixel(0.0, 0.0, {{250, T, 60.0}});
    EXPECT_EQ(breaks(run(medium, ccd::HarmonicOptions{})), 1);
    EXPECT_EQ(breaks(run(medium, smaller)), 0);
}

TEST(Cadence, SeasonalRmseCount)
{
    ccd::HarmonicOptions fewer, more;
    fewer.SEASONAL_RMSE_COUNT = 6;
    more.SEASONAL_RMSE_COUNT = 96;

    const auto small = make_pixel(0.0, 0.0, {{250, T, 50.0}});
    EXPECT_EQ(breaks(run(small, ccd::HarmonicOptions{})), 0);
    EXPECT_EQ(breaks(run(small, fewer)), 1);

    const auto medium = make_pixel(0.0, 0.0, {{250, T, 60.0}});
    EXPECT_EQ(breaks(run(medium, ccd::HarmonicOptions{})), 1);
    EXPECT_EQ(breaks(run(medium, more)), 0);
}

//==============================================================================
// CHANGE_PROBABILITY: a higher probability raises the threshold of a peek
// window longer than the default. Read by adjust_change_threshold(), tested
// there and through detect() on a dense series.
//==============================================================================

TEST(Cadence, ChangeProbability)
{
    constexpr double DEFAULT = 15.086272469388987;

    // At or below the default window, the default threshold whatever the
    // probability.
    EXPECT_EQ(ccd::adjust_change_threshold(6, 6, DEFAULT, 5, 0.999), DEFAULT);

    // Above it, the chi2 quantile of 1 - (1 - p)^(defpeek / peek).
    for (const double p : {0.9, 0.99, 0.999})
    {
        EXPECT_EQ(
            ccd::adjust_change_threshold(12, 6, DEFAULT, 5, p),
            ccd::chi2_quantile(1.0 - std::pow(1.0 - p, 6.0 / 12.0), 5));
    }

    EXPECT_LT(ccd::adjust_change_threshold(12, 6, DEFAULT, 5, 0.9),
              ccd::adjust_change_threshold(12, 6, DEFAULT, 5, 0.99));
    EXPECT_LT(ccd::adjust_change_threshold(12, 6, DEFAULT, 5, 0.99),
              ccd::adjust_change_threshold(12, 6, DEFAULT, 5, 0.999));

    // A higher probability, a higher quantile.
    EXPECT_LT(ccd::chi2_quantile(0.99, 5), ccd::chi2_quantile(0.999, 5));
}

// At an 8-day cadence the peek window is 12 and the threshold is the chi2(5)
// quantile at 1 - (1 - p)^(6 / 12): 9.24 at 0.99, 12.24 at 0.999. A step of 58
// clears the first and not the second; a step of 65 clears both.
TEST(Cadence, ChangeProbabilityThroughDetect)
{
    ccd::HarmonicOptions higher;
    higher.CHANGE_PROBABILITY = 0.999;

    const auto small = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 8);
    const auto at99 = run(small, ccd::HarmonicOptions{});
    ASSERT_EQ(breaks(at99), 1);
    EXPECT_EQ(first_break(at99), small.dates[250]);
    EXPECT_EQ(breaks(run(small, higher)), 0);

    const auto larger = make_pixel(0.0, 0.0, {{250, T, 65.0}}, 8);
    const auto at999 = run(larger, higher);
    ASSERT_EQ(breaks(at999), 1);
    EXPECT_EQ(first_break(at999), larger.dates[250]);
}

//==============================================================================
// PeekThreshold: the threshold of an adjusted peek window.
//==============================================================================

TEST(PeekThreshold, AdjustChangeThreshold)
{
    constexpr double DEFAULT = 15.086272469388987;

    // pyccd: chi2.ppf(1 - (1 - 0.99) ** (6 / 12), 5).
    const double at12 = ccd::adjust_change_threshold(12, 6, DEFAULT, 5, 0.99);
    EXPECT_EQ(at12, ccd::chi2_quantile(1.0 - std::pow(1.0 - 0.99, 6.0 / 12.0), 5));
    EXPECT_NEAR(at12, 9.236356899781123, 1e-12);

    // pyccd: chi2.ppf(1 - (1 - 0.99) ** (6 / 8), 5).
    EXPECT_NEAR(
        ccd::adjust_change_threshold(8, 6, DEFAULT, 5, 0.99),
        12.241629783293535,
        1e-12);

    // A window no longer than the default keeps the default threshold.
    EXPECT_EQ(ccd::adjust_change_threshold(6, 6, DEFAULT, 5, 0.99), DEFAULT);
}

// An 8-day series runs with a peek window of 12 and the threshold of that
// window, so it gives the same output as options that fix both: PEEK_SIZE 12
// with a PEEK_REFERENCE_DAYS that leaves it at 12, and CHANGE_THRESHOLD at
// the adjusted value. Holding the threshold at its default instead misses a
// step the adjusted one finds.
TEST(PeekThreshold, DenseSeriesUsesAdjustedThreshold)
{
    ccd::HarmonicOptions fixed;
    fixed.PEEK_SIZE = 12;
    fixed.PEEK_REFERENCE_DAYS = 8.0;
    fixed.CHANGE_THRESHOLD = ccd::adjust_change_threshold(12, 6, ccd::HarmonicOptions{}.CHANGE_THRESHOLD, 5, 0.99);

    ccd::HarmonicOptions unadjusted = fixed;
    unadjusted.CHANGE_THRESHOLD = ccd::HarmonicOptions{}.CHANGE_THRESHOLD;

    const auto step = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 8);

    std::string adjusted_dump, fixed_dump;
    golden::dump_pixel(adjusted_dump, "step", run(step, ccd::HarmonicOptions{}));
    golden::dump_pixel(fixed_dump, "step", run(step, fixed));
    EXPECT_EQ(adjusted_dump, fixed_dump);

    const auto adjusted = run(step, ccd::HarmonicOptions{});
    ASSERT_EQ(breaks(adjusted), 1);
    EXPECT_EQ(first_break(adjusted), step.dates[250]);
    EXPECT_EQ(breaks(run(step, unadjusted)), 0);
}

// The adjustment starts from the caller's options on every call: a sparse
// pixel run after a dense one gives the output it gets alone, and the
// options passed in keep their values.
TEST(PeekThreshold, EachPixelStartsFromTheCallerOptions)
{
    const auto dense = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 8);
    const auto sparse = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 16);

    std::string alone;
    golden::dump_pixel(alone, "sparse", run(sparse, ccd::HarmonicOptions{}));

    const ccd::HarmonicOptions h;
    run(dense, h);
    std::string after;
    golden::dump_pixel(after, "sparse", run(sparse, h));
    EXPECT_EQ(alone, after);
    EXPECT_EQ(h.PEEK_SIZE, 6u);
    EXPECT_EQ(h.CHANGE_THRESHOLD, ccd::HarmonicOptions{}.CHANGE_THRESHOLD);
}
