//==============================================================================
//
// The cadence and change-probability parameters of the "ccdc" configuration.
//
// change_test's peek_reference_days, seasonal_count, seasonal_divisor and
// model_rmse_up_to; noise_floor's min_lag_days; refit's min_obs and factor;
// and the change probability of change_test, stability_rule and lookback.
// These tests hold two claims:
//
//   * Cadence.DefaultsSpelledOutEqualDefault: each parameter set explicitly
//     to its default gives output bit for bit equal to the preset's, on the
//     Landsat corpus of test_golden.cpp in DN and in physical units.
//
//   * One test per parameter: moving it moves the result in the direction
//     the parameter's meaning predicts, on a synthetic pixel built so that
//     the parameter decides the outcome. The step sizes and drift rates were
//     chosen by scanning; each case sits on the side of a boundary the
//     parameter moves.
//
// The PeekThreshold tests hold the threshold that goes with an adjusted peek
// window: a series denser than change_test's peek_reference_days gets a
// longer peek window and, with it, the lower change threshold
// adjust_change_threshold() takes from the configured peek_size and
// change_threshold.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/machine/fit.hpp"

#include "golden.hpp"
#include "specs.hpp"
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

ccd::FitResult run(const Pixel& p, const ccd::config::ConfigTree& config)
{
    return ccd::run_pixel(
        golden::spec_of(ccd::HarmonicOptions{}, config),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})).result;
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

// Every parameter of this file set to its default value by hand.
ccd::config::ConfigTree spelled_out(ccd::config::ConfigTree c)
{
    using ccd::config::integer;
    using ccd::config::real;
    c["change_test"].params["peek_reference_days"] = real(16.0);
    c["noise_floor"].params["min_lag_days"] = integer(30);
    c["refit"].params["min_obs"] = integer(24);
    c["change_test"].params["model_rmse_up_to"] = integer(24);
    c["refit"].params["factor"] = real(1.33);
    c["change_test"].params["seasonal_count"] = integer(24);
    c["change_test"].params["seasonal_divisor"] = real(4.0);
    c["change_test"].params["change_probability"] = real(0.99);
    c["stability_rule"].params["change_probability"] = real(0.99);
    c["lookback"].params["change_probability"] = real(0.99);
    return c;
}

std::string dump_cube(
    golden::Cube c,
    const ccd::HarmonicOptions& h,
    const ccd::config::ConfigTree& config,
    bool physical
)
{
    std::string out;
    for (std::size_t p = 0; p < c.H * c.W; ++p)
    {
        double* block = c.spectral.data() + p * 7 * c.T;
        if (physical)
            golden::to_physical(block, c.T);

        golden::dump_pixel(out, std::to_string(p), ccd::run_pixel(
            golden::spec_of(h, config),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {c.T}),
            ccd::ArrayView<const double, 2>::contiguous(block, {7, c.T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data() + p * c.T, {c.T})).result);
    }
    return out;
}

} // namespace

//==============================================================================
// Defaults
//==============================================================================

TEST(Cadence, Defaults)
{
    using golden::parameter;
    const ccd::config::ConfigTree c = golden::ccdc();
    EXPECT_EQ(parameter<double>(c, "change_test", "peek_reference_days"), 16.0);
    EXPECT_EQ(parameter<std::int64_t>(c, "noise_floor", "min_lag_days"), 30);
    EXPECT_EQ(parameter<std::int64_t>(c, "refit", "min_obs"), 24);
    EXPECT_EQ(parameter<std::int64_t>(c, "change_test", "model_rmse_up_to"), 24);
    EXPECT_EQ(parameter<double>(c, "refit", "factor"), 1.33);
    EXPECT_EQ(parameter<std::int64_t>(c, "change_test", "seasonal_count"), 24);
    EXPECT_EQ(parameter<double>(c, "change_test", "seasonal_divisor"), 4.0);

    // Each change threshold is the chi2(5) quantile at its stage's change
    // probability, 0.99, to the last bit.
    for (const char* stage : {"change_test", "stability_rule", "lookback"})
    {
        const double probability = parameter<double>(c, stage, "change_probability");
        EXPECT_EQ(probability, 0.99) << stage;
        EXPECT_EQ(ccd::chi2_quantile(probability, 5),
                  parameter<double>(c, stage, "change_threshold"))
            << stage;
    }
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
            dump_cube(cube, ccd::HarmonicOptions{}, spelled_out(golden::ccdc()), false),
            dump_cube(cube, ccd::HarmonicOptions{}, golden::ccdc(), false));

        EXPECT_EQ(
            dump_cube(cube, golden::physical_harmonic(), spelled_out(golden::physical_config()), true),
            dump_cube(cube, golden::physical_harmonic(), golden::physical_config(), true));
    }
}

//==============================================================================
// change_test's peek_reference_days: a longer reference interval makes a
// longer peek window, which a short-lived disturbance does not fill.
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
    EXPECT_EQ(ccd::adjust_peek(d16, 6, 8.0), 6u);    // never below peek_size
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

    const ccd::config::ConfigTree longer =
        golden::with(golden::ccdc(), "change_test", "peek_reference_days", ccd::config::real(32.0));

    // Eight disturbed observations fill a peek window of 6, not one of 12.
    const auto transient = make_pixel(300.0, 0.0, {{250, 258, 600.0}});
    EXPECT_EQ(breaks(run(transient, golden::ccdc())), 1);
    EXPECT_EQ(breaks(run(transient, longer)), 0);

    // A lasting step is found by both, on the same day.
    const auto step = make_pixel(300.0, 0.0, {{250, T, 600.0}});
    const auto at6 = run(step, golden::ccdc());
    const auto at12 = run(step, longer);
    ASSERT_EQ(breaks(at6), 1);
    ASSERT_EQ(breaks(at12), 1);
    EXPECT_EQ(first_break(at6), step.dates[250]);
    EXPECT_EQ(first_break(at12), step.dates[250]);
}

//==============================================================================
// noise_floor's min_lag_days: a longer minimum lag measures the variogram
// over pairs further apart, which on a seasonal signal is larger, so a step
// must be larger to count as change.
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

    const ccd::config::ConfigTree shorter =
        golden::with(golden::ccdc(), "noise_floor", "min_lag_days", ccd::config::integer(10));
    const ccd::config::ConfigTree longer =
        golden::with(golden::ccdc(), "noise_floor", "min_lag_days", ccd::config::integer(60));

    const auto small = make_pixel(300.0, 0.0, {{250, T, 150.0}});
    EXPECT_EQ(breaks(run(small, shorter)), 1);
    EXPECT_EQ(breaks(run(small, golden::ccdc())), 0);

    const auto large = make_pixel(300.0, 0.0, {{250, T, 300.0}});
    EXPECT_EQ(breaks(run(large, golden::ccdc())), 1);
    EXPECT_EQ(breaks(run(large, longer)), 0);
}

//==============================================================================
// refit's factor and min_obs: on a pixel with accelerating drift, a model
// refitted less often departs from the data sooner, so the first break comes
// earlier. A larger span factor refits less often; a larger minimum count
// refits more often.
//==============================================================================

TEST(Cadence, RetrainSpanFactor)
{
    const auto drifting = make_pixel(0.0, 0.5, {});

    const ccd::config::ConfigTree often =
        golden::with(golden::ccdc(), "refit", "factor", ccd::config::real(1.2));
    EXPECT_EQ(breaks(run(drifting, often)), 0);

    std::int64_t previous = 0;
    for (const double factor : {1.33, 1.6, 2.0, 4.0})
    {
        const auto r =
            run(drifting, golden::with(golden::ccdc(), "refit", "factor", ccd::config::real(factor)));
        ASSERT_GE(breaks(r), 1) << "factor " << factor;
        if (previous != 0)
            EXPECT_LT(first_break(r), previous) << "factor " << factor;
        previous = first_break(r);
    }
}

// The span rule disabled, so that only refit's min_obs refits: the models
// are refitted while the window holds at most that many observations (and
// change_test's model RMSE is the model's own up to the same count).
TEST(Cadence, RetrainMinObs)
{
    const auto drifting = make_pixel(0.0, 0.5, {});

    std::int64_t previous = 0;
    std::size_t segments = 0;
    for (const std::int64_t count : {24, 48, 96, 200})
    {
        ccd::config::ConfigTree c = golden::ccdc();
        c["refit"].params["factor"] = ccd::config::real(1e6);
        c["refit"].params["min_obs"] = ccd::config::integer(count);
        c["change_test"].params["model_rmse_up_to"] = ccd::config::integer(count);
        const auto r = run(drifting, c);
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
// change_test's seasonal_divisor and seasonal_count: the seasonal RMSE is the
// norm of seasonal_count residuals over seasonal_divisor, and it normalises
// the change magnitude. On a pixel without seasonality, where it outweighs
// the variogram, a larger divisor or a smaller count makes a smaller step
// count as change.
//==============================================================================

TEST(Cadence, SeasonalRmseDivisor)
{
    const ccd::config::ConfigTree larger =
        golden::with(golden::ccdc(), "change_test", "seasonal_divisor", ccd::config::real(8.0));
    const ccd::config::ConfigTree smaller =
        golden::with(golden::ccdc(), "change_test", "seasonal_divisor", ccd::config::real(2.0));

    const auto small = make_pixel(0.0, 0.0, {{250, T, 50.0}});
    EXPECT_EQ(breaks(run(small, golden::ccdc())), 0);
    EXPECT_EQ(breaks(run(small, larger)), 1);

    const auto medium = make_pixel(0.0, 0.0, {{250, T, 60.0}});
    EXPECT_EQ(breaks(run(medium, golden::ccdc())), 1);
    EXPECT_EQ(breaks(run(medium, smaller)), 0);
}

TEST(Cadence, SeasonalRmseCount)
{
    const ccd::config::ConfigTree fewer =
        golden::with(golden::ccdc(), "change_test", "seasonal_count", ccd::config::integer(6));
    const ccd::config::ConfigTree more =
        golden::with(golden::ccdc(), "change_test", "seasonal_count", ccd::config::integer(96));

    const auto small = make_pixel(0.0, 0.0, {{250, T, 50.0}});
    EXPECT_EQ(breaks(run(small, golden::ccdc())), 0);
    EXPECT_EQ(breaks(run(small, fewer)), 1);

    const auto medium = make_pixel(0.0, 0.0, {{250, T, 60.0}});
    EXPECT_EQ(breaks(run(medium, golden::ccdc())), 1);
    EXPECT_EQ(breaks(run(medium, more)), 0);
}

//==============================================================================
// The change probability: a higher probability raises the threshold of a
// peek window longer than the default. Read by adjust_change_threshold(),
// tested there and through run_pixel() on a dense series.
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
    ccd::config::ConfigTree higher = golden::ccdc();
    higher["change_test"].params["change_probability"] = ccd::config::real(0.999);
    higher["stability_rule"].params["change_probability"] = ccd::config::real(0.999);
    higher["lookback"].params["change_probability"] = ccd::config::real(0.999);

    const auto small = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 8);
    const auto at99 = run(small, golden::ccdc());
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
// window, so it gives the same output as a configuration that fixes both:
// change_test's peek_size 12 with a peek_reference_days that leaves it at 12,
// and the change threshold at the adjusted value. Holding the threshold at
// its default instead misses a step the adjusted one finds.
TEST(PeekThreshold, DenseSeriesUsesAdjustedThreshold)
{
    const double threshold = golden::parameter<double>(golden::ccdc(), "change_test", "change_threshold");
    const auto with_threshold = [](ccd::config::ConfigTree c, double value)
    {
        for (const char* stage : {"change_test", "stability_rule", "lookback"})
            c[stage].params["change_threshold"] = ccd::config::real(value);
        return c;
    };

    ccd::config::ConfigTree fixed = golden::ccdc();
    fixed["change_test"].params["peek_size"] = ccd::config::integer(12);
    fixed["change_test"].params["peek_reference_days"] = ccd::config::real(8.0);
    fixed = with_threshold(fixed, ccd::adjust_change_threshold(12, 6, threshold, 5, 0.99));

    const ccd::config::ConfigTree unadjusted = with_threshold(fixed, threshold);

    const auto step = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 8);

    std::string adjusted_dump, fixed_dump;
    golden::dump_pixel(adjusted_dump, "step", run(step, golden::ccdc()));
    golden::dump_pixel(fixed_dump, "step", run(step, fixed));
    EXPECT_EQ(adjusted_dump, fixed_dump);

    const auto adjusted = run(step, golden::ccdc());
    ASSERT_EQ(breaks(adjusted), 1);
    EXPECT_EQ(first_break(adjusted), step.dates[250]);
    EXPECT_EQ(breaks(run(step, unadjusted)), 0);
}

// The adjustment starts from the caller's configuration on every call: a
// sparse pixel run after a dense one gives the output it gets alone, and the
// configuration passed in keeps its values.
TEST(PeekThreshold, EachPixelStartsFromTheCallerConfiguration)
{
    const auto dense = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 8);
    const auto sparse = make_pixel(0.0, 0.0, {{250, T, 58.0}}, 16);

    std::string alone;
    golden::dump_pixel(alone, "sparse", run(sparse, golden::ccdc()));

    const ccd::config::ConfigTree config = golden::ccdc();
    run(dense, config);
    std::string after;
    golden::dump_pixel(after, "sparse", run(sparse, config));
    EXPECT_EQ(alone, after);
    EXPECT_EQ(golden::parameter<std::int64_t>(config, "change_test", "peek_size"), 6);
    EXPECT_EQ(config, golden::ccdc());
}

