//==============================================================================
//
// Input validation at the entry points.
//
// run_pixel() and run_cube() reject a band layout ccd::validate() refuses --
// a band count other than HarmonicOptions::NUM_BANDS, an index off the band
// axis, a repeated or missing band -- inputs whose timestep counts
// disagree, and dates that decrease, with std::invalid_argument before any
// pixel runs. Inside
// detect_cube a bad layout would otherwise be out-of-bounds access within
// the OpenMP region, where nothing reports it.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../golden/specs.hpp"

#include "ccd/ccd.hpp"

#include "../golden/characterisation.hpp"

namespace
{

constexpr std::int64_t START = 724276; // 1984-01-01

std::vector<std::int64_t> make_dates(ccd::index_t T)
{
    std::vector<std::int64_t> dates(T);
    for (ccd::index_t t = 0; t < T; ++t)
        dates[t] = START + 16 * static_cast<std::int64_t>(t);
    return dates;
}

// (B, T) of in-range DN: reflectance 1000 + 100 * band, thermal 2900.
template<typename Value>
std::vector<Value> make_spectral(ccd::index_t B, ccd::index_t T)
{
    std::vector<Value> spectral(B * T);
    for (ccd::index_t b = 0; b < B; ++b)
        for (ccd::index_t t = 0; t < T; ++t)
            spectral[b * T + t] = static_cast<Value>(
                b == 6 ? 2900 : 1000 + 100 * b);
    return spectral;
}

ccd::FitResult run_detect(
    ccd::index_t B,
    ccd::index_t T,
    ccd::index_t date_count,
    ccd::index_t qa_count
)
{
    const auto dates = make_dates(date_count);
    const auto spectral = make_spectral<double>(B, T);
    const std::vector<std::uint8_t> qas(qa_count, 0);

    return ccd::run_pixel(
        ccd::RunSpec{},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {date_count}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {B, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {qa_count})).result;
}

// A 2 x 3 cube, (H, W, B, T).
template<typename Value>
ccd::CubeResult run_cube(
    ccd::index_t B,
    ccd::index_t T,
    ccd::index_t date_count
)
{
    constexpr ccd::index_t H = 2;
    constexpr ccd::index_t W = 3;

    const auto dates = make_dates(date_count);
    const auto pixel = make_spectral<Value>(B, T);

    std::vector<Value> spectral;
    for (ccd::index_t p = 0; p < H * W; ++p)
        spectral.insert(spectral.end(), pixel.begin(), pixel.end());

    const std::vector<std::uint8_t> qas(H * W * T, 0);

    return ccd::run_cube(
        ccd::RunSpec{},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {date_count}),
        ccd::ArrayView<const Value, 4>::contiguous(spectral.data(), {H, W, B, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {H, W, T}),
        2);
}

// The message names both counts.
template<typename F>
void expect_band_error(F&& f, ccd::index_t bands)
{
    try
    {
        f();
        FAIL() << "no exception for " << bands << " bands";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find("has " + std::to_string(bands) + " bands"),
                  std::string::npos) << message;
        EXPECT_NE(message.find("expects 7"), std::string::npos) << message;
    }
}

// A configuration of `a` refused for an input of `bands` bands, the message
// holding `where` and `what`.
void expect_config_error(const golden::Algorithm& a, ccd::index_t bands,
                         const std::string& where, const std::string& what)
{
    try
    {
        const ccd::config::ConfigTree tree = golden::config_of(a);
        ccd::strategies::library_registry().check_bands(tree, bands);
        ccd::strategies::build_strategies(ccd::strategies::library_registry(), tree);
        FAIL() << "no exception for " << where;
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find(where), std::string::npos) << message;
        EXPECT_NE(message.find(what), std::string::npos) << message;
    }
}

} // namespace

TEST(Validation, DefaultLayoutAcceptsSevenOnly)
{
    const ccd::HarmonicOptions h;
    EXPECT_NO_THROW(ccd::validate(h, 7));
    expect_band_error([&] { ccd::validate(h, 6); }, 6);
    expect_band_error([&] { ccd::validate(h, 8); }, 8);
    expect_band_error([&] { ccd::validate(h, 0); }, 0);
}

//------------------------------------------------------------------------------
// One case per rule of ccd::validate(). Each starts from a layout that
// passes and breaks one thing; the message names what broke.
//------------------------------------------------------------------------------

namespace
{

// A 10-band layout without thermal, which passes.
ccd::HarmonicOptions ten_band_layout()
{
    ccd::HarmonicOptions h;
    h.NUM_BANDS = 10;
    h.DETECTION_BANDS = {1, 2, 6, 8, 9};
    h.SATURATION_BANDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    h.GREEN_IDX = 1;
    h.THERMAL_IDX = ccd::NO_BAND;
    return h;
}

void expect_layout_error(
    const ccd::HarmonicOptions& h,
    ccd::index_t bands,
    const std::string& expected
)
{
    try
    {
        ccd::validate(h, bands);
        FAIL() << "no exception; expected \"" << expected << "\"";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        EXPECT_NE(message.find(expected), std::string::npos) << message;
    }
}

} // namespace

TEST(Validation, LayoutRules)
{
    EXPECT_NO_THROW(ccd::validate(ten_band_layout(), 10));

    // B != NUM_BANDS
    expect_layout_error(ten_band_layout(), 7,
        "spectral has 7 bands; HarmonicOptions.NUM_BANDS expects 10");

    // B > CCD_MAX_BANDS
    {
        auto h = ten_band_layout();
        h.NUM_BANDS = ccd::CCD_MAX_BANDS + 1;
        expect_layout_error(h, ccd::CCD_MAX_BANDS + 1,
            "spectral has 33 bands; ccdcpp accepts at most 32");

        h.NUM_BANDS = ccd::CCD_MAX_BANDS;
        EXPECT_NO_THROW(ccd::validate(h, ccd::CCD_MAX_BANDS));
    }

    // An index >= B, in each list.
    {
        auto h = ten_band_layout();
        h.DETECTION_BANDS = {1, 2, 10};
        expect_layout_error(h, 10, "DETECTION_BANDS holds band 10; spectral has 10 bands");
    }
    {
        golden::Algorithm a;
        a.TMASK_BANDS = {1, 12};
        expect_config_error(a, 10, "window_screen 'tmask'", "holds band 12; the input has 10 bands");
    }
    {
        auto h = ten_band_layout();
        h.SATURATION_BANDS = {0, 31};
        expect_layout_error(h, 10, "SATURATION_BANDS holds band 31; spectral has 10 bands");
    }

    // Duplicates, in each list.
    {
        auto h = ten_band_layout();
        h.DETECTION_BANDS = {1, 2, 6, 2};
        expect_layout_error(h, 10, "DETECTION_BANDS holds band 2 more than once");
    }
    {
        golden::Algorithm a;
        a.TMASK_BANDS = {8, 8};
        expect_config_error(a, 10, "window_screen 'tmask'", "holds band 8 twice");
    }
    {
        auto h = ten_band_layout();
        h.SATURATION_BANDS = {0, 1, 0};
        expect_layout_error(h, 10, "SATURATION_BANDS holds band 0 more than once");
    }

    // Empty detection or tmask list; an empty saturation list is allowed.
    {
        auto h = ten_band_layout();
        h.DETECTION_BANDS = {};
        expect_layout_error(h, 10, "DETECTION_BANDS is empty");
    }
    {
        golden::Algorithm a;
        a.TMASK_BANDS = {};
        expect_config_error(a, 10, "window_screen 'tmask'", "is empty");
    }
    {
        auto h = ten_band_layout();
        h.SATURATION_BANDS = {};
        EXPECT_NO_THROW(ccd::validate(h, 10));
    }

    // GREEN_IDX absent or out of range.
    {
        auto h = ten_band_layout();
        h.GREEN_IDX = ccd::NO_BAND;
        expect_layout_error(h, 10, "GREEN_IDX is NO_BAND");
    }
    {
        auto h = ten_band_layout();
        h.GREEN_IDX = 10;
        expect_layout_error(h, 10, "GREEN_IDX is 10; spectral has 10 bands");
    }

    // THERMAL_IDX out of range, or in SATURATION_BANDS.
    {
        auto h = ten_band_layout();
        h.THERMAL_IDX = 10;
        expect_layout_error(h, 10, "THERMAL_IDX is 10; spectral has 10 bands");
    }
    {
        auto h = ten_band_layout();
        h.THERMAL_IDX = 9;
        expect_layout_error(h, 10, "SATURATION_BANDS holds THERMAL_IDX (9)");

        h.SATURATION_BANDS = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        EXPECT_NO_THROW(ccd::validate(h, 10));
    }

    // The default layout's thermal band is outside its saturation list.
    {
        ccd::HarmonicOptions h;
        h.SATURATION_BANDS = {0, 1, 2, 3, 4, 5, 6};
        expect_layout_error(h, 7, "SATURATION_BANDS holds THERMAL_IDX (6)");
    }
}

// The configuration's cadence and change-probability parameters, as 0.10's
// options set them: zero or negative days, counts below 1, a probability
// outside (0, 1), a non-positive factor or divisor, a peek below 2.
TEST(Validation, CadenceRules)
{
    const double nan = std::nan("");

    const auto expect_error = [](auto set, const std::string& where, const std::string& what)
    {
        golden::Algorithm a;
        set(a);
        expect_config_error(a, 7, where, what);
    };

    for (const double days : {0.0, -16.0, nan})
        expect_error([&](auto& a) { a.PEEK_REFERENCE_DAYS = days; },
            "change_test 'chi2' parameter 'peek_reference_days'", "it must be greater than 0");

    for (const std::int64_t days : {0, -30})
        expect_error([&](auto& a) { a.VARIOGRAM_MIN_LAG_DAYS = days; },
            "noise_floor 'variogram' parameter 'min_lag_days'", "it must be");

    // RETRAIN_MIN_OBS is two parameters; the first stage refused names its own.
    expect_error([](auto& a) { a.RETRAIN_MIN_OBS = 0; },
        "change_test 'chi2' parameter 'model_rmse_up_to' is 0", "it must be at least 1");
    expect_error([](auto& a) { a.SEASONAL_RMSE_COUNT = 0; },
        "change_test 'chi2' parameter 'seasonal_count'", "it must be at least 1");

    for (const double factor : {0.0, -1.33, nan})
        expect_error([&](auto& a) { a.RETRAIN_SPAN_FACTOR = factor; },
            "refit 'span-factor' parameter 'factor'", "it must be greater than 0");
    for (const double divisor : {0.0, -4.0, nan})
        expect_error([&](auto& a) { a.SEASONAL_RMSE_DIVISOR = divisor; },
            "change_test 'chi2' parameter 'seasonal_divisor'", "it must be greater than 0");

    for (const double p : {0.0, 1.0, -0.5, 1.5, nan})
        expect_error([&](auto& a) { a.CHANGE_PROBABILITY = p; },
            "parameter 'change_probability'", "");

    // A peek of 0 or 1 would leave a peek empty.
    expect_error([](auto& a) { a.PEEK_SIZE = 0; },
        "change_test 'chi2' parameter 'peek_size' is 0", "it must be at least 2");
    expect_error([](auto& a) { a.PEEK_SIZE = 1; },
        "change_test 'chi2' parameter 'peek_size' is 1", "it must be at least 2");

    // The smallest accepted values.
    golden::Algorithm a;
    a.PEEK_SIZE = 2;
    a.PEEK_REFERENCE_DAYS = 1e-9;
    a.VARIOGRAM_MIN_LAG_DAYS = 1;
    a.RETRAIN_MIN_OBS = 1;
    a.RETRAIN_SPAN_FACTOR = 1e-9;
    a.SEASONAL_RMSE_COUNT = 1;
    a.SEASONAL_RMSE_DIVISOR = 1e-9;
    a.CHANGE_PROBABILITY = 1e-9;
    EXPECT_NO_THROW(golden::strategies_of(a));
    a.CHANGE_PROBABILITY = 1.0 - 1e-9;
    EXPECT_NO_THROW(golden::strategies_of(a));
}

// The model's coefficient rule (CoefficientRule::check()).
TEST(Validation, CoefficientRules)
{
    const auto expect_error = [](auto set, const std::string& what)
    {
        golden::Algorithm a;
        set(a);
        expect_config_error(a, 7, "model 'lasso' ", what);
    };

    // COEFFICIENT_MIN is also the edge fit's count, refused first.
    {
        golden::Algorithm a;
        a.COEFFICIENT_MIN = 0;
        expect_config_error(a, 7, "edge_fit 'lasso' parameter 'coefficients' is 0",
                            "it must be at least 1");
    }
    expect_error([](auto& a) { a.COEFFICIENT_MIN = 7; },
        "model 'lasso' parameters 'coefficient_min', 'coefficient_mid', 'coefficient_max' "
        "are 7, 6, 8; they must not decrease");
    expect_error([](auto& a) { a.COEFFICIENT_MAX = 5; },
        "model 'lasso' parameters 'coefficient_min', 'coefficient_mid', 'coefficient_max' "
        "are 4, 6, 5; they must not decrease");
    expect_error([](auto& a) { a.COEFFICIENT_MAX = 9; },
        "model 'lasso' parameter 'coefficient_max' is 9; the harmonic basis has at most 8");

    // The extremes accepted: a count of 1 throughout (the trend and annual
    // basis, as 4, with 1 degree of freedom), and 8.
    golden::Algorithm a;
    a.COEFFICIENT_MIN = a.COEFFICIENT_MID = a.COEFFICIENT_MAX = 1;
    EXPECT_NO_THROW(golden::strategies_of(a));
    a.COEFFICIENT_MIN = a.COEFFICIENT_MID = a.COEFFICIENT_MAX = 8;
    EXPECT_NO_THROW(golden::strategies_of(a));
}

// The configuration is checked at the entry points too.
TEST(Validation, DetectRejectsACadenceOption)
{
    constexpr ccd::index_t T = 40;
    const auto dates = make_dates(T);
    const auto spectral = make_spectral<double>(7, T);
    const std::vector<std::uint8_t> qas(T, 0);

    golden::Algorithm a;
    a.SEASONAL_RMSE_DIVISOR = 0.0;

    EXPECT_THROW(
        ccd::run_pixel(
            golden::spec_of(ccd::HarmonicOptions{}, a),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T})),
        std::invalid_argument);

    EXPECT_THROW(ccd::run_cube(
        golden::spec_of(ccd::HarmonicOptions{}, a),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(spectral.data(), {1, 1, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {1, 1, T}),
        1),
        std::invalid_argument);
}

// PEEK_SIZE 1 crashed run_pixel() before the check; the smallest accepted
// value runs every transition the characterization pixels take.
TEST(Validation, PeekSizeAtLeastTwoThroughDetect)
{
    for (auto p : golden::characterisation_pixels())
    {
        const std::size_t T = p.dates.size();
        const auto run = [&]()
        {
            return ccd::run_pixel(
                golden::spec_of(p.h, p.a),
                ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
                ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})).result;
        };

        p.a.PEEK_SIZE = 1;
        EXPECT_THROW(run(), std::invalid_argument) << p.name;

        p.a.PEEK_SIZE = 2;
        EXPECT_NO_THROW(run()) << p.name;
    }
}

// The extreme coefficient counts the rule accepts run every transition the
// characterization pixels take.
TEST(Validation, CoefficientExtremesThroughDetect)
{
    for (const ccd::index_t count : {ccd::index_t{1}, ccd::index_t{8}})
    {
        for (auto p : golden::characterisation_pixels())
        {
            const std::size_t T = p.dates.size();
            p.a.COEFFICIENT_MIN = p.a.COEFFICIENT_MID = p.a.COEFFICIENT_MAX = count;
            ccd::FitResult r;
            EXPECT_NO_THROW(r = ccd::run_pixel(
                golden::spec_of(p.h, p.a),
                ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
                ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})).result) << p.name << " " << count;
            for (const auto& m : r.models)
                for (const auto& b : m.bands)
                    EXPECT_TRUE(std::isfinite(b.rmse)) << p.name << " " << count;
        }
    }
}

TEST(Validation, BandListRejectsWhatItCannotHold)
{
    ccd::BandList list;
    EXPECT_THROW(list.push_back(ccd::CCD_MAX_BANDS), std::invalid_argument);
    EXPECT_THROW(list.push_back(ccd::NO_BAND), std::invalid_argument);

    for (ccd::index_t i = 0; i < ccd::CCD_MAX_BANDS; ++i)
        list.push_back(i % 4);
    EXPECT_EQ(list.size(), ccd::CCD_MAX_BANDS);
    EXPECT_THROW(list.push_back(0), std::invalid_argument);

    // Stored order is kept.
    const ccd::BandList ordered = {5, 1, 3};
    EXPECT_EQ(ordered.to_vector(), (std::vector<ccd::index_t>{5, 1, 3}));
}

// The layout is checked before any pixel runs, for run_pixel() and for
// run_cube() of every dtype.
TEST(Validation, EntryPointsRejectALayout)
{
    constexpr ccd::index_t T = 40;
    const auto dates = make_dates(T);
    const auto spectral = make_spectral<double>(7, T);
    const std::vector<std::uint8_t> qas(T, 0);

    ccd::HarmonicOptions h;
    h.DETECTION_BANDS = {1, 2, 7};

    EXPECT_THROW(
        ccd::run_pixel(
            golden::spec_of(h),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T})),
        std::invalid_argument);

    const std::vector<std::uint8_t> cube_qas(T, 0);
    const std::vector<float> f32(spectral.begin(), spectral.end());
    const std::vector<std::uint16_t> u16(spectral.begin(), spectral.end());
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T});
    const auto q = ccd::ArrayView<const std::uint8_t, 3>::contiguous(cube_qas.data(), {1, 1, T});

    EXPECT_THROW(ccd::run_cube(
        golden::spec_of(h),
        d,
        ccd::ArrayView<const double, 4>::contiguous(spectral.data(), {1, 1, 7, T}),
        q,
        1), std::invalid_argument);
    EXPECT_THROW(ccd::run_cube(
        golden::spec_of(h),
        d,
        ccd::ArrayView<const float, 4>::contiguous(f32.data(), {1, 1, 7, T}),
        q,
        1), std::invalid_argument);
    EXPECT_THROW(ccd::run_cube(
        golden::spec_of(h),
        d,
        ccd::ArrayView<const std::uint16_t, 4>::contiguous(u16.data(), {1, 1, 7, T}),
        q,
        1), std::invalid_argument);
}

TEST(Validation, DetectRejectsBandCount)
{
    expect_band_error([] { run_detect(6, 40, 40, 40); }, 6);
    expect_band_error([] { run_detect(8, 40, 40, 40); }, 8);
    EXPECT_NO_THROW(run_detect(7, 40, 40, 40));
}

TEST(Validation, DetectRejectsTimestepMismatch)
{
    EXPECT_THROW(run_detect(7, 40, 39, 40), std::invalid_argument);
    EXPECT_THROW(run_detect(7, 40, 41, 40), std::invalid_argument);
    EXPECT_THROW(run_detect(7, 40, 40, 39), std::invalid_argument);
    EXPECT_THROW(run_detect(7, 40, 40, 41), std::invalid_argument);
}

TEST(Validation, DetectCubeRejectsBandCountForEveryDtype)
{
    expect_band_error([] { run_cube<double>(6, 40, 40); }, 6);
    expect_band_error([] { run_cube<double>(8, 40, 40); }, 8);
    expect_band_error([] { run_cube<float>(6, 40, 40); }, 6);
    expect_band_error([] { run_cube<float>(8, 40, 40); }, 8);
    expect_band_error([] { run_cube<std::uint16_t>(6, 40, 40); }, 6);
    expect_band_error([] { run_cube<std::uint16_t>(8, 40, 40); }, 8);

    const auto ok = run_cube<double>(7, 40, 40);
    EXPECT_EQ(ok.pixels.size(), 6u);
    EXPECT_TRUE(ok.failures.empty());
}

TEST(Validation, DetectCubeRejectsDateCountMismatch)
{
    EXPECT_THROW(run_cube<double>(7, 40, 39), std::invalid_argument);
    EXPECT_THROW(run_cube<double>(7, 40, 41), std::invalid_argument);
}

//------------------------------------------------------------------------------
// A series no longer than the band count is valid input with too few
// observations to fit, in every procedure: it returns no models.
//------------------------------------------------------------------------------

TEST(Validation, SeriesNoLongerThanBandCountRuns)
{
    for (ccd::index_t T : {1, 3, 6, 7})
    {
        const auto dates = make_dates(T);
        const auto spectral = make_spectral<double>(7, T);

        // clear -> Standard, snow -> PermanentSnow, cloud -> InsufficientClear
        for (const std::uint8_t qa : {0, 3, 4})
        {
            const std::vector<std::uint8_t> qas(T, qa);

            const auto result = ccd::run_pixel(
                ccd::RunSpec{},
                ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
                ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T})).result;

            EXPECT_TRUE(result.models.empty()) << "T=" << T << " qa=" << int(qa);
        }
    }
}

//------------------------------------------------------------------------------
// Dates must be non-decreasing; equal dates are allowed and deduplicated.
//------------------------------------------------------------------------------

namespace
{

template<typename Value>
ccd::CubeResult run_cube_dates(const std::vector<std::int64_t>& dates)
{
    constexpr ccd::index_t H = 2;
    constexpr ccd::index_t W = 3;
    const auto T = static_cast<ccd::index_t>(dates.size());

    const auto pixel = make_spectral<Value>(7, T);
    std::vector<Value> spectral;
    for (ccd::index_t p = 0; p < H * W; ++p)
        spectral.insert(spectral.end(), pixel.begin(), pixel.end());

    const std::vector<std::uint8_t> qas(H * W * T, 0);

    return ccd::run_cube(
        ccd::RunSpec{},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const Value, 4>::contiguous(spectral.data(), {H, W, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {H, W, T}),
        2);
}

ccd::FitResult run_detect_dates(const std::vector<std::int64_t>& dates)
{
    const auto T = static_cast<ccd::index_t>(dates.size());
    const auto spectral = make_spectral<double>(7, T);
    const std::vector<std::uint8_t> qas(T, 0);

    return ccd::run_pixel(
        ccd::RunSpec{},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T})).result;
}

std::vector<std::int64_t> swapped(ccd::index_t T, ccd::index_t at)
{
    auto dates = make_dates(T);
    std::swap(dates[at], dates[at + 1]);
    return dates;
}

} // namespace

TEST(Validation, ValidateDates)
{
    const auto check = [](const std::vector<std::int64_t>& dates) {
        ccd::validate_dates(ccd::ArrayView<const std::int64_t, 1>::contiguous(
            dates.data(), {static_cast<ccd::index_t>(dates.size())}));
    };

    EXPECT_NO_THROW(check({}));
    EXPECT_NO_THROW(check({5}));
    EXPECT_NO_THROW(check({1, 2, 3}));
    EXPECT_NO_THROW(check({1, 1, 2, 2, 2, 3}));
    EXPECT_THROW(check({2, 1}), std::invalid_argument);
    EXPECT_THROW(check({1, 2, 3, 3, 2}), std::invalid_argument);

    try
    {
        check({10, 20, 30, 25, 40});
        FAIL() << "no exception";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_STREQ(e.what(),
            "dates must be non-decreasing; dates[3] = 25 follows dates[2] = 30");
    }

    // A strided view is read by its strides.
    const std::vector<std::int64_t> interleaved = {1, 99, 2, 0, 3, 50};
    EXPECT_NO_THROW(ccd::validate_dates(ccd::ArrayView<const std::int64_t, 1>::strided(
        interleaved.data(), {3}, {2})));
    EXPECT_THROW(ccd::validate_dates(ccd::ArrayView<const std::int64_t, 1>::strided(
        interleaved.data() + 1, {3}, {2})), std::invalid_argument);
}

TEST(Validation, DetectRejectsUnsortedDates)
{
    EXPECT_THROW(run_detect_dates(swapped(40, 0)), std::invalid_argument);
    EXPECT_THROW(run_detect_dates(swapped(40, 20)), std::invalid_argument);
    EXPECT_THROW(run_detect_dates(swapped(40, 38)), std::invalid_argument);

    // uint16 raw QA with a table.
    const auto dates = swapped(40, 10);
    const auto spectral = make_spectral<double>(7, 40);
    const std::vector<std::uint16_t> raw(40, 0);
    EXPECT_THROW(ccd::run_pixel(
        [] { ccd::RunSpec r; r.qa_table = std::make_shared<ccd::QaTable>(); return r; }(),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {40}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, 40}),
        ccd::ArrayView<const std::uint16_t, 1>::contiguous(raw.data(), {40})), std::invalid_argument);
}

TEST(Validation, DetectAcceptsEqualDates)
{
    auto dates = make_dates(40);
    dates[11] = dates[10];
    dates[12] = dates[10];
    const auto result = run_detect_dates(dates);
    ASSERT_EQ(result.mask.size(), 40u);
    EXPECT_TRUE(result.mask[10]);
    EXPECT_FALSE(result.mask[11]);
    EXPECT_FALSE(result.mask[12]);
}

TEST(Validation, DetectCubeRejectsUnsortedDatesForEveryDtype)
{
    EXPECT_THROW(run_cube_dates<double>(swapped(40, 5)), std::invalid_argument);
    EXPECT_THROW(run_cube_dates<float>(swapped(40, 5)), std::invalid_argument);
    EXPECT_THROW(run_cube_dates<std::uint16_t>(swapped(40, 5)), std::invalid_argument);

    auto equal = make_dates(40);
    equal[21] = equal[20];
    const auto ok = run_cube_dates<double>(equal);
    EXPECT_EQ(ok.pixels.size(), 6u);
    EXPECT_TRUE(ok.failures.empty());
}
