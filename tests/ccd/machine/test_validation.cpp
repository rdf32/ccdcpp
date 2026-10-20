//==============================================================================
//
// Input validation at the entry points.
//
// detect() and detect_cube() reject an input ccd::validate() refuses -- a
// band count other than its roles', an unknown, repeated or missing role, a
// saturation row off the band axis -- a configuration the input does not
// fit, inputs whose timestep counts disagree, and dates that decrease, with
// std::invalid_argument before any pixel runs. Inside
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

    return ccd::detect(
        ccd::RunSpec{},
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {date_count}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {B, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {qa_count})).result;
}

// A 2 x 3 cube, (H, W, B, T).
template<typename Value>
ccd::CubeOutput detect_cube(
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

    return ccd::detect_cube(
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
        EXPECT_NE(message.find("the input has 7"), std::string::npos) << message;
    }
}

// `config` refused on `input`, the message holding `where` and `what`.
void expect_config_error(const ccd::config::Config& config, const ccd::Input& input,
                         const std::string& where, const std::string& what)
{
    try
    {
        ccd::strategies::library_registry().resolve(config, input);
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
    const ccd::Input h;
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
ccd::Input ten_band_layout()
{
    ccd::Input h;
    h.roles = {"blue", "green", "red", "red_edge_1", "red_edge_2", "red_edge_3",
               "nir", "other", "swir1", "swir2"};
    return h;
}

void expect_layout_error(
    const ccd::Input& h,
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

    // B != the roles'
    expect_layout_error(ten_band_layout(), 7,
        "spectral has 7 bands; the input has 10 (a row per role)");

    // B > CCD_MAX_BANDS
    {
        auto h = ten_band_layout();
        h.roles.resize(ccd::CCD_MAX_BANDS + 1, "other");
        expect_layout_error(h, ccd::CCD_MAX_BANDS + 1,
            "spectral has 33 bands; ccdcpp accepts at most 32");

        h.roles.resize(ccd::CCD_MAX_BANDS);
        EXPECT_NO_THROW(ccd::validate(h, ccd::CCD_MAX_BANDS));
    }

    // A row >= B, in each of the configuration's lists (against the input,
    // when it is resolved).
    expect_config_error(
        golden::with_shared(golden::pyccd(), "detection_bands", ccd::config::bands({1, 2, 10})),
        ten_band_layout(), "shared value 'detection_bands'", "holds band 10; the input has 10 bands");
    expect_config_error(
        golden::with(golden::pyccd(), "window_screen", "bands", ccd::config::bands({1, 12})),
        ten_band_layout(), "window_screen 'tmask'", "holds band 12; the input has 10 bands");
    expect_config_error(
        golden::with_steps(golden::pyccd(), "band_range", "bands", ccd::config::bands({0, 31})),
        ten_band_layout(), "preprocess standard step 2 'band_range' parameter 'bands'",
        "holds band 31; the input has 10 bands");

    // Duplicates, in each list.
    expect_config_error(
        golden::with_shared(golden::pyccd(), "detection_bands", ccd::config::bands({1, 2, 6, 2})),
        ten_band_layout(), "shared value 'detection_bands'", "holds band 2 twice");
    expect_config_error(
        golden::with(golden::pyccd(), "window_screen", "bands", ccd::config::bands({8, 8})),
        ten_band_layout(), "window_screen 'tmask'", "holds band 8 twice");
    expect_config_error(
        golden::with_steps(golden::pyccd(), "band_range", "bands", ccd::config::bands({0, 1, 0})),
        ten_band_layout(), "preprocess standard step 2 'band_range' parameter 'bands'",
        "holds band 0 twice");

    // An empty list, in each.
    expect_config_error(
        golden::with_shared(golden::pyccd(), "detection_bands", ccd::config::bands({})),
        ten_band_layout(), "shared value 'detection_bands'", "is empty");
    expect_config_error(
        golden::with(golden::pyccd(), "window_screen", "bands", ccd::config::bands({})),
        ten_band_layout(), "window_screen 'tmask'", "is empty");
    for (const ccd::config::Value& empty : {ccd::config::bands({}), ccd::config::roles({})})
        expect_config_error(golden::with_steps(golden::pyccd(), "band_range", "bands", empty),
                            ten_band_layout(), "preprocess standard step 2 'band_range'",
                            "parameter 'bands' is empty");

    // band_range's one selector is "reflective".
    expect_config_error(
        golden::with_steps(golden::pyccd(), "band_range", "bands",
                           ccd::config::Value(std::string("optical"))),
        ten_band_layout(), "preprocess standard step 2 'band_range' parameter 'bands'",
        "is 'optical'; a band list is roles, rows or \"reflective\"");

    // Roles: green required, each known, none but "other" twice.
    {
        auto h = ten_band_layout();
        h.roles[1] = "other";
        expect_layout_error(h, 10, "roles has no 'green'");
    }
    {
        auto h = ten_band_layout();
        h.roles[7] = "nir8a";
        expect_layout_error(h, 10, "roles holds 'nir8a', which is not a role; the roles are blue");
    }
    {
        auto h = ten_band_layout();
        h.roles[7] = "green";
        expect_layout_error(h, 10, "roles holds 'green' more than once");
    }

    // A unit of reflectance is a positive number.
    for (const double unit : {0.0, -1.0, std::nan("")})
    {
        ccd::Input h;
        h.reflectance_scale = unit;
        expect_layout_error(h, 7, "reflectance_scale is");
    }

    // The value of reflectance 0 is a finite number.
    for (const double offset : {std::nan(""), HUGE_VAL, -HUGE_VAL})
    {
        ccd::Input h;
        h.reflectance_offset = offset;
        expect_layout_error(h, 7, "reflectance_offset is");
    }

    // The value of 100 degrees Celsius is a positive number.
    for (const double unit : {0.0, -1.0, std::nan(""), HUGE_VAL})
    {
        ccd::Input h;
        h.thermal_at_100c = unit;
        expect_layout_error(h, 7, "thermal_at_100c is");
    }
}

// The configuration's cadence and change-probability parameters: zero or
// negative days, counts below 1, a probability outside (0, 1), a
// non-positive factor or divisor, a peek below 2.
TEST(Validation, CadenceRules)
{
    using ccd::config::integer;
    using ccd::config::real;
    const double nan = std::nan("");

    const auto expect_error = [](const std::string& stage, const std::string& name,
                                 ccd::config::Value value, const std::string& where,
                                 const std::string& what)
    { expect_config_error(golden::with(golden::pyccd(), stage, name, value), ccd::Input{}, where, what); };

    for (const double days : {0.0, -16.0, nan})
        expect_error("change_test", "peek_reference_days", real(days),
            "change_test 'chi2' parameter 'peek_reference_days'", "it must be greater than 0");

    for (const std::int64_t days : {0, -30})
        expect_error("noise_floor", "min_lag_days", integer(days),
            "noise_floor 'variogram' parameter 'min_lag_days'", "it must be");

    expect_error("refit", "min_obs", integer(0),
        "refit 'span-factor' parameter 'min_obs' is 0", "it must be at least 1");
    expect_error("change_test", "model_rmse_up_to", integer(0),
        "change_test 'chi2' parameter 'model_rmse_up_to' is 0", "it must be at least 1");
    expect_error("change_test", "seasonal_count", integer(0),
        "change_test 'chi2' parameter 'seasonal_count'", "it must be at least 1");

    for (const double factor : {0.0, -1.33, nan})
        expect_error("refit", "factor", real(factor),
            "refit 'span-factor' parameter 'factor'", "it must be greater than 0");
    for (const double divisor : {0.0, -4.0, nan})
        expect_error("change_test", "seasonal_divisor", real(divisor),
            "change_test 'chi2' parameter 'seasonal_divisor'", "it must be greater than 0");

    for (const double p : {0.0, 1.0, -0.5, 1.5, nan})
        for (const char* name : {"change_probability", "outlier_probability"})
            expect_config_error(golden::with_shared(golden::pyccd(), name, real(p)), ccd::Input{},
                std::string("shared parameter '") + name + "'", "it must be in (0, 1)");

    // A peek of 0 or 1 would leave a peek empty.
    expect_error("change_test", "peek_size", integer(0),
        "change_test 'chi2' parameter 'peek_size' is 0", "it must be at least 2");
    expect_error("change_test", "peek_size", integer(1),
        "change_test 'chi2' parameter 'peek_size' is 1", "it must be at least 2");

    // The smallest accepted values.
    ccd::config::Config c = golden::pyccd();
    c.stages["change_test"].params["peek_size"] = integer(2);
    c.stages["change_test"].params["peek_reference_days"] = real(1e-9);
    c.stages["noise_floor"].params["min_lag_days"] = integer(1);
    c.stages["refit"].params["min_obs"] = integer(1);
    c.stages["change_test"].params["model_rmse_up_to"] = integer(1);
    c.stages["refit"].params["factor"] = real(1e-9);
    c.stages["change_test"].params["seasonal_count"] = integer(1);
    c.stages["change_test"].params["seasonal_divisor"] = real(1e-9);
    for (const double p : {1e-9, 1.0 - 1e-9})
    {
        c.shared["change_probability"] = real(p);
        c.shared["outlier_probability"] = real(p);
        EXPECT_NO_THROW(golden::strategies_of(c)) << p;
    }
}

namespace
{

// `config` with every coefficient count set to `count`: the shared
// coefficient rule, which the stability fit, the model and the edge fit read.
ccd::config::Config with_coefficients(ccd::config::Config config, std::int64_t count)
{
    for (const char* name : {"coefficient_min", "coefficient_mid", "coefficient_max"})
        config.shared[name] = ccd::config::integer(count);
    return config;
}

} // namespace

// The model's coefficient rule (CoefficientRule::check()).
TEST(Validation, CoefficientRules)
{
    const auto expect_error = [](const std::string& name, std::int64_t value,
                                 const std::string& what)
    {
        expect_config_error(
            golden::with_shared(golden::pyccd(), name, ccd::config::integer(value)), ccd::Input{},
            "model 'lasso' ", what);
    };

    expect_config_error(
        golden::with_shared(golden::pyccd(), "coefficient_min", ccd::config::integer(0)),
        ccd::Input{}, "shared parameter 'coefficient_min' is 0", "it must be at least 1");
    expect_error("coefficient_min", 7,
        "model 'lasso' parameters 'coefficient_min', 'coefficient_mid', 'coefficient_max' "
        "are 7, 6, 8; they must not decrease");
    expect_error("coefficient_max", 5,
        "model 'lasso' parameters 'coefficient_min', 'coefficient_mid', 'coefficient_max' "
        "are 4, 6, 5; they must not decrease");
    expect_error("coefficient_max", 9,
        "model 'lasso' parameter 'coefficient_max' is 9; the harmonic basis has at most 8");

    // The extremes accepted: a count of 1 throughout (the trend and annual
    // basis, as 4, with 1 degree of freedom), and 8.
    EXPECT_NO_THROW(golden::strategies_of(with_coefficients(golden::pyccd(), 1)));
    EXPECT_NO_THROW(golden::strategies_of(with_coefficients(golden::pyccd(), 8)));
}

// The configuration is checked at the entry points too.
TEST(Validation, DetectRejectsACadenceParameter)
{
    constexpr ccd::index_t T = 40;
    const auto dates = make_dates(T);
    const auto spectral = make_spectral<double>(7, T);
    const std::vector<std::uint8_t> qas(T, 0);

    const ccd::config::Config config =
        golden::with(golden::pyccd(), "change_test", "seasonal_divisor", ccd::config::real(0.0));

    // The Algorithm refuses it as it is built.
    EXPECT_THROW(golden::algorithm(config), std::invalid_argument);

    // A configuration the input does not fit: at the entry points.
    const ccd::config::Config off_input =
        golden::with_shared(golden::pyccd(), "detection_bands", ccd::config::bands({1, 2, 9}));
    const ccd::AlgorithmPtr algorithm = golden::algorithm(off_input);
    EXPECT_THROW(
        ccd::detect(
            golden::spec_of(ccd::Input{}, algorithm),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T})),
        std::invalid_argument);

    EXPECT_THROW(ccd::detect_cube(
        golden::spec_of(ccd::Input{}, algorithm),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(spectral.data(), {1, 1, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {1, 1, T}),
        1),
        std::invalid_argument);
}

// A peek_size of 1 is refused before the run; the smallest accepted value
// runs every transition the characterization pixels take.
TEST(Validation, PeekSizeAtLeastTwoThroughDetect)
{
    for (auto p : golden::characterisation_pixels())
    {
        const std::size_t T = p.dates.size();
        const auto run = [&]()
        {
            return ccd::detect(
                golden::spec_of(p.h, p.config),
                ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
                ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})).result;
        };

        p.config.stages["change_test"].params["peek_size"] = ccd::config::integer(1);
        EXPECT_THROW(run(), std::invalid_argument) << p.name;

        p.config.stages["change_test"].params["peek_size"] = ccd::config::integer(2);
        EXPECT_NO_THROW(run()) << p.name;
    }
}

// The extreme coefficient counts the rule accepts run every transition the
// characterization pixels take.
TEST(Validation, CoefficientExtremesThroughDetect)
{
    for (const std::int64_t count : {1, 8})
    {
        for (auto p : golden::characterisation_pixels())
        {
            const std::size_t T = p.dates.size();
            p.config = with_coefficients(p.config, count);

            ccd::FitResult r;
            EXPECT_NO_THROW(r = ccd::detect(
                golden::spec_of(p.h, p.config),
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

// The layout is checked before any pixel runs, for detect() and for
// detect_cube() of every dtype.
TEST(Validation, EntryPointsRejectALayout)
{
    constexpr ccd::index_t T = 40;
    const auto dates = make_dates(T);
    const auto spectral = make_spectral<double>(7, T);
    const std::vector<std::uint8_t> qas(T, 0);

    const ccd::Input h;
    const ccd::AlgorithmPtr off = golden::algorithm(
        golden::with_shared(golden::pyccd(), "detection_bands", ccd::config::bands({1, 2, 7})));

    EXPECT_THROW(
        ccd::detect(
            golden::spec_of(h, off),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T})),
        std::invalid_argument);

    const std::vector<std::uint8_t> cube_qas(T, 0);
    const std::vector<float> f32(spectral.begin(), spectral.end());
    const std::vector<std::uint16_t> u16(spectral.begin(), spectral.end());
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T});
    const auto q = ccd::ArrayView<const std::uint8_t, 3>::contiguous(cube_qas.data(), {1, 1, T});

    EXPECT_THROW(ccd::detect_cube(
        golden::spec_of(h, off),
        d,
        ccd::ArrayView<const double, 4>::contiguous(spectral.data(), {1, 1, 7, T}),
        q,
        1), std::invalid_argument);
    EXPECT_THROW(ccd::detect_cube(
        golden::spec_of(h, off),
        d,
        ccd::ArrayView<const float, 4>::contiguous(f32.data(), {1, 1, 7, T}),
        q,
        1), std::invalid_argument);
    EXPECT_THROW(ccd::detect_cube(
        golden::spec_of(h, off),
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
    expect_band_error([] { detect_cube<double>(6, 40, 40); }, 6);
    expect_band_error([] { detect_cube<double>(8, 40, 40); }, 8);
    expect_band_error([] { detect_cube<float>(6, 40, 40); }, 6);
    expect_band_error([] { detect_cube<float>(8, 40, 40); }, 8);
    expect_band_error([] { detect_cube<std::uint16_t>(6, 40, 40); }, 6);
    expect_band_error([] { detect_cube<std::uint16_t>(8, 40, 40); }, 8);

    const auto ok = detect_cube<double>(7, 40, 40);
    EXPECT_EQ(ok.pixels.size(), 6u);
    EXPECT_TRUE(ok.failures.empty());
}

TEST(Validation, DetectCubeRejectsDateCountMismatch)
{
    EXPECT_THROW(detect_cube<double>(7, 40, 39), std::invalid_argument);
    EXPECT_THROW(detect_cube<double>(7, 40, 41), std::invalid_argument);
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

            const auto result = ccd::detect(
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
ccd::CubeOutput detect_cube_dates(const std::vector<std::int64_t>& dates)
{
    constexpr ccd::index_t H = 2;
    constexpr ccd::index_t W = 3;
    const auto T = static_cast<ccd::index_t>(dates.size());

    const auto pixel = make_spectral<Value>(7, T);
    std::vector<Value> spectral;
    for (ccd::index_t p = 0; p < H * W; ++p)
        spectral.insert(spectral.end(), pixel.begin(), pixel.end());

    const std::vector<std::uint8_t> qas(H * W * T, 0);

    return ccd::detect_cube(
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
    ccd::RunSpec spec;
    spec.masks = true;

    return ccd::detect(
        spec,
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
    EXPECT_THROW(ccd::detect(
        [] { ccd::RunSpec r; r.encoding.qa_table = std::make_shared<const ccd::QaTable>(ccd::QaTable::Codes{}); return r; }(),
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
    EXPECT_THROW(detect_cube_dates<double>(swapped(40, 5)), std::invalid_argument);
    EXPECT_THROW(detect_cube_dates<float>(swapped(40, 5)), std::invalid_argument);
    EXPECT_THROW(detect_cube_dates<std::uint16_t>(swapped(40, 5)), std::invalid_argument);

    auto equal = make_dates(40);
    equal[21] = equal[20];
    const auto ok = detect_cube_dates<double>(equal);
    EXPECT_EQ(ok.pixels.size(), 6u);
    EXPECT_TRUE(ok.failures.empty());
}
