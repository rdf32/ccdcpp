//==============================================================================
//
// Input validation at the entry points.
//
// detect() and detect_cube() reject a spectral array whose band count is not
// CCD_NUM_BANDS, and inputs whose timestep counts disagree, with
// std::invalid_argument before any pixel runs. Inside detect_cube a wrong
// band count would otherwise be out-of-bounds access within the OpenMP
// region, where nothing reports it.
//
//==============================================================================

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

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
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {date_count}),
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {B, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {qa_count}),
        ccd::HarmonicOptions(),
        ccd::LassoOptions()
    );
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

    return ccd::detect_cube(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {date_count}),
        ccd::ArrayView<const Value, 4>::contiguous(spectral.data(), {H, W, B, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {H, W, T}),
        ccd::HarmonicOptions(),
        ccd::LassoOptions(),
        2
    );
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

} // namespace

TEST(Validation, ValidateBandsAcceptsSevenOnly)
{
    EXPECT_NO_THROW(ccd::validate_bands(ccd::CCD_NUM_BANDS));
    expect_band_error([] { ccd::validate_bands(6); }, 6);
    expect_band_error([] { ccd::validate_bands(8); }, 8);
    expect_band_error([] { ccd::validate_bands(0); }, 0);
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
