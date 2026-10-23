//==============================================================================
//
// The worked example of docs/writing-a-strategy.md: the outlier rule
// "residual-cap" (tests/ccd/extension/outlier/residual_cap.*), registered in
// a registry beside the library's strategies, as an extension's registration
// function registers it:
//
//   * the registry lists it with its parameter at its physical default, and
//     a configuration naming it resolves `limit` in the spec's units;
//   * judge() drops a first peek observation whose residual in a detection
//     band is above the limit, and keeps it otherwise;
//   * a run with it drops more observations at a smaller limit, and
//     detect_cube() gives detect()'s dump of every pixel, on 1 and 4 threads;
//   * a limit outside its bounds is refused, naming the stage, strategy and
//     parameter;
//   * every block docs/writing-a-strategy.md quotes from a file (the
//     "<!-- from: path -->" line before it) is in that file, word for word.
//
//==============================================================================

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "config.hpp"
#include "dump.hpp"
#include "outlier/residual_cap.hpp"
#include "synthetic.hpp"
#include "units.hpp"

namespace
{

namespace cfg = ccd::config;

// The library's strategies and the example's.
const cfg::Registry& registry()
{
    static const cfg::Registry r = []
    {
        cfg::Registry r;
        ccd::strategies::register_core_strategies(r);
        ccd::strategies::register_outlier_residual_cap(r);
        return r;
    }();
    return r;
}

// pyccd's configuration with the outlier rule "residual-cap" at `limit`.
cfg::Config capped(double limit)
{
    cfg::Config config = golden::pyccd();
    config.stages["outlier"] = {"residual-cap", {{"limit", cfg::real(limit)}}};
    return config;
}

} // namespace

TEST(ResidualCap, IsRegisteredWithItsPhysicalDefault)
{
    const auto names = registry().names().at("outlier");
    EXPECT_NE(std::find(names.begin(), names.end(), "residual-cap"), names.end());
    EXPECT_EQ(registry().defaults("outlier", "residual-cap"),
              (cfg::Values{{"limit", cfg::real(0.2)}}));

    // 0.2 reflectance is 2000 on Collection-1 DN and 0.2 on reflectance.
    const auto limit = [](const ccd::DataSpec& spec)
    {
        const cfg::Resolved r = cfg::resolve(registry(), capped(0.2), spec);
        return golden::parameter<double>(r.stages, "outlier", "limit");
    };
    EXPECT_EQ(limit(ccd::DataSpec{}), 2000.0);
    EXPECT_EQ(limit(golden::physical_spec()), 0.2);
}

TEST(ResidualCap, DropsAResidualAboveTheLimit)
{
    const auto rule =
        registry().make<ccd::OutlierRule>({"residual-cap", {{"limit", cfg::real(2000.0)}}});
    const auto model = registry().make<ccd::MonitoringModel>({"lasso", {}})->make_model();

    const std::int64_t day = 730000;
    const double value = 0;
    const ccd::BandNoise noise;
    const ccd::BandList bands = {1, 3};
    std::vector<std::vector<double>> residuals(7, std::vector<double>{0.0});
    const ccd::Peek peek{*model, ccd::ArrayView<const std::int64_t, 1>::contiguous(&day, {1}),
                         ccd::ArrayView<const double, 2>::contiguous(&value, {1, 1}),
                         {}, {}, 0, {}, residuals, noise, bands};
    const ccd::PeekResult result;

    residuals[3][0] = 1999.0;
    EXPECT_EQ(rule->judge(peek, result), ccd::OutlierOutcome::Keep);
    residuals[3][0] = 2001.0;
    EXPECT_EQ(rule->judge(peek, result), ccd::OutlierOutcome::Drop);
    // A band it does not test.
    residuals[3][0] = 0.0;
    residuals[2][0] = 1e9;
    EXPECT_EQ(rule->judge(peek, result), ccd::OutlierOutcome::Keep);
}

TEST(ResidualCap, ASmallerLimitDropsMore)
{
    const golden::Cube c = golden::make_cube(7, golden::Regime::Cadence16, 1, 1);
    const ccd::index_t T = c.T;
    const auto kept = [&](double limit)
    {
        const ccd::PixelOutput out = ccd::detect(
            golden::spec_of(ccd::DataSpec{},
                            std::make_shared<const ccd::Algorithm>(capped(limit), registry())),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(c.spectral.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data(), {T}));
        EXPECT_FALSE(out.result.models.empty());
        return out.result.mask.count();
    };
    EXPECT_LT(kept(0.02), kept(1.0));
}

TEST(ResidualCap, ACubeIsItsPixels)
{
    const std::size_t H = 2, W = 2;
    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, H, W);
    const ccd::index_t T = c.T;
    const ccd::RunSpec spec = golden::spec_of(
        ccd::DataSpec{}, std::make_shared<const ccd::Algorithm>(capped(0.05), registry()));
    const auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T});

    for (const int threads : {1, 4})
    {
        const ccd::CubeOutput cube = ccd::detect_cube(
            spec, dates,
            ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {H, W, 7, T}),
            ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {H, W, T}), threads);
        ASSERT_TRUE(cube.failures.empty());
        for (const ccd::CubePixel& px : cube.pixels)
        {
            const std::size_t at = px.row * W + px.col;
            const ccd::PixelOutput pixel = ccd::detect(
                spec, dates,
                ccd::ArrayView<const double, 2>::contiguous(c.spectral.data() + at * 7 * T,
                                                            {7, T}),
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data() + at * T, {T}));
            EXPECT_EQ(golden::dump(cube, px), golden::dump(pixel.result))
                << "pixel " << at << ", " << threads << " threads";
        }
    }
}

TEST(ResidualCap, RefusesALimitOutsideItsBounds)
{
    try
    {
        ccd::Algorithm(capped(0.0), registry());
        ADD_FAILURE() << "a limit of 0 accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find(
                      "outlier 'residual-cap' parameter 'limit' is 0; it must be greater than 0"),
                  std::string::npos)
            << e.what();
    }
}

namespace
{

std::string read(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    EXPECT_TRUE(in) << "cannot read " << path;
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

TEST(ResidualCap, TheGuideQuotesItsFiles)
{
    const std::string marker = "<!-- from: ";
    std::istringstream guide(read(CCD_SOURCE_DIR "/docs/writing-a-strategy.md"));
    std::string line;
    int quotes = 0;
    while (std::getline(guide, line))
    {
        if (line.rfind(marker, 0) != 0)
            continue;
        const std::string path = line.substr(marker.size(), line.find(" -->") - marker.size());
        ASSERT_TRUE(std::getline(guide, line) && line.rfind("```", 0) == 0) << path;
        std::string block;
        while (std::getline(guide, line) && line != "```")
            block += line + "\n";
        EXPECT_NE(read(CCD_SOURCE_DIR "/" + path).find(block), std::string::npos)
            << "docs/writing-a-strategy.md quotes " << path << " as\n" << block;
        ++quotes;
    }
    EXPECT_GE(quotes, 10);
}
