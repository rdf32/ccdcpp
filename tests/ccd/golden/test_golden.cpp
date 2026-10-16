//==============================================================================
//
// The bit-exact golden test.
//
// The other procedure tests compare against pyccd to 1e-6. This one compares
// against ccdcpp itself, exactly: every coefficient, intercept, rmse,
// magnitude, iteration count, segment boundary and processing mask bit, as
// hexfloat text, byte for byte. It is the gate for changes that are meant to
// be pure refactors -- performance work above all -- where "close" is not the
// claim being made.
//
// Inputs:
//
//   * the pyccd reference pixel (DN defaults and the physical-unit preset),
//   * the same pixel with saturation-band values on the reflectance bounds
//     and a NaN (golden::put_bound_values), both ways,
//   * the Collection-2 reference pixel under Collection-2 bounds,
//   * synthetic cubes from synthetic.hpp covering Standard, PermanentSnow and
//     InsufficientClear, 0-4 breaks, unflagged outliers, saturation, fill
//     gaps, duplicate dates, short series and a dense 8-day cadence,
//
// run three ways: detect() per pixel, and detect_cube() with 1 and 4 threads.
// detect() is compared with the fixture; each detect_cube() run is compared
// with detect(), pixel for pixel in raster order.
//
// THE FIXTURE BELONGS TO A PLATFORM. libm, floating point contraction and
// Eigen's kernels all reach the low bits, so exact equality is only a
// meaningful claim on the platform that wrote the fixture; its first line
// names it. On a different platform a mismatch is reported as a skip that says
// so, not as a failure. To regenerate (from a build of the code whose output
// is the reference, never from the change under test):
//
//     CCD_GOLDEN_WRITE=1 ./test_golden
//
//==============================================================================

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

#include "golden.hpp"
#include "synthetic.hpp"

#ifndef CCD_GOLDEN_FIXTURE
#define CCD_GOLDEN_FIXTURE "golden_fixture.txt"
#endif

namespace
{

struct Pixel
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, T)
    std::vector<std::uint8_t> qas;
};

Pixel read_csv(const std::string& filename)
{
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);
    if (!file)
        throw std::runtime_error("Unable to open " + filename);

    std::array<std::vector<double>, 7> bands;
    Pixel out;

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        out.dates.push_back(static_cast<std::int64_t>(std::stod(value)));

        for (std::size_t b = 0; b < 7; ++b)
        {
            std::getline(ss, value, ',');
            bands[b].push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        out.qas.push_back(static_cast<std::uint8_t>(std::stod(value)));
    }

    const std::size_t T = out.dates.size();
    out.spectral.resize(7 * T);
    for (std::size_t b = 0; b < 7; ++b)
        std::copy(bands[b].begin(), bands[b].end(), out.spectral.begin() + b * T);

    return out;
}

ccd::FitResult run_detect(
    const std::vector<std::int64_t>& dates,
    const double* spectral,
    const std::uint8_t* qas,
    const ccd::HarmonicOptions& h,
    const ccd::LassoOptions& l
)
{
    const std::size_t T = dates.size();

    // A private copy: the reference output is the one a fresh buffer gives.
    std::vector<double> block(spectral, spectral + 7 * T);

    return ccd::detect(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<double, 2>::contiguous(block.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas, {T}),
        h,
        l
    );
}

struct Options
{
    const char* name;
    ccd::HarmonicOptions h;
    ccd::LassoOptions l;
    bool physical;
};

std::vector<Options> option_sets()
{
    return {
        {"dn", ccd::HarmonicOptions{}, ccd::LassoOptions{}, false},
        {"phys", golden::physical_harmonic(), golden::physical_lasso(), true},
    };
}

struct CubeSpec
{
    const char* name;
    std::uint64_t seed;
    golden::Regime regime;
    std::size_t H;
    std::size_t W;
};

const CubeSpec CUBES[] = {
    {"c16",    11, golden::Regime::Cadence16,  10, 16},
    {"c8",     12, golden::Regime::Cadence8,    6,  8},
    {"dup",    13, golden::Regime::Duplicates,  5,  8},
    {"short",  14, golden::Regime::Short,       4,  5},
    {"medium", 15, golden::Regime::Medium,      5,  8},
};

std::string pixel_label(const char* cube, const char* opts, std::size_t row, std::size_t col)
{
    return std::string(cube) + ":" + opts + ":" + std::to_string(row) + "," + std::to_string(col);
}

//------------------------------------------------------------------------------
// The whole corpus through detect(), plus every detect_cube() run checked
// against it. Returns the detect() dump.
//------------------------------------------------------------------------------

std::string run_corpus()
{
    std::string dump;

    //--------------------------------------------------------------------------
    // Reference pixels
    //--------------------------------------------------------------------------
    {
        const Pixel p = read_csv("test_3657_3610_observations.csv");
        const std::size_t T = p.dates.size();

        golden::dump_pixel(dump, "pyccd:dn",
            run_detect(p.dates, p.spectral.data(), p.qas.data(),
                ccd::HarmonicOptions{}, ccd::LassoOptions{}));

        std::vector<double> phys = p.spectral;
        golden::to_physical(phys.data(), T);
        golden::dump_pixel(dump, "pyccd:phys",
            run_detect(p.dates, phys.data(), p.qas.data(),
                golden::physical_harmonic(), golden::physical_lasso()));
    }
    {
        // The reference pixel with saturation-band values on the bounds and a
        // NaN: the reflectance filter's default exclusive ends drop all five.
        Pixel p = read_csv("test_3657_3610_observations.csv");
        const std::size_t T = p.dates.size();
        golden::put_bound_values(p.spectral.data(), p.qas.data(), T);

        golden::dump_pixel(dump, "bounds:dn",
            run_detect(p.dates, p.spectral.data(), p.qas.data(),
                ccd::HarmonicOptions{}, ccd::LassoOptions{}));

        std::vector<double> phys = p.spectral;
        golden::to_physical(phys.data(), T);
        golden::dump_pixel(dump, "bounds:phys",
            run_detect(p.dates, phys.data(), p.qas.data(),
                golden::physical_harmonic(), golden::physical_lasso()));
    }
    {
        const Pixel p = read_csv("test002_c2.csv");
        golden::dump_pixel(dump, "c2:c2",
            run_detect(p.dates, p.spectral.data(), p.qas.data(),
                golden::c2_harmonic(), ccd::LassoOptions{}));
    }

    //--------------------------------------------------------------------------
    // Synthetic cubes
    //--------------------------------------------------------------------------
    for (const auto& spec : CUBES)
    {
        const golden::Cube base =
            golden::make_cube(spec.seed, spec.regime, spec.H, spec.W);

        for (const auto& o : option_sets())
        {
            golden::Cube c = base;
            const std::size_t T = c.T;
            const std::size_t N = c.H * c.W;

            if (o.physical)
                for (std::size_t p = 0; p < N; ++p)
                    golden::to_physical(c.spectral.data() + p * 7 * T, T);

            std::string single;
            for (std::size_t p = 0; p < N; ++p)
            {
                golden::dump_pixel(
                    single,
                    pixel_label(spec.name, o.name, p / c.W, p % c.W),
                    run_detect(c.dates, c.spectral.data() + p * 7 * T,
                        c.qas.data() + p * T, o.h, o.l));
            }

            for (int threads : {1, 4})
            {
                std::vector<double> cube_spectral = c.spectral;

                ccd::CubeResult cube = ccd::detect_cube(
                    ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
                    ccd::ArrayView<double, 4>::contiguous(cube_spectral.data(), {c.H, c.W, 7, T}),
                    ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
                    o.h,
                    o.l,
                    threads
                );

                EXPECT_TRUE(cube.failures.empty());
                EXPECT_EQ(cube.pixels.size(), N);

                for (std::size_t i = 0; i < cube.pixels.size(); ++i)
                {
                    EXPECT_EQ(cube.pixels[i].row * c.W + cube.pixels[i].col, i)
                        << "detect_cube(threads=" << threads << ") is not in raster order";
                }

                std::sort(
                    cube.pixels.begin(),
                    cube.pixels.end(),
                    [](const ccd::PixelResult& a, const ccd::PixelResult& b)
                    {
                        return a.row != b.row ? a.row < b.row : a.col < b.col;
                    });

                std::string from_cube;
                for (const auto& px : cube.pixels)
                    golden::dump_pixel(
                        from_cube,
                        pixel_label(spec.name, o.name, px.row, px.col),
                        px.result);

                EXPECT_TRUE(from_cube == single)
                    << "detect_cube(threads=" << threads << ") differs from detect() on "
                    << spec.name << ":" << o.name;
            }

            dump += single;
        }
    }

    return dump;
}

std::vector<std::string> split_pixels(const std::string& dump)
{
    std::vector<std::string> out;
    std::istringstream in(dump);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("P ", 0) == 0 || out.empty())
            out.emplace_back();
        out.back() += line;
        out.back() += '\n';
    }
    return out;
}

} // namespace

TEST(Golden, BitIdenticalToFixture)
{
    const std::string dump = run_corpus();
    const std::string header = "# platform: " + golden::platform() + "\n";

    if (std::getenv("CCD_GOLDEN_WRITE"))
    {
        std::ofstream out(CCD_GOLDEN_FIXTURE, std::ios::binary);
        out << header << dump;
        GTEST_SKIP() << "fixture written to " << CCD_GOLDEN_FIXTURE;
    }

    std::ifstream in(CCD_GOLDEN_FIXTURE, std::ios::binary);
    ASSERT_TRUE(in) << "missing fixture " << CCD_GOLDEN_FIXTURE;

    std::string fixture_header;
    std::getline(in, fixture_header);
    fixture_header += '\n';
    const std::string expected(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    if (dump == expected)
        return;

    if (fixture_header != header)
    {
        GTEST_SKIP()
            << "output differs from a fixture written on another platform\n"
            << "  fixture: " << fixture_header
            << "  this:    " << header
            << "exact equality is only claimed on the fixture's platform";
    }

    const auto actual_px = split_pixels(dump);
    const auto expected_px = split_pixels(expected);

    ASSERT_EQ(actual_px.size(), expected_px.size()) << "pixel count differs";

    std::size_t differing = 0;
    for (std::size_t i = 0; i < actual_px.size(); ++i)
    {
        if (actual_px[i] == expected_px[i])
            continue;

        if (++differing <= 3)
        {
            ADD_FAILURE()
                << "expected:\n" << expected_px[i]
                << "actual:\n" << actual_px[i];
        }
    }
    FAIL() << differing << " of " << actual_px.size() << " pixels differ from the fixture";
}

//------------------------------------------------------------------------------
// detect_cube reads float64, float32 and uint16, in any strides, and gives
// the results of the same values as contiguous float64. Nothing writes to
// the input.
//------------------------------------------------------------------------------

namespace
{

std::string cube_dump(ccd::CubeResult cube)
{
    std::sort(
        cube.pixels.begin(),
        cube.pixels.end(),
        [](const ccd::PixelResult& a, const ccd::PixelResult& b)
        {
            return a.row != b.row ? a.row < b.row : a.col < b.col;
        });

    std::string out;
    for (const auto& px : cube.pixels)
        golden::dump_pixel(
            out,
            std::to_string(px.row) + "," + std::to_string(px.col),
            px.result);
    return out;
}

} // namespace

TEST(Golden, CubeDtypesAndLayouts)
{
    golden::Cube c = golden::make_cube(21, golden::Regime::Cadence16, 4, 6);
    const std::size_t H = c.H, W = c.W, T = c.T;

    // Values every input type represents exactly.
    for (auto& v : c.spectral)
        v = std::min(std::max(v, 0.0), 65535.0);

    const std::vector<double> original = c.spectral;

    const auto dates =
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T});
    const auto qas =
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {H, W, T});
    const ccd::HarmonicOptions h;
    const ccd::LassoOptions l;

    const std::string reference = cube_dump(ccd::detect_cube(
        dates,
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {H, W, 7, T}),
        qas, h, l, 2));

    EXPECT_EQ(c.spectral, original) << "detect_cube wrote to its input";
    EXPECT_EQ(reference, cube_dump(ccd::detect_cube(
        dates,
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {H, W, 7, T}),
        qas, h, l, 2))) << "a second call on the same buffer differs";

    std::vector<float> f32(c.spectral.begin(), c.spectral.end());
    EXPECT_EQ(reference, cube_dump(ccd::detect_cube(
        dates,
        ccd::ArrayView<const float, 4>::contiguous(f32.data(), {H, W, 7, T}),
        qas, h, l, 2))) << "float32";

    std::vector<std::uint16_t> u16(c.spectral.size());
    for (std::size_t i = 0; i < u16.size(); ++i)
        u16[i] = static_cast<std::uint16_t>(c.spectral[i]);
    EXPECT_EQ(reference, cube_dump(ccd::detect_cube(
        dates,
        ccd::ArrayView<const std::uint16_t, 4>::contiguous(u16.data(), {H, W, 7, T}),
        qas, h, l, 2))) << "uint16";

    // (H, W, T, B) storage seen as (H, W, B, T).
    std::vector<double> hwtb(c.spectral.size());
    for (std::size_t p = 0; p < H * W; ++p)
        for (std::size_t b = 0; b < 7; ++b)
            for (std::size_t t = 0; t < T; ++t)
                hwtb[(p * T + t) * 7 + b] = c.spectral[(p * 7 + b) * T + t];
    EXPECT_EQ(reference, cube_dump(ccd::detect_cube(
        dates,
        ccd::ArrayView<const double, 4>::strided(hwtb.data(), {H, W, 7, T}, {W * T * 7, T * 7, 1, 7}),
        qas, h, l, 2))) << "strided float64";
}

TEST(Golden, DetectDoesNotWriteItsInput)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const std::size_t T = p.dates.size();

    std::vector<double> buffer = p.spectral;
    const auto view = ccd::ArrayView<const double, 2>::contiguous(buffer.data(), {7, T});
    const auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T});
    const auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T});

    std::string first, second;
    golden::dump_pixel(first, "", ccd::detect(dates, view, qas, ccd::HarmonicOptions{}, ccd::LassoOptions{}));
    EXPECT_EQ(buffer, p.spectral);
    golden::dump_pixel(second, "", ccd::detect(dates, view, qas, ccd::HarmonicOptions{}, ccd::LassoOptions{}));
    EXPECT_EQ(first, second);
}
