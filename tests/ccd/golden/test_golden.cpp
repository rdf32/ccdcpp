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
// so, not as a failure. A fixture is a same-machine before-and-after check
// (this build against the reference code built on the same platform), never
// a machine's expected values: the fixtures here, in test_bands.cpp and in
// test_transitions.cpp are the only platform-bound tests. Every other test is
// machine-agnostic, with no per-platform expected values; where a platform
// difference needs a tolerance, the test says what differs (libm's last bit,
// a compiler's contraction). To regenerate (from a build of the code whose
// output is the reference, never from the change under test):
//
//     CCD_GOLDEN_WRITE=1 ./test_golden
//
//==============================================================================

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.hpp"

#include "ccd/ccd.hpp"

#include "dump.hpp"
#include "series.hpp"
#include "fixture.hpp"
#include "units.hpp"
#include "synthetic.hpp"

#ifndef CCD_GOLDEN_FIXTURE
#define CCD_GOLDEN_FIXTURE "golden_fixture.txt"
#endif

namespace
{

struct Options
{
    const char* name;
    ccd::DataSpec h;
    ccd::config::Config config;
    bool physical;
};

std::vector<Options> option_sets()
{
    return {
        {"dn", ccd::DataSpec{}, golden::pyccd(), false},
        {"phys", golden::physical_spec(), golden::pyccd(), true},
    };
}

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
        const golden::Series p = golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", true);
        golden::dump_pixel(dump, "pyccd:dn", golden::detect(p));

        golden::Series phys = golden::to_physical(p);
        phys.h = golden::physical_spec();
        golden::dump_pixel(dump, "pyccd:phys", golden::detect(phys));
    }
    {
        // The reference pixel with saturation-band values on the bounds and a
        // NaN: the reflectance filter's default exclusive ends drop all five.
        golden::Series p = golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", true);
        golden::put_bound_values(p.spectral.data(), p.qas.data(), p.size());
        golden::dump_pixel(dump, "bounds:dn", golden::detect(p));

        golden::Series phys = golden::to_physical(p);
        phys.h = golden::physical_spec();
        golden::dump_pixel(dump, "bounds:phys", golden::detect(phys));
    }
    {
        golden::Series p = golden::read_csv(CCD_TEST_DATA_DIR "/test002_c2.csv", true);
        p.h = golden::c2_spec();
        p.config = golden::c2_config();
        golden::dump_pixel(dump, "c2:c2", golden::detect(p));
    }

    //--------------------------------------------------------------------------
    // Synthetic cubes
    //--------------------------------------------------------------------------
    for (const auto& spec : golden::GOLDEN_CUBES)
    {
        const golden::FlatCube base =
            golden::flat(golden::make_cube(spec.seed, spec.regime, spec.H, spec.W));

        for (const auto& o : option_sets())
        {
            const golden::FlatCube c = o.physical ? golden::to_physical(base) : base;
            const ccd::RunSpec run = golden::spec_of(o.h, o.config);
            const auto label = [&](std::size_t row, std::size_t col)
            {
                return pixel_label(spec.name, o.name, row, col);
            };

            std::string single;
            for (std::size_t p = 0; p < c.N; ++p)
                golden::dump_pixel(single, label(p / spec.W, p % spec.W), golden::detect(run, c.view(p)));

            golden::expect_cube_equals_detect(run, c, spec.H, spec.W, single, label,
                                              std::string(spec.name) + ":" + o.name);

            dump += single;
        }
    }

    return dump;
}

} // namespace

TEST(Golden, BitIdenticalToFixture)
{
    golden::expect_matches_fixture(run_corpus(), CCD_GOLDEN_FIXTURE, "CCD_GOLDEN_WRITE");
}

//------------------------------------------------------------------------------
// detect_cube reads float64, float32 and uint16, in any strides, and gives
// the results of the same values as contiguous float64. Nothing writes to
// the input.
//------------------------------------------------------------------------------

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
    const ccd::DataSpec h;
    const ccd::config::Config config = golden::pyccd();


    const std::string reference = golden::dump_cube(ccd::detect_cube(
        golden::spec_of(h, config),
        dates,
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {H, W, 7, T}),
        qas,
        2));

    EXPECT_EQ(c.spectral, original) << "detect_cube wrote to its input";
    EXPECT_EQ(reference, golden::dump_cube(ccd::detect_cube(
        golden::spec_of(h, config),
        dates,
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {H, W, 7, T}),
        qas,
        2))) << "a second call on the same buffer differs";

    std::vector<float> f32(c.spectral.begin(), c.spectral.end());
    EXPECT_EQ(reference, golden::dump_cube(ccd::detect_cube(
        golden::spec_of(h, config),
        dates,
        ccd::ArrayView<const float, 4>::contiguous(f32.data(), {H, W, 7, T}),
        qas,
        2))) << "float32";

    std::vector<std::uint16_t> u16(c.spectral.size());
    for (std::size_t i = 0; i < u16.size(); ++i)
        u16[i] = static_cast<std::uint16_t>(c.spectral[i]);
    EXPECT_EQ(reference, golden::dump_cube(ccd::detect_cube(
        golden::spec_of(h, config),
        dates,
        ccd::ArrayView<const std::uint16_t, 4>::contiguous(u16.data(), {H, W, 7, T}),
        qas,
        2))) << "uint16";

    // (H, W, T, B) storage seen as (H, W, B, T).
    std::vector<double> hwtb(c.spectral.size());
    for (std::size_t p = 0; p < H * W; ++p)
        for (std::size_t b = 0; b < 7; ++b)
            for (std::size_t t = 0; t < T; ++t)
                hwtb[(p * T + t) * 7 + b] = c.spectral[(p * 7 + b) * T + t];
    EXPECT_EQ(reference, golden::dump_cube(ccd::detect_cube(
        golden::spec_of(h, config),
        dates,
        ccd::ArrayView<const double, 4>::strided(hwtb.data(), {H, W, 7, T}, {W * T * 7, T * 7, 1, 7}),
        qas,
        2))) << "strided float64";
}

TEST(Golden, DetectDoesNotWriteItsInput)
{
    const golden::Series p = golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", true);
    const golden::Series buffer = p;

    const std::string first = golden::dump(golden::detect(buffer));
    EXPECT_EQ(buffer.spectral, p.spectral);
    EXPECT_EQ(first, golden::dump(golden::detect(buffer)));
}
