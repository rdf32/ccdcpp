//==============================================================================
//
// Band layouts other than Landsat's.
//
// HarmonicOptions describes the band layout -- NUM_BANDS, the detection,
// Tmask and saturation lists, GREEN_IDX and THERMAL_IDX -- and the
// procedures read bands only through it. These tests hold that claim from
// three sides:
//
//   * Bands.BitIdenticalToFixture: synthetic profiles with 7, 8 and 10 bands,
//     with and without thermal, and an 8-band detection list, through
//     run_pixel() and run_cube(), against their own hexfloat fixture. The
//     profiles live here, in the test; ccdcpp ships no presets for them.
//
//   * Equivalences that need no fixture, on the Landsat corpus and on the
//     10-band profile: the default layout spelled out field by field equals
//     HarmonicOptions{}; permuting the rows and remapping every index gives
//     the same fits, un-permuted, bit for bit; bands that are in no list
//     leave the other bands' results bit for bit as they were; and an input
//     without a thermal row matches one whose thermal row nothing rejects.
//
//   * adjust_change_threshold() takes its chi-squared df from the caller.
//
// The fixture belongs to a platform, as the Landsat one does (see
// test_golden.cpp). To regenerate it, from a build of the code whose output
// is the reference:
//
//     CCD_BANDS_WRITE=1 ./test_bands
//
//==============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/math/distributions/chi_squared.hpp>
#include <gtest/gtest.h>

#include "specs.hpp"

#include "ccd/ccd.hpp"

#include "golden.hpp"
#include "synthetic.hpp"
#include "synthetic_bands.hpp"
#include "band_profiles.hpp"

#ifndef CCD_BANDS_FIXTURE
#define CCD_BANDS_FIXTURE "bands_fixture.txt"
#endif

namespace
{

using namespace golden::bands;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

// run_pixel() on a private copy of one (B, T) block.
ccd::FitResult run_detect(
    const std::vector<std::int64_t>& dates,
    const double* spectral,
    std::size_t B,
    const std::uint8_t* qas,
    const ccd::HarmonicOptions& h,
    const ccd::config::ConfigTree& config
)
{
    const std::size_t T = dates.size();
    std::vector<double> block(spectral, spectral + B * T);

    return ccd::run_pixel(
        golden::spec_of(h, config),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(block.data(), {B, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas, {T})).result;
}

// `result` with each model's bands replaced by bands[rows[0]], bands[rows[1]],
// ... -- a subset, a reordering, or both.
ccd::FitResult select_bands(
    const ccd::FitResult& result,
    const std::vector<ccd::index_t>& rows
)
{
    ccd::FitResult out = result;
    for (std::size_t m = 0; m < out.models.size(); ++m)
    {
        out.models[m].bands.clear();
        for (const auto r : rows)
            out.models[m].bands.push_back(result.models[m].bands[r]);
    }
    return out;
}

// A cube as flat arrays, B rows per pixel.
struct Series
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (N, B, T)
    std::vector<std::uint8_t> qas;  // (N, T)
    std::size_t N = 0;
    std::size_t B = 0;
    std::size_t T = 0;

    const double* pixel(std::size_t p) const { return spectral.data() + p * B * T; }
    const std::uint8_t* qa(std::size_t p) const { return qas.data() + p * T; }
};

Series from_cube(const golden::Cube& c)
{
    return {c.dates, c.spectral, c.qas, c.H * c.W, 7, c.T};
}

Series from_cube(const golden::BandCube& c)
{
    return {c.dates, c.spectral, c.qas, c.H * c.W, c.B, c.T};
}

Series to_physical(Series s)
{
    for (std::size_t p = 0; p < s.N; ++p)
        golden::to_physical(s.spectral.data() + p * s.B * s.T, s.T);
    return s;
}

// The rows `rows` of every pixel, in that order.
Series take_rows(const Series& s, const std::vector<ccd::index_t>& rows)
{
    Series out{s.dates, {}, s.qas, s.N, rows.size(), s.T};
    out.spectral.reserve(s.N * rows.size() * s.T);
    for (std::size_t p = 0; p < s.N; ++p)
        for (const auto r : rows)
            out.spectral.insert(
                out.spectral.end(),
                s.pixel(p) + r * s.T,
                s.pixel(p) + (r + 1) * s.T);
    return out;
}

// `s` with `k` rows of independent noise appended to every pixel.
Series with_inert_rows(const Series& s, std::size_t k, std::uint64_t seed)
{
    golden::Rng rng(seed);
    Series out{s.dates, {}, s.qas, s.N, s.B + k, s.T};
    out.spectral.reserve(s.N * out.B * s.T);
    for (std::size_t p = 0; p < s.N; ++p)
    {
        out.spectral.insert(out.spectral.end(), s.pixel(p), s.pixel(p) + s.B * s.T);
        for (std::size_t i = 0; i < k * s.T; ++i)
            out.spectral.push_back(std::floor(rng.uniform(0.0, 5000.0)));
    }
    return out;
}

// Every pixel's run_pixel() dump, with each result passed through `map` first.
template<typename Map>
std::string dump_series(
    const Series& s,
    const ccd::HarmonicOptions& h,
    const ccd::config::ConfigTree& config,
    Map map
)
{
    std::string out;
    for (std::size_t p = 0; p < s.N; ++p)
    {
        golden::dump_pixel(
            out,
            std::to_string(p),
            map(run_detect(s.dates, s.pixel(p), s.B, s.qa(p), h, config)));
    }
    return out;
}

std::string dump_series(
    const Series& s,
    const ccd::HarmonicOptions& h,
    const ccd::config::ConfigTree& config
)
{
    return dump_series(s, h, config, [](ccd::FitResult r) { return r; });
}

// The Landsat corpus of test_golden.cpp, with its seeds.
std::vector<Series> landsat_corpus()
{
    return {
        from_cube(golden::make_cube(11, golden::Regime::Cadence16, 10, 16)),
        from_cube(golden::make_cube(12, golden::Regime::Cadence8,   6,  8)),
        from_cube(golden::make_cube(13, golden::Regime::Duplicates, 5,  8)),
        from_cube(golden::make_cube(14, golden::Regime::Short,      4,  5)),
        from_cube(golden::make_cube(15, golden::Regime::Medium,     5,  8)),
    };
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

// Which procedures and outcomes a profile's corpus reached.
struct Coverage
{
    std::size_t standard = 0;
    std::size_t breaks = 0;
    std::size_t snow = 0;
    std::size_t insufficient = 0;
};

} // namespace

//==============================================================================
// The fixture
//==============================================================================

TEST(Bands, BitIdenticalToFixture)
{
    std::string dump;

    for (std::size_t pi = 0; pi < profiles().size(); ++pi)
    {
        const Profile profile = profiles()[pi];
        const ccd::config::ConfigTree& config = profile.config;
        const std::size_t B = profile.specs.size();
        Coverage coverage;

        for (const auto& spec : CUBES)
        {
            const golden::BandCube c = golden::make_band_cube(
                100 * (pi + 1) + spec.seed, spec.regime, spec.H, spec.W, profile.specs);
            const Series s = from_cube(c);

            std::string single;
            for (std::size_t p = 0; p < s.N; ++p)
            {
                const ccd::FitResult r =
                    run_detect(s.dates, s.pixel(p), B, s.qa(p), profile.h, config);

                for (const auto& m : r.models)
                {
                    EXPECT_EQ(m.bands.size(), B);
                    switch (m.curve_qa)
                    {
                        case ccd::CurveQA::PersistentSnow:    ++coverage.snow; break;
                        case ccd::CurveQA::InsufficientClear: ++coverage.insufficient; break;
                        default:
                            ++coverage.standard;
                            if (m.change_probability == 1.0)
                                ++coverage.breaks;
                    }
                }

                golden::dump_pixel(
                    single,
                    std::string(profile.name) + ":" + spec.name + ":" +
                        std::to_string(p / c.W) + "," + std::to_string(p % c.W),
                    r);
            }

            for (int threads : {1, 4})
            {
                ccd::CubeResult cube = ccd::run_cube(
                    golden::spec_of(profile.h, config),
                    ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {c.T}),
                    ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, B, c.T}),
                    ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, c.T}),
                    threads);

                EXPECT_TRUE(cube.failures.empty());
                ASSERT_EQ(cube.pixels.size(), s.N);

                std::string from_cube_dump;
                for (const auto& px : cube.pixels)
                    golden::dump_pixel(
                        from_cube_dump,
                        std::string(profile.name) + ":" + spec.name + ":" +
                            std::to_string(px.row) + "," + std::to_string(px.col),
                        px.result);

                EXPECT_TRUE(from_cube_dump == single)
                    << "run_cube(threads=" << threads << ") differs from run_pixel() on "
                    << profile.name << ":" << spec.name;
            }

            dump += single;
        }

        // The corpus reaches every procedure, and Standard finds breaks.
        EXPECT_GT(coverage.standard, 0u) << profile.name;
        EXPECT_GT(coverage.breaks, 0u) << profile.name;
        EXPECT_GT(coverage.snow, 0u) << profile.name;
        EXPECT_GT(coverage.insufficient, 0u) << profile.name;
    }

    const std::string header = "# platform: " + golden::platform() + "\n";

    if (std::getenv("CCD_BANDS_WRITE"))
    {
        std::ofstream out(CCD_BANDS_FIXTURE, std::ios::binary);
        out << header << dump;
        GTEST_SKIP() << "fixture written to " << CCD_BANDS_FIXTURE;
    }

    std::ifstream in(CCD_BANDS_FIXTURE, std::ios::binary);
    ASSERT_TRUE(in) << "missing fixture " << CCD_BANDS_FIXTURE;

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

//==============================================================================
// Generic equals default
//
// The Landsat layout written out through every band-layout field gives what
// HarmonicOptions{} gives, under the DN defaults and the physical preset.
//==============================================================================

namespace
{

ccd::HarmonicOptions landsat_layout(ccd::HarmonicOptions h)
{
    h.NUM_BANDS = 7;
    h.DETECTION_BANDS = band_list({1, 2, 3, 4, 5});
    h.SATURATION_BANDS = band_list({0, 1, 2, 3, 4, 5});
    h.GREEN_IDX = 1;
    h.THERMAL_IDX = 6;
    return h;
}

} // namespace

TEST(Bands, GenericLayoutEqualsDefault)
{
    for (const auto& s : landsat_corpus())
    {
        EXPECT_EQ(
            dump_series(s, landsat_layout(ccd::HarmonicOptions{}), golden::ccdc()),
            dump_series(s, ccd::HarmonicOptions{}, golden::ccdc()));

        const Series phys = to_physical(s);
        EXPECT_EQ(
            dump_series(phys, landsat_layout(golden::physical_harmonic()), golden::physical_config()),
            dump_series(phys, golden::physical_harmonic(), golden::physical_config()));
    }
}

//==============================================================================
// Permutation
//
// Row r of the permuted input holds row perm[r] of the original. Every index
// is remapped to follow its band, and each list keeps its order of roles, so
// the detection sums run over the same bands in the same order. The fits
// come back in the permuted row order; put back, they are bit-identical.
//==============================================================================

namespace
{

ccd::HarmonicOptions permute_layout(
    ccd::HarmonicOptions h,
    const std::vector<ccd::index_t>& inv
)
{
    const auto remap = [&](const ccd::BandList& list)
    {
        ccd::BandList out;
        for (const ccd::index_t b : list)
            out.push_back(inv[b]);
        return out;
    };

    h.DETECTION_BANDS = remap(h.DETECTION_BANDS);
    h.SATURATION_BANDS = remap(h.SATURATION_BANDS);
    h.GREEN_IDX = inv[h.GREEN_IDX];
    if (h.THERMAL_IDX != ccd::NO_BAND)
        h.THERMAL_IDX = inv[h.THERMAL_IDX];
    return h;
}

void expect_permutation_invariant(
    const Series& s,
    const ccd::HarmonicOptions& h,
    const ccd::config::ConfigTree& config,
    const std::vector<ccd::index_t>& perm
)
{
    ASSERT_EQ(perm.size(), s.B);

    std::vector<ccd::index_t> inv(perm.size());
    for (ccd::index_t r = 0; r < perm.size(); ++r)
        inv[perm[r]] = r;

    const Series permuted = take_rows(s, perm);
    const ccd::HarmonicOptions hp = permute_layout(h, inv);
    ccd::config::ConfigTree permuted_config = config;   // the Tmask bands follow theirs too
    std::vector<std::int64_t> tmask;
    for (const std::int64_t b :
         golden::parameter<std::vector<std::int64_t>>(config, "window_screen", "bands"))
        tmask.push_back(static_cast<std::int64_t>(inv[static_cast<std::size_t>(b)]));
    permuted_config["window_screen"].params["bands"] = tmask;

    EXPECT_EQ(
        dump_series(permuted, hp, permuted_config,

            [&](const ccd::FitResult& r) { return select_bands(r, inv); }),
        dump_series(s, h, config));
}

} // namespace

TEST(Bands, PermutationInvariantSevenBands)
{
    const std::vector<ccd::index_t> perm = {6, 3, 0, 5, 1, 4, 2};

    for (const auto& s : landsat_corpus())
    {
        expect_permutation_invariant(s, ccd::HarmonicOptions{}, golden::ccdc(), perm);
        expect_permutation_invariant(
            to_physical(s), golden::physical_harmonic(), golden::physical_config(), perm);
    }
}

TEST(Bands, PermutationInvariantTenBands)
{
    const std::vector<ccd::index_t> perm = {9, 4, 7, 0, 2, 8, 1, 5, 3, 6};

    for (const auto& profile : profiles())
    {
        if (profile.specs.size() != 10)
            continue;

        for (const auto& spec : CUBES)
        {
            const Series s = from_cube(golden::make_band_cube(
                500 + spec.seed, spec.regime, spec.H, spec.W, profile.specs));
            expect_permutation_invariant(s, profile.h, profile.config, perm);
        }
    }
}

//==============================================================================
// Inert bands
//
// Rows in no list are fitted and reported, and touch nothing else: the
// original rows' results are bit-identical with k noise rows appended.
//==============================================================================

TEST(Bands, InertBandsChangeNothingElse)
{
    for (const std::size_t k : {1u, 3u})
    {
        ccd::HarmonicOptions h;
        h.NUM_BANDS = 7 + k;

        ccd::HarmonicOptions hp = golden::physical_harmonic();
        hp.NUM_BANDS = 7 + k;

        for (const auto& s : landsat_corpus())
        {
            const Series wide = with_inert_rows(s, k, 900 + k);

            EXPECT_EQ(
                dump_series(wide, h, golden::ccdc(),
                    [](const ccd::FitResult& r) { return select_bands(r, iota(7)); }),
                dump_series(s, ccd::HarmonicOptions{}, golden::ccdc()));

            const Series phys = to_physical(s);
            EXPECT_EQ(
                dump_series(with_inert_rows(phys, k, 900 + k), hp, golden::physical_config(),
                    [](const ccd::FitResult& r) { return select_bands(r, iota(7)); }),
                dump_series(phys, golden::physical_harmonic(), golden::physical_config()));
        }
    }

    // A non-Landsat layout, its thermal row now followed by the inert rows.
    const Profile l30 = profiles()[1];
    for (const auto& spec : CUBES)
    {
        const Series s = from_cube(golden::make_band_cube(
            700 + spec.seed, spec.regime, spec.H, spec.W, l30.specs));

        ccd::HarmonicOptions h = l30.h;
        h.NUM_BANDS = 10;

        EXPECT_EQ(
            dump_series(with_inert_rows(s, 2, 77), h, l30.config,
                [](const ccd::FitResult& r) { return select_bands(r, iota(8)); }),
            dump_series(s, l30.h, l30.config));
    }
}

//==============================================================================
// Thermal absent
//
// THERMAL_IDX = NO_BAND skips the thermal filter and the Standard thermal
// transform. Its reference is the same series WITH the thermal row, under an
// identity transform and infinite thermal bounds: a filter that rejects no
// finite value. The thermal row is in no list, so the six optical bands, the
// masks and the segments are bit-identical.
//
// What the equivalence needs, and what it does not claim:
//
//   * Every thermal value is finite. NaN fails both bound comparisons, so an
//     infinite-bounds filter still drops a NaN observation that NO_BAND
//     keeps. The generator writes no NaN.
//   * The reference needs the infinite bounds, not the defaults: the
//     generator plants out-of-range thermal values, which the default
//     bounds reject and NO_BAND does not. That difference is checked below,
//     so the test is not passing because the filter never fired.
//   * The transform plays no part in the optical results (thermal is in no
//     list); NO_BAND is run under the DN transform defaults to show the
//     transform is skipped rather than applied to a row that is not there.
//   * PermanentSnow and InsufficientClear apply the thermal filter too; the
//     corpus reaches both, and the equivalence covers them.
//==============================================================================

TEST(Bands, ThermalAbsentMatchesAThermalFilterThatRejectsNothing)
{
    constexpr double INF = std::numeric_limits<double>::infinity();
    const std::vector<ccd::index_t> optical = iota(6);

    const auto no_thermal = [](ccd::HarmonicOptions h)
    {
        h.NUM_BANDS = 6;
        h.THERMAL_IDX = ccd::NO_BAND;
        return h;
    };

    const auto open_thermal = [&](ccd::HarmonicOptions h)
    {
        h.THERMAL_SCALE = 1.0;
        h.THERMAL_OFFSET = 0.0;
        h.THERM_MIN = -INF;
        h.THERM_MAX = INF;
        return h;
    };

    const auto optical_only = [&](const ccd::FitResult& r) { return select_bands(r, optical); };

    bool default_bounds_differ = false;
    Coverage coverage;

    for (const auto& s : landsat_corpus())
    {
        for (std::size_t p = 0; p < s.N; ++p)
            for (std::size_t t = 0; t < s.T; ++t)
                ASSERT_TRUE(std::isfinite(s.pixel(p)[6 * s.T + t]));

        const Series stripped = take_rows(s, optical);
        const std::string absent =
            dump_series(stripped, no_thermal(ccd::HarmonicOptions{}), golden::ccdc());

        EXPECT_EQ(
            absent,
            dump_series(s, open_thermal(ccd::HarmonicOptions{}), golden::ccdc(), optical_only));

        default_bounds_differ = default_bounds_differ ||
            absent != dump_series(s, ccd::HarmonicOptions{}, golden::ccdc(), optical_only);

        const Series phys = to_physical(s);
        EXPECT_EQ(
            dump_series(take_rows(phys, optical), no_thermal(golden::physical_harmonic()),
                golden::physical_config()),
            dump_series(phys, open_thermal(golden::physical_harmonic()),
                golden::physical_config(), optical_only));

        for (std::size_t p = 0; p < s.N; ++p)
        {
            for (const auto& m : run_detect(stripped.dates, stripped.pixel(p), 6,
                     stripped.qa(p), no_thermal(ccd::HarmonicOptions{}), golden::ccdc()).models)
            {
                if (m.curve_qa == ccd::CurveQA::PersistentSnow) ++coverage.snow;
                else if (m.curve_qa == ccd::CurveQA::InsufficientClear) ++coverage.insufficient;
                else ++coverage.standard;
            }
        }
    }

    EXPECT_TRUE(default_bounds_differ)
        << "the default thermal bounds rejected nothing; the test shows nothing";
    EXPECT_GT(coverage.standard, 0u);
    EXPECT_GT(coverage.snow, 0u);
    EXPECT_GT(coverage.insufficient, 0u);
}

//==============================================================================
// adjust_change_threshold() and its df
//==============================================================================

namespace
{

// The number of doubles between a and b: 0 when they are equal, 1 for
// neighbours. Both finite and of one sign.
std::uint64_t ulp_distance(double a, double b)
{
    std::uint64_t ia, ib;
    std::memcpy(&ia, &a, sizeof a);
    std::memcpy(&ib, &b, sizeof b);
    return ia > ib ? ia - ib : ib - ia;
}

// adjust_change_threshold() takes a chi-squared quantile through
// std::pow and Boost, and libm's last bit differs between platforms (glibc
// and Apple's libm give neighbouring doubles at peek 7 and 40). The pinned
// values hold to 4 ulp on any platform; a df mix-up moves them by far more.
constexpr std::uint64_t LIBM_ULP = 4;

} // namespace

TEST(Bands, ChangeThresholdTakesItsDegreesOfFreedom)
{
    constexpr double DEFAULT = 15.086272469388987;

    // df = 5: the values it gives with the detection count of the default
    // layout.
    EXPECT_EQ(ccd::adjust_change_threshold(6, 6, DEFAULT, 5, 0.99), DEFAULT);
    const std::pair<ccd::index_t, double> pinned[] = {
        {7, 0x1.af382cc5ddf21p+3},
        {9, 0x1.6866ca4a577a6p+3},
        {12, 0x1.27903c5857cd8p+3},
        {20, 0x1.a71fc718bf22bp+2},
        {40, 0x1.15f079295b2c3p+2},
    };
    for (const auto& [peek, expected] : pinned)
    {
        const double actual = ccd::adjust_change_threshold(peek, 6, DEFAULT, 5, 0.99);
        EXPECT_LE(ulp_distance(actual, expected), LIBM_ULP)
            << "peek " << peek << ": " << actual << " vs " << expected;
    }

    // df = 8: the chi2(8) quantile of the same probability.
    const double pt = 1.0 - std::pow(1.0 - 0.99, 6.0 / 12.0);
    EXPECT_EQ(ccd::adjust_change_threshold(12, 6, DEFAULT, 8, 0.99), chi2_quantile(8, pt));
    EXPECT_GT(ccd::adjust_change_threshold(12, 6, DEFAULT, 8, 0.99),
              ccd::adjust_change_threshold(12, 6, DEFAULT, 5, 0.99));
}
