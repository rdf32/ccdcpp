//==============================================================================
//
// QA lookup tables.
//
// The claim a table makes is that decoding raw QA through it is the same run
// as passing the decoded codes as uint8. These tests take the golden corpus's
// QA codes, encode each one into a raw value of each product -- cycling
// through several raw values per code, flag combinations that exercise the
// precedence among them -- and compare detect_cube() on the raw values plus
// the table with detect_cube() on the codes, exactly, as hexfloat dumps.
//
// The corpus uses code 8 (QA_CIRRUS1) beside the six classes. The procedures
// read no code but QA_FILL, QA_CLEAR, QA_WATER, QA_SNOW and QA_CLOUD, so 8 is
// read as QA_SHADOW is, and it is encoded as each product's shadow.
//
//==============================================================================

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/qa_table.hpp"

#include "golden.hpp"
#include "synthetic.hpp"

namespace
{

using ccd::HarmonicOptions;
using ccd::LassoOptions;
using ccd::QaTablePtr;

//------------------------------------------------------------------------------
// Raw values per class, for one product.
//------------------------------------------------------------------------------

struct Scheme
{
    const char* name;
    std::function<QaTablePtr(const HarmonicOptions&)> build;

    std::vector<std::uint16_t> fill;
    std::vector<std::uint16_t> clear;
    std::vector<std::uint16_t> water;
    std::vector<std::uint16_t> shadow;
    std::vector<std::uint16_t> snow;
    std::vector<std::uint16_t> cloud;

    // Every raw value fits in uint8.
    bool narrow;
};

std::vector<Scheme> schemes()
{
    return {
        {
            "landsat_c2_qa_pixel",
            [](const HarmonicOptions& o) { return ccd::qa_tables::landsat_c2_qa_pixel(o); },
            {1, 1 | 8, 0xFFFF},
            {21824, 64, 0},
            {21952, 128},
            {16, 16 | 32, 16 | 128, 23888},
            {32, 32 | 128, 30048},
            {8, 2, 4, 8 | 16 | 128, 2 | 32, 22280, 55052},
            false
        },
        {
            "landsat_c1_pixel_qa",
            [](const HarmonicOptions& o) { return ccd::qa_tables::landsat_c1_pixel_qa(o); },
            {1, 1 | 32, 0xFFFF},
            {2, 2 | 64, 322, 0x300, 0x400, 0x300 | 64},
            {4, 4 | 2, 324},
            {8, 8 | 16 | 4, 8 | 2, 328},
            {16, 16 | 4 | 2, 336},
            {32, 32 | 8 | 4, 32 | 2, 352, 480, 992},
            false
        },
        {
            "hls_fmask",
            [](const HarmonicOptions& o) { return ccd::qa_tables::hls_fmask(o); },
            {255},
            {0, 0x40, 0x80, 0xC0},
            {32, 32 | 0x40},
            {8, 8 | 16 | 32, 8 | 0x40},
            {16, 16 | 32, 16 | 0x80},
            {2, 4, 1, 2 | 8 | 32, 4 | 16, 1 | 0xC0},
            true
        },
        {
            "sentinel2_scl",
            [](const HarmonicOptions& o) { return ccd::qa_tables::sentinel2_scl(o); },
            {0, 1},
            {4, 5},
            {6},
            {2, 3},
            {11},
            {7, 8, 9, 10},
            true
        },
    };
}

// The class a corpus code is read as; see the header.
std::uint8_t read_as(std::uint8_t code, const HarmonicOptions& o)
{
    switch (code)
    {
        case 255: return o.QA_FILL;
        case 0:   return o.QA_CLEAR;
        case 1:   return o.QA_WATER;
        case 2:   return o.QA_SHADOW;
        case 3:   return o.QA_SNOW;
        case 4:   return o.QA_CLOUD;
        case 8:   return o.QA_SHADOW;
    }
    throw std::logic_error("corpus code " + std::to_string(code));
}

// The i-th raw value of corpus code `code` (default numbering).
std::uint16_t encode(const Scheme& s, std::uint8_t code, std::size_t i)
{
    const std::vector<std::uint16_t>* values = nullptr;
    switch (code)
    {
        case 255: values = &s.fill;   break;
        case 0:   values = &s.clear;  break;
        case 1:   values = &s.water;  break;
        case 2:
        case 8:   values = &s.shadow; break;
        case 3:   values = &s.snow;   break;
        case 4:   values = &s.cloud;  break;
        default:  throw std::logic_error("corpus code " + std::to_string(code));
    }
    return (*values)[i % values->size()];
}

//------------------------------------------------------------------------------
// Runs and dumps.
//------------------------------------------------------------------------------

std::string dump(const ccd::CubeResult& cube)
{
    EXPECT_TRUE(cube.failures.empty());

    std::string out;
    for (const auto& px : cube.pixels)
        golden::dump_pixel(
            out,
            std::to_string(px.row) + "," + std::to_string(px.col),
            px.result);
    return out;
}

template<typename Qa>
std::string run_cube(
    const golden::Cube& c,
    const std::vector<Qa>& qas,
    const HarmonicOptions& h,
    int threads,
    const QaTablePtr& table
)
{
    return dump(ccd::detect_cube(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {c.T}),
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, c.T}),
        ccd::ArrayView<const Qa, 3>::contiguous(qas.data(), {c.H, c.W, c.T}),
        h, LassoOptions{}, threads, true, table));
}

struct CubeSpec
{
    const char* name;
    std::uint64_t seed;
    golden::Regime regime;
    std::size_t H;
    std::size_t W;
};

// The golden corpus's cubes, smaller.
const CubeSpec CUBES[] = {
    {"c16",    11, golden::Regime::Cadence16,  5, 8},
    {"c8",     12, golden::Regime::Cadence8,   3, 4},
    {"dup",    13, golden::Regime::Duplicates, 3, 4},
    {"short",  14, golden::Regime::Short,      2, 4},
    {"medium", 15, golden::Regime::Medium,     3, 4},
};

HarmonicOptions renumbered()
{
    HarmonicOptions h;
    h.QA_FILL   = 200;
    h.QA_CLEAR  = 10;
    h.QA_WATER  = 11;
    h.QA_SHADOW = 12;
    h.QA_SNOW   = 13;
    h.QA_CLOUD  = 14;
    return h;
}

} // namespace

//------------------------------------------------------------------------------
// Decoding is the codes: every scheme, both QA dtypes where the raw values
// fit, threads 1 and 8, against the corpus codes passed directly.
//------------------------------------------------------------------------------

TEST(QaTables, DecodedEqualsPreDecoded)
{
    const HarmonicOptions h;

    for (const auto& spec : CUBES)
    {
        const golden::Cube c =
            golden::make_cube(spec.seed, spec.regime, spec.H, spec.W);

        const std::string reference = run_cube(c, c.qas, h, 1, nullptr);
        ASSERT_FALSE(reference.empty());

        for (const auto& s : schemes())
        {
            const QaTablePtr table = s.build(h);

            std::vector<std::uint16_t> raw(c.qas.size());
            for (std::size_t i = 0; i < raw.size(); ++i)
            {
                raw[i] = encode(s, c.qas[i], i);
                ASSERT_EQ(table->code[raw[i]], read_as(c.qas[i], h))
                    << s.name << " raw " << raw[i];
            }
            const std::vector<std::uint16_t> raw_before = raw;

            for (int threads : {1, 8})
            {
                EXPECT_EQ(reference, run_cube(c, raw, h, threads, table))
                    << s.name << " uint16 " << spec.name << " threads=" << threads;

                if (s.narrow)
                {
                    const std::vector<std::uint8_t> raw8(raw.begin(), raw.end());
                    EXPECT_EQ(reference, run_cube(c, raw8, h, threads, table))
                        << s.name << " uint8 " << spec.name << " threads=" << threads;
                }
            }

            EXPECT_EQ(raw, raw_before) << "detect_cube wrote to qas";
        }
    }
}

//------------------------------------------------------------------------------
// detect() on one pixel, raw plus table, against the codes.
//------------------------------------------------------------------------------

TEST(QaTables, DetectDecodes)
{
    const golden::Cube c =
        golden::make_cube(11, golden::Regime::Cadence16, 2, 4);
    const std::size_t T = c.T;
    const HarmonicOptions h;
    const auto dates =
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T});

    for (const auto& s : schemes())
    {
        const QaTablePtr table = s.build(h);

        for (std::size_t p = 0; p < c.H * c.W; ++p)
        {
            const auto spectral = ccd::ArrayView<const double, 2>::contiguous(
                c.spectral.data() + p * 7 * T, {7, T});
            const std::uint8_t* codes = c.qas.data() + p * T;

            std::vector<std::uint16_t> raw(T);
            for (std::size_t t = 0; t < T; ++t)
                raw[t] = encode(s, codes[t], t);

            std::string expected, actual;
            golden::dump_pixel(expected, "", ccd::detect(dates, spectral,
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(codes, {T}),
                h, LassoOptions{}));
            golden::dump_pixel(actual, "", ccd::detect(dates, spectral,
                ccd::ArrayView<const std::uint16_t, 1>::contiguous(raw.data(), {T}),
                h, LassoOptions{}, table));
            EXPECT_EQ(expected, actual) << s.name << " pixel " << p;

            // Caller-held scratch, reused across a table and no table.
            ccd::DetectScratch scratch;
            std::string with_scratch;
            golden::dump_pixel(with_scratch, "", ccd::detect(dates, spectral,
                ccd::ArrayView<const std::uint16_t, 1>::contiguous(raw.data(), {T}),
                h, LassoOptions{}, scratch, table));
            EXPECT_EQ(expected, with_scratch) << s.name << " pixel " << p;

            std::string after;
            golden::dump_pixel(after, "", ccd::detect(dates, spectral,
                ccd::ArrayView<const std::uint8_t, 1>::contiguous(codes, {T}),
                h, LassoOptions{}, scratch));
            EXPECT_EQ(expected, after) << s.name << " pixel " << p;
        }
    }
}

//------------------------------------------------------------------------------
// A table built from renumbered QA_* codes writes those codes, and the run
// under those options is the run under the defaults.
//------------------------------------------------------------------------------

TEST(QaTables, HonourTheOptionsCodes)
{
    const golden::Cube c =
        golden::make_cube(11, golden::Regime::Cadence16, 4, 6);
    const HarmonicOptions defaults;
    const HarmonicOptions h = renumbered();

    const std::string reference = run_cube(c, c.qas, defaults, 2, nullptr);

    std::vector<std::uint8_t> codes(c.qas.size());
    for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = read_as(c.qas[i], h);
    EXPECT_EQ(reference, run_cube(c, codes, h, 2, nullptr));

    for (const auto& s : schemes())
    {
        const QaTablePtr table = s.build(h);

        std::vector<std::uint16_t> raw(c.qas.size());
        for (std::size_t i = 0; i < raw.size(); ++i)
            raw[i] = encode(s, c.qas[i], i);

        EXPECT_EQ(reference, run_cube(c, raw, h, 2, table)) << s.name;

        // Every entry is one of the six codes.
        for (std::size_t v = 0; v < ccd::QA_TABLE_SIZE; ++v)
        {
            const std::uint8_t code = table->code[v];
            ASSERT_TRUE(
                code == h.QA_FILL || code == h.QA_CLEAR || code == h.QA_WATER ||
                code == h.QA_SHADOW || code == h.QA_SNOW || code == h.QA_CLOUD)
                << s.name << " entry " << v << " is " << int(code);
        }
    }
}

//------------------------------------------------------------------------------
// Strided raw QA: (H, W, T) seen through (T, H, W) storage.
//------------------------------------------------------------------------------

TEST(QaTables, StridedRawQa)
{
    const golden::Cube c =
        golden::make_cube(21, golden::Regime::Cadence16, 3, 5);
    const std::size_t H = c.H, W = c.W, T = c.T;
    const HarmonicOptions h;
    const Scheme s = schemes()[0];
    const QaTablePtr table = s.build(h);

    const std::string reference = run_cube(c, c.qas, h, 2, nullptr);

    std::vector<std::uint16_t> thw(H * W * T);
    for (std::size_t p = 0; p < H * W; ++p)
        for (std::size_t t = 0; t < T; ++t)
            thw[t * H * W + p] = encode(s, c.qas[p * T + t], p * T + t);

    EXPECT_EQ(reference, dump(ccd::detect_cube(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {H, W, 7, T}),
        ccd::ArrayView<const std::uint16_t, 3>::strided(thw.data(), {H, W, T}, {W, 1, H * W}),
        h, LassoOptions{}, 2, true, table)));
}

//------------------------------------------------------------------------------
// The tables' own definitions, value by value where a product is small, and
// at the precedence boundaries where it is a bitmask.
//------------------------------------------------------------------------------

TEST(QaTables, Definitions)
{
    const HarmonicOptions o;
    const auto FILL = o.QA_FILL, CLEAR = o.QA_CLEAR, WATER = o.QA_WATER,
               SHADOW = o.QA_SHADOW, SNOW = o.QA_SNOW, CLOUD = o.QA_CLOUD;

    {
        const auto t = ccd::qa_tables::landsat_c2_qa_pixel(o);
        EXPECT_EQ(t->code[0], CLEAR);
        EXPECT_EQ(t->code[1], FILL);
        EXPECT_EQ(t->code[1 | 8], FILL);
        EXPECT_EQ(t->code[2], CLOUD);
        EXPECT_EQ(t->code[4], CLOUD);
        EXPECT_EQ(t->code[8], CLOUD);
        EXPECT_EQ(t->code[8 | 16 | 32 | 128], CLOUD);
        EXPECT_EQ(t->code[16 | 32 | 128], SHADOW);
        EXPECT_EQ(t->code[32 | 128], SNOW);
        EXPECT_EQ(t->code[128], WATER);
        EXPECT_EQ(t->code[64], CLEAR);
        EXPECT_EQ(t->code[0xFF00], CLEAR);
    }
    {
        const auto t = ccd::qa_tables::landsat_c1_pixel_qa(o);
        EXPECT_EQ(t->code[0], FILL);
        EXPECT_EQ(t->code[1], FILL);
        EXPECT_EQ(t->code[1 | 32], FILL);
        EXPECT_EQ(t->code[32 | 8 | 16 | 4 | 2], CLOUD);
        EXPECT_EQ(t->code[8 | 16 | 4 | 2], SHADOW);
        EXPECT_EQ(t->code[16 | 4 | 2], SNOW);
        EXPECT_EQ(t->code[4 | 2], WATER);
        EXPECT_EQ(t->code[2], CLEAR);
        EXPECT_EQ(t->code[0x300], CLEAR);
        EXPECT_EQ(t->code[0x100], FILL);
        EXPECT_EQ(t->code[0x200], FILL);
        EXPECT_EQ(t->code[0x400], CLEAR);
        EXPECT_EQ(t->code[64], FILL);
        EXPECT_EQ(t->code[0x800], FILL);
    }
    {
        const auto t = ccd::qa_tables::hls_fmask(o);
        const auto a = ccd::qa_tables::hls_fmask(o, true);
        EXPECT_EQ(t->code[255], FILL);
        EXPECT_EQ(t->code[256], FILL);
        EXPECT_EQ(t->code[0], CLEAR);
        EXPECT_EQ(t->code[1], CLOUD);
        EXPECT_EQ(t->code[2 | 8 | 16 | 32], CLOUD);
        EXPECT_EQ(t->code[4], CLOUD);
        EXPECT_EQ(t->code[8 | 16 | 32], SHADOW);
        EXPECT_EQ(t->code[16 | 32], SNOW);
        EXPECT_EQ(t->code[32], WATER);
        EXPECT_EQ(t->code[0xC0], CLEAR);
        EXPECT_EQ(t->code[0xC0 | 32], WATER);
        EXPECT_EQ(a->code[0xC0], CLOUD);
        EXPECT_EQ(a->code[0xC0 | 8], CLOUD);
        EXPECT_EQ(a->code[0x80], CLEAR);
        EXPECT_EQ(a->code[0x40 | 32], WATER);

        // Only the aerosol entries differ.
        for (std::size_t v = 0; v < ccd::QA_TABLE_SIZE; ++v)
        {
            const bool high = v < 255 && (v >> 6) == 3;
            if (!high)
                ASSERT_EQ(t->code[v], a->code[v]) << v;
        }
    }
    {
        const auto t = ccd::qa_tables::sentinel2_scl(o);
        const std::uint8_t expected[12] = {
            FILL, FILL, SHADOW, SHADOW, CLEAR, CLEAR,
            WATER, CLOUD, CLOUD, CLOUD, CLOUD, SNOW};
        for (std::size_t v = 0; v < 12; ++v)
            EXPECT_EQ(t->code[v], expected[v]) << v;
        for (std::size_t v = 12; v < ccd::QA_TABLE_SIZE; ++v)
            ASSERT_EQ(t->code[v], FILL) << v;
    }
}

//------------------------------------------------------------------------------
// uint16 QA without a table has no meaning, and is refused before any work.
//------------------------------------------------------------------------------

TEST(QaTables, Uint16NeedsATable)
{
    const golden::Cube c =
        golden::make_cube(11, golden::Regime::Cadence16, 1, 2);
    const std::size_t T = c.T;
    const std::vector<std::uint16_t> raw(c.qas.begin(), c.qas.end());
    const HarmonicOptions h;
    const LassoOptions l;

    const auto dates =
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T});
    const auto qa1 =
        ccd::ArrayView<const std::uint16_t, 1>::contiguous(raw.data(), {T});
    const auto qa3 =
        ccd::ArrayView<const std::uint16_t, 3>::contiguous(raw.data(), {1, 2, T});
    const auto pixel =
        ccd::ArrayView<const double, 2>::contiguous(c.spectral.data(), {7, T});

    auto expect_refused = [](auto call)
    {
        try
        {
            call();
            ADD_FAILURE() << "uint16 qas without a table was accepted";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find("uint16"), std::string::npos) << e.what();
        }
    };

    expect_refused([&] { ccd::detect(dates, pixel, qa1, h, l); });
    expect_refused([&] { ccd::detect(dates, pixel, qa1, h, l, nullptr); });

    ccd::DetectScratch scratch;
    expect_refused([&] { ccd::detect(dates, pixel, qa1, h, l, scratch); });

    const std::vector<float> f32(c.spectral.begin(), c.spectral.end());
    std::vector<std::uint16_t> u16(c.spectral.size());
    for (std::size_t i = 0; i < u16.size(); ++i)
        u16[i] = static_cast<std::uint16_t>(std::max(0.0, c.spectral[i]));

    expect_refused([&] { ccd::detect_cube(dates,
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {1, 2, 7, T}),
        qa3, h, l, 1); });
    expect_refused([&] { ccd::detect_cube(dates,
        ccd::ArrayView<const float, 4>::contiguous(f32.data(), {1, 2, 7, T}),
        qa3, h, l, 1); });
    expect_refused([&] { ccd::detect_cube(dates,
        ccd::ArrayView<const std::uint16_t, 4>::contiguous(u16.data(), {1, 2, 7, T}),
        qa3, h, l, 1); });

    // With a table it runs, for every spectral type.
    const QaTablePtr table = ccd::qa_tables::hls_fmask(h);
    EXPECT_EQ(ccd::detect_cube(dates,
        ccd::ArrayView<const std::uint16_t, 4>::contiguous(u16.data(), {1, 2, 7, T}),
        qa3, h, l, 1, true, table).pixels.size(), 2u);
    EXPECT_EQ(ccd::detect_cube(dates,
        ccd::ArrayView<const float, 4>::contiguous(f32.data(), {1, 2, 7, T}),
        qa3, h, l, 1, true, table).pixels.size(), 2u);
}
