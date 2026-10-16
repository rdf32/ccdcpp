//==============================================================================
//
// Prefix stability: new observations never rewrite a final segment.
//
// The Standard machine reads the length of the series only in its size
// guards (room for a start window, a full peek after the window, enough
// observations after the last break), each of the form position + k <= size.
// Observations dated after T turn no guard that held into one that fails,
// and every read before the first failing guard lies inside a window those
// guards bounded. So a run on the observations up to T and a run on the
// whole series make the same decisions, step for step, up to the first size
// guard that fails in the shorter run. A segment written before that point
// is final: the longer run writes it too, bit for bit, with the same
// processing mask below its break.
//
// The one input that is not causal is the statistics window (the QA
// statistics, the cadence, the variogram and the green filter all read the
// observations dated <= STAT_ORD). Both runs here take STAT_ORD as
// min(STAT_ORD, T), so they see the same window.
//
// Read from the output alone, the final segments of a run are the leading
// ones up to its last change-ended segment (change probability 1), or the
// Start segment when no segment ended in a change. The series-end segment
// and the End segment are provisional: a longer series can rewrite them.
//
// PrefixStability.* split each pixel at many dates T and check, for each:
//
//   * the shorter run's final segments equal the full run's leading
//     segments, every field bit for bit (the hexfloat dump of golden.hpp);
//   * its processing mask equals the full run's below the last final
//     segment's break day.
//
// Every change to the procedure must keep this. A change that breaks it lets
// new data rewrite the past, and makes resuming from stored results unsound.
//
//==============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

#include "characterisation.hpp"
#include "golden.hpp"
#include "synthetic.hpp"

namespace
{

// One pixel's whole series and the options it runs under.
struct Series
{
    std::string name;
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, T), band-major
    std::vector<std::uint8_t> qas;
    ccd::HarmonicOptions h;
    ccd::LassoOptions l;
};

// The first `count` observations of `s`, with STAT_ORD at `stat_ord`.
ccd::FitResult run_prefix(const Series& s, std::size_t count, std::int64_t stat_ord)
{
    const std::size_t T = s.dates.size();

    std::vector<double> block(7 * count);
    for (std::size_t b = 0; b < 7; ++b)
        std::copy_n(s.spectral.begin() + b * T, count, block.begin() + b * count);

    ccd::HarmonicOptions h = s.h;
    h.STAT_ORD = stat_ord;

    return ccd::detect(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(s.dates.data(), {count}),
        ccd::ArrayView<double, 2>::contiguous(block.data(), {7, count}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(s.qas.data(), {count}),
        h,
        s.l
    );
}

// The number of leading final segments of `r` (see the header).
std::size_t final_count(const ccd::FitResult& r)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < r.models.size(); ++i)
        if (r.models[i].change_probability == 1.0)
            n = i + 1;

    if (n == 0 && !r.models.empty() && r.models[0].curve_qa == ccd::CurveQA::Start)
        n = 1;

    return n;
}

std::string dump_leading(const ccd::FitResult& r, std::size_t n)
{
    ccd::FitResult leading;
    leading.models.assign(r.models.begin(), r.models.begin() + n);
    std::string out;
    golden::dump_models(out, leading);
    return out;
}

// What one pixel's sweep saw.
struct Sweep
{
    std::size_t splits = 0;            // split dates tested
    std::size_t with_final = 0;        // of those, with at least one final segment
    std::size_t provisional = 0;       // of those, with a segment the full run rewrote
};

// Splits `s` after each observation index in `ends` (an index i keeps
// observations [0, i]; an index followed by another observation of the same
// date is skipped, so a split never cuts a date in two) and checks prefix
// stability at each, counting into `out`.
void sweep(const Series& s, const std::vector<std::size_t>& ends, Sweep& out)
{
    const std::size_t T = s.dates.size();

    // The full run under each statistics cutoff, computed once.
    std::map<std::int64_t, ccd::FitResult> full;
    auto full_at = [&](std::int64_t stat_ord) -> const ccd::FitResult&
    {
        auto it = full.find(stat_ord);
        if (it == full.end())
            it = full.emplace(stat_ord, run_prefix(s, T, stat_ord)).first;
        return it->second;
    };

    for (const std::size_t last : ends)
    {
        if (last + 1 >= T || s.dates[last + 1] == s.dates[last])
            continue;

        const std::int64_t split_day = s.dates[last];
        const std::int64_t stat_ord = std::min(s.h.STAT_ORD, split_day);
        const ccd::FitResult shorter = run_prefix(s, last + 1, stat_ord);
        const ccd::FitResult& longer = full_at(stat_ord);
        ++out.splits;

        const std::size_t n = final_count(shorter);
        if (n == 0)
            continue;
        ++out.with_final;

        const std::string at = s.name + " split after " + std::to_string(split_day);

        ASSERT_LE(n, longer.models.size()) << at;
        EXPECT_EQ(dump_leading(shorter, n), dump_leading(longer, n)) << at;

        if (shorter.models.size() > n &&
            dump_leading(shorter, n + 1) != dump_leading(longer, std::min(n + 1, longer.models.size())))
            ++out.provisional;

        // The mask below the last final segment's break day.
        const std::int64_t break_day = shorter.models[n - 1].break_day;
        ASSERT_EQ(shorter.mask.size(), last + 1) << at;
        ASSERT_EQ(longer.mask.size(), T) << at;
        for (std::size_t t = 0; t <= last && s.dates[t] < break_day; ++t)
        {
            ASSERT_EQ(shorter.mask[t], longer.mask[t])
                << at << ": mask differs at day " << s.dates[t];
        }
    }
}

// Every observation index.
std::vector<std::size_t> every(std::size_t T)
{
    std::vector<std::size_t> out(T);
    for (std::size_t i = 0; i < T; ++i)
        out[i] = i;
    return out;
}

// `count` indexes spread evenly over [0, T).
std::vector<std::size_t> spread(std::size_t T, std::size_t count)
{
    std::vector<std::size_t> out;
    for (std::size_t k = 1; k <= count; ++k)
        out.push_back(k * T / (count + 1));
    return out;
}

// A test CSV: date, seven bands, QA per row; `header` skips the first line.
Series read_csv(const std::string& name, const std::string& path, bool header)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Unable to open " + path);

    Series s;
    s.name = name;
    std::array<std::vector<double>, 7> bands;

    std::string line;
    if (header)
        std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        s.dates.push_back(static_cast<std::int64_t>(std::stod(value)));
        for (std::size_t b = 0; b < 7; ++b)
        {
            std::getline(ss, value, ',');
            bands[b].push_back(std::stod(value));
        }
        std::getline(ss, value, ',');
        s.qas.push_back(static_cast<std::uint8_t>(std::stod(value)));
    }

    const std::size_t T = s.dates.size();
    s.spectral.resize(7 * T);
    for (std::size_t b = 0; b < 7; ++b)
        std::copy(bands[b].begin(), bands[b].end(), s.spectral.begin() + b * T);
    return s;
}

void report(const std::string& corpus, const Sweep& total)
{
    std::cout << "[ prefix   ] " << corpus << ": " << total.splits << " splits, "
              << total.with_final << " with final segments, " << total.provisional
              << " with a provisional segment the full run rewrote\n";
}

} // namespace

// The characterization pixels (characterisation.hpp), every split: they take
// every reachable transition, Lookback below the break and the re-entry after
// a series end among them.
TEST(PrefixStability, CharacterizationPixelsEverySplit)
{
    Sweep total;
    for (const auto& p : golden::characterisation_pixels())
    {
        Series s{p.name, p.dates, p.spectral, p.qas, p.h, p.l};
        sweep(s, every(s.dates.size()), total);
    }
    report("characterization", total);
    EXPECT_GT(total.with_final, 0u);
    EXPECT_GT(total.provisional, 0u);
}

// The CSV pixels, every split: the pyccd reference pixel, the Collection-2
// pixel and the Atlanta pixel with a half-day median gap.
TEST(PrefixStability, CsvPixelsEverySplit)
{
    const std::string dir = CCD_PREFIX_DATA_DIR;
    std::vector<Series> pixels;
    pixels.push_back(read_csv("pyccd", dir + "/test_3657_3610_observations.csv", false));
    pixels.push_back(read_csv("atlanta_481_393", dir + "/test_atlanta_024013_00030008_481_393.csv", false));
    {
        Series c2 = read_csv("c2", dir + "/test002_c2.csv", true);
        c2.h = golden::c2_harmonic();
        pixels.push_back(std::move(c2));
    }

    Sweep total;
    for (const auto& s : pixels)
        sweep(s, every(s.dates.size()), total);
    report("csv", total);
    EXPECT_GT(total.with_final, 0u);
    EXPECT_GT(total.provisional, 0u);
}

// The golden synthetic cubes (synthetic.hpp), every pixel at 12 split dates,
// under the DN defaults and the physical-unit preset.
TEST(PrefixStability, SyntheticCubesTwelveSplits)
{
    struct Spec { const char* name; std::uint64_t seed; golden::Regime regime; std::size_t H, W; };
    const Spec cubes[] = {
        {"c16",    11, golden::Regime::Cadence16,  10, 16},
        {"c8",     12, golden::Regime::Cadence8,    6,  8},
        {"dup",    13, golden::Regime::Duplicates,  5,  8},
        {"short",  14, golden::Regime::Short,       4,  5},
        {"medium", 15, golden::Regime::Medium,      5,  8},
    };

    Sweep total;
    for (const auto& spec : cubes)
    {
        const golden::Cube c = golden::make_cube(spec.seed, spec.regime, spec.H, spec.W);
        const std::size_t T = c.T;

        for (const bool physical : {false, true})
        {
            for (std::size_t p = 0; p < c.H * c.W; ++p)
            {
                Series s;
                s.name = std::string(spec.name) + (physical ? ":phys:" : ":dn:") + std::to_string(p);
                s.dates = c.dates;
                s.spectral.assign(c.spectral.begin() + p * 7 * T, c.spectral.begin() + (p + 1) * 7 * T);
                s.qas.assign(c.qas.begin() + p * T, c.qas.begin() + (p + 1) * T);
                if (physical)
                {
                    golden::to_physical(s.spectral.data(), T);
                    s.h = golden::physical_harmonic();
                    s.l = golden::physical_lasso();
                }
                sweep(s, spread(T, 12), total);
            }
        }
    }
    report("synthetic", total);
    EXPECT_GT(total.with_final, 0u);
    EXPECT_GT(total.provisional, 0u);
}
