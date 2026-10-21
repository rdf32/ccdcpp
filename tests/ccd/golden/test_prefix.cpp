//==============================================================================
//
// Prefix stability: new observations never rewrite a final segment.
//
// The Standard machine reads the length of the series only in its size
// guards (enough observations, room for a start window, a full peek after
// the window, enough observations after the last break), each a comparison
// of the series size with a position plus a count.
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
// observations dated <= STATISTICS_END). Both runs here take STATISTICS_END as
// min(STATISTICS_END, T), so they see the same window.
//
// Read from the output alone, the final segments of a run are the leading
// ones up to its last change-ended segment (change probability 1), or the
// Start segment when no segment ended in a change. The series-end segment
// and the End segment are provisional: a longer series can rewrite them.
//
// PrefixStability.* split each pixel at many dates T and check, for each:
//
//   * the shorter run's final segments equal the full run's leading
//     segments, every field bit for bit (the hexfloat dump of dump.hpp);
//   * its processing mask equals the full run's below the last final
//     segment's break day;
//   * its resume record (record.hpp) names the same number of final
//     segments.
//
// Each runs as golden::SHARDS tests (series_corpus.hpp): the sample in the
// quick tier, the rest labelled corpus.
//
// Every change to the procedure must keep this. A change that breaks it lets
// new data rewrite the past, and makes resuming from stored results unsound.
//
//==============================================================================

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

#include "dump.hpp"
#include "series_corpus.hpp"

namespace
{

using golden::Series;

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
// observations [0, i]; a split never cuts a date in two) and checks prefix
// stability at each, counting into `out`.
void sweep(const Series& s, const std::vector<std::size_t>& ends, Sweep& out)
{
    const std::size_t T = s.size();

    // The full run under each statistics cutoff, computed once.
    std::map<std::int64_t, ccd::FitResult> full;
    auto full_at = [&](std::int64_t statistics_end) -> const ccd::FitResult&
    {
        auto it = full.find(statistics_end);
        if (it == full.end())
            it = full.emplace(statistics_end, golden::detect_at(s, statistics_end)).first;
        return it->second;
    };

    for (const std::size_t last : ends)
    {
        if (!golden::splits_cleanly(s, last))
            continue;

        const std::int64_t split_day = s.dates[last];
        const std::int64_t statistics_end = std::min(s.statistics_end(), split_day);
        const ccd::PixelOutput recorded = golden::recorded_at(s.prefix(last + 1), statistics_end);
        const ccd::FitResult& shorter = recorded.result;
        const ccd::FitResult& longer = full_at(statistics_end);
        ++out.splits;

        const std::string at = s.name + " split after " + std::to_string(split_day);
        const std::size_t n = final_count(shorter);

        // The resume record names the same final segments.
        const std::size_t recorded_final =
            recorded.record.procedure == ccd::PixelProcedure::Standard
                ? recorded.record.standard.final_segments
                : 0;
        EXPECT_EQ(recorded_final, n) << at;
        if (n == 0)
            continue;
        ++out.with_final;

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

void report(const std::string& corpus, const Sweep& total)
{
    std::cout << "[ prefix   ] " << corpus << ": " << total.splits << " splits, "
              << total.with_final << " with final segments, " << total.provisional
              << " with a provisional segment the full run rewrote\n";
}

} // namespace

class PrefixStability : public ::testing::TestWithParam<std::size_t>
{
};

// The characterization pixels (characterization.hpp), every split: they take
// every reachable transition, Lookback near the break and the series end at
// a dense cadence among them.
TEST_P(PrefixStability, CharacterizationPixelsEverySplit)
{
    Sweep total;
    std::size_t ordinal = 0;
    for (const auto& s : golden::characterization_series())
        sweep(s, golden::shard_of(s, ordinal++, golden::every(s.size()), GetParam()), total);
    report("characterization", total);
    EXPECT_GT(total.with_final, 0u);
    EXPECT_GT(total.provisional, 0u);
}

// The CSV pixels, every split: the pyccd reference pixel, the Collection-2
// pixel and the Atlanta pixel with a half-day median gap.
TEST_P(PrefixStability, CsvPixelsEverySplit)
{
    Sweep total;
    std::size_t ordinal = 0;
    for (const auto& s : golden::csv_series(CCD_TEST_DATA_DIR))
        sweep(s, golden::shard_of(s, ordinal++, golden::every(s.size()), GetParam()), total);
    report("csv", total);
    EXPECT_GT(total.with_final, 0u);
    EXPECT_GT(total.provisional, 0u);
}

// The golden synthetic cubes (synthetic.hpp), every pixel at 12 split dates,
// under the DN defaults and the physical-unit preset.
TEST_P(PrefixStability, SyntheticCubesTwelveSplits)
{
    Sweep total;
    std::size_t ordinal = 0;
    for (const auto& s : golden::synthetic_series())
        sweep(s, golden::shard_of(s, ordinal++, golden::spread(s.size(), 12), GetParam()), total);
    report("synthetic", total);
    EXPECT_GT(total.with_final, 0u);
    EXPECT_GT(total.provisional, 0u);
}

INSTANTIATE_TEST_SUITE_P(Shards, PrefixStability,
                         ::testing::Range(std::size_t{0}, golden::SHARDS),
                         [](const auto& info) { return golden::shard_name(info.param); });
