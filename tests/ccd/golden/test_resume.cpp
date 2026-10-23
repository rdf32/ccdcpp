//==============================================================================
//
// Resume equals the full run.
//
// For a pixel series P and a date T, the stored run is detect() with RunSpec::record on
// the observations up to T; its continuation is detect() with that record
// and that run's segments as its resume, on the whole of P. The contract
// (run/run.hpp, detect()) is that the continuation equals, bit for bit,
// detect() on P with every
// statistics cutoff capped at T: every segment and the processing mask (the
// hexfloat dump of dump.hpp), and the record it writes equals the one a
// run made that way writes, so a resumed run resumes again.
//
//   * ResumeEqualsTheFullRun.*: that, at every split of the
//     characterization and CSV pixels and at twelve of each golden cube
//     pixel; and a chain, P cut at two dates and resumed twice.
//   * ResumeOfChangedInput.*: an observation up to T reprocessed is rerun,
//     and says so; a scene added up to T that the pixel's QA marks fill is
//     not a change.
//   * ResumeWithNothingNew.*: observations after T that the filters drop
//     give the stored run's output again (the series ends where it did).
//   * ResumeRefuses/WhatItCannotContinue.Case/<case>, one test per case:
//     another format, results version or configuration, a shorter series,
//     too few stored segments, two records for one pixel; ResumeAccepts.*:
//     exactly the final segments.
//   * ResumeFreezesTheStatistics.*: a configuration whose statistics cutoff
//     the capping would change (none, or a later date) is refused
//     with a message naming both dates, and resumed with freeze_statistics;
//     one at or before the stored end needs no flag; the same for a test
//     mode rule's extra-mask green median (ModeRule::statistics_end()).
//     ResumeCube.RefusesTo*: the same for a cube.
//
// The sweeps above cut before the pyccd cutoff they run under, so they
// resume with freeze_statistics: the capped run is what they compare with.
//
//==============================================================================

#include <algorithm>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>


#include "ccd/ccd.hpp"
#include "ccd/strategies/mode/pyccd.hpp"
#include "config.hpp"

#include "dump.hpp"
#include "series_corpus.hpp"

namespace
{

using golden::Series;
using golden::expect_same_record;
using golden::resume_on;

struct Counts
{
    std::size_t splits = 0;
    std::size_t continued = 0;   // the machine continued (not a rerun)
};

// Cuts `s` after each index in `ends` and checks resume against the full run
// with its cutoffs capped at the cut.
void sweep(const Series& s, const std::vector<std::size_t>& ends, Counts& counts)
{
    for (const std::size_t last : ends)
    {
        if (!golden::splits_cleanly(s, last))
            continue;
        const std::int64_t T = s.dates[last];
        const std::string at = s.name + " cut after " + std::to_string(T);

        const ccd::PixelOutput stored = golden::recorded_at(s.prefix(last + 1), s.statistics_end());
        const ccd::PixelOutput resumed = resume_on(s, stored);
        const ccd::PixelOutput full = golden::recorded_at(s, std::min(s.statistics_end(), T));
        ++counts.splits;

        EXPECT_EQ(resumed.status, ccd::ResumeStatus::Resumed) << at;
        EXPECT_EQ(golden::dump(resumed.result), golden::dump(full.result)) << at;
        expect_same_record(resumed.record, full.record, at);
        EXPECT_EQ(resumed.header.statistics_end, T) << at;
        EXPECT_EQ(resumed.header.series_end, s.dates.back()) << at;

        const auto& sus = stored.record.standard;
        if (stored.record.procedure == ccd::PixelProcedure::Standard &&
            (sus.exit == ccd::machine::Exit::NoRoomToInitialize ||
             sus.exit == ccd::machine::Exit::InitializeExhausted ||
             sus.exit == ccd::machine::Exit::PeekBeyondEnd ||
             sus.exit == ccd::machine::Exit::SeriesEnd))
            ++counts.continued;
    }
}

void report(const std::string& corpus, const Counts& c)
{
    std::cout << "[ resume   ] " << corpus << ": " << c.splits << " cuts, " << c.continued
              << " continued from the record\n";
}

} // namespace

//------------------------------------------------------------------------------
// ResumeEqualsTheFullRun
//------------------------------------------------------------------------------

// Each sweep over cuts runs as golden::SHARDS tests (series_corpus.hpp): the
// sample in the quick tier, the rest labelled corpus.
class ResumeEqualsTheFullRun : public ::testing::TestWithParam<std::size_t>
{
};

TEST_P(ResumeEqualsTheFullRun, CharacterizationPixelsEveryCut)
{
    Counts c;
    std::size_t ordinal = 0;
    for (const auto& s : golden::characterization_pixels())
        sweep(s, golden::shard_of(s, ordinal++, golden::every(s.size()), GetParam()), c);
    report("characterization", c);
    EXPECT_GT(c.continued, c.splits / 2);
}

TEST_P(ResumeEqualsTheFullRun, CsvPixelsEveryCut)
{
    Counts c;
    std::size_t ordinal = 0;
    for (const auto& s : golden::csv_series(CCD_TEST_DATA_DIR))
        sweep(s, golden::shard_of(s, ordinal++, golden::every(s.size()), GetParam()), c);
    report("csv", c);
    EXPECT_GT(c.continued, c.splits / 2);
}

TEST_P(ResumeEqualsTheFullRun, SyntheticCubesTwelveCuts)
{
    Counts c;
    std::size_t ordinal = 0;
    for (const auto& s : golden::synthetic_series())
        sweep(s, golden::shard_of(s, ordinal++, golden::spread(s.size(), 12), GetParam()), c);
    report("synthetic", c);
    EXPECT_GT(c.continued, 0u);
}

// Cut at T1, resumed to T2, resumed to the end: the full run with its
// cutoffs capped at T1, and the record of each step that full run's.
TEST(ResumeEqualsTheFullRun, AChainOfResumes)
{
    std::size_t chains = 0;
    for (const auto& s : golden::characterization_pixels())
    {
        const std::size_t T = s.size();
        for (std::size_t k = 1; k <= 6; ++k)
        {
            const std::size_t a = k * T / 9, b = (k + 2) * T / 9;
            if (!golden::splits_cleanly(s, a) || !golden::splits_cleanly(s, b))
                continue;
            const std::int64_t T1 = s.dates[a];
            const std::string at = s.name + " cut after " + std::to_string(T1) +
                                   " and " + std::to_string(s.dates[b]);

            const ccd::PixelOutput first = golden::recorded_at(s.prefix(a + 1), s.statistics_end());
            const ccd::PixelOutput second = resume_on(s.prefix(b + 1), first);
            const ccd::PixelOutput third = resume_on(s, second);

            const std::int64_t capped = std::min(s.statistics_end(), T1);
            const ccd::PixelOutput middle = golden::recorded_at(s.prefix(b + 1), capped);
            const ccd::PixelOutput full = golden::recorded_at(s, capped);

            EXPECT_EQ(golden::dump(second.result), golden::dump(middle.result)) << at;
            expect_same_record(second.record, middle.record, at);
            EXPECT_EQ(golden::dump(third.result), golden::dump(full.result)) << at;
            expect_same_record(third.record, full.record, at);
            EXPECT_EQ(third.header.statistics_end, T1) << at;
            ++chains;
        }
    }
    EXPECT_GT(chains, 50u);
}

// A plan without OnePerDate keeps two observations of one date, so a dropped
// day may not say which of them the stored run dropped: such a pixel is rerun
// rather than continued, and either way the result is the full run's.
TEST_P(ResumeEqualsTheFullRun, APlanThatKeepsDuplicateDates)
{
    // The golden duplicates cube, and three more of its regime, whose
    // ambiguous drops come after final segments.
    std::vector<Series> pixels;
    for (const auto& s : golden::synthetic_series())
        if (s.name.rfind("dup:", 0) == 0)
            pixels.push_back(s);
    for (const std::uint64_t seed : {33u, 34u, 35u})
    {
        const golden::Cube c = golden::make_cube(seed, golden::Regime::Duplicates, 4, 5);
        for (std::size_t p = 0; p < c.H * c.W; ++p)
        {
            Series s;
            s.name = "dup" + std::to_string(seed) + ":" + std::to_string(p);
            s.dates = c.dates;
            s.spectral.assign(c.spectral.begin() + p * 7 * c.T, c.spectral.begin() + (p + 1) * 7 * c.T);
            s.qas.assign(c.qas.begin() + p * c.T, c.qas.begin() + (p + 1) * c.T);
            pixels.push_back(std::move(s));
        }
    }

    std::size_t cuts = 0;
    std::size_t ordinal = 0;
    for (const auto& s : pixels)
    {
        // The Standard chain without its one_per_date: the same date kept twice.
        ccd::config::Config given = s.config;
        auto& steps = given.preprocess.at("standard").steps;
        steps.erase(std::remove_if(steps.begin(), steps.end(),
                                   [](const ccd::config::StepConfig& step)
                                   { return step.step == "one_per_date"; }),
                    steps.end());

        const bool more_cuts = s.name.rfind("dup:", 0) != 0;
        for (const std::size_t last :
             golden::shard_of(s, ordinal++, golden::spread(s.size(), more_cuts ? 60 : 12), GetParam()))
        {
            const Series cut = s.prefix(last + 1);
            const std::int64_t T = cut.dates.back();
            const std::string at = s.name + " cut after " + std::to_string(T);

            const ccd::PixelOutput stored = ccd::detect(
                golden::spec_of(s.h, given, true),
                cut.date_view(),
                cut.spectral_view(),
                cut.qa_view());
            const ccd::PixelOutput resumed = ccd::detect(
                golden::spec_of(s.h, given),
                s.date_view(),
                s.spectral_view(),
                s.qa_view(),
                ccd::Resume{stored.header, {stored.record}, {stored.result.models}, true});

            // Every statistics cutoff capped at T: the shared statistics_end, which
            // the green median step reads too.
            const ccd::config::Config capped = golden::with_shared(
                given, "statistics_end", ccd::config::integer(std::min(s.statistics_end(), T)));
            const ccd::FitResult full = ccd::detect(
                golden::spec_of(s.h, capped),
                s.date_view(),
                s.spectral_view(),
                s.qa_view()).result;

            EXPECT_EQ(golden::dump(resumed.result), golden::dump(full)) << at;
            ++cuts;
        }
    }
    EXPECT_GT(cuts, 100u);
}

INSTANTIATE_TEST_SUITE_P(Shards, ResumeEqualsTheFullRun,
                         ::testing::Range(std::size_t{0}, golden::SHARDS),
                         [](const auto& info) { return golden::shard_name(info.param); });

//------------------------------------------------------------------------------
// ResumeOfChangedInput
//------------------------------------------------------------------------------

TEST(ResumeOfChangedInput, AReprocessedObservationIsRerun)
{
    const auto all = golden::characterization_pixels();
    for (const char* name : {"start_catch", "lookback_near_break", "series_end_after_outlier"})
    {
        const Series& s = golden::named(all, name);
        const std::size_t last = 2 * s.size() / 3;
        const std::int64_t T = s.dates[last];
        const ccd::PixelOutput stored = golden::recorded_at(s.prefix(last + 1), s.statistics_end());

        // One value of one observation before T, reprocessed.
        Series changed = s;
        changed.spectral[3 * s.size() + last / 2] += 1.0;

        const ccd::PixelOutput resumed = resume_on(changed, stored);
        const ccd::PixelOutput full = golden::recorded_at(changed, std::min(s.statistics_end(), T));

        EXPECT_EQ(resumed.status, ccd::ResumeStatus::RerunInputChanged) << name;
        EXPECT_EQ(golden::dump(resumed.result), golden::dump(full.result)) << name;
        expect_same_record(resumed.record, full.record, name);
    }
}

TEST(ResumeOfChangedInput, AFillSceneBeforeTheCutIsNotAChange)
{
    const auto all = golden::characterization_pixels();
    const Series& s = golden::named(all, "start_catch");
    const std::size_t last = 2 * s.size() / 3;
    const std::int64_t T = s.dates[last];
    const ccd::PixelOutput stored = golden::recorded_at(s.prefix(last + 1), s.statistics_end());

    // A scene back-filled between two observations, fill for this pixel.
    const std::size_t at = last / 2;
    Series filled;
    filled.name = s.name;
    filled.h = s.h;
    filled.config = s.config;
    const std::size_t N = s.size();
    for (std::size_t t = 0; t < N; ++t)
    {
        if (t == at)
        {
            filled.dates.push_back(s.dates[t - 1] + 1);
            filled.qas.push_back(s.h.qa_fill);
        }
        filled.dates.push_back(s.dates[t]);
        filled.qas.push_back(s.qas[t]);
    }
    filled.spectral.resize(7 * (N + 1));
    for (std::size_t b = 0; b < 7; ++b)
        for (std::size_t t = 0, u = 0; t < N + 1; ++t)
            filled.spectral[b * (N + 1) + t] = t == at ? 0.0 : s.spectral[b * N + u++];

    const ccd::PixelOutput resumed = resume_on(filled, stored);
    const ccd::PixelOutput full = golden::recorded_at(filled, std::min(s.statistics_end(), T));

    EXPECT_EQ(resumed.status, ccd::ResumeStatus::Resumed);
    EXPECT_EQ(golden::dump(resumed.result), golden::dump(full.result));
    expect_same_record(resumed.record, full.record, "filled");
}

//------------------------------------------------------------------------------
// ResumeWithNothingNew
//------------------------------------------------------------------------------

// The same series, and the same series with cloudy observations after it:
// the filters keep nothing new, every guard that failed fails again, and the
// output is the stored run's (the series end's provisional segment written
// again from the record's last test).
TEST(ResumeWithNothingNew, TheStoredOutputAgain)
{
    std::size_t seen_series_end = 0;
    for (const auto& s : golden::characterization_pixels())
    {
        for (const std::size_t last : golden::spread(s.size(), 8))
        {
            if (!golden::splits_cleanly(s, last))
                continue;
            const Series cut = s.prefix(last + 1);
            const std::int64_t T = cut.dates.back();
            const ccd::PixelOutput stored = golden::recorded_at(cut, s.statistics_end());
            const std::string at = s.name + " cut after " + std::to_string(T);

            const ccd::PixelOutput again = resume_on(cut, stored);
            EXPECT_EQ(golden::dump(again.result), golden::dump(golden::detect_at(cut, std::min(s.statistics_end(), T)))) << at;

            Series cloudy = s.prefix(std::min(s.size(), last + 1 + 10));
            for (std::size_t t = last + 1; t < cloudy.size(); ++t)
                cloudy.qas[t] = s.h.qa_cloud;
            const ccd::PixelOutput more = resume_on(cloudy, stored);
            EXPECT_EQ(golden::dump(more.result),
                      golden::dump(golden::detect_at(cloudy, std::min(s.statistics_end(), T)))) << at;

            if (stored.record.standard.exit == ccd::machine::Exit::SeriesEnd)
                ++seen_series_end;
        }
    }
    EXPECT_GT(seen_series_end, 10u);
}

//------------------------------------------------------------------------------
// ResumeRefuses
//------------------------------------------------------------------------------

namespace
{

// start_catch cut two thirds of the way, and its run recorded on the cut.
struct Stored
{
    Series s;
    Series cut;
    ccd::PixelOutput stored;
};

const Stored& stored_cut()
{
    static const Stored st = []
    {
        const auto all = golden::characterization_pixels();
        const Series& s = golden::named(all, "start_catch");
        const Series cut = s.prefix(2 * s.size() / 3);
        return Stored{s, cut, golden::recorded_at(cut, s.statistics_end())};
    }();
    return st;
}

// detect() on `input` resuming the stored run under `header`, with its first
// `segments` segments, throws invalid_argument.
void expect_refused(const ccd::ResumeHeader& header, const Series& input, std::size_t segments)
{
    const Stored& st = stored_cut();
    const std::vector<ccd::ChangeModel> given(
        st.stored.result.models.begin(), st.stored.result.models.begin() + segments);
    EXPECT_THROW(
        ccd::detect(
            golden::spec_of(st.s.h, input.config),
            input.date_view(),
            input.spectral_view(),
            input.qa_view(),
            ccd::Resume{header, {st.stored.record}, {given}, true}),
        std::invalid_argument);
}

// The stored run's final segments.
std::vector<ccd::ChangeModel> final_only(const Stored& st)
{
    return std::vector<ccd::ChangeModel>(
        st.stored.result.models.begin(),
        st.stored.result.models.begin() + st.stored.record.standard.final_segments);
}

// What a resume cannot continue, named.
struct Refusal
{
    const char* name;
    void (*refuse)(const Stored& st);
};

void PrintTo(const Refusal& refusal, std::ostream* os)
{
    *os << refusal.name;
}

const Refusal REFUSALS[] = {
    {"another_format", [](const Stored& st)
     {
         ccd::ResumeHeader h = st.stored.header;
         h.format += 1;
         expect_refused(h, st.s, st.stored.result.models.size());
     }},
    {"another_results_version", [](const Stored& st)
     {
         ccd::ResumeHeader h = st.stored.header;
         h.results_version += 1;
         expect_refused(h, st.s, st.stored.result.models.size());
     }},
    {"another_configuration", [](const Stored& st)
     {
         Series other = st.s;
         other.config = golden::with_shared(st.s.config, "change_probability", ccd::config::real(0.95));
         expect_refused(st.stored.header, other, st.stored.result.models.size());
     }},
    {"a_shorter_series", [](const Stored& st)
     { expect_refused(st.stored.header, st.cut.prefix(st.cut.size() - 1), st.stored.result.models.size()); }},
    {"fewer_than_the_final_segments", [](const Stored& st)
     { expect_refused(st.stored.header, st.s, st.stored.record.standard.final_segments - 1); }},
    // A pixel's resume holds one record and one segment list.
    {"two_records_for_one_pixel", [](const Stored& st)
     {
         EXPECT_THROW(ccd::detect(
             golden::spec_of(st.s.h, st.s.config),
             st.s.date_view(),
             st.s.spectral_view(),
             st.s.qa_view(),
             ccd::Resume{st.stored.header, {st.stored.record, st.stored.record},
                         {final_only(st), final_only(st)}, true}),
                      std::invalid_argument);
     }},
};

class WhatItCannotContinue : public ::testing::TestWithParam<Refusal>
{
};

} // namespace

// One test per refusal, named for it.
TEST_P(WhatItCannotContinue, Case)
{
    const Stored& st = stored_cut();
    ASSERT_GT(st.stored.record.standard.final_segments, 0u);
    GetParam().refuse(st);
}

INSTANTIATE_TEST_SUITE_P(ResumeRefuses, WhatItCannotContinue, ::testing::ValuesIn(REFUSALS),
                         [](const auto& info) { return std::string(info.param.name); });

// What it does accept: the same, with exactly the final segments.
TEST(ResumeAccepts, ExactlyTheFinalSegments)
{
    const Stored& st = stored_cut();
    ASSERT_GT(st.stored.record.standard.final_segments, 0u);
    EXPECT_NO_THROW(ccd::detect(
        golden::spec_of(st.s.h, st.s.config),
        st.s.date_view(),
        st.s.spectral_view(),
        st.s.qa_view(),
        ccd::Resume{st.stored.header, {st.stored.record}, {final_only(st)}, true}));
}

//------------------------------------------------------------------------------
// ResumeFreezesTheStatistics
//------------------------------------------------------------------------------

namespace
{

// start_catch moved to 2006-2026 under `config` (its strategies from
// `registry`), and its run recorded on the series cut two thirds of the way,
// on 2022-12-31.
struct LateCut
{
    Series s;
    Series cut;
    ccd::AlgorithmPtr algorithm;
    ccd::PixelOutput stored;
};

LateCut late_cut(const ccd::config::Config& config,
                 const ccd::config::Registry& registry = ccd::strategies::library_registry())
{
    const auto all = golden::characterization_pixels();
    Series s = golden::named(all, "start_catch");
    for (std::int64_t& day : s.dates)
        day += 5984;
    s.config = config;
    const Series cut = s.prefix(2 * s.size() / 3);
    auto algorithm = std::make_shared<const ccd::Algorithm>(config, registry);
    return LateCut{s, cut, algorithm,
                   ccd::detect(golden::spec_of(cut.h, algorithm, true), cut.date_view(),
                               cut.spectral_view(), cut.qa_view())};
}

// detect() resuming `late` on its whole series; std::invalid_argument's
// message, empty when it runs.
std::string refusal_of(const LateCut& late, bool freeze)
{
    try
    {
        ccd::detect(golden::spec_of(late.s.h, late.algorithm), late.s.date_view(),
                    late.s.spectral_view(), late.s.qa_view(),
                    ccd::Resume{late.stored.header, {late.stored.record},
                                {late.stored.result.models}, freeze});
    }
    catch (const std::invalid_argument& e)
    {
        return e.what();
    }
    return "";
}

// The ccdc preset with the shared statistics_end at `end`.
ccd::config::Config ccdc_until(ccd::config::Value end)
{
    return golden::with_shared(golden::ccdc(), "statistics_end", std::move(end));
}

} // namespace

// The default, no cutoff: refused, naming the stored end and the two ways
// on; with freeze_statistics, the full run with its statistics capped there.
TEST(ResumeFreezesTheStatistics, RefusesToCapNoCutoff)
{
    const LateCut late = late_cut(golden::ccdc());
    ASSERT_EQ(late.stored.header.statistics_end, 738520);   // 2022-12-31
    EXPECT_EQ(refusal_of(late, false),
              "this configuration's statistics cover the whole series; a resume keeps them at "
              "the stored run's end (2022-12-31). Pass freeze_statistics=True to resume that "
              "way, or rerun in full to use the new dates in the statistics");
    EXPECT_EQ(refusal_of(late, true), "");

    const ccd::PixelOutput frozen = ccd::detect(
        golden::spec_of(late.s.h, late.s.config), late.s.date_view(), late.s.spectral_view(),
        late.s.qa_view(),
        ccd::Resume{late.stored.header, {late.stored.record}, {late.stored.result.models}, true});
    const ccd::PixelOutput full = golden::recorded_at(late.s, 738520);
    EXPECT_EQ(golden::dump(frozen.result), golden::dump(full.result));
    expect_same_record(frozen.record, full.record, "frozen");
    EXPECT_EQ(frozen.header.statistics_end, 738520);
}

// A cutoff after the stored end: refused, naming both dates.
TEST(ResumeFreezesTheStatistics, RefusesToCapALaterCutoff)
{
    const LateCut late = late_cut(ccdc_until(std::string("2024-06-30")));
    EXPECT_EQ(refusal_of(late, false),
              "this configuration's statistics run to 2024-06-30; a resume keeps them at the "
              "stored run's end (2022-12-31). Pass freeze_statistics=True to resume that way, "
              "or rerun in full to use the new dates in the statistics");
    EXPECT_EQ(refusal_of(late, true), "");
}

// A cutoff at or before the stored end is not capped: the resume needs no
// flag, and equals the plain full run (the flag changes nothing).
TEST(ResumeFreezesTheStatistics, ACutoffAtOrBeforeTheStoredEndNeedsNoFlag)
{
    for (const std::int64_t end : {golden::PYCCD_STATISTICS_END, std::int64_t{738520}})
    {
        const std::string at = "statistics_end " + std::to_string(end);
        const LateCut late = late_cut(ccdc_until(ccd::config::integer(end)));
        EXPECT_EQ(refusal_of(late, false), "") << at;
        const ccd::PixelOutput resumed = resume_on(late.s, late.stored);
        const ccd::PixelOutput plain = ccd::detect(
            golden::spec_of(late.s.h, late.s.config), late.s.date_view(),
            late.s.spectral_view(), late.s.qa_view(),
            ccd::Resume{late.stored.header, {late.stored.record}, {late.stored.result.models}});
        const ccd::PixelOutput full = golden::recorded_at(late.s, end);
        EXPECT_EQ(golden::dump(plain.result), golden::dump(full.result)) << at;
        EXPECT_EQ(golden::dump(resumed.result), golden::dump(full.result)) << at;
        expect_same_record(plain.record, full.record, at);
    }
}

namespace
{

// mode "late-green-median": pyccd's choice, and a Standard pixel's extra mask
// a green median step (row 1, 50 DN over the median of the observations
// dated through median_through, at every date), which ModeRule::statistics_end()
// names.
struct LateGreenMedianMode final : ccd::ModeRule
{
    struct Params
    {
        std::int64_t median_through = 0;
    };

    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s = {
            {"median_through", &Params::median_through, ccd::config::Unit::Date,
             ccd::config::any(), "the last date the green median reads"},
        };
        return s;
    }

    explicit LateGreenMedianMode(const Params& p) : p_(p) {}

    ccd::PixelMode choose(const ccd::QaStatistics& statistics) const override
    {
        ccd::PixelMode mode = pyccd_.choose(statistics);
        if (mode.procedure == ccd::PixelProcedure::Standard)
            mode.extra_mask.push_back(ccd::preprocess::Step::green_median(
                1, 50.0, p_.median_through, ccd::GreenFilterScope::All));
        return mode;
    }

    std::optional<std::int64_t> statistics_end() const override { return p_.median_through; }

private:
    Params p_;
    ccd::strategies::PyccdMode pyccd_{ccd::strategies::PyccdMode::Params{}};
};

const ccd::config::Registry& mode_registry()
{
    static const ccd::config::Registry r = []
    {
        ccd::config::Registry r;
        ccd::strategies::register_core_strategies(r);
        r.add<ccd::ModeRule, LateGreenMedianMode>("late-green-median");
        return r;
    }();
    return r;
}

// ccdc_until(2022-12-31) with mode "late-green-median" through `through`.
ccd::config::Config late_green_median(ccd::config::Value through)
{
    ccd::config::Config config = ccdc_until(ccd::config::integer(738520));
    config.stages["mode"] = {"late-green-median", {{"median_through", std::move(through)}}};
    return config;
}

// detect() on `late`'s whole series under `config`, recorded.
ccd::PixelOutput full_run(const LateCut& late, const ccd::config::Config& config)
{
    return ccd::detect(
        golden::spec_of(late.s.h, std::make_shared<const ccd::Algorithm>(config, mode_registry()),
                        true),
        late.s.date_view(), late.s.spectral_view(), late.s.qa_view());
}

// detect() resuming `late` on its whole series.
ccd::PixelOutput resumed(const LateCut& late, bool freeze)
{
    return ccd::detect(
        golden::spec_of(late.s.h, late.algorithm), late.s.date_view(), late.s.spectral_view(),
        late.s.qa_view(),
        ccd::Resume{late.stored.header, {late.stored.record}, {late.stored.result.models}, freeze});
}

} // namespace

// The mode rule's extra-mask green median past the stored end: refused,
// naming that cutoff; with freeze_statistics, the full run with its date
// capped at the stored end.
TEST(ResumeFreezesTheStatistics, RefusesToCapTheModeRulesExtraMask)
{
    const LateCut late = late_cut(late_green_median(std::string("2024-06-30")), mode_registry());
    ASSERT_EQ(late.stored.header.statistics_end, 738520);   // 2022-12-31
    EXPECT_EQ(refusal_of(late, false),
              "the mode rule's extra mask: this configuration's statistics run to 2024-06-30; a "
              "resume keeps them at the stored run's end (2022-12-31). Pass "
              "freeze_statistics=True to resume that way, or rerun in full to use the new dates "
              "in the statistics");
    EXPECT_EQ(refusal_of(late, true), "");

    const ccd::PixelOutput frozen = resumed(late, true);
    const ccd::PixelOutput full = full_run(late, late_green_median(ccd::config::integer(738520)));
    EXPECT_EQ(golden::dump(frozen.result), golden::dump(full.result));
    expect_same_record(frozen.record, full.record, "frozen");
    // The cap changes the run: the uncapped one differs.
    EXPECT_NE(golden::dump(full.result), golden::dump(full_run(late, late.s.config).result));
}

// Its date at or before the stored end: the resume needs no flag and equals
// the full run.
TEST(ResumeFreezesTheStatistics, AModeRuleCutoffAtOrBeforeTheStoredEndNeedsNoFlag)
{
    for (const std::int64_t through : {golden::PYCCD_STATISTICS_END, std::int64_t{738520}})
    {
        const std::string at = "median_through " + std::to_string(through);
        const LateCut late =
            late_cut(late_green_median(ccd::config::integer(through)), mode_registry());
        EXPECT_EQ(refusal_of(late, false), "") << at;
        const ccd::PixelOutput plain = resumed(late, false);
        const ccd::PixelOutput full = full_run(late, late.s.config);
        EXPECT_EQ(golden::dump(plain.result), golden::dump(full.result)) << at;
        expect_same_record(plain.record, full.record, at);
    }
}

//------------------------------------------------------------------------------
// ResumeCube
//------------------------------------------------------------------------------

namespace
{

// The golden c8 cube cut after observation `last`, as a (H, W, 7, T) block.
struct CutCube
{
    golden::Cube cube;
    std::size_t T = 0;
    std::vector<double> spectral;
    std::vector<std::uint8_t> qas;

    CutCube(const golden::Cube& c, std::size_t count) : cube(c), T(count)
    {
        const std::size_t N = c.H * c.W;
        spectral.resize(N * 7 * T);
        qas.resize(N * T);
        for (std::size_t p = 0; p < N; ++p)
        {
            for (std::size_t b = 0; b < 7; ++b)
                std::copy_n(c.spectral.begin() + (p * 7 + b) * c.T, T,
                            spectral.begin() + (p * 7 + b) * T);
            std::copy_n(c.qas.begin() + p * c.T, T, qas.begin() + p * T);
        }
    }

    ccd::ArrayView<const std::int64_t, 1> dates() const
    {
        return ccd::ArrayView<const std::int64_t, 1>::contiguous(cube.dates.data(), {T});
    }
    ccd::ArrayView<const double, 4> block() const
    {
        return ccd::ArrayView<const double, 4>::contiguous(spectral.data(), {cube.H, cube.W, 7, T});
    }
    ccd::ArrayView<const std::uint8_t, 3> qa() const
    {
        return ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {cube.H, cube.W, T});
    }
};

} // namespace

TEST(ResumeCube, EachPixelAsResumeAlone)
{
    const golden::Cube c = golden::make_cube(12, golden::Regime::Cadence8, 6, 8);
    const ccd::DataSpec h;
    const std::size_t N = c.H * c.W;

    for (const std::size_t count : {c.T / 3, c.T / 2, (3 * c.T) / 4})
    {
        const CutCube cut(c, count);
        const CutCube whole(c, c.T);

        for (const int threads : {1, 4})
        {
            const ccd::CubeOutput stored = ccd::detect_cube(
                golden::spec_of(h, golden::pyccd(), true),
                cut.dates(),
                cut.block(),
                cut.qa(),
                threads);
            ASSERT_TRUE(stored.failures.empty());
            ASSERT_EQ(stored.pixels.size(), N);

            ccd::Resume from;
            from.header = stored.header;
            from.freeze_statistics = true;
            from.records.resize(N);
            from.segments.resize(N);
            for (const auto& px : stored.pixels)
            {
                const std::size_t at = px.row * c.W + px.col;
                from.records[at] = px.record;
                from.segments[at] = px.result.models;
            }

            const ccd::CubeOutput resumed = ccd::detect_cube(
                golden::spec_of(h),
                whole.dates(),
                whole.block(),
                whole.qa(),
                threads,
                &from);
            ASSERT_TRUE(resumed.failures.empty());
            ASSERT_EQ(resumed.pixels.size(), N);
            EXPECT_EQ(resumed.header.statistics_end, stored.header.series_end);

            for (std::size_t p = 0; p < N; ++p)
            {
                const std::string at = "pixel " + std::to_string(p) + " cut " +
                                       std::to_string(count) + " threads " + std::to_string(threads);
                Series s;
                s.name = at;
                s.dates = c.dates;
                s.spectral.assign(c.spectral.begin() + p * 7 * c.T, c.spectral.begin() + (p + 1) * 7 * c.T);
                s.qas.assign(c.qas.begin() + p * c.T, c.qas.begin() + (p + 1) * c.T);

                const ccd::PixelOutput alone = golden::recorded_at(s.prefix(count), s.statistics_end());
                expect_same_record(stored.pixels[p].record, alone.record, at + " stored");

                const ccd::PixelOutput one = resume_on(s, alone);
                EXPECT_EQ(golden::dump(resumed, resumed.pixels[p]), golden::dump(one.result)) << at;
                expect_same_record(resumed.pixels[p].record, one.record, at);
                EXPECT_EQ(resumed.pixels[p].status, one.status) << at;
            }
        }
    }
}

TEST(ResumeCube, APixelWithTooFewSegmentsFailsAlone)
{
    const golden::Cube c = golden::make_cube(12, golden::Regime::Cadence8, 6, 8);
    const std::size_t N = c.H * c.W;
    const CutCube cut(c, c.T / 2);
    const CutCube whole(c, c.T);
    const ccd::CubeOutput stored = ccd::detect_cube(
        golden::spec_of(ccd::DataSpec{}, golden::pyccd(), true),
        cut.dates(),
        cut.block(),
        cut.qa(),
        1);

    ccd::Resume from;
    from.header = stored.header;
    from.freeze_statistics = true;
    from.records.resize(N);
    from.segments.resize(N);
    std::size_t victim = N;
    for (const auto& px : stored.pixels)
    {
        const std::size_t at = px.row * c.W + px.col;
        from.records[at] = px.record;
        from.segments[at] = px.result.models;
        if (victim == N && px.record.standard.final_segments > 0)
            victim = at;
    }
    ASSERT_LT(victim, N);
    from.segments[victim].clear();

    const ccd::CubeOutput resumed = ccd::detect_cube(
        golden::spec_of(ccd::DataSpec{}),
        whole.dates(),
        whole.block(),
        whole.qa(),
        2,
        &from);
    ASSERT_EQ(resumed.failures.size(), 1u);
    EXPECT_EQ(resumed.failures[0].row * c.W + resumed.failures[0].col, victim);
    EXPECT_EQ(resumed.pixels.size(), N - 1);

    // The call itself refuses records that do not cover the cube.
    from.records.pop_back();
    EXPECT_THROW(ccd::detect_cube(
        golden::spec_of(ccd::DataSpec{}),
        whole.dates(),
        whole.block(),
        whole.qa(),
        1,
        &from),
                 std::invalid_argument);
}

// A cube's resume is refused as a pixel's, before any pixel runs.
TEST(ResumeCube, RefusesToCapNoCutoff)
{
    const golden::Cube c = golden::make_cube(12, golden::Regime::Cadence8, 2, 3);
    const std::size_t N = c.H * c.W;
    const CutCube cut(c, c.T / 2);
    const CutCube whole(c, c.T);
    const ccd::CubeOutput stored = ccd::detect_cube(
        golden::spec_of(ccd::DataSpec{}, golden::ccdc(), true), cut.dates(), cut.block(), cut.qa(), 1);

    ccd::Resume from;
    from.header = stored.header;
    from.records.resize(N);
    from.segments.resize(N);
    for (const auto& px : stored.pixels)
    {
        from.records[px.row * c.W + px.col] = px.record;
        from.segments[px.row * c.W + px.col] = px.result.models;
    }
    const auto resumed = [&]
    {
        return ccd::detect_cube(golden::spec_of(ccd::DataSpec{}, golden::ccdc()), whole.dates(),
                                whole.block(), whole.qa(), 2, &from);
    };
    try
    {
        resumed();
        ADD_FAILURE() << "a resume capping no cutoff ran";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_EQ(std::string(e.what()).rfind(
                      "this configuration's statistics cover the whole series; a resume keeps "
                      "them at the stored run's end (", 0), 0u) << e.what();
    }

    from.freeze_statistics = true;
    const ccd::CubeOutput frozen = resumed();
    EXPECT_TRUE(frozen.failures.empty());
    EXPECT_EQ(frozen.pixels.size(), N);
    EXPECT_EQ(frozen.header.statistics_end, stored.header.series_end);
}

