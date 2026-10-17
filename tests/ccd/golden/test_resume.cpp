//==============================================================================
//
// Resume equals the full run.
//
// For a pixel series P and a date T, the stored run is detect_recorded() on
// the observations up to T; its continuation is resume() on the whole of P
// with that record and that run's segments. The contract (ccd.hpp, resume())
// is that the continuation equals, bit for bit, detect() on P with every
// statistics cutoff capped at T: every segment and the processing mask (the
// hexfloat dump of golden.hpp), and the record it writes equals the one a
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
//   * ResumeRefuses.*: another format, results version or options, a
//     shorter series, too few stored segments.
//
//==============================================================================

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

#include "golden.hpp"
#include "series_corpus.hpp"

namespace
{

using golden::Series;

std::string dump(const ccd::FitResult& r)
{
    std::string out;
    golden::dump_pixel(out, "pixel", r);
    return out;
}

ccd::Resumed resume_on(const Series& whole, const ccd::Recorded& stored)
{
    return ccd::resume(whole.date_view(), whole.spectral_view(), whole.qa_view(),
                       whole.h, whole.l, stored.header, stored.record, stored.result.models);
}

ccd::Resumed resume_on(const Series& whole, const ccd::Resumed& stored)
{
    return ccd::resume(whole.date_view(), whole.spectral_view(), whole.qa_view(),
                       whole.h, whole.l, stored.header, stored.record, stored.result.models);
}

// Two records say the same, field by field.
void expect_same_record(const ccd::PixelRecord& a, const ccd::PixelRecord& b, const std::string& at)
{
    EXPECT_EQ(a.input_fingerprint, b.input_fingerprint) << at;
    EXPECT_EQ(a.procedure, b.procedure) << at;
    EXPECT_EQ(a.excluded_days, b.excluded_days) << at;
    const auto& x = a.standard;
    const auto& y = b.standard;
    EXPECT_EQ(x.exit, y.exit) << at;
    EXPECT_EQ(x.window.start, y.window.start) << at;
    EXPECT_EQ(x.window.stop, y.window.stop) << at;
    EXPECT_EQ(x.prev_break, y.prev_break) << at;
    EXPECT_EQ(x.fit_stop, y.fit_stop) << at;
    EXPECT_EQ(x.grow, y.grow) << at;
    EXPECT_EQ(x.num_coef, y.num_coef) << at;
    EXPECT_EQ(x.magnitude, y.magnitude) << at;
    EXPECT_EQ(x.final_segments, y.final_segments) << at;
}

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

        const ccd::Recorded stored = golden::recorded_at(s.prefix(last + 1), s.h.STAT_ORD);
        const ccd::Resumed resumed = resume_on(s, stored);
        const ccd::Recorded full = golden::recorded_at(s, std::min(s.h.STAT_ORD, T));
        ++counts.splits;

        EXPECT_EQ(resumed.status, ccd::ResumeStatus::Resumed) << at;
        EXPECT_EQ(dump(resumed.result), dump(full.result)) << at;
        expect_same_record(resumed.record, full.record, at);
        EXPECT_EQ(resumed.header.statistics_end, T) << at;
        EXPECT_EQ(resumed.header.series_end, s.dates.back()) << at;

        const auto& sus = stored.record.standard;
        if (stored.record.procedure == ccd::PixelProcedure::Standard &&
            (sus.exit == ccd::standard::Exit::NoRoomToInitialize ||
             sus.exit == ccd::standard::Exit::InitializeExhausted ||
             sus.exit == ccd::standard::Exit::PeekBeyondEnd ||
             sus.exit == ccd::standard::Exit::SeriesEnd))
            ++counts.continued;
    }
}

void report(const std::string& corpus, const Counts& c)
{
    std::cout << "[ resume   ] " << corpus << ": " << c.splits << " cuts, " << c.continued
              << " continued from the record\n";
}

const Series& named(const std::vector<Series>& all, const std::string& name)
{
    for (const auto& s : all)
        if (s.name == name)
            return s;
    throw std::runtime_error("no series " + name);
}

} // namespace

//------------------------------------------------------------------------------
// ResumeEqualsTheFullRun
//------------------------------------------------------------------------------

TEST(ResumeEqualsTheFullRun, CharacterizationPixelsEveryCut)
{
    Counts c;
    for (const auto& s : golden::characterization_series())
        sweep(s, golden::every(s.size()), c);
    report("characterization", c);
    EXPECT_GT(c.continued, c.splits / 2);
}

TEST(ResumeEqualsTheFullRun, CsvPixelsEveryCut)
{
    Counts c;
    for (const auto& s : golden::csv_series(CCD_PREFIX_DATA_DIR))
        sweep(s, golden::every(s.size()), c);
    report("csv", c);
    EXPECT_GT(c.continued, c.splits / 2);
}

TEST(ResumeEqualsTheFullRun, SyntheticCubesTwelveCuts)
{
    Counts c;
    for (const auto& s : golden::synthetic_series())
        sweep(s, golden::spread(s.size(), 12), c);
    report("synthetic", c);
    EXPECT_GT(c.continued, 0u);
}

// Cut at T1, resumed to T2, resumed to the end: the full run with its
// cutoffs capped at T1, and the record of each step that full run's.
TEST(ResumeEqualsTheFullRun, AChainOfResumes)
{
    std::size_t chains = 0;
    for (const auto& s : golden::characterization_series())
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

            const ccd::Recorded first = golden::recorded_at(s.prefix(a + 1), s.h.STAT_ORD);
            const ccd::Resumed second = resume_on(s.prefix(b + 1), first);
            const ccd::Resumed third = resume_on(s, second);

            const std::int64_t capped = std::min(s.h.STAT_ORD, T1);
            const ccd::Recorded middle = golden::recorded_at(s.prefix(b + 1), capped);
            const ccd::Recorded full = golden::recorded_at(s, capped);

            EXPECT_EQ(dump(second.result), dump(middle.result)) << at;
            expect_same_record(second.record, middle.record, at);
            EXPECT_EQ(dump(third.result), dump(full.result)) << at;
            expect_same_record(third.record, full.record, at);
            EXPECT_EQ(third.header.statistics_end, T1) << at;
            ++chains;
        }
    }
    EXPECT_GT(chains, 50u);
}

// A plan without Dedupe keeps two observations of one date, so a dropped
// day may not say which of them the stored run dropped: such a pixel is rerun
// rather than continued, and either way the result is the full run's.
TEST(ResumeEqualsTheFullRun, APlanThatKeepsDuplicateDates)
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
    for (const auto& s : pixels)
    {

        auto plan = std::make_shared<ccd::preprocess::PreprocessPlan>(
            ccd::preprocess::default_plan(s.h));
        ccd::preprocess::Chain chain;
        for (ccd::index_t i = 0; i < plan->standard.chain.size(); ++i)
            if (plan->standard.chain.steps[i].kind != ccd::preprocess::StepKind::Dedupe)
                chain.push_back(plan->standard.chain.steps[i]);
        plan->standard.chain = chain;
        const ccd::PreprocessPlanPtr given = plan;

        const bool more_cuts = s.name.rfind("dup:", 0) != 0;
        for (const std::size_t last : golden::spread(s.size(), more_cuts ? 60 : 12))
        {
            if (!golden::splits_cleanly(s, last))
                continue;
            const Series cut = s.prefix(last + 1);
            const std::int64_t T = cut.dates.back();
            const std::string at = s.name + " cut after " + std::to_string(T);

            const ccd::Recorded stored = ccd::detect_recorded(
                cut.date_view(), cut.spectral_view(), cut.qa_view(), s.h, s.l, nullptr, given);
            const ccd::Resumed resumed = ccd::resume(
                s.date_view(), s.spectral_view(), s.qa_view(), s.h, s.l,
                stored.header, stored.record, stored.result.models, nullptr, given);

            // Every statistics cutoff capped at T: STAT_ORD and the plan's
            // own GreenMedian dates.
            ccd::HarmonicOptions capped = s.h;
            capped.STAT_ORD = std::min(s.h.STAT_ORD, T);
            auto capped_plan = std::make_shared<ccd::preprocess::PreprocessPlan>(*plan);
            for (auto* procedure : {&capped_plan->standard, &capped_plan->snow,
                                    &capped_plan->insufficient_clear})
                for (ccd::index_t i = 0; i < procedure->chain.size(); ++i)
                    if (procedure->chain.steps[i].kind == ccd::preprocess::StepKind::GreenMedian)
                        procedure->chain.steps[i].max_date =
                            std::min(procedure->chain.steps[i].max_date, T);
            const ccd::FitResult full = ccd::detect(
                s.date_view(), s.spectral_view(), s.qa_view(), capped, s.l, nullptr,
                capped_plan);

            EXPECT_EQ(dump(resumed.result), dump(full)) << at;
            ++cuts;
        }
    }
    EXPECT_GT(cuts, 100u);
}

//------------------------------------------------------------------------------
// ResumeOfChangedInput
//------------------------------------------------------------------------------

TEST(ResumeOfChangedInput, AReprocessedObservationIsRerun)
{
    const auto all = golden::characterization_series();
    for (const char* name : {"start_catch", "lookback_near_break", "series_end_after_outlier"})
    {
        const Series& s = named(all, name);
        const std::size_t last = 2 * s.size() / 3;
        const std::int64_t T = s.dates[last];
        const ccd::Recorded stored = golden::recorded_at(s.prefix(last + 1), s.h.STAT_ORD);

        // One value of one observation before T, reprocessed.
        Series changed = s;
        changed.spectral[3 * s.size() + last / 2] += 1.0;

        const ccd::Resumed resumed = resume_on(changed, stored);
        const ccd::Recorded full = golden::recorded_at(changed, std::min(s.h.STAT_ORD, T));

        EXPECT_EQ(resumed.status, ccd::ResumeStatus::RerunInputChanged) << name;
        EXPECT_EQ(dump(resumed.result), dump(full.result)) << name;
        expect_same_record(resumed.record, full.record, name);
    }
}

TEST(ResumeOfChangedInput, AFillSceneBeforeTheCutIsNotAChange)
{
    const auto all = golden::characterization_series();
    const Series& s = named(all, "start_catch");
    const std::size_t last = 2 * s.size() / 3;
    const std::int64_t T = s.dates[last];
    const ccd::Recorded stored = golden::recorded_at(s.prefix(last + 1), s.h.STAT_ORD);

    // A scene back-filled between two observations, fill for this pixel.
    const std::size_t at = last / 2;
    Series filled;
    filled.name = s.name;
    filled.h = s.h;
    filled.l = s.l;
    const std::size_t N = s.size();
    for (std::size_t t = 0; t < N; ++t)
    {
        if (t == at)
        {
            filled.dates.push_back(s.dates[t - 1] + 1);
            filled.qas.push_back(s.h.QA_FILL);
        }
        filled.dates.push_back(s.dates[t]);
        filled.qas.push_back(s.qas[t]);
    }
    filled.spectral.resize(7 * (N + 1));
    for (std::size_t b = 0; b < 7; ++b)
        for (std::size_t t = 0, u = 0; t < N + 1; ++t)
            filled.spectral[b * (N + 1) + t] = t == at ? 0.0 : s.spectral[b * N + u++];

    const ccd::Resumed resumed = resume_on(filled, stored);
    const ccd::Recorded full = golden::recorded_at(filled, std::min(s.h.STAT_ORD, T));

    EXPECT_EQ(resumed.status, ccd::ResumeStatus::Resumed);
    EXPECT_EQ(dump(resumed.result), dump(full.result));
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
    for (const auto& s : golden::characterization_series())
    {
        for (const std::size_t last : golden::spread(s.size(), 8))
        {
            if (!golden::splits_cleanly(s, last))
                continue;
            const Series cut = s.prefix(last + 1);
            const std::int64_t T = cut.dates.back();
            const ccd::Recorded stored = golden::recorded_at(cut, s.h.STAT_ORD);
            const std::string at = s.name + " cut after " + std::to_string(T);

            const ccd::Resumed again = resume_on(cut, stored);
            EXPECT_EQ(dump(again.result), dump(golden::detect_at(cut, std::min(s.h.STAT_ORD, T)))) << at;

            Series cloudy = s.prefix(std::min(s.size(), last + 1 + 10));
            for (std::size_t t = last + 1; t < cloudy.size(); ++t)
                cloudy.qas[t] = s.h.QA_CLOUD;
            const ccd::Resumed more = resume_on(cloudy, stored);
            EXPECT_EQ(dump(more.result),
                      dump(golden::detect_at(cloudy, std::min(s.h.STAT_ORD, T)))) << at;

            if (stored.record.standard.exit == ccd::standard::Exit::SeriesEnd)
                ++seen_series_end;
        }
    }
    EXPECT_GT(seen_series_end, 10u);
}

//------------------------------------------------------------------------------
// ResumeRefuses
//------------------------------------------------------------------------------

TEST(ResumeRefuses, WhatItCannotContinue)
{
    const auto all = golden::characterization_series();
    const Series& s = named(all, "start_catch");
    const Series cut = s.prefix(2 * s.size() / 3);
    const ccd::Recorded stored = golden::recorded_at(cut, s.h.STAT_ORD);
    ASSERT_GT(stored.record.standard.final_segments, 0u);

    const auto refuse = [&](ccd::ResumeHeader header, const Series& input,
                            const ccd::HarmonicOptions& h, std::size_t segments)
    {
        const std::vector<ccd::ChangeModel> given(
            stored.result.models.begin(), stored.result.models.begin() + segments);
        EXPECT_THROW(
            ccd::resume(input.date_view(), input.spectral_view(), input.qa_view(), h, input.l,
                        header, stored.record, given),
            std::invalid_argument);
    };
    const std::size_t all_segments = stored.result.models.size();

    ccd::ResumeHeader h = stored.header;
    h.format += 1;
    refuse(h, s, s.h, all_segments);

    h = stored.header;
    h.results_version += 1;
    refuse(h, s, s.h, all_segments);

    ccd::HarmonicOptions other = s.h;
    other.CHANGE_THRESHOLD += 1.0;
    refuse(stored.header, s, other, all_segments);

    refuse(stored.header, cut.prefix(cut.size() - 1), s.h, all_segments);
    refuse(stored.header, s, s.h, stored.record.standard.final_segments - 1);

    // What it does accept: the same, with exactly the final segments.
    const std::vector<ccd::ChangeModel> final_only(
        stored.result.models.begin(),
        stored.result.models.begin() + stored.record.standard.final_segments);
    EXPECT_NO_THROW(ccd::resume(s.date_view(), s.spectral_view(), s.qa_view(), s.h, s.l,
                                stored.header, stored.record, final_only));
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
    const ccd::HarmonicOptions h;
    const ccd::LassoOptions l;
    const std::size_t N = c.H * c.W;

    for (const std::size_t count : {c.T / 3, c.T / 2, (3 * c.T) / 4})
    {
        const CutCube cut(c, count);
        const CutCube whole(c, c.T);

        for (const int threads : {1, 4})
        {
            const ccd::CubeResult stored = ccd::detect_cube(
                cut.dates(), cut.block(), cut.qa(), h, l, threads, true, nullptr, nullptr, true);
            ASSERT_TRUE(stored.failures.empty());
            ASSERT_EQ(stored.pixels.size(), N);

            ccd::CubeResumeFrom from;
            from.header = stored.header;
            from.pixels.resize(N);
            from.segments.resize(N);
            for (const auto& px : stored.pixels)
            {
                const std::size_t at = px.row * c.W + px.col;
                from.pixels[at] = px.record;
                from.segments[at] = px.result.models;
            }

            const ccd::CubeResult resumed = ccd::resume_cube(
                whole.dates(), whole.block(), whole.qa(), h, l, from, threads);
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

                const ccd::Recorded alone = golden::recorded_at(s.prefix(count), h.STAT_ORD);
                expect_same_record(stored.pixels[p].record, alone.record, at + " stored");

                const ccd::Resumed one = resume_on(s, alone);
                EXPECT_EQ(dump(resumed.pixels[p].result), dump(one.result)) << at;
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
    const ccd::CubeResult stored = ccd::detect_cube(
        cut.dates(), cut.block(), cut.qa(), ccd::HarmonicOptions{}, ccd::LassoOptions{},
        1, true, nullptr, nullptr, true);

    ccd::CubeResumeFrom from;
    from.header = stored.header;
    from.pixels.resize(N);
    from.segments.resize(N);
    std::size_t victim = N;
    for (const auto& px : stored.pixels)
    {
        const std::size_t at = px.row * c.W + px.col;
        from.pixels[at] = px.record;
        from.segments[at] = px.result.models;
        if (victim == N && px.record.standard.final_segments > 0)
            victim = at;
    }
    ASSERT_LT(victim, N);
    from.segments[victim].clear();

    const ccd::CubeResult resumed = ccd::resume_cube(
        whole.dates(), whole.block(), whole.qa(), ccd::HarmonicOptions{}, ccd::LassoOptions{}, from, 2);
    ASSERT_EQ(resumed.failures.size(), 1u);
    EXPECT_EQ(resumed.failures[0].row * c.W + resumed.failures[0].col, victim);
    EXPECT_EQ(resumed.pixels.size(), N - 1);

    // The call itself refuses records that do not cover the cube.
    from.pixels.pop_back();
    EXPECT_THROW(ccd::resume_cube(whole.dates(), whole.block(), whole.qa(), ccd::HarmonicOptions{},
                                  ccd::LassoOptions{}, from, 1),
                 std::invalid_argument);
}
