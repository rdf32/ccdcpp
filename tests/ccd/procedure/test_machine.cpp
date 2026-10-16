//==============================================================================
//
// The Standard machine's routing, from hand-built records, its exit reasons,
// on whole pixels, and the Standard detector's per-pixel choice.
//
//   * Routing: each guard between phases (room_to_initialize, peek_fits,
//     start_catch_due) and each routing step (after_stable_start,
//     after_lookback, peek_guard, restart, end_catch, start_catch,
//     prepare) on a State built by hand at a chosen cursor, against the
//     phase, the exit and the record the procedure's description gives.
//   * Exits: standard::run() on the characterisation pixels that end the
//     segment loop each way, with the same preprocessing ccd::detect() runs,
//     against the Exit the run leaves in the record. The result of each run
//     is also checked against detect()'s, field for field.
//   * Detector::select(): the clear percentage first, then the snow
//     percentage, each threshold on its >= side.
//
// The outputs themselves are pinned by the golden and transitions fixtures;
// these tests say which route moved if one ever does.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/preprocess/preprocess.hpp"
#include "ccd/procedure/machine.hpp"
#include "ccd/procedure/standard.hpp"
#include "ccd/regression/lasso_solver.hpp"

#include "../golden/characterisation.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;
using ccd::standard::Exit;
using ccd::standard::Phase;

//------------------------------------------------------------------------------
// A hand-built record
//
// `n` clear observations 16 days apart from `first_day` with a smooth
// seasonal signal, every one kept by the mask, and a Context over them. The record's phase, cursor,
// cadence and segments are set by each test.
//------------------------------------------------------------------------------
struct Record
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, n)
    std::vector<std::uint8_t> qas;
    ccd::HarmonicOptions options;
    ccd::LassoOptions lasso_options;

    ccd::HarmonicWorkspace workspace;
    ccd::LassoWorkspace lasso{0};
    ccd::LassoSolver solver;
    ccd::standard::Context context;
    ccd::standard::State state;

    explicit Record(index_t n, std::int64_t first_day = 730120)
        : dates(make_dates(n, first_day)),
          spectral(make_spectral(n)),
          qas(n, 0),
          workspace(
              ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n}),
              ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, n}),
              ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {n}),
              options),
          solver(lasso_options),
          context{options, workspace, lasso, solver}
    {
        ccd::ProcessingMask mask(n);
        for (index_t i = 0; i < n; ++i)
            mask.set(i, true);
        state.reset(workspace, std::move(mask));
    }

    // The record at a cursor, with a cadence, past Prepare.
    void at(index_t start, index_t stop, index_t prev_break, index_t peek = 6)
    {
        state.cursor = ccd::Cursor{ccd::Window(start, stop), prev_break};
        state.cadence = ccd::standard::Cadence{peek, options.CHANGE_THRESHOLD};
    }

    // A segment already written, as Extend or StartCatch leaves one.
    void with_segment()
    {
        state.segments.push_back(ccd::ChangeModel{});
    }

    static std::vector<std::int64_t> make_dates(index_t n, std::int64_t first_day)
    {
        std::vector<std::int64_t> d(n);
        for (index_t i = 0; i < n; ++i)
            d[i] = first_day + 16 * static_cast<std::int64_t>(i);
        return d;
    }

    static std::vector<double> make_spectral(index_t n)
    {
        std::vector<double> s(7 * n);
        for (index_t b = 0; b < 7; ++b)
            for (index_t i = 0; i < n; ++i)
                s[b * n + i] =
                    1000.0 + 100.0 * b +
                    200.0 * std::sin(2.0 * 3.141592653589793 * 16.0 * i / 365.25) +
                    (i % 3) * 7.0;
        return s;
    }
};

//------------------------------------------------------------------------------
// Guards
//------------------------------------------------------------------------------

TEST(StandardRouting, RoomToInitializeIsAMeowWindowAfterTheCursor)
{
    Record r(40);   // MEOW_SIZE 12
    r.at(16, 28, 16);
    EXPECT_TRUE(ccd::standard::room_to_initialize(r.context, r.state));   // 28 + 12 == 40
    r.at(17, 29, 17);
    EXPECT_FALSE(ccd::standard::room_to_initialize(r.context, r.state));
}

TEST(StandardRouting, PeekFitsIsAFullPeekAfterTheWindow)
{
    Record r(40);
    r.at(0, 34, 0, 6);
    EXPECT_TRUE(ccd::standard::peek_fits(r.state));    // 34 + 6 == 40
    r.at(0, 35, 0, 6);
    EXPECT_FALSE(ccd::standard::peek_fits(r.state));
    r.at(0, 30, 0, 11);
    EXPECT_FALSE(ccd::standard::peek_fits(r.state));   // the cadence's peek, not PEEK_SIZE
}

TEST(StandardRouting, StartCatchIsDueOnlyBeforeTheFirstSegmentPastAPeek)
{
    Record r(60);
    r.at(7, 19, 0, 6);
    EXPECT_TRUE(ccd::standard::start_catch_due(r.state));    // 7 - 0 > 6
    r.at(6, 18, 0, 6);
    EXPECT_FALSE(ccd::standard::start_catch_due(r.state));   // 6 - 0 == 6

    // A later segment never gets one, however long the gap.
    r.with_segment();
    r.at(40, 52, 20, 6);
    EXPECT_FALSE(ccd::standard::start_catch_due(r.state));
}

//------------------------------------------------------------------------------
// Routing steps
//------------------------------------------------------------------------------

TEST(StandardRouting, AfterAStableStartLookbackRunsOnlyAboveTheBreak)
{
    Record r(60);
    r.at(3, 15, 0);
    EXPECT_EQ(ccd::standard::after_stable_start(r.context, r.state), Phase::Lookback);

    // At the break: straight on to the start catch and peek guards.
    r.at(0, 12, 0);
    EXPECT_EQ(ccd::standard::after_stable_start(r.context, r.state), Phase::Extend);
    EXPECT_EQ(r.state.exit, Exit::None);
}

TEST(StandardRouting, AfterLookbackTheStartCatchComesBeforeThePeekGuard)
{
    Record r(30);
    // Gap 8 > peek 6 and no segment, though no full peek fits (26 + 6 > 30):
    // StartCatch first.
    r.at(8, 26, 0, 6);
    EXPECT_EQ(ccd::standard::after_lookback(r.context, r.state), Phase::StartCatch);
    EXPECT_EQ(r.state.exit, Exit::None);

    r.at(2, 14, 0, 6);
    EXPECT_EQ(ccd::standard::after_lookback(r.context, r.state), Phase::Extend);

    r.at(2, 26, 0, 6);
    EXPECT_EQ(ccd::standard::after_lookback(r.context, r.state), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::PeekBeyondEnd);
}

TEST(StandardRouting, StartCatchWritesTheLeadingSegmentThenThePeekGuard)
{
    Record r(60);
    r.at(9, 21, 0, 6);
    EXPECT_EQ(ccd::standard::start_catch(r.context, r.state), Phase::Extend);
    ASSERT_EQ(r.state.segments.size(), 1u);

    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.curve_qa, ccd::CurveQA::Start);
    EXPECT_EQ(s.start_day, r.dates[0]);
    EXPECT_EQ(s.end_day, r.dates[8]);
    EXPECT_EQ(s.break_day, r.dates[9]);
    EXPECT_EQ(s.observation_count, 9u);
    EXPECT_EQ(s.change_probability, 0.0);
    EXPECT_EQ(s.bands.size(), 7u);

    Record t(30);
    t.at(9, 26, 0, 6);
    EXPECT_EQ(ccd::standard::start_catch(t.context, t.state), Phase::EndCatch);
    EXPECT_EQ(t.state.exit, Exit::PeekBeyondEnd);
}

TEST(StandardRouting, RestartMovesTheCursorToTheBreak)
{
    Record r(60);
    r.with_segment();
    r.at(5, 30, 0);
    EXPECT_EQ(ccd::standard::restart(r.context, r.state), Phase::Initialize);
    EXPECT_EQ(r.state.cursor.prev_break, 30u);
    EXPECT_EQ(r.state.cursor.window.start, 30u);
    EXPECT_EQ(r.state.cursor.window.stop, 42u);
    EXPECT_EQ(r.state.exit, Exit::None);

    r.at(5, 37, 0);
    EXPECT_EQ(ccd::standard::restart(r.context, r.state), Phase::EndCatch);
    EXPECT_EQ(r.state.cursor.prev_break, 37u);
    EXPECT_EQ(r.state.exit, Exit::NoRoomToInitialize);
}

TEST(StandardRouting, EndCatchWritesTheTrailingSegmentPastAPeek)
{
    Record r(60);
    r.with_segment();
    r.at(40, 52, 40, 6);
    EXPECT_EQ(ccd::standard::end_catch(r.context, r.state), Phase::Done);
    ASSERT_EQ(r.state.segments.size(), 2u);

    const ccd::ChangeModel& s = r.state.segments[1];
    EXPECT_EQ(s.curve_qa, ccd::CurveQA::End);
    EXPECT_EQ(s.start_day, r.dates[40]);
    EXPECT_EQ(s.end_day, r.dates[59]);
    EXPECT_EQ(s.break_day, r.dates[59]);   // the window reaches the end
    EXPECT_EQ(s.observation_count, 20u);

    // prev_break + peek == size: nothing to catch.
    Record t(60);
    t.with_segment();
    t.at(54, 60, 54, 6);
    EXPECT_EQ(ccd::standard::end_catch(t.context, t.state), Phase::Done);
    EXPECT_EQ(t.state.segments.size(), 1u);
}

//------------------------------------------------------------------------------
// Prepare
//------------------------------------------------------------------------------

TEST(StandardPrepare, AtMostMeowObservationsIsDoneWithNothing)
{
    Record r(12);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Done);
    EXPECT_EQ(r.state.exit, Exit::TooFewObservations);

    r.state.phase = Phase::Done;
    const ccd::FitResult result = r.state.take_result();
    EXPECT_TRUE(result.models.empty());
    EXPECT_EQ(result.mask.size(), 0u);   // quirk 5
}

TEST(StandardPrepare, ANaNVariogramIsDoneWithNothing)
{
    // One observation dated <= STAT_ORD: the statistics window's variogram
    // has no pair to difference, and is NaN.
    Record r(40, ccd::HarmonicOptions{}.STAT_ORD);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Done);
    EXPECT_EQ(r.state.exit, Exit::VariogramNaN);

    r.state.phase = Phase::Done;
    const ccd::FitResult result = r.state.take_result();
    EXPECT_TRUE(result.models.empty());
    EXPECT_EQ(result.mask.size(), 0u);   // quirk 5
}

TEST(StandardPrepare, SetsTheCadenceAndTheCursor)
{
    Record r(40);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Initialize);
    EXPECT_EQ(r.state.exit, Exit::None);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 12u);
    EXPECT_EQ(r.state.cursor.prev_break, 0u);

    // 16-day cadence: the caller's peek and threshold.
    EXPECT_EQ(r.state.cadence.peek, r.options.PEEK_SIZE);
    EXPECT_EQ(r.state.cadence.change_threshold, r.options.CHANGE_THRESHOLD);
    EXPECT_EQ(r.state.variogram.bands.size(), 7u);
}

TEST(StandardPrepare, NoRoomForAStartWindowGoesToTheEndCatch)
{
    Record r(23);   // 12 + 12 > 23
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::NoRoomToInitialize);
}

//------------------------------------------------------------------------------
// Exits on whole pixels
//------------------------------------------------------------------------------

// standard::run() on a characterisation pixel, preprocessed as detect() does
// for the Standard procedure: the plan's transform on a copy, its chain for
// the mask.
struct MachineRun
{
    ccd::FitResult result;
    Exit exit;
};

MachineRun run_machine(const golden::CharPixel& p)
{
    const index_t T = p.dates.size();
    const ccd::preprocess::PreprocessPlan plan = ccd::preprocess::default_plan(p.h);
    const ccd::preprocess::ProcedurePlan& standard = plan.standard;

    std::vector<double> block = p.spectral;
    if (standard.transform)
    {
        const auto& affine = *standard.transform;
        for (index_t t = 0; t < T; ++t)
            block[affine.band * T + t] = block[affine.band * T + t] * affine.scale + affine.offset;
    }

    const ccd::HarmonicWorkspace workspace(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
        p.h
    );

    ccd::ProcessingMask mask;
    ccd::preprocess::apply(
        standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

    ccd::LassoWorkspace lasso(0);
    ccd::LassoSolver solver(p.l);
    const ccd::standard::Context context{p.h, workspace, lasso, solver};

    ccd::standard::State state;
    MachineRun out;
    out.result = ccd::standard::run(context, state, std::move(mask));
    out.exit = state.exit;
    EXPECT_EQ(state.phase, Phase::Done);
    return out;
}

ccd::FitResult run_detect(const golden::CharPixel& p)
{
    const std::size_t T = p.dates.size();
    return ccd::detect(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
        p.h,
        p.l
    );
}

void expect_same(const ccd::FitResult& a, const ccd::FitResult& b, const std::string& name)
{
    ASSERT_EQ(a.models.size(), b.models.size()) << name;
    for (std::size_t m = 0; m < a.models.size(); ++m)
    {
        const auto& x = a.models[m];
        const auto& y = b.models[m];
        EXPECT_EQ(x.start_day, y.start_day) << name << " " << m;
        EXPECT_EQ(x.end_day, y.end_day) << name << " " << m;
        EXPECT_EQ(x.break_day, y.break_day) << name << " " << m;
        EXPECT_EQ(x.observation_count, y.observation_count) << name << " " << m;
        EXPECT_EQ(x.change_probability, y.change_probability) << name << " " << m;
        EXPECT_EQ(x.curve_qa, y.curve_qa) << name << " " << m;
        ASSERT_EQ(x.bands.size(), y.bands.size()) << name << " " << m;
        for (std::size_t b = 0; b < x.bands.size(); ++b)
        {
            EXPECT_EQ(x.bands[b].score.rmse, y.bands[b].score.rmse) << name << " " << m;
            EXPECT_EQ(x.bands[b].score.magn, y.bands[b].score.magn) << name << " " << m;
            EXPECT_EQ(x.bands[b].model.intercept(), y.bands[b].model.intercept()) << name;
            EXPECT_EQ(x.bands[b].model.coefficients(), y.bands[b].model.coefficients()) << name;
        }
    }
    ASSERT_EQ(a.mask.size(), b.mask.size()) << name;
    for (index_t i = 0; i < a.mask.size(); ++i)
        EXPECT_EQ(a.mask[i], b.mask[i]) << name << " mask " << i;
}

const golden::CharPixel& pixel(const std::vector<golden::CharPixel>& all, const std::string& name)
{
    for (const auto& p : all)
        if (p.name == name)
            return p;
    throw std::runtime_error("no characterisation pixel " + name);
}

TEST(StandardExit, EachWayTheMachineStops)
{
    const auto all = golden::characterisation_pixels();

    const std::vector<std::pair<std::string, Exit>> expected{
        {"standard_too_few", Exit::TooFewObservations},
        {"variogram_nan", Exit::VariogramNaN},
        {"standard_no_room", Exit::NoRoomToInitialize},         // from Prepare
        {"start_catch", Exit::NoRoomToInitialize},              // from Restart
        {"constant_detection_band", Exit::InitializeExhausted}, // no segment
        {"lookback_below_break_decides", Exit::InitializeExhausted},  // after one
        {"peek_guard_tail", Exit::PeekBeyondEnd},
    };

    for (const auto& [name, exit] : expected)
    {
        const golden::CharPixel& p = pixel(all, name);
        const MachineRun run = run_machine(p);
        EXPECT_EQ(run.exit, exit) << name << ": " << ccd::standard::to_string(run.exit);
        expect_same(run.result, run_detect(p), name);
    }
}

TEST(StandardExit, TheEarlyExitsReturnNoMask)
{
    const auto all = golden::characterisation_pixels();
    for (const char* name : {"standard_too_few", "variogram_nan"})
    {
        const MachineRun run = run_machine(pixel(all, name));
        EXPECT_TRUE(run.result.models.empty()) << name;
        EXPECT_EQ(run.result.mask.size(), 0u) << name;   // quirk 5
    }
}

TEST(StandardMachine, OneStateServesManyPixels)
{
    // A record reused across pixels gives each pixel the result a fresh one
    // does: reset() starts every pixel from Prepare with nothing carried.
    const auto all = golden::characterisation_pixels();
    ccd::StandardProcedure reused;

    for (const auto& p : all)
    {
        const index_t T = p.dates.size();
        const ccd::preprocess::PreprocessPlan plan = ccd::preprocess::default_plan(p.h);
        std::vector<double> block = p.spectral;
        if (plan.standard.transform)
        {
            const auto& affine = *plan.standard.transform;
            for (index_t t = 0; t < T; ++t)
                block[affine.band * T + t] = block[affine.band * T + t] * affine.scale + affine.offset;
        }
        const ccd::HarmonicWorkspace workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
            p.h
        );
        ccd::ProcessingMask mask;
        ccd::preprocess::apply(
            plan.standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

        ccd::LassoWorkspace lasso(0);
        ccd::LassoSolver solver(p.l);
        ccd::StandardProcedure fresh;
        const ccd::FitResult once = fresh.run(workspace, mask, lasso, solver);
        const ccd::FitResult again = reused.run(workspace, mask, lasso, solver);
        expect_same(again, once, p.name);
    }
}

} // namespace

//------------------------------------------------------------------------------
// The detector's per-pixel choice
//------------------------------------------------------------------------------

TEST(StandardDetector, SelectsByTheClearThenTheSnowPercentage)
{
    const ccd::HarmonicOptions options;   // clear 0.25, snow 0.75
    const ccd::standard::Detector detector{};

    auto stats = [](scalar_t clear, scalar_t snow)
    {
        ccd::Quality::Statistics s;
        s.clear_probability = clear;
        s.snow_probability = snow;
        return s;
    };

    EXPECT_EQ(detector.select(stats(0.25, 0.9), options), ccd::PixelProcedure::Standard);
    EXPECT_EQ(detector.select(stats(0.2499, 0.75), options), ccd::PixelProcedure::PermanentSnow);
    EXPECT_EQ(detector.select(stats(0.2499, 0.7499), options),
              ccd::PixelProcedure::InsufficientClear);
}
