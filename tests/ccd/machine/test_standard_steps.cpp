//==============================================================================
//
// The Standard machine's routing and its steps, from hand-built records.
//
//   * Routing: each guard between phases (room_to_initialize, peek_fits,
//     start_catch_due) and each routing step (after_stable_start,
//     after_lookback, peek_guard, restart, end_catch, start_catch,
//     prepare) on a State built by hand at a chosen cursor, against the
//     phase, the exit and the record the procedure's description gives.
//   * Steps: one transition call of Initialize, Lookback and Extend per row
//     of their guard tables (the brief's 1.4, 1.5 and 1.8), on a short
//     smooth series with the observation that decides it edited in, against
//     the next phase and the record, and the same record's step function
//     (initialize_step, lookback_step, extend_step) against the outcome it
//     reports; lookback_peek()'s three cases.
//
// The outputs themselves are pinned by the golden and transitions fixtures;
// these tests say which route moved if one ever does.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/input/procedure.hpp"
#include "ccd/machine/standard.hpp"   // internal (src/)
#include "ccd/regression/lasso_solver.hpp"
#include "config.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/pyccd/peek_statistics.hpp"
#include "ccd/strategies/library.hpp"

#include "find_closest_doy.hpp"
#include "standard_record.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;
using ccd::standard::Exit;
using ccd::standard::InitializeOutcome;
using ccd::standard::Phase;
using machine_test::Record;
using machine_test::in_extend;
using machine_test::kVariogram;

//------------------------------------------------------------------------------
// Guards
//------------------------------------------------------------------------------

TEST(StandardRouting, RoomToInitializeIsAMeowWindowAfterTheCursor)
{
    Record r(40);   // min_observations 12
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
    EXPECT_FALSE(ccd::standard::peek_fits(r.state));   // the plan's peek, not peek_size
}

TEST(StandardRouting, StartCatchIsDueOnlyBeforeTheFirstSegmentPastAPeek)
{
    Record r(60);
    r.at(7, 19, 0, 6);
    EXPECT_TRUE(ccd::standard::start_catch_due(r.context, r.state));    // 7 - 0 > 6
    r.at(6, 18, 0, 6);
    EXPECT_FALSE(ccd::standard::start_catch_due(r.context, r.state));   // 6 - 0 == 6

    // A later segment never gets one, however long the gap.
    r.with_segment();
    r.at(40, 52, 20, 6);
    EXPECT_FALSE(ccd::standard::start_catch_due(r.context, r.state));
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

TEST(StandardPrepare, AtMostMeowObservationsIsDoneWithTheMaskOnly)
{
    Record r(12);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Done);
    EXPECT_EQ(r.state.exit, Exit::TooFewObservations);

    r.state.phase = Phase::Done;
    const ccd::FitResult result = r.state.take_result();
    EXPECT_TRUE(result.models.empty());
    EXPECT_EQ(result.mask.size(), 12u);
    EXPECT_EQ(result.mask.count(), 12u);
}

TEST(StandardPrepare, ANaNVariogramIsDoneWithTheMaskOnly)
{
    // One observation dated <= statistics_end: the statistics window's variogram
    // has no pair to difference, and is NaN.
    Record r(40, golden::PYCCD_STATISTICS_END);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Done);
    EXPECT_EQ(r.state.exit, Exit::VariogramNaN);

    r.state.phase = Phase::Done;
    const ccd::FitResult result = r.state.take_result();
    EXPECT_TRUE(result.models.empty());
    EXPECT_EQ(result.mask.size(), 40u);
    EXPECT_EQ(result.mask.count(), 40u);
}

TEST(StandardPrepare, SetsThePeekPlanAndTheCursor)
{
    Record r(40);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Initialize);
    EXPECT_EQ(r.state.exit, Exit::None);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 12u);
    EXPECT_EQ(r.state.cursor.prev_break, 0u);

    // 16-day cadence: the caller's peek.
    EXPECT_EQ(r.state.peek_plan.peek, r.count("change_test", "peek_size"));

    // Initialize's coefficients, once for the pixel: the stability fit's
    // (coefficients.min coefficients in a coefficients.max-wide basis).
    const ccd::CoefficientRule& c = r.state.stability_coefficients;
    EXPECT_EQ(c.min, r.count("stability_fit", "coefficients.min"));
    EXPECT_EQ(c.max, r.count("stability_fit", "coefficients.max"));
    EXPECT_EQ(r.state.noise.bands.size(), 7u);
}

TEST(StandardPrepare, NoRoomForAStartWindowGoesToTheEndCatch)
{
    Record r(23);   // 12 + 12 > 23
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::NoRoomToInitialize);
}

//------------------------------------------------------------------------------
// Initialize: one candidate window per call, one test per guard
//------------------------------------------------------------------------------

// A record in Initialize at (start, stop), past Prepare.
void in_initialize(Record& r, index_t start, index_t stop, index_t prev_break = 0)
{
    r.with_variogram(kVariogram);
    r.at(start, stop, prev_break);
    r.state.phase = Phase::Initialize;
}

// What the shared step reports for a record set up as `setup` leaves it,
// on a record of its own (the step changes the record it is given).
template<typename Setup>
InitializeOutcome outcome_of(index_t n, Setup setup)
{
    Record r(n);
    setup(r);
    return ccd::standard::initialize_step(r.context, r.state);
}

TEST(StandardInitialize, NoCandidateLeftIsExhausted)
{
    // window.stop + min_observations < size is strict: 28 + 12 == 40 fails.
    const auto setup = [](Record& r)
    {
        in_initialize(r, 16, 28, 16);
    };
    EXPECT_EQ(outcome_of(40, setup), InitializeOutcome::Exhausted);

    Record r(40);
    setup(r);
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::InitializeExhausted);
    EXPECT_EQ(r.state.cursor.window.start, 16u);
    EXPECT_EQ(r.state.cursor.window.stop, 28u);
    EXPECT_EQ(r.state.series.size(), 40u);
}

TEST(StandardInitialize, AShortSpanGrows)
{
    // 12 observations 16 days apart span 176 days < day_delta.
    const auto setup = [](Record& r)
    {
        in_initialize(r, 0, 12);
    };
    EXPECT_EQ(outcome_of(60, setup), InitializeOutcome::ShortSpan);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Initialize);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 13u);
    EXPECT_EQ(r.state.series.size(), 60u);
    EXPECT_EQ(r.state.exit, Exit::None);
}

TEST(StandardInitialize, TmaskFlaggingEverythingGrows)
{
    // A variogram of almost 0: every Tmask residual is above it.
    const auto setup = [](Record& r)
    {
        in_initialize(r, 0, 24);
        r.with_variogram(1e-12);
    };
    EXPECT_EQ(outcome_of(60, setup), InitializeOutcome::AllOutliers);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Initialize);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 25u);
    EXPECT_EQ(r.state.series.size(), 60u);
}

TEST(StandardInitialize, ARejectedTmaskDropGrowsWithNothingDropped)
{
    // (0, 24) spans 368 days; without its last observation it spans 352 <
    // day_delta, so dropping a Tmask outlier there is refused.
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i == 23 && (b == 1 || b == 4)) ? v + 5000 : v; });
        in_initialize(r, 0, 24);
    };
    EXPECT_EQ(outcome_of(60, setup), InitializeOutcome::RejectedOutliers);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Initialize);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 25u);
    EXPECT_EQ(r.state.series.size(), 60u);
    EXPECT_TRUE(r.state.series.mask()[23]);
}

TEST(StandardInitialize, AnAcceptedTmaskDropCommitsThenTests)
{
    // An interior outlier: dropped from the series, the window one shorter,
    // and the shortened window is stable in the same step. At the break,
    // with no gap, the route goes straight to the peek guard.
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i == 10 && (b == 1 || b == 4)) ? v + 5000 : v; });
        in_initialize(r, 0, 24);
    };
    EXPECT_EQ(outcome_of(60, setup), InitializeOutcome::DroppedThenStable);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Extend);
    EXPECT_EQ(r.state.series.size(), 59u);
    EXPECT_FALSE(r.state.series.mask()[10]);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 23u);
    ASSERT_TRUE(r.state.stability_model);
    EXPECT_FALSE(r.state.stability_model->residuals(1).empty());   // a detection band
}

TEST(StandardInitialize, AnAcceptedTmaskDropThenAnUnstableWindowShifts)
{
    // The same interior outlier, and a steep trend in detection bands Tmask
    // does not read: one step drops the outlier, shortening the window to
    // (0, 23), then fits it, finds it unstable and shifts it.
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v)
        {
            if (i == 10 && (b == 1 || b == 4))
                return v + 5000;
            return (b == 2 || b == 3 || b == 5) ? v + 40.0 * i : v;
        });
        in_initialize(r, 0, 24);
    };
    EXPECT_EQ(outcome_of(60, setup), InitializeOutcome::DroppedThenUnstable);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Initialize);
    EXPECT_EQ(r.state.series.size(), 59u);
    EXPECT_FALSE(r.state.series.mask()[10]);
    EXPECT_EQ(r.state.cursor.window.start, 1u);
    EXPECT_EQ(r.state.cursor.window.stop, 24u);
}

TEST(StandardInitialize, AStableWindowLeavesThroughTheRouting)
{
    // Above the break: Lookback next, with the window's stable fits.
    const auto setup = [](Record& r)
    {
        in_initialize(r, 20, 44, 10);
        r.with_segment();
    };
    EXPECT_EQ(outcome_of(80, setup), InitializeOutcome::Stable);

    Record r(80);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Lookback);
    EXPECT_EQ(r.state.cursor.window.start, 20u);
    EXPECT_EQ(r.state.cursor.window.stop, 44u);
    EXPECT_EQ(r.state.series.size(), 80u);
}

TEST(StandardInitialize, AnUnstableWindowShifts)
{
    // A steep trend in detection bands Tmask does not read: the stability rule fails.
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (b == 2 || b == 3 || b == 5) ? v + 40.0 * i : v; });
        in_initialize(r, 0, 24);
    };
    EXPECT_EQ(outcome_of(60, setup), InitializeOutcome::Unstable);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Initialize);
    EXPECT_EQ(r.state.cursor.window.start, 1u);
    EXPECT_EQ(r.state.cursor.window.stop, 25u);
    EXPECT_EQ(r.state.series.size(), 60u);
}

//------------------------------------------------------------------------------
// Lookback: one observation per call, one test per guard
//------------------------------------------------------------------------------

TEST(StandardLookback, ThePeekWindowNeverReadsBelowTheBreak)
{
    using ccd::strategies::lookback_peek;

    // Far from the break: start - 1 down to start - peek, the adjusted peek
    // of observations.
    auto far = lookback_peek(20, 0, 6);
    EXPECT_EQ(far.start, 19);
    EXPECT_EQ(far.stop, 13);

    // Near the break: down to prev_break, at the series start (prev_break 0)
    // as after a segment, however small the window's start.
    auto start = lookback_peek(5, 0, 6);
    EXPECT_EQ(start.start, 4);
    EXPECT_EQ(start.stop, -1);
    auto small_start = lookback_peek(17, 13, 19);
    EXPECT_EQ(small_start.start, 16);
    EXPECT_EQ(small_start.stop, 12);
    auto near = lookback_peek(30, 26, 6);
    EXPECT_EQ(near.start, 29);
    EXPECT_EQ(near.stop, 25);
}

// A record in Lookback at (start, stop) with the window's stable fits.
void in_lookback(Record& r, index_t start, index_t stop, index_t prev_break = 0)
{
    r.with_variogram(kVariogram);
    r.at(start, stop, prev_break);
    r.with_stability_fit(start, stop);
    r.state.phase = Phase::Lookback;
}

// What lookback_step() reports for a record set up as `setup` leaves it, on
// a record of its own; also the window the step leaves, which the
// transition keeps whatever it routes to.
template<typename Setup>
std::pair<ccd::standard::LookbackOutcome, ccd::Window> lookback_outcome_of(index_t n, Setup setup)
{
    Record r(n);
    setup(r);
    const auto outcome = ccd::standard::lookback_step(r.context, r.state);
    return {outcome, r.state.cursor.window};
}

TEST(StandardLookback, AChangeStopsWithTheWindowWhereItIs)
{
    // Every observation before position 10 is far off the models. The gap
    // to the series start is 10 > peek: the start catch is next.
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i < 10 && b >= 1 && b <= 5) ? v + 3000 : v; });
        in_lookback(r, 10, 34);
    };
    EXPECT_EQ(lookback_outcome_of(60, setup).first, ccd::standard::LookbackOutcome::Change);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::StartCatch);
    EXPECT_EQ(r.state.cursor.window.start, 10u);
    EXPECT_EQ(r.state.series.size(), 60u);
}

TEST(StandardLookback, AnOutlierIsDroppedAndTheWindowMovesWithTheSeries)
{
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i == 9 && b >= 1 && b <= 5) ? v + 3000 : v; });
        in_lookback(r, 10, 34);
    };
    EXPECT_EQ(lookback_outcome_of(60, setup).first, ccd::standard::LookbackOutcome::Dropped);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Lookback);
    EXPECT_EQ(r.state.series.size(), 59u);
    EXPECT_FALSE(r.state.series.mask()[9]);
    EXPECT_EQ(r.state.cursor.window.start, 9u);
    EXPECT_EQ(r.state.cursor.window.stop, 33u);
}

TEST(StandardLookback, AnObservationThatFitsIsIncluded)
{
    const auto setup = [](Record& r)
    {
        in_lookback(r, 10, 34);
    };
    EXPECT_EQ(lookback_outcome_of(60, setup).first, ccd::standard::LookbackOutcome::Included);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Lookback);
    EXPECT_EQ(r.state.series.size(), 60u);
    EXPECT_EQ(r.state.cursor.window.start, 9u);
    EXPECT_EQ(r.state.cursor.window.stop, 34u);
}

TEST(StandardLookback, ReachingThePreviousBreakRoutesOn)
{
    // The include that reaches prev_break is the last Lookback step.
    const auto setup = [](Record& r)
    {
        r.with_segment();
        in_lookback(r, 21, 45, 20);
    };
    const auto [outcome, window] = lookback_outcome_of(60, setup);
    EXPECT_EQ(outcome, ccd::standard::LookbackOutcome::Included);
    EXPECT_EQ(window.start, 20u);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Extend);
    EXPECT_EQ(r.state.cursor.window.start, 20u);
    EXPECT_EQ(r.state.exit, Exit::None);
}

//------------------------------------------------------------------------------
// Extend: one peek test per call, one test per guard
//------------------------------------------------------------------------------

// What extend_step() reports for a record set up as `setup` leaves it, on a
// record of its own, and the segments it leaves (it writes none).
template<typename Setup>
std::pair<ccd::standard::ExtendOutcome, std::size_t> extend_outcome_of(index_t n, Setup setup)
{
    Record r(n);
    setup(r);
    const auto outcome = ccd::standard::extend_step(r.context, r.state);
    return {outcome, r.state.segments.size()};
}

TEST(StandardExtend, TheFirstTestTrainsThenAFittingPeekGrows)
{
    const auto setup = [](Record& r)
    {
        in_extend(r, 0, 24);
    };
    EXPECT_EQ(extend_outcome_of(80, setup).first, ccd::standard::ExtendOutcome::Grown);

    Record r(80);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Extend);
    const ccd::standard::ExtendState& e = r.state.extend;
    EXPECT_TRUE(e.trained);
    EXPECT_EQ(e.fit_window.start, 0u);
    EXPECT_EQ(e.fit_window.stop, 24u);
    ASSERT_TRUE(e.model);
    EXPECT_EQ(e.peek.start, 24u);
    EXPECT_EQ(e.peek.stop, 30u);
    EXPECT_EQ(e.num_coef, r.strategies.model->coefficients().count(24));
    EXPECT_EQ(r.state.cursor.window.stop, 25u);
    EXPECT_EQ(r.state.series.size(), 80u);
    EXPECT_TRUE(r.state.segments.empty());
}

TEST(StandardExtend, ShortWindowsRefitEveryTestAndLongerOnesOnTheSpanFactor)
{
    // Up to refit's min_obs every test refits on the window.
    Record r(80);
    in_extend(r, 0, 20);
    r.step();
    EXPECT_EQ(r.state.extend.fit_window.stop, 20u);
    r.step();
    EXPECT_EQ(r.state.extend.fit_window.stop, 21u);

    // At 24 observations it refits; from 25 on, only once the window spans
    // 1.33 times the last fit: (0, 25) spans 384 days < 1.33 * 368.
    Record t(80);
    in_extend(t, 0, 24);
    t.step();
    t.step();
    EXPECT_EQ(t.state.extend.fit_window.stop, 24u);
    EXPECT_EQ(t.state.cursor.window.stop, 26u);
}

TEST(StandardExtend, TheComparisonRmseIsTheModelsUpToRetrainMinObs)
{
    // A window of at most model_rmse_up_to compares against the model RMSE,
    // the one it was just refitted on; a longer one against the seasonal
    // RMSE of the fit window.
    Record r(80);
    in_extend(r, 0, 24);
    r.step();
    // The magnitudes the test wrote are change_magnitude() of those RMSEs.
    const ccd::standard::ExtendState& e = r.state.extend;
    const auto& bands = r.shared.detection_bands;
    const auto magnitude_with = [&](const std::vector<double>& rmse)
    {
        std::vector<double> m;
        ccd::change_magnitude(e.residuals, bands, r.state.noise, rmse, m);
        return m;
    };
    std::vector<double> rmse(bands.size());
    for (index_t i = 0; i < bands.size(); ++i)
        rmse[i] = e.model->rmse(bands[i]);
    EXPECT_EQ(e.result.comparison_rmse, rmse);
    EXPECT_EQ(*e.result.find("change_magnitude"), magnitude_with(rmse));

    r.step();   // window (0, 25): seasonal
    const auto closest = reference::find_closest_doy(
        r.state.series.dates(), e.peek.stop - 1, e.fit_window, r.count("change_test", "seasonal_count"));
    for (index_t i = 0; i < bands.size(); ++i)
        rmse[i] = ccd::seasonal_rmse(e.model->residuals(bands[i]), closest,
                                     r.real("change_test", "seasonal_divisor"));
    EXPECT_EQ(e.result.comparison_rmse, rmse);
    EXPECT_EQ(*e.result.find("change_magnitude"), magnitude_with(rmse));
}

TEST(StandardExtend, TheTestsModelRmseWindowIsItsOwnParameter)
{
    // The change test's model_rmse_up_to set apart from the refit's min_obs:
    // a 24-observation window, refitted (min_obs 24), compares against the
    // seasonal RMSE when the test's limit is 10.
    Record r(80);
    auto tree = ccd::strategies::library_registry().preset("ccdc");
    tree.stages["change_test"].params["model_rmse_up_to"] = ccd::config::integer(10);
    r.strategies = ccd::config::resolve(golden::core_registry(), tree, r.options).strategies;
    in_extend(r, 0, 24);
    r.step();

    const ccd::standard::ExtendState& e = r.state.extend;
    EXPECT_EQ(e.fit_window.stop, 24u);
    const auto& bands = r.shared.detection_bands;
    const auto closest = reference::find_closest_doy(
        r.state.series.dates(), e.peek.stop - 1, e.fit_window, r.count("change_test", "seasonal_count"));
    std::vector<double> rmse(bands.size());
    for (index_t i = 0; i < bands.size(); ++i)
        rmse[i] = ccd::seasonal_rmse(e.model->residuals(bands[i]), closest,
                                     r.real("change_test", "seasonal_divisor"));
    std::vector<double> want;
    ccd::change_magnitude(e.residuals, bands, r.state.noise, rmse, want);
    EXPECT_EQ(e.result.comparison_rmse, rmse);
    EXPECT_EQ(*e.result.find("change_magnitude"), want);
}

TEST(StandardExtend, AChangeWritesTheSegmentAtTheWindow)
{
    // The step decides; the transition writes the segment.
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i >= 24 && b >= 1 && b <= 5) ? v + 3000 : v; });
        in_extend(r, 0, 24);
    };
    const auto [outcome, segments] = extend_outcome_of(80, setup);
    EXPECT_EQ(outcome, ccd::standard::ExtendOutcome::Change);
    EXPECT_EQ(segments, 0u);

    Record r(80);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Restart);
    ASSERT_EQ(r.state.segments.size(), 1u);

    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.change_probability, 1.0);
    EXPECT_EQ(s.start_day, r.dates[0]);
    EXPECT_EQ(s.end_day, r.dates[23]);
    EXPECT_EQ(s.break_day, r.dates[24]);
    EXPECT_EQ(s.observation_count, 24u);
    EXPECT_EQ(s.curve_qa, static_cast<ccd::CurveQA>(r.strategies.model->coefficients().count(24)));
    EXPECT_EQ(s.bands.size(), 7u);
    EXPECT_FALSE(r.state.extend.trained);   // the next segment starts untrained
    EXPECT_EQ(r.state.cursor.window.stop, 24u);
}

TEST(StandardExtend, AnOutlierIsDroppedAndTheWindowKeepsItsCoordinates)
{
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i == 24 && b >= 1 && b <= 5) ? v + 3000 : v; });
        in_extend(r, 0, 24);
    };
    EXPECT_EQ(extend_outcome_of(80, setup).first, ccd::standard::ExtendOutcome::Dropped);

    Record r(80);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Extend);
    EXPECT_EQ(r.state.series.size(), 79u);
    EXPECT_FALSE(r.state.series.mask()[24]);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 24u);
    EXPECT_TRUE(r.state.segments.empty());
}

TEST(StandardExtend, TheLastFullPeekEndsTheCurve)
{
    // (0, 74) of 80 with peek 6: the peek reaches the last observation and
    // holds no change, so the curve ends at the window, with change
    // probability 0, and the machine goes to EndCatch: no Restart.
    const auto setup = [](Record& r)
    {
        in_extend(r, 0, 74);
    };
    const auto [outcome, segments] = extend_outcome_of(80, setup);
    EXPECT_EQ(outcome, ccd::standard::ExtendOutcome::SeriesEnd);
    EXPECT_EQ(segments, 0u);

    Record r(80);
    setup(r);
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::SeriesEnd);
    ASSERT_EQ(r.state.segments.size(), 1u);

    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.change_probability, 0.0);
    EXPECT_EQ(s.end_day, r.dates[73]);
    EXPECT_EQ(s.break_day, r.dates[74]);   // the first observation after it
    EXPECT_EQ(s.observation_count, 74u);
    EXPECT_EQ(s.curve_qa, static_cast<ccd::CurveQA>(r.strategies.model->coefficients().count(74)));
    EXPECT_EQ(r.state.cursor.window.stop, 74u);
    EXPECT_EQ(r.state.cursor.prev_break, 74u);

    // The End segment's guard, prev_break + peek < size, fails: 74 + 6.
    EXPECT_EQ(r.step(), Phase::Done);
    EXPECT_EQ(r.state.segments.size(), 1u);
}

TEST(StandardExtend, TheSeriesEndAfterADropBreaksAfterTheDroppedObservation)
{
    const auto setup = [](Record& r)
    {
        r.edit([](index_t b, index_t i, double v) { return (i == 74 && b >= 1 && b <= 5) ? v + 3000 : v; });
        in_extend(r, 0, 74);
    };
    EXPECT_EQ(extend_outcome_of(80, setup).first, ccd::standard::ExtendOutcome::Dropped);

    Record r(80);
    setup(r);
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::SeriesEnd);
    ASSERT_EQ(r.state.segments.size(), 1u);

    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.change_probability, 0.0);
    EXPECT_EQ(r.state.series.size(), 79u);
    EXPECT_EQ(s.end_day, r.dates[73]);
    EXPECT_EQ(s.break_day, r.dates[75]);   // masked position 74
    EXPECT_EQ(s.observation_count, 74u);
}

} // namespace
