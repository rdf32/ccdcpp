//==============================================================================
//
// The Standard machine's routing and its steps, from hand-built records, its
// exit reasons, on whole pixels, the per-pixel procedure choice and the
// Standard detector's fit of each procedure.
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
//   * Whole pixels stepped by hand: every self-transition is one decision,
//     and the result is the driven machine's.
//   * Exits: standard::run() on the characterization pixels that end the
//     segment loop each way, with the same preprocessing ccd::detect() runs,
//     against the Exit the run leaves in the record. The result of each run
//     is also checked against detect()'s, field for field.
//   * the mode's choose(): the clear percentage first, then the snow
//     percentage, each threshold on its >= side; Detector::fit(): the
//     machine for Standard, the shared whole-series fit for the others.
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

#include "../golden/specs.hpp"

#include "ccd/ccd.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/machine/machine.hpp"
#include "ccd/machine/pixel_procedure.hpp"
#include "ccd/machine/standard.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "../golden/config.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/register_core.hpp"

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
    ccd::Input options;
    ccd::config::Config config = golden::pyccd();   // the strategies' configuration
    ccd::config::Resolved resolved = golden::resolved(config, options);
    const ccd::config::Shared& shared = resolved.shared;
    ccd::LassoOptions lasso_options;   // the kernels' solver, at the preset's values

    ccd::PixelSeries workspace;
    ccd::LassoWorkspace lasso{0};
    ccd::LassoSolver solver;
    ccd::Strategies strategies;
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
              options, shared),
          solver(lasso_options),
          strategies(resolved.strategies),
          context{shared, workspace, strategies}
    {
        ccd::ProcessingMask mask(n);
        for (index_t i = 0; i < n; ++i)
            mask.set(i, true);
        state.reset(workspace, std::move(mask));
    }

    // A whole-number and a real parameter of the record's configuration.
    index_t count(const char* stage, const char* name) const
    {
        return static_cast<index_t>(golden::parameter<std::int64_t>(resolved.stages, stage, name));
    }
    double real(const char* stage, const char* name) const
    {
        return golden::parameter<double>(resolved.stages, stage, name);
    }

    // The record at a cursor, with a cadence, past Prepare.
    void at(index_t start, index_t stop, index_t prev_break, index_t peek = 6)
    {
        state.cursor = ccd::Cursor{ccd::Window(start, stop), prev_break};
        state.cadence = ccd::standard::Cadence{peek, peek, 0};   // chi2: full peeks only
        state.initialize_criteria = ccd::standard::initialize_criteria(
            shared, strategies.stability_fit->coefficients());
        state.coefficients = strategies.model->coefficients();

        // The change test as Prepare leaves it on a series at the reference
        // cadence (16 days): its threshold, change_threshold.
        const std::int64_t reference[] = {730000, 730016};
        strategies.change_test->prepare(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(reference, {2}),
            shared.detection_bands.size(),
            state.extension.test.get(strategies.change_test,
                                     [](const ccd::ChangeTest& c) { return c.make_scratch(); }));
        // The stability rule's threshold, change_threshold (a peek of the
        // change test's peek_size), whatever peek the record has.
        const index_t peek_size = count("change_test", "peek_size");
        const ccd::PeekPlan plan{peek_size, true, 0, peek_size};
        strategies.stability_rule->prepare(
            plan, shared.detection_bands.size(), stability_scratch());
        // Lookback over `peek` observations against its change_threshold:
        // pyccd's at a peek of its own, so its threshold is unadjusted.
        strategies.lookback = std::make_shared<const ccd::strategies::PyccdLookback>(
            ccd::strategies::PyccdLookback::Params{real("lookback", "change_threshold"),
                                                   real("lookback", "change_probability"),
                                                   real("lookback", "outlier_threshold"), peek});
        strategies.lookback->prepare(plan, shared.detection_bands.size(),
                                     ccd::standard::lookback_scratch(context, state));
    }

    // A segment already written, as Extend or StartCatch leaves one.
    void with_segment()
    {
        state.segments.push_back(ccd::ChangeModel{});
    }

    // Changes the observations, spectral(band, i) = f(band, i, old value),
    // and restarts the record on them (every observation kept).
    template<typename F>
    void edit(F f)
    {
        const index_t n = dates.size();
        for (index_t b = 0; b < 7; ++b)
            for (index_t i = 0; i < n; ++i)
                spectral[b * n + i] = f(b, i, spectral[b * n + i]);

        ccd::ProcessingMask mask(n);
        for (index_t i = 0; i < n; ++i)
            mask.set(i, true);
        state.reset(workspace, std::move(mask));
    }

    // A variogram of `value` in every band, as Prepare would leave one.
    void with_variogram(scalar_t value)
    {
        state.variogram.bands.assign(7, value);
    }

    // Initialize's stable fits over (start, stop), as a Stable step leaves
    // them for Lookback: the detection bands, the stability fit's
    // coefficient_min coefficients, residuals kept.
    void with_stability_fit(index_t start, index_t stop)
    {
        stability_fit().fit(
            state.series.dates().slice(ccd::range(start, stop)),
            state.series.spectral().slice(ccd::all(), ccd::range(start, stop)),
            count("stability_fit", "coefficient_min"),
            count("stability_fit", "coefficient_max"),
            shared.detection_bands,
            ccd::KeepResiduals::Yes
        );
    }

    // The record's stability fit and stability-rule scratch.
    ccd::Model& stability_fit() { return ccd::standard::stability_fit(context, state); }
    ccd::Scratch& stability_scratch() { return ccd::standard::stability_scratch(context, state); }

    // One transition of the record's phase, as drive() makes it.
    Phase step()
    {
        const Phase next = ccd::standard::Machine::step(context, state);
        state.phase = next;
        return next;
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
    Record r(40);   // meow_size 12
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
    EXPECT_FALSE(ccd::standard::peek_fits(r.state));   // the cadence's peek, not peek_size
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

TEST(StandardPrepare, SetsTheCadenceAndTheCursor)
{
    Record r(40);
    EXPECT_EQ(ccd::standard::prepare(r.context, r.state), Phase::Initialize);
    EXPECT_EQ(r.state.exit, Exit::None);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 12u);
    EXPECT_EQ(r.state.cursor.prev_break, 0u);

    // 16-day cadence: the caller's peek.
    EXPECT_EQ(r.state.cadence.peek, r.count("change_test", "peek_size"));

    // Initialize's criteria, once for the pixel: the stability fit's
    // (coefficient_min coefficients in a coefficient_max-wide basis among
    // them).
    const ccd::InitializeCriteria& c = r.state.initialize_criteria;
    EXPECT_EQ(c.detection_bands.size(), r.shared.detection_bands.size());
    EXPECT_EQ(c.num_coefficients, r.count("stability_fit", "coefficient_min"));
    EXPECT_EQ(c.max_coefficients, r.count("stability_fit", "coefficient_max"));
    EXPECT_EQ(r.state.variogram.bands.size(), 7u);
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

constexpr scalar_t kVariogram = 50.0;   // Tmask flags a residual above 244.5

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
ccd::InitializeOutcome outcome_of(index_t n, Setup setup)
{
    Record r(n);
    setup(r);
    return ccd::initialize_step(r.state.series, r.state.cursor.window, r.state.initialize_criteria,
                                r.state.variogram,
                                ccd::InitializeStages{*r.strategies.start_search,
                                                      *r.strategies.window_screen, r.stability_fit(),
                                                      *r.strategies.stability_rule,
                                                      r.stability_scratch()});
}

TEST(StandardInitialize, NoCandidateLeftIsExhausted)
{
    // window.stop + meow_size < size is strict: 28 + 12 == 40 fails.
    const auto setup = [](Record& r)
    {
        in_initialize(r, 16, 28, 16);
    };
    EXPECT_EQ(outcome_of(40, setup), ccd::InitializeOutcome::Exhausted);

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
    EXPECT_EQ(outcome_of(60, setup), ccd::InitializeOutcome::ShortSpan);

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
    EXPECT_EQ(outcome_of(60, setup), ccd::InitializeOutcome::AllOutliers);

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
    EXPECT_EQ(outcome_of(60, setup), ccd::InitializeOutcome::RejectedOutliers);

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
    EXPECT_EQ(outcome_of(60, setup), ccd::InitializeOutcome::DroppedThenStable);

    Record r(60);
    setup(r);
    EXPECT_EQ(r.step(), Phase::Extend);
    EXPECT_EQ(r.state.series.size(), 59u);
    EXPECT_FALSE(r.state.series.mask()[10]);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 23u);
    ASSERT_TRUE(r.state.init_model);
    EXPECT_FALSE(r.state.init_model->residuals(1).empty());   // a detection band
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
    EXPECT_EQ(outcome_of(60, setup), ccd::InitializeOutcome::DroppedThenUnstable);

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
    EXPECT_EQ(outcome_of(80, setup), ccd::InitializeOutcome::Stable);

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
    EXPECT_EQ(outcome_of(60, setup), ccd::InitializeOutcome::Unstable);

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

// A record in Extend at (start, stop), untrained, as the peek guard leaves it.
void in_extend(Record& r, index_t start, index_t stop, index_t prev_break = 0)
{
    r.with_variogram(kVariogram);
    r.at(start, stop, prev_break);
    r.state.phase = Phase::Extend;
}

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
    const ccd::standard::Extension& e = r.state.extension;
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
    EXPECT_EQ(r.state.extension.fit_window.stop, 20u);
    r.step();
    EXPECT_EQ(r.state.extension.fit_window.stop, 21u);

    // At 24 observations it refits; from 25 on, only once the window spans
    // 1.33 times the last fit: (0, 25) spans 384 days < 1.33 * 368.
    Record t(80);
    in_extend(t, 0, 24);
    t.step();
    t.step();
    EXPECT_EQ(t.state.extension.fit_window.stop, 24u);
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
    const ccd::standard::Extension& e = r.state.extension;
    const auto& bands = r.shared.detection_bands;
    const auto magnitude_with = [&](const std::vector<double>& rmse)
    {
        std::vector<double> m;
        ccd::change_magnitude(e.residuals, bands, r.state.variogram, rmse, m);
        return m;
    };
    std::vector<double> rmse(bands.size());
    for (index_t i = 0; i < bands.size(); ++i)
        rmse[i] = e.model->rmse(bands[i]);
    EXPECT_EQ(e.result.comparison_rmse, rmse);
    EXPECT_EQ(*e.result.find("change_magnitude"), magnitude_with(rmse));

    r.step();   // window (0, 25): seasonal
    const auto closest = ccd::find_closest_doy(
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
    r.strategies = ccd::strategies::core_registry().resolve(tree, r.options).strategies;
    in_extend(r, 0, 24);
    r.step();

    const ccd::standard::Extension& e = r.state.extension;
    EXPECT_EQ(e.fit_window.stop, 24u);
    const auto& bands = r.shared.detection_bands;
    const auto closest = ccd::find_closest_doy(
        r.state.series.dates(), e.peek.stop - 1, e.fit_window, r.count("change_test", "seasonal_count"));
    std::vector<double> rmse(bands.size());
    for (index_t i = 0; i < bands.size(); ++i)
        rmse[i] = ccd::seasonal_rmse(e.model->residuals(bands[i]), closest,
                                     r.real("change_test", "seasonal_divisor"));
    std::vector<double> want;
    ccd::change_magnitude(e.residuals, bands, r.state.variogram, rmse, want);
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
    EXPECT_FALSE(r.state.extension.trained);   // the next segment starts untrained
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

//------------------------------------------------------------------------------
// The routes of other strategies
//
// Test-only strategies on the record, each with chi2's or pyccd's decisions
// but for the hook it sets: shortened peeks, a test's minimum window, its
// Undecided and a break past the peek's first observation, the outlier
// rule's EndSeries, a model that cannot fit, a forced and a restoring start
// search, the mode's cap and green filter.
//------------------------------------------------------------------------------

void expect_same(const ccd::FitResult& a, const ccd::FitResult& b, const std::string& name);

// chi2's test with the hooks a test sets.
struct HookTest final : ccd::ChangeTest
{
    std::shared_ptr<const ccd::ChangeTest> chi2;
    bool full = true;            // PeekPlan::full
    index_t min = 0;             // PeekPlan::min_window
    bool undecided = false;      // every test Undecided
    bool change = false;         // every test a Change...
    index_t skip = 0;            // ...breaking this far into the peek

    explicit HookTest(std::shared_ptr<const ccd::ChangeTest> chi2) : chi2(std::move(chi2)) {}

    ccd::Capabilities needs() const override { return chi2->needs(); }
    ccd::Capabilities provides() const override { return chi2->provides(); }
    std::vector<std::string> statistics() const override { return chi2->statistics(); }
    std::unique_ptr<ccd::Scratch> make_scratch() const override { return chi2->make_scratch(); }
    ccd::PeekPlan prepare(ccd::ArrayView<const std::int64_t, 1> dates, index_t bands,
                          ccd::Scratch& scratch) const override
    {
        ccd::PeekPlan plan = chi2->prepare(dates, bands, scratch);
        plan.full = full;
        plan.min_window = min;
        return plan;
    }
    ccd::TestOutcome test(const ccd::Peek& peek, ccd::Scratch& scratch,
                          ccd::PeekResult& result) const override
    {
        const ccd::TestOutcome outcome = chi2->test(peek, scratch, result);
        if (undecided)
            return ccd::TestOutcome::Undecided;
        return change ? ccd::TestOutcome::Change : outcome;
    }
    index_t break_at(const ccd::Peek& peek, const ccd::Scratch&) const override
    {
        return peek.peek.start + skip;
    }
    double magnitude(const ccd::Peek& peek, const ccd::Scratch& scratch,
                     index_t band) const override
    {
        return chi2->magnitude(peek, scratch, band);
    }
};

// A record whose change test is a HookTest, set up by `hooks`, in Extend at
// (start, stop) with the cadence the test's hooks give.
template<typename Hooks>
void with_hooks(Record& r, index_t start, index_t stop, Hooks hooks)
{
    auto test = std::make_shared<HookTest>(r.strategies.change_test);
    hooks(*test);
    r.strategies.change_test = test;
    in_extend(r, start, stop);
    r.state.cadence = ccd::standard::Cadence{6, test->full ? 6u : 0u, test->min};
}

TEST(StandardHooks, ShortenedPeeksGrowTheCurveToTheLastObservation)
{
    Record r(80);
    with_hooks(r, 0, 74, [](HookTest& t) { t.full = false; });
    while (r.state.phase == Phase::Extend)
        r.step();

    // The window reached the series end; the step after it had no
    // observation left to test.
    EXPECT_EQ(r.state.phase, Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::SeriesEnd);
    ASSERT_EQ(r.state.segments.size(), 1u);
    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.observation_count, 80u);
    EXPECT_EQ(s.end_day, r.dates[79]);
    EXPECT_EQ(s.break_day, r.dates[79]);   // no observation after it
    EXPECT_EQ(s.change_probability, 0.0);
    EXPECT_EQ(r.step(), Phase::Done);      // no End segment: none left
}

TEST(StandardHooks, AWindowBelowTheTestsMinimumGrowsUntested)
{
    Record r(80);
    with_hooks(r, 0, 24, [](HookTest& t) { t.min = 30; });
    EXPECT_EQ(ccd::standard::extend_step(r.context, r.state), ccd::standard::ExtendOutcome::Grown);
    EXPECT_FALSE(r.state.extension.trained);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 25u);

    // At the series end: no fit, no segment, Done.
    Record end(30);
    with_hooks(end, 0, 30, [](HookTest& t) { t.full = false; t.min = 40; });
    EXPECT_EQ(end.step(), Phase::Done);
    EXPECT_EQ(end.state.exit, Exit::NoFit);
    EXPECT_TRUE(end.state.segments.empty());
}

TEST(StandardHooks, AnUndecidedTestEndsTheSeries)
{
    Record r(80);
    with_hooks(r, 0, 40, [](HookTest& t) { t.undecided = true; });
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::SeriesEnd);
    ASSERT_EQ(r.state.segments.size(), 1u);
    EXPECT_EQ(r.state.segments[0].observation_count, 40u);
    EXPECT_EQ(r.state.segments[0].break_day, r.dates[40]);
}

TEST(StandardHooks, ABreakIntoThePeekDropsTheObservationsBeforeIt)
{
    Record r(80);
    with_hooks(r, 0, 40, [](HookTest& t) { t.change = true; t.skip = 2; });
    EXPECT_EQ(r.step(), Phase::Restart);
    ASSERT_EQ(r.state.segments.size(), 1u);
    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.change_probability, 1.0);
    EXPECT_EQ(s.end_day, r.dates[39]);
    EXPECT_EQ(s.break_day, r.dates[42]);
    // Masked positions 40 and 41 left; the change's first is at the window's
    // stop.
    EXPECT_EQ(r.state.series.size(), 78u);
    EXPECT_EQ(r.state.series.dates()(40), r.dates[42]);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 40u);
}

// pyccd's outlier rule, or EndSeries at every test.
struct Ending final : ccd::OutlierRule
{
    ccd::Capabilities needs() const override { return {}; }
    ccd::OutlierOutcome judge(const ccd::Peek&, const ccd::PeekResult&) const override
    {
        return ccd::OutlierOutcome::EndSeries;
    }
};

TEST(StandardHooks, AnOutlierRuleThatCannotTellEndsTheSeries)
{
    Record r(80);
    in_extend(r, 0, 40);
    r.strategies.outlier = std::make_shared<Ending>();
    EXPECT_EQ(ccd::standard::extend_step(r.context, r.state),
              ccd::standard::ExtendOutcome::SeriesEnd);
}

// A monitoring model whose every fit fails.
struct Unfittable final : ccd::MonitoringModel
{
    class Fits final : public ccd::Model
    {
    public:
        bool fit(ccd::ArrayView<const std::int64_t, 1>, ccd::ArrayView<const double, 2>, index_t,
                 index_t, const ccd::BandList&, ccd::KeepResiduals) override
        {
            return false;
        }
        void peek_residuals(ccd::ArrayView<const std::int64_t, 1>, ccd::ArrayView<const double, 2>,
                            const ccd::BandList&, std::vector<std::vector<double>>&) override {}
        double rmse(index_t) const override { return 0; }
        const std::vector<double>& residuals(index_t) const override { return none_; }
        ccd::BandFit band_fit(index_t) const override { return {}; }

    private:
        std::vector<double> none_;
    };

    ccd::Capabilities provides() const override { return {}; }
    std::unique_ptr<ccd::Model> make_model() const override { return std::make_unique<Fits>(); }
    ccd::CoefficientRule coefficients() const override { return {}; }
};

TEST(StandardHooks, AModelThatCannotFitEndsTheSeriesWithoutTheCurve)
{
    Record r(80);
    in_extend(r, 0, 40);
    r.strategies.model = std::make_shared<Unfittable>();
    EXPECT_EQ(r.step(), Phase::Done);
    EXPECT_EQ(r.state.exit, Exit::NoFit);
    EXPECT_TRUE(r.state.segments.empty());
    EXPECT_FALSE(r.state.extension.trained);
}

// pyccd's start search, but an exhausted search forces a start of the 10
// observations from where it began, or restores its series.
struct Forcing final : ccd::StartSearch
{
    std::shared_ptr<const ccd::StartSearch> pyccd;
    ccd::Exhaustion when;

    Forcing(std::shared_ptr<const ccd::StartSearch> pyccd, ccd::Exhaustion when)
        : pyccd(std::move(pyccd)), when(when) {}

    bool too_few(index_t n) const override { return pyccd->too_few(n); }
    ccd::Window first(index_t at) const override { return pyccd->first(at); }
    bool room(const ccd::Window& w, index_t n) const override { return pyccd->room(w, n); }
    ccd::SearchPlan begin(ccd::ArrayView<const std::int64_t, 1>, index_t,
                          std::int64_t) const override
    {
        ccd::SearchPlan plan;
        plan.exhausted = when;
        return plan;
    }
    bool exhausted(const ccd::Window& w, index_t n) const override { return pyccd->exhausted(w, n); }
    ccd::Window forced(ccd::ArrayView<const std::int64_t, 1>, index_t at,
                       std::int64_t) const override
    {
        return ccd::Window(at, at + 10);
    }
    bool short_span(ccd::ArrayView<const std::int64_t, 1> d) const override
    {
        return pyccd->short_span(d);
    }
    bool usable(ccd::ArrayView<const std::int64_t, 1> d) const override { return pyccd->usable(d); }
    index_t min_window() const override { return pyccd->min_window(); }
};

TEST(StandardHooks, AForcedStartTheSeriesEndsBeforeAFitReportsItsOwnFit)
{
    Record r(80);
    with_hooks(r, 0, 24, [](HookTest& t) { t.full = false; t.min = 30; });
    r.strategies.start_search =
        std::make_shared<Forcing>(r.strategies.start_search, ccd::Exhaustion::Force);
    r.at(58, 70, 58);   // exhausted: 70 + 12 >= 80
    r.state.cadence = ccd::standard::Cadence{6, 0, 30};
    EXPECT_EQ(ccd::standard::begin_search(r.context, r.state), Phase::Initialize);
    r.state.phase = Phase::Initialize;

    EXPECT_EQ(r.step(), Phase::Extend);
    EXPECT_EQ(r.state.cursor.window.start, 58u);
    EXPECT_EQ(r.state.cursor.window.stop, 68u);
    ASSERT_TRUE(r.state.forced);
    while (r.state.phase == Phase::Extend)
        r.step();

    // Below the test's minimum to the series end: the forced window's own fit.
    EXPECT_EQ(r.state.phase, Phase::Done);
    ASSERT_EQ(r.state.segments.size(), 1u);
    const ccd::ChangeModel& s = r.state.segments[0];
    EXPECT_EQ(s.start_day, r.dates[58]);
    EXPECT_EQ(s.end_day, r.dates[79]);
    EXPECT_EQ(s.break_day, r.dates[79]);
    EXPECT_EQ(s.observation_count, 10u);
    EXPECT_EQ(s.curve_qa, static_cast<ccd::CurveQA>(r.count("model", "coefficient_min")));
    EXPECT_FALSE(r.state.forced);
    EXPECT_EQ(r.state.exit, Exit::NoFit);
    EXPECT_FALSE(r.state.resumable);   // a record keeps no forced start
}

TEST(StandardHooks, ARestoringSearchGivesBackItsDropsWhenExhausted)
{
    Record r(80);
    r.strategies.start_search =
        std::make_shared<Forcing>(r.strategies.start_search, ccd::Exhaustion::Restore);
    r.with_variogram(kVariogram);
    // A window of 13: a window screen drops only what leaves a usable window
    // of at least min_window (12) observations.
    r.at(60, 73, 60);
    EXPECT_EQ(ccd::standard::begin_search(r.context, r.state), Phase::Initialize);
    r.state.series.drop(65, r.state.cursor.window);   // as a window screen's
    ASSERT_EQ(r.state.series.size(), 79u);
    ASSERT_EQ(r.state.cursor.window.size(), r.strategies.start_search->min_window());

    r.state.phase = Phase::Initialize;
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::InitializeExhausted);
    EXPECT_EQ(r.state.series.size(), 80u);
    EXPECT_FALSE(r.state.resumable);
}

TEST(StandardHooks, TheModesCapRefitsTheSegmentWithFewerCoefficients)
{
    const auto curve = [](index_t cap)
    {
        Record r(80);
        in_extend(r, 0, 74);
        r.context.max_coefficients = cap;
        r.step();
        return r.state.segments.at(0);
    };
    const ccd::ChangeModel full = curve(0);
    const ccd::ChangeModel capped = curve(4);
    EXPECT_EQ(full.curve_qa, static_cast<ccd::CurveQA>(8));
    EXPECT_EQ(capped.curve_qa, static_cast<ccd::CurveQA>(4));
    for (const ccd::BandFit& band : capped.bands)
        for (index_t c = 3; c < band.coefficients.size(); ++c)
            EXPECT_EQ(band.coefficients[c], 0.0);
    // The magnitudes are the test's, whatever the cap.
    for (index_t b = 0; b < full.bands.size(); ++b)
        EXPECT_EQ(capped.bands[b].magnitude, full.bands[b].magnitude);
}

TEST(StandardHooks, TheModesExtraMaskStepsFollowTheChain)
{
    // Two observations far above the green median: the mode's extra green
    // step after the chain drops them, as the chain with that step does.
    Record r(80);
    r.edit([](index_t b, index_t i, double v) { return (b == 1 && (i == 30 || i == 50)) ? v + 5000 : v; });
    const auto green = ccd::preprocess::Step::green_median(
        r.options.index("green"),
        r.resolved.plan.insufficient_clear.chain[4].green_offset,
        r.shared.statistics_end,
        ccd::GreenFilterScope::All);
    const ccd::preprocess::Chain chain{ccd::preprocess::Step::dedupe()};
    ccd::PixelMode mode(ccd::PixelProcedure::Standard);
    mode.extra_mask.push_back(green);

    ccd::ProcessingMask after;
    ccd::preprocess::apply(chain, r.workspace.dates(), r.workspace.spectral(), r.workspace.qas(),
                           after);
    ccd::preprocess::apply_steps(mode.extra_mask, r.workspace.dates(), r.workspace.spectral(),
                                 r.workspace.qas(), after);
    ccd::preprocess::Chain joined = chain;
    joined.push_back(green);
    ccd::ProcessingMask whole;
    ccd::preprocess::apply(joined, r.workspace.dates(), r.workspace.spectral(), r.workspace.qas(),
                           whole);
    EXPECT_EQ(after.data(), whole.data());
    EXPECT_FALSE(after[30]);
    EXPECT_FALSE(after[50]);
    EXPECT_EQ(after.count(), 78u);
}

TEST(StandardHooks, ARunTheRecordCannotContinueStoresNoSuspension)
{
    // Shortened peeks: no full-peek suspension to continue from.
    Record r(80);
    ccd::ProcessingMask mask(80);
    for (index_t i = 0; i < 80; ++i)
        mask.set(i, true);
    auto test = std::make_shared<HookTest>(r.strategies.change_test);
    ccd::standard::Procedure plain;
    plain.run(r.workspace, mask, r.strategies);
    EXPECT_NE(plain.suspended().exit, Exit::None);

    test->full = false;
    r.strategies.change_test = test;
    ccd::standard::Procedure shortened;
    shortened.run(r.workspace, mask, r.strategies);
    EXPECT_EQ(shortened.suspended().exit, Exit::None);

    // A mode's cap.
    r.strategies = golden::strategies_of(r.config);
    ccd::standard::Procedure capped;
    capped.run(r.workspace, mask, r.strategies, {}, 4);
    EXPECT_EQ(capped.suspended().exit, Exit::None);
}

//------------------------------------------------------------------------------
// Exits on whole pixels
//------------------------------------------------------------------------------

// standard::run() on a characterization pixel, preprocessed as detect() does
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
    const ccd::config::Resolved resolved = golden::resolved(p.config, p.h);
    const ccd::preprocess::PreprocessPlan plan =
        resolved.plan;
    const ccd::preprocess::ProcedurePlan& standard = plan.standard;

    std::vector<double> block = p.spectral;
    if (standard.transform)
    {
        const auto& affine = *standard.transform;
        for (index_t t = 0; t < T; ++t)
            block[affine.band * T + t] = block[affine.band * T + t] * affine.scale + affine.offset;
    }

    const ccd::PixelSeries workspace(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
        p.h, resolved.shared
    );

    ccd::ProcessingMask mask;
    ccd::preprocess::apply(
        standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

    const ccd::Strategies& strategies = resolved.strategies;
    const ccd::standard::Context context{resolved.shared, workspace, strategies};

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
        golden::spec_of(p.h, p.config),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})).result;
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
            EXPECT_EQ(x.bands[b].rmse, y.bands[b].rmse) << name << " " << m;
            EXPECT_EQ(x.bands[b].magnitude, y.bands[b].magnitude) << name << " " << m;
            EXPECT_EQ(x.bands[b].intercept, y.bands[b].intercept) << name;
            EXPECT_EQ(x.bands[b].coefficients, y.bands[b].coefficients) << name;
            EXPECT_EQ(x.bands[b].iterations, y.bands[b].iterations) << name;
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
    throw std::runtime_error("no characterization pixel " + name);
}

TEST(StandardExit, EachWayTheMachineStops)
{
    const auto all = golden::characterisation_pixels();

    // Two random pixels for the routes after a change: Restart with no room
    // left, and an Initialize that finds no stable window after a segment.
    std::vector<golden::CharPixel> pixels = all;
    pixels.push_back(golden::sparse_random("restart_no_room", 12));
    pixels.push_back(golden::sparse_random("exhausted_after_a_change", 8));

    const std::vector<std::pair<std::string, Exit>> expected{
        {"standard_too_few", Exit::TooFewObservations},
        {"variogram_nan", Exit::VariogramNaN},
        {"standard_no_room", Exit::NoRoomToInitialize},         // from Prepare
        {"restart_no_room", Exit::NoRoomToInitialize},          // from Restart
        {"constant_detection_band", Exit::InitializeExhausted}, // no segment
        {"exhausted_after_a_change", Exit::InitializeExhausted},  // after one
        {"peek_guard_tail", Exit::PeekBeyondEnd},
        {"start_catch", Exit::SeriesEnd},                       // after a grow
        {"series_end_after_outlier", Exit::SeriesEnd},          // after a drop
    };

    for (const auto& [name, exit] : expected)
    {
        const golden::CharPixel& p = pixel(pixels, name);
        const MachineRun run = run_machine(p);
        EXPECT_EQ(run.exit, exit) << name << ": " << ccd::standard::to_string(run.exit);
        expect_same(run.result, run_detect(p), name);
    }
}

TEST(StandardExit, TheEarlyExitsReturnTheMask)
{
    const auto all = golden::characterisation_pixels();
    for (const char* name : {"standard_too_few", "variogram_nan"})
    {
        const auto& p = pixel(all, name);
        const MachineRun run = run_machine(p);
        EXPECT_TRUE(run.result.models.empty()) << name;
        EXPECT_EQ(run.result.mask.size(), p.dates.size()) << name;
        EXPECT_GT(run.result.mask.count(), 0u) << name;
    }
}

// Every self-transition of a whole pixel is one decision, and the stepped
// machine is the driven one: on each characterization pixel, stepped by
// hand, each Initialize -> Initialize grows by one, shifts by one, or shifts
// after a Tmask drop; each Lookback -> Lookback moves window.start down one
// (an include, or a drop that moves the window with the series); each Extend
// -> Extend grows by one or drops the observation at window.stop; and the
// result equals standard::run()'s.
TEST(StandardMachine, EachStepIsOneDecision)
{
    const auto all = golden::characterisation_pixels();
    std::size_t initialize_steps = 0, lookback_steps = 0, extend_steps = 0;

    for (const auto& p : all)
    {
        const index_t T = p.dates.size();
        const ccd::config::Resolved resolved = golden::resolved(p.config, p.h);
        const ccd::preprocess::PreprocessPlan plan =
            resolved.plan;
        std::vector<double> block = p.spectral;
        if (plan.standard.transform)
        {
            const auto& affine = *plan.standard.transform;
            for (index_t t = 0; t < T; ++t)
                block[affine.band * T + t] = block[affine.band * T + t] * affine.scale + affine.offset;
        }
        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
            p.h, resolved.shared
        );
        ccd::ProcessingMask mask;
        ccd::preprocess::apply(
            plan.standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

        const ccd::Strategies& strategies = resolved.strategies;
        const ccd::standard::Context context{resolved.shared, workspace, strategies};

        ccd::standard::State driven;
        const ccd::FitResult want = ccd::standard::run(context, driven, mask);

        ccd::standard::State s;
        s.reset(workspace, mask);
        while (s.phase != Phase::Done)
        {
            const ccd::Window w = s.cursor.window;
            const index_t n = s.series.size();
            const std::size_t segments = s.segments.size();
            const Phase phase = s.phase;

            const Phase next = ccd::standard::Machine::step(context, s);
            const ccd::Window v = s.cursor.window;

            if (phase == Phase::Initialize && next == Phase::Initialize)
            {
                ++initialize_steps;
                const bool grew = s.series.size() == n && v.start == w.start && v.stop == w.stop + 1;
                const bool shifted = s.series.size() == n && v.start == w.start + 1 && v.stop == w.stop + 1;
                const bool dropped_then_shifted =
                    s.series.size() < n && v.start == w.start + 1 &&
                    v.stop == w.stop - (n - s.series.size()) + 1;
                EXPECT_TRUE(grew || shifted || dropped_then_shifted) << p.name;
            }
            if (phase == Phase::Lookback && next == Phase::Lookback)
            {
                ++lookback_steps;
                EXPECT_EQ(v.start, w.start - 1) << p.name;
                EXPECT_LE(n - s.series.size(), 1u) << p.name;
                EXPECT_EQ(v.stop, w.stop - (n - s.series.size())) << p.name;
            }
            if (phase == Phase::Extend && next == Phase::Extend)
            {
                ++extend_steps;
                const bool grew = s.series.size() == n && v.stop == w.stop + 1;
                const bool dropped = s.series.size() == n - 1 && v.stop == w.stop;
                EXPECT_TRUE(grew || dropped) << p.name;
                EXPECT_EQ(v.start, w.start) << p.name;
            }
            if (phase == Phase::Extend && next == Phase::Restart)
                EXPECT_EQ(s.segments.size(), segments + 1) << p.name;

            s.phase = next;
        }
        expect_same(s.take_result(), want, p.name);
    }

    // The corpus takes every self-transition many times.
    EXPECT_GT(initialize_steps, 100u);
    EXPECT_GT(lookback_steps, 10u);
    EXPECT_GT(extend_steps, 1000u);
}

TEST(StandardMachine, OneStateServesManyPixels)
{
    // A record reused across pixels gives each pixel the result a fresh one
    // does: reset() starts every pixel from Prepare with nothing carried.
    const auto all = golden::characterisation_pixels();
    ccd::standard::Procedure reused;

    for (const auto& p : all)
    {
        const index_t T = p.dates.size();
        const ccd::config::Resolved resolved = golden::resolved(p.config, p.h);
        const ccd::preprocess::PreprocessPlan plan =
            resolved.plan;
        std::vector<double> block = p.spectral;
        if (plan.standard.transform)
        {
            const auto& affine = *plan.standard.transform;
            for (index_t t = 0; t < T; ++t)
                block[affine.band * T + t] = block[affine.band * T + t] * affine.scale + affine.offset;
        }
        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
            p.h, resolved.shared
        );
        ccd::ProcessingMask mask;
        ccd::preprocess::apply(
            plan.standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

        ccd::standard::Procedure fresh;
        const ccd::Strategies strategies = golden::strategies_of(p.config);
        const ccd::FitResult once = fresh.run(workspace, mask, strategies);
        const ccd::FitResult again = reused.run(workspace, mask, strategies);
        expect_same(again, once, p.name);
    }
}

} // namespace

//------------------------------------------------------------------------------
// The per-pixel procedure and the detector
//------------------------------------------------------------------------------

TEST(SelectProcedure, TheClearThenTheSnowPercentage)
{
    const auto mode = golden::strategies_of().mode;   // clear 0.25, snow 0.75

    auto stats = [](scalar_t clear, scalar_t snow)
    {
        ccd::QaStatistics s;
        s.clear_probability = clear;
        s.snow_probability = snow;
        return s;
    };

    // At, just above and just below each threshold; enough clear wins
    // whatever the snow percentage.
    EXPECT_EQ(mode->choose(stats(0.25, 0.9)).procedure, ccd::PixelProcedure::Standard);
    EXPECT_EQ(mode->choose(stats(0.2501, 0.0)).procedure, ccd::PixelProcedure::Standard);
    EXPECT_EQ(mode->choose(stats(0.2499, 0.75)).procedure,
              ccd::PixelProcedure::PermanentSnow);
    EXPECT_EQ(mode->choose(stats(0.2499, 0.7501)).procedure,
              ccd::PixelProcedure::PermanentSnow);
    EXPECT_EQ(mode->choose(stats(0.2499, 0.7499)).procedure,
              ccd::PixelProcedure::InsufficientClear);
}

TEST(StandardDetector, FitsStandardWithTheMachineAndTheRestWholeSeries)
{
    // Standard is standard::Procedure::run(); PermanentSnow and
    // InsufficientClear are the shared whole-series fit, whose curve_qa
    // names the procedure.
    Record r(80);
    ccd::ProcessingMask mask(80);
    for (index_t i = 0; i < 80; ++i)
        mask.set(i, true);

    ccd::standard::Detector detector;
    ccd::standard::Procedure standard;
    expect_same(
        detector.fit(ccd::PixelProcedure::Standard, r.workspace, mask,
                     r.strategies),
        standard.run(r.workspace, mask, r.strategies),
        "standard");

    for (const auto procedure :
         {ccd::PixelProcedure::PermanentSnow, ccd::PixelProcedure::InsufficientClear})
    {
        const ccd::FitResult fit =
            detector.fit(procedure, r.workspace, mask, r.strategies);
        expect_same(
            fit,
            ccd::fit_whole_series(procedure, r.workspace, mask,
                                  r.count("edge_fit", "min_observations"),
                                  r.count("edge_fit", "coefficients"), r.lasso, r.solver),

            "whole series");
        ASSERT_EQ(fit.models.size(), 1u);
        EXPECT_EQ(fit.models[0].curve_qa,
                  procedure == ccd::PixelProcedure::PermanentSnow
                      ? ccd::CurveQA::PersistentSnow
                      : ccd::CurveQA::InsufficientClear);
        EXPECT_EQ(fit.models[0].observation_count, 80u);
        EXPECT_EQ(fit.mask.count(), 80u);
    }
}
