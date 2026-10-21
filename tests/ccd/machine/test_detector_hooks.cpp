//==============================================================================
//
// The machine's routes under test-only strategies, from hand-built
// records: each strategy has chi2's or pyccd's decisions but for the hook it
// sets, and the machine takes the route that hook leads to.
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
#include "ccd/machine/detector.hpp"   // internal (src/)
#include "ccd/regression/lasso_solver.hpp"
#include "config.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/pyccd/peek_statistics.hpp"
#include "ccd/strategies/library.hpp"

#include "detector_record.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;
using ccd::machine::Exit;
using ccd::machine::Phase;
using machine_test::Record;
using machine_test::expect_same;
using machine_test::in_extend;
using machine_test::kVariogram;

//------------------------------------------------------------------------------
// The routes of other strategies
//
// Test-only strategies on the record, each with chi2's or pyccd's decisions
// but for the hook it sets: shortened peeks, a test's minimum window, its
// Undecided and a break past the peek's first observation, the outlier
// rule's EndSeries, a model that cannot fit, a forced and a restoring start
// search, the mode's cap and green filter.
//------------------------------------------------------------------------------

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
// (start, stop) with the peek plan the test's hooks give.
template<typename Hooks>
void with_hooks(Record& r, index_t start, index_t stop, Hooks hooks)
{
    auto test = std::make_shared<HookTest>(r.strategies.change_test);
    hooks(*test);
    r.strategies.change_test = test;
    in_extend(r, start, stop);
    r.state.peek_plan = ccd::PeekPlan{6, test->full, test->min};
}

TEST(MachineHooks, ShortenedPeeksGrowTheCurveToTheLastObservation)
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

TEST(MachineHooks, AWindowBelowTheTestsMinimumGrowsUntested)
{
    Record r(80);
    with_hooks(r, 0, 24, [](HookTest& t) { t.min = 30; });
    EXPECT_EQ(ccd::machine::extend_step(r.context, r.state), ccd::machine::ExtendOutcome::Grown);
    EXPECT_FALSE(r.state.extend.trained);
    EXPECT_EQ(r.state.cursor.window.start, 0u);
    EXPECT_EQ(r.state.cursor.window.stop, 25u);

    // At the series end: no fit, no segment, Done.
    Record end(30);
    with_hooks(end, 0, 30, [](HookTest& t) { t.full = false; t.min = 40; });
    EXPECT_EQ(end.step(), Phase::Done);
    EXPECT_EQ(end.state.exit, Exit::NoFit);
    EXPECT_TRUE(end.state.segments.empty());
}

TEST(MachineHooks, AnUndecidedTestEndsTheSeries)
{
    Record r(80);
    with_hooks(r, 0, 40, [](HookTest& t) { t.undecided = true; });
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::SeriesEnd);
    ASSERT_EQ(r.state.segments.size(), 1u);
    EXPECT_EQ(r.state.segments[0].observation_count, 40u);
    EXPECT_EQ(r.state.segments[0].break_day, r.dates[40]);
}

TEST(MachineHooks, ABreakIntoThePeekDropsTheObservationsBeforeIt)
{
    Record r(80);
    r.with_reasons();
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
    // Their code is BeforeChange.
    std::vector<std::uint8_t> reasons(80, static_cast<std::uint8_t>(ccd::Reason::Kept));
    reasons[40] = reasons[41] = static_cast<std::uint8_t>(ccd::Reason::BeforeChange);
    EXPECT_EQ(r.state.series.reasons(), reasons);
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

TEST(MachineHooks, AnOutlierRuleThatCannotTellEndsTheSeries)
{
    Record r(80);
    in_extend(r, 0, 40);
    r.strategies.outlier = std::make_shared<Ending>();
    EXPECT_EQ(ccd::machine::extend_step(r.context, r.state),
              ccd::machine::ExtendOutcome::SeriesEnd);
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

TEST(MachineHooks, AModelThatCannotFitEndsTheSeriesWithoutTheCurve)
{
    Record r(80);
    in_extend(r, 0, 40);
    r.strategies.model = std::make_shared<Unfittable>();
    EXPECT_EQ(r.step(), Phase::Done);
    EXPECT_EQ(r.state.exit, Exit::NoFit);
    EXPECT_TRUE(r.state.segments.empty());
    EXPECT_FALSE(r.state.extend.trained);
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

TEST(MachineHooks, AForcedStartTheSeriesEndsBeforeAFitReportsItsOwnFit)
{
    Record r(80);
    with_hooks(r, 0, 24, [](HookTest& t) { t.full = false; t.min = 30; });
    r.strategies.start_search =
        std::make_shared<Forcing>(r.strategies.start_search, ccd::Exhaustion::Force);
    r.at(58, 70, 58);   // exhausted: 70 + 12 >= 80
    r.state.peek_plan = ccd::PeekPlan{6, false, 30};
    EXPECT_EQ(ccd::machine::begin_search(r.context, r.state), Phase::Initialize);
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
    EXPECT_EQ(s.curve_qa, static_cast<ccd::CurveQA>(r.count("model", "coefficients.min")));
    EXPECT_FALSE(r.state.forced);
    EXPECT_EQ(r.state.exit, Exit::NoFit);
    EXPECT_FALSE(r.state.resumable);   // a record keeps no forced start
}

TEST(MachineHooks, ARestoringSearchGivesBackItsDropsWhenExhausted)
{
    Record r(80);
    r.with_reasons();
    r.strategies.start_search =
        std::make_shared<Forcing>(r.strategies.start_search, ccd::Exhaustion::Restore);
    r.with_variogram(kVariogram);
    // A window of 13: a window screen drops only what leaves a usable window
    // of at least min_window (12) observations.
    r.at(60, 73, 60);
    EXPECT_EQ(ccd::machine::begin_search(r.context, r.state), Phase::Initialize);
    r.state.series.drop(65, r.state.cursor.window, ccd::Reason::WindowScreen);   // as a window screen's
    ASSERT_EQ(r.state.series.size(), 79u);
    ASSERT_EQ(r.state.series.reasons()[65], static_cast<std::uint8_t>(ccd::Reason::WindowScreen));
    ASSERT_EQ(r.state.cursor.window.size(), r.strategies.start_search->min_window());

    r.state.phase = Phase::Initialize;
    EXPECT_EQ(r.step(), Phase::EndCatch);
    EXPECT_EQ(r.state.exit, Exit::InitializeExhausted);
    EXPECT_EQ(r.state.series.size(), 80u);
    // Its reason codes come back with it.
    EXPECT_EQ(r.state.series.reasons(),
              std::vector<std::uint8_t>(80, static_cast<std::uint8_t>(ccd::Reason::Kept)));
    EXPECT_FALSE(r.state.resumable);
}

TEST(MachineHooks, TheModesCapRefitsTheSegmentWithFewerCoefficients)
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

TEST(MachineHooks, TheModesExtraMaskStepsFollowTheChain)
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
    const ccd::preprocess::Chain chain{ccd::preprocess::Step::one_per_date()};
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

TEST(MachineHooks, ARunTheRecordCannotContinueStoresNoSuspension)
{
    // Shortened peeks: no full-peek suspension to continue from.
    Record r(80);
    ccd::ProcessingMask mask(80);
    for (index_t i = 0; i < 80; ++i)
        mask.set(i, true);
    auto test = std::make_shared<HookTest>(r.strategies.change_test);
    ccd::machine::Detector plain;
    plain.fit(ccd::PixelProcedure::Standard, r.workspace, mask, r.shared, r.strategies);
    EXPECT_NE(plain.suspended().exit, Exit::None);

    test->full = false;
    r.strategies.change_test = test;
    ccd::machine::Detector shortened;
    shortened.fit(ccd::PixelProcedure::Standard, r.workspace, mask, r.shared, r.strategies);
    EXPECT_EQ(shortened.suspended().exit, Exit::None);

    // A mode's cap.
    r.strategies = golden::strategies_of(r.config);
    ccd::PixelMode cap(ccd::PixelProcedure::Standard);
    cap.max_coefficients = 4;
    ccd::machine::Detector capped;
    capped.fit(cap, r.workspace, mask, r.shared, r.strategies);
    EXPECT_EQ(capped.suspended().exit, Exit::None);
}

} // namespace
