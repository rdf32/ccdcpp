//==============================================================================
//
// The Standard machine on whole pixels, the per-pixel procedure choice and
// the Standard detector's fit of each procedure.
//
//   * Whole pixels stepped by hand: every self-transition is one decision,
//     and the result is the driven machine's.
//   * Exits: Detector::fit() on the characterization pixels that end the
//     segment loop each way, one test per pixel, with the same preprocessing
//     ccd::detect() runs, against the Exit the run leaves in the record. The
//     result of each run is also checked against detect()'s, field for field.
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
#include <ostream>
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

#include "characterization.hpp"
#include "standard_record.hpp"

namespace
{

using ccd::index_t;
using ccd::scalar_t;
using ccd::standard::Exit;
using ccd::standard::Phase;
using machine_test::Record;
using machine_test::expect_same;

//------------------------------------------------------------------------------
// Exits on whole pixels
//------------------------------------------------------------------------------

// Detector::fit() on a characterization pixel, preprocessed as detect() does
// for the Standard procedure: the plan's transform on a copy, its chain for
// the mask.
struct MachineRun
{
    ccd::FitResult result;
    Exit exit;
};

MachineRun run_machine(const golden::Series& p)
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
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})
    );

    ccd::ProcessingMask mask;
    ccd::preprocess::apply(
        standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

    ccd::standard::Detector detector;
    MachineRun out;
    out.result = detector.fit(ccd::PixelProcedure::Standard, workspace, std::move(mask),
                              resolved.shared, resolved.strategies);
    out.exit = detector.exit();
    return out;
}

// A pixel and the way the machine stops on it.
struct Stop
{
    const char* pixel;
    Exit exit;
};

void PrintTo(const Stop& stop, std::ostream* os)
{
    *os << stop.pixel << ": " << ccd::standard::to_string(stop.exit);
}

const Stop STOPS[] = {
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

class EachWayTheMachineStops : public ::testing::TestWithParam<Stop>
{
};

// One test per pixel, named for it: the exit Detector::fit() leaves, and the
// run's result against detect()'s.
TEST_P(EachWayTheMachineStops, Pixel)
{
    // The characterization pixels, and two random pixels for the routes after
    // a change: Restart with no room left, and an Initialize that finds no
    // stable window after a segment.
    static const std::vector<golden::Series> pixels = []
    {
        std::vector<golden::Series> all = golden::characterization_pixels();
        all.push_back(golden::sparse_random("restart_no_room", 12));
        all.push_back(golden::sparse_random("exhausted_after_a_change", 8));
        return all;
    }();

    const auto& [name, exit] = GetParam();
    const golden::Series& p = golden::named(pixels, name);
    const MachineRun run = run_machine(p);
    EXPECT_EQ(run.exit, exit) << name << ": " << ccd::standard::to_string(run.exit);
    expect_same(run.result, golden::detect(p), name);
}

INSTANTIATE_TEST_SUITE_P(StandardExit, EachWayTheMachineStops, ::testing::ValuesIn(STOPS),
                         [](const auto& info) { return std::string(info.param.pixel); });

TEST(StandardExit, TheEarlyExitsReturnTheMask)
{
    const auto all = golden::characterization_pixels();
    for (const char* name : {"standard_too_few", "variogram_nan"})
    {
        const auto& p = golden::named(all, name);
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
// result equals Detector::fit()'s.
TEST(StandardMachine, EachStepIsOneDecision)
{
    const auto all = golden::characterization_pixels();
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
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})
        );
        ccd::ProcessingMask mask;
        ccd::preprocess::apply(
            plan.standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

        const ccd::Strategies& strategies = resolved.strategies;
        const ccd::standard::Context context{resolved.shared, workspace, strategies};

        ccd::standard::Detector driven;
        const ccd::FitResult want =
            driven.fit(ccd::PixelProcedure::Standard, workspace, mask, resolved.shared, strategies);

        ccd::standard::State s;
        s.bind(strategies);
        s.reset(workspace, mask);
        while (s.phase != Phase::Done)
        {
            const ccd::Window w = s.cursor.window;
            const index_t n = s.series.size();
            const std::size_t segments = s.segments.size();
            const Phase phase = s.phase;

            const Phase next = ccd::standard::step(context, s);
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
    const auto all = golden::characterization_pixels();
    ccd::standard::Detector reused;

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
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T})
        );
        ccd::ProcessingMask mask;
        ccd::preprocess::apply(
            plan.standard.chain, workspace.dates(), workspace.spectral(), workspace.qas(), mask);

        ccd::standard::Detector fresh;
        const ccd::Strategies strategies = golden::strategies_of(p.config);
        const ccd::FitResult once = fresh.fit(ccd::PixelProcedure::Standard, workspace, mask,
                                              resolved.shared, strategies);
        const ccd::FitResult again = reused.fit(ccd::PixelProcedure::Standard, workspace, mask,
                                                resolved.shared, strategies);
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
    // Standard is the machine stepped from Prepare to Done; PermanentSnow
    // and InsufficientClear are the shared whole-series fit, whose curve_qa
    // names the procedure.
    Record r(80);
    ccd::ProcessingMask mask(80);
    for (index_t i = 0; i < 80; ++i)
        mask.set(i, true);

    ccd::standard::Detector detector;
    const ccd::FitResult standard =
        detector.fit(ccd::PixelProcedure::Standard, r.workspace, mask, r.shared, r.strategies);
    r.state.reset(r.workspace, mask);
    while (r.state.phase != Phase::Done)
        r.step();
    expect_same(standard, r.state.take_result(), "standard");

    for (const auto procedure :
         {ccd::PixelProcedure::PermanentSnow, ccd::PixelProcedure::InsufficientClear})
    {
        const ccd::FitResult fit =
            detector.fit(procedure, r.workspace, mask, r.shared, r.strategies);
        expect_same(
            fit,
            ccd::fit_whole_series(procedure, r.workspace, mask,
                                  r.count("edge_segments", "min_observations"),
                                  r.count("edge_segments", "coefficients.min"), r.lasso, r.solver),

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
