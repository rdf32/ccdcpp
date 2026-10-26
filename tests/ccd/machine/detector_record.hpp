#pragma once

// The hand-built record test_machine's sources (test_detector_steps.cpp,
// test_detector_hooks.cpp, test_detector_whole_pixel.cpp) step the machine
// on, a record in Extend, and two fit results compared field for
// field. Not part of the library.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/machine/detector.hpp"   // internal (src/)
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "config.hpp"

namespace machine_test
{

using ccd::index_t;
using ccd::scalar_t;
using ccd::machine::Phase;

//------------------------------------------------------------------------------
// A hand-built record
//
// `n` clear observations 16 days apart from `first_day` with a smooth
// seasonal signal, every one kept by the mask, and a Context over them. The record's phase, cursor,
// peek plan and segments are set by each test.
//------------------------------------------------------------------------------
struct Record
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, n)
    std::vector<std::uint8_t> qas;
    ccd::DataSpec options;
    ccd::config::Config config = golden::pyccd();   // the strategies' configuration
    ccd::config::Resolved resolved = golden::resolved(config, options);
    const ccd::config::Shared& shared = resolved.shared;
    ccd::LassoOptions lasso_options;   // the kernels' solver, at the preset's values

    ccd::PixelSeries workspace;
    ccd::LassoWorkspace lasso{0};
    ccd::LassoSolver solver;
    ccd::Strategies strategies;
    ccd::machine::Context context;
    ccd::machine::State state;

    explicit Record(index_t n, std::int64_t first_day = 730120)
        : dates(make_dates(n, first_day)),
          spectral(make_spectral(n)),
          qas(n, 0),
          workspace(
              ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n}),
              ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, n}),
              ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {n})),
          solver(lasso_options),
          strategies(resolved.strategies),
          context{shared, workspace, strategies}
    {
        ccd::ProcessingMask mask(n);
        for (index_t i = 0; i < n; ++i)
            mask.set(i, true);
        state.bind(strategies);
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

    // The record at a cursor, with a peek plan, past Prepare.
    void at(index_t start, index_t stop, index_t prev_break, index_t peek = 6)
    {
        state.cursor = ccd::machine::Cursor{ccd::Window(start, stop), prev_break};
        state.peek_plan = ccd::PeekPlan{.peek = peek, .full = true};   // chi2: full peeks only
        state.stability_coefficients = strategies.stability_fit->coefficients();
        state.coefficients = strategies.model->coefficients();

        // The change test as Prepare leaves it on a series at the reference
        // cadence (16 days): its threshold, change_threshold.
        const std::int64_t reference[] = {730000, 730016};
        state.bind(strategies);
        strategies.change_test->prepare(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(reference, {2}),
            shared.detection_bands.size(), *state.extend.test);
        // The stability rule's threshold, change_threshold (a peek of the
        // change test's peek_size), whatever peek the record has.
        const index_t peek_size = count("change_test", "peek_size");
        const ccd::PeekPlan plan{.peek = peek_size, .reference = peek_size};
        strategies.stability_rule->prepare(
            plan, shared.detection_bands.size(), *state.stability);
        // Lookback over `peek` observations against its change_threshold:
        // pyccd's at a peek of its own, so its threshold is unadjusted.
        strategies.lookback = std::make_shared<const ccd::strategies::PyccdLookback>(
            ccd::strategies::PyccdLookback::Params{
                .change_threshold = real("lookback", "change_threshold"),
                .change_probability = real("lookback", "change_probability"),
                .outlier_threshold = real("lookback", "outlier_threshold"),
                .peek = peek});
        state.bind(strategies);
        strategies.lookback->prepare(plan, shared.detection_bands.size(), *state.lookback);
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

    // Restarts the record keeping reason codes (every observation kept,
    // Reason::InSegment), which its drops write.
    void with_reasons()
    {
        const index_t n = dates.size();
        ccd::ProcessingMask mask(n);
        for (index_t i = 0; i < n; ++i)
            mask.set(i, true);
        state.reset(workspace, std::move(mask),
                    std::vector<std::uint8_t>(
                        n, static_cast<std::uint8_t>(ccd::Reason::InSegment)));
    }

    // A variogram of `value` in every band, as Prepare would leave one.
    void with_variogram(scalar_t value)
    {
        state.noise.bands.assign(7, value);
    }

    // Initialize's stable fits over (start, stop), as a Stable step leaves
    // them for Lookback: the detection bands, the stability fit's
    // coefficients.min coefficients, residuals kept.
    void with_stability_fit(index_t start, index_t stop)
    {
        state.stability_model->fit(
            state.series.dates().slice(ccd::range(start, stop)),
            state.series.spectral().slice(ccd::all(), ccd::range(start, stop)),
            count("stability_fit", "coefficients.min"),
            count("stability_fit", "coefficients.max"),
            shared.detection_bands,
            ccd::KeepResiduals::Yes
        );
    }

    // One transition of the record's phase, as the Detector makes it, with
    // the record's strategies bound (a test may have replaced one).
    Phase step()
    {
        state.bind(strategies);
        const Phase next = ccd::machine::step(context, state);
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

inline constexpr scalar_t kVariogram = 50.0;   // Tmask flags a residual above 244.5

// A record in Extend at (start, stop), untrained, as the peek guard leaves it.
inline void in_extend(Record& r, index_t start, index_t stop, index_t prev_break = 0)
{
    r.with_variogram(kVariogram);
    r.at(start, stop, prev_break);
    r.state.phase = Phase::Extend;
}

inline void expect_same(const ccd::FitResult& a, const ccd::FitResult& b, const std::string& name)
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

} // namespace machine_test
