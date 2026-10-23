//==============================================================================
//
// Each library strategy against what it stands for, built from the registry:
//
//   * the lasso monitoring model fits, predicts and scores as the shared
//     kernels do, bit for bit, with the solver options it was configured with;
//   * the chi2 change test's peek is the adjusted peek of its parameters, and
//     the threshold memo is adjust_change_threshold() for any arguments;
//   * a configuration whose strategy needs what no earlier stage provides is
//     refused, naming both; the floored outlier rule judges the change
//     test's statistic against its threshold;
//   * the pyccd stability rule is pyccd's check of the stability fit, against
//     its threshold for the pixel's peek;
//   * the tmask window screen is tmask() of its bands and t_const;
//   * the pyccd start search's decisions are pyccd's comparisons with
//     min_observations (the shared value) and day_delta;
//   * pyccd's Lookback takes its thresholds from the shared values, adjusted
//     for a longer peek; the lasso edge segments are the catch-segment and
//     whole-series kernels; the variogram noise is the adjusted variogram.
//
//==============================================================================

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/config/registry.hpp"
#include "ccd/config/resolve.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/core/maths.hpp"
#include "ccd/core/window.hpp"
#include "ccd/results/segment.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"
#include "ccd/strategies/noise/variogram.hpp"
#include "ccd/strategies/pyccd/peek_statistics.hpp"
#include "ccd/strategies/window_screen/tmask.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/refit/span_factor.hpp"
#include "config.hpp"
#include "units.hpp"
#include "synthetic.hpp"
#include "ccd/strategies/library.hpp"

#include "expect_refused.hpp"

namespace
{

using registry_test::expect_refused;

} // namespace

TEST(MonitoringModel, TheLassoIsTheKernelsFit)
{
    ccd::config::Config tree = golden::ccdc();
    tree.shared["lasso.alpha"] = ccd::config::real(0.5);
    tree.shared["lasso.max_iter"] = ccd::config::integer(300);
    EXPECT_EQ(tree.stages.at("model").strategy, "lasso");
    ccd::DataSpec reflectance;
    reflectance.reflectance_scale = 1.0;
    const auto s = golden::strategies_of(tree, reflectance);
    const auto& lasso = static_cast<const ccd::strategies::Lasso&>(*s.model);
    EXPECT_EQ(lasso.solver_options().alpha, 0.5);
    EXPECT_EQ(lasso.solver_options().max_iter, 300u);

    // 40 observations of 3 bands, 16 days apart; a 6-observation peek after.
    const ccd::index_t n = 40, p = 6, bands = 3;
    std::vector<std::int64_t> dates(n + p);
    std::vector<double> spectral(bands * (n + p));
    for (ccd::index_t t = 0; t < n + p; ++t)
    {
        dates[t] = 730000 + 16 * static_cast<std::int64_t>(t);
        for (ccd::index_t b = 0; b < bands; ++b)
            spectral[b * (n + p) + t] =
                1000.0 * (b + 1) + 300.0 * std::sin(0.27 * t + b) + 17.0 * ((t * 7 + b) % 5);
    }
    const auto all_dates =
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n + p});
    const auto all_spectral =
        ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {bands, n + p});
    const auto window = ccd::range(ccd::index_t{0}, n), peek = ccd::range(n, n + p);
    const ccd::BandList fitted{0, 2};

    auto model = s.model->make_model();
    model->fit(all_dates.slice(window), all_spectral.slice(ccd::all(), window), 6, 8, fitted,
               ccd::KeepResiduals::Yes);
    std::vector<std::vector<double>> got(bands);
    model->peek_residuals(all_dates.slice(peek), all_spectral.slice(ccd::all(), peek), fitted,
                          got);

    // The kernels' solver with the model's parameters.
    ccd::LassoOptions l;
    l.alpha = 0.5;
    l.max_iter = 300;
    ccd::LassoWorkspace workspace(0);
    ccd::LassoSolver solver(l);
    std::vector<ccd::LassoResult> fits(bands);
    ccd::fit_bands(all_dates.slice(window), all_spectral.slice(ccd::all(), window), 6, fitted,
                   ccd::KeepResiduals::Yes, workspace, solver, fits);
    ccd::HarmonicBasis basis;
    basis.build(all_dates.slice(peek), 8);
    std::vector<std::vector<double>> want(bands);
    ccd::peek_residuals(basis, all_spectral.slice(ccd::all(), peek), fits, fitted, want);

    for (const ccd::index_t b : fitted)
    {
        EXPECT_EQ(got[b], want[b]) << "band " << b;
        EXPECT_EQ(model->rmse(b), fits[b].score.rmse) << "band " << b;
        EXPECT_EQ(model->residuals(b), fits[b].score.residuals) << "band " << b;
        const ccd::BandFit a = model->band_fit(b), c = ccd::band_fit(fits[b]);
        EXPECT_EQ(a.intercept, c.intercept) << "band " << b;
        EXPECT_EQ(a.coefficients, c.coefficients) << "band " << b;
        EXPECT_EQ(a.iterations, c.iterations) << "band " << b;
    }
    EXPECT_TRUE(got[1].empty());   // band 1 was not fitted

    const auto& r = golden::core_registry();
    expect_refused([&] { golden::strategies_of(golden::with_shared(tree, "lasso.alpha",
                                                                   ccd::config::real(NAN))); },
                   "shared lasso parameter 'alpha' is nan");
    expect_refused(
        [&] { r.make<ccd::MonitoringModel>({"lasso", {{"lasso.alpha", ccd::config::real(-1)}}}); },
        "model 'lasso' parameter 'lasso.alpha' is -1; it must be at least 0");
}

// chi2's peek is adjust_peek() of its parameters (peek_size and
// peek_reference_days) at the reference cadence (no adjustment) and at
// denser ones; the threshold memo gives adjust_change_threshold()'s value
// for every sequence of arguments.
TEST(ChangeTest, Chi2PeekIsTheAdjustedPeek)
{
    const ccd::config::ConfigTree tree = golden::resolved().stages;
    const auto peek_size =
        static_cast<ccd::index_t>(golden::parameter<std::int64_t>(tree, "change_test", "peek_size"));
    const double reference_days =
        golden::parameter<double>(tree, "change_test", "peek_reference_days");
    const double threshold = golden::parameter<double>(tree, "change_test", "change_threshold");
    const auto s = golden::strategies_of();
    const ccd::index_t detection = ccd::config::Shared{}.detection_bands.size();
    const auto scratch = s.change_test->make_scratch();
    for (const std::int64_t gap : {16, 8, 5})
    {
        std::vector<std::int64_t> dates(100);
        for (std::size_t i = 0; i < dates.size(); ++i)
            dates[i] = 730000 + gap * static_cast<std::int64_t>(i);
        const auto view = ccd::ArrayView<const std::int64_t, 1>::contiguous(
            dates.data(), {dates.size()});

        const ccd::PeekPlan plan =
            s.change_test->prepare(view, detection, *scratch);
        EXPECT_TRUE(plan.full);
        EXPECT_EQ(plan.min_window, 0u);
        const ccd::index_t peek = plan.peek;
        EXPECT_EQ(peek, ccd::adjust_peek(view, peek_size, reference_days)) << gap;
        if (gap == 16)
            EXPECT_EQ(peek, peek_size);
        else
            EXPECT_GT(peek, peek_size) << gap;
    }

    // Fewer than two dates: the default peek.
    const std::int64_t one = 730000;
    EXPECT_EQ(s.change_test->prepare(ccd::ArrayView<const std::int64_t, 1>::contiguous(&one, {1}),
                                     detection, *scratch).peek,
              peek_size);

    ccd::ChangeThresholdMemo memo;
    for (const ccd::index_t peek : {6u, 9u, 9u, 6u, 12u, 12u, 9u})
        for (const double probability : {0.99, 0.95, 0.99})
            EXPECT_EQ(memo(peek, 6, threshold, 7, probability),
                      ccd::adjust_change_threshold(peek, 6, threshold, 7, probability))
                << peek << " " << probability;
}

namespace
{

// Test-only strategies: a monitoring model and a stability fit that provide
// nothing, and a change test that provides nothing but needs no more than
// chi2.
struct Opaque final : ccd::MonitoringModel
{
    struct Params
    {
    };
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s;
        return s;
    }
    explicit Opaque(const Params&) {}
    ccd::Capabilities provides() const override { return {}; }
    std::unique_ptr<ccd::Model> make_model() const override { return nullptr; }
    ccd::CoefficientRule coefficients() const override { return {}; }
};

struct OpaqueFit final : ccd::StabilityFit
{
    struct Params
    {
    };
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s;
        return s;
    }
    explicit OpaqueFit(const Params&) {}
    ccd::Capabilities provides() const override { return {}; }
    std::unique_ptr<ccd::Model> make_model() const override { return nullptr; }
    ccd::CoefficientRule coefficients() const override { return {}; }
};

// A stability fit with residuals and RMSE but no peek residuals: enough for
// the stability rule, not for Lookback.
struct NoPeekFit final : ccd::StabilityFit
{
    struct Params
    {
    };
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s;
        return s;
    }
    explicit NoPeekFit(const Params&) {}
    ccd::Capabilities provides() const override
    {
        return ccd::Capability::ModelRmse | ccd::Capability::InWindowResiduals;
    }
    std::unique_ptr<ccd::Model> make_model() const override { return nullptr; }
    ccd::CoefficientRule coefficients() const override { return {}; }
};

struct Silent final : ccd::ChangeTest
{
    struct Params
    {
    };
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s;
        return s;
    }
    explicit Silent(const Params&) {}
    ccd::Capabilities needs() const override { return {}; }
    ccd::Capabilities provides() const override { return {}; }
    std::unique_ptr<ccd::Scratch> make_scratch() const override { return nullptr; }
    ccd::PeekPlan prepare(ccd::ArrayView<const std::int64_t, 1>, ccd::index_t,
                          ccd::Scratch&) const override
    {
        return ccd::PeekPlan{6};
    }
    ccd::TestOutcome test(const ccd::Peek&, ccd::Scratch&, ccd::PeekResult&) const override
    {
        return ccd::TestOutcome::NoChange;
    }
    ccd::index_t break_at(const ccd::Peek& peek, const ccd::Scratch&) const override
    {
        return peek.peek.start;
    }
    double magnitude(const ccd::Peek&, const ccd::Scratch&, ccd::index_t) const override
    {
        return 0;
    }
};

} // namespace

TEST(Capabilities, AStrategyNeedingWhatNoEarlierStageProvidesIsRefused)
{
    ccd::config::Registry r;
    ccd::strategies::register_core_strategies(r);
    r.add<ccd::MonitoringModel, Opaque>("opaque");
    auto tree = golden::ccdc();
    const ccd::DataSpec spec;
    EXPECT_NO_THROW(ccd::config::resolve(r, tree, spec));
    tree.stages["model"] = {"opaque", {}};
    expect_refused([&] { ccd::config::resolve(r, tree, spec); },
                   "change_test 'chi2' needs peek_residuals, model_rmse, in_window_residuals, "
                   "which model 'opaque' does not provide");

    // The outlier rule's statistic, by name, from the change test.
    r.add<ccd::ChangeTest, Silent>("silent");
    tree.stages["model"] = {"lasso", {}};
    tree.stages["change_test"] = {"silent", {}};
    expect_refused([&] { ccd::config::check(r, tree); },
                   "outlier 'floored' reads the statistic 'change_magnitude', which "
                   "change_test 'silent' does not provide");
    tree = golden::ccdc();
    tree.stages["outlier"].params["statistic"] = ccd::config::Value(std::string("other"));
    expect_refused([&] { ccd::config::check(r, tree); },
                   "outlier 'floored' reads the statistic 'other', which change_test 'chi2' "
                   "does not provide; it provides change_magnitude");

    // The stability rule's needs, from the stability fit; Lookback's too.
    r.add<ccd::StabilityFit, OpaqueFit>("opaque");
    tree = golden::ccdc();
    tree.stages["stability_fit"] = {"opaque", {}};
    expect_refused([&] { ccd::config::resolve(r, tree, spec); },
                   "stability_rule 'pyccd' needs model_rmse, in_window_residuals, which "
                   "stability_fit 'opaque' does not provide");

    r.add<ccd::StabilityFit, NoPeekFit>("no-peek");
    tree.stages["stability_fit"] = {"no-peek", {}};
    expect_refused([&] { ccd::config::resolve(r, tree, spec); },
                   "lookback 'pyccd' needs peek_residuals, which stability_fit 'no-peek' does not "
                   "provide");

    // A configuration without an outlier stage.
    tree = golden::ccdc();
    tree.stages.erase("outlier");
    expect_refused([&] { ccd::config::check(r, tree); },
                   "the configuration has no 'outlier' stage");
}

TEST(OutlierRule, FlooredJudgesTheStatisticAgainstItsThreshold)
{
    // A resolved stage's threshold, as the registry builds it.
    const auto outlier = golden::core_registry().make<ccd::OutlierRule>(
        {"floored", {{"threshold", ccd::config::real(10.0)}}});

    const std::int64_t day = 730000;
    const std::vector<std::vector<double>> residuals;
    const ccd::BandNoise variogram;
    const ccd::BandList bands;
    const auto model = golden::core_registry()
                           .make<ccd::MonitoringModel>({"lasso", {}})
                           ->make_model();
    const double value = 0;
    const ccd::Peek peek{*model, ccd::ArrayView<const std::int64_t, 1>::contiguous(&day, {1}),
                         ccd::ArrayView<const double, 2>::contiguous(&value, {1, 1}),
                         {}, {}, 0, {}, residuals, variogram, bands};
    // It reads its statistic by name, whatever else the result holds.
    ccd::PeekResult result;
    result.provide("other") = {99.0, 99.0};
    result.provide("change_magnitude") = {10.0, 99.0};
    EXPECT_EQ(outlier->statistics(), std::vector<std::string>{"change_magnitude"});
    EXPECT_EQ(outlier->judge(peek, result), ccd::OutlierOutcome::Keep);
    result.provide("change_magnitude") = {10.5, 0.0};
    EXPECT_EQ(result.statistics.size(), 2u);
    EXPECT_EQ(outlier->judge(peek, result), ccd::OutlierOutcome::Drop);

    expect_refused(
        [&]
        {
            golden::core_registry().make<ccd::OutlierRule>(
                {"floored", {{"threshold", ccd::config::real(NAN)}}});
        },
        "outlier 'floored' parameter 'threshold' is nan");
}

TEST(StabilityRule, PyccdIsTheSumOfSquaresBelowTheThreshold)
{
    const ccd::config::ConfigTree tree = golden::resolved().stages;
    const auto s = golden::strategies_of();
    EXPECT_EQ(tree.at("stability_rule").strategy, "pyccd");
    const auto count = [&](const char* stage, const char* name)
    { return static_cast<ccd::index_t>(golden::parameter<std::int64_t>(tree, stage, name)); };
    const ccd::index_t peek_size = count("change_test", "peek_size");
    const ccd::index_t coefficient_min = count("stability_fit", "coefficients.min");
    const ccd::index_t coefficient_max = count("stability_fit", "coefficients.max");
    const double threshold = golden::parameter<double>(tree, "stability_rule", "change_threshold");

    // 24 observations, 16 days apart, of a flat series with noise: a fit
    // whose check is well below the threshold, and one whose slope is not.
    const ccd::index_t n = 24, bands = 7;
    std::vector<std::int64_t> dates(n);
    std::vector<double> flat(bands * n), steep(bands * n);
    for (ccd::index_t t = 0; t < n; ++t)
    {
        dates[t] = 730000 + 16 * static_cast<std::int64_t>(t);
        for (ccd::index_t b = 0; b < bands; ++b)
        {
            flat[b * n + t] = 1000.0 + 10.0 * ((t * 3 + b) % 4);
            steep[b * n + t] = flat[b * n + t] + 40.0 * t;
        }
    }
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n});
    const ccd::BandNoise variogram{std::vector<double>(bands, 20.0)};
    const ccd::BandList detection = ccd::config::Shared{}.detection_bands;

    const auto scratch = s.stability_rule->make_scratch();
    s.stability_rule->prepare(ccd::PeekPlan{peek_size, true, 0, peek_size}, detection.size(),
                              *scratch);
    const auto fit = s.stability_fit->make_model();

    // The check by hand, as pyccd computes it.
    const auto check = [&](const ccd::Model& m)
    {
        double norm = 0;
        for (const ccd::index_t b : detection)
        {
            const auto& r = m.residuals(b);
            const double v = (std::abs(m.band_fit(b).coefficients[0] * (dates[n - 1] - dates[0])) +
                              std::abs(r.front()) + std::abs(r.back())) /
                             std::max(variogram[b], m.rmse(b));
            norm += v * v;
        }
        return norm;
    };
    for (const auto* series : {&flat, &steep})
    {
        fit->fit(d, ccd::ArrayView<const double, 2>::contiguous(series->data(), {bands, n}),
                 coefficient_min, coefficient_max, detection, ccd::KeepResiduals::Yes);
        EXPECT_EQ(s.stability_rule->stable(*fit, d, variogram, detection, *scratch),
                  check(*fit) < threshold);
    }
    fit->fit(d, ccd::ArrayView<const double, 2>::contiguous(flat.data(), {bands, n}),
             coefficient_min, coefficient_max, detection, ccd::KeepResiduals::Yes);
    EXPECT_TRUE(s.stability_rule->stable(*fit, d, variogram, detection, *scratch));
    fit->fit(d, ccd::ArrayView<const double, 2>::contiguous(steep.data(), {bands, n}),
             coefficient_min, coefficient_max, detection, ccd::KeepResiduals::Yes);
    EXPECT_FALSE(s.stability_rule->stable(*fit, d, variogram, detection, *scratch));
}

TEST(WindowScreen, TmaskIsTmaskOfItsBands)
{
    ccd::config::Config tree = golden::ccdc();
    tree.stages["window_screen"].params["bands"] = ccd::config::bands({2, 5});
    tree.stages["window_screen"].params["t_const"] = ccd::config::real(3.5);
    const auto s = golden::strategies_of(tree);
    const ccd::config::ConfigTree resolved = golden::resolved(tree).stages;
    const auto& screen = resolved.at("window_screen");
    EXPECT_EQ(screen.strategy, "tmask");
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(screen.params.at("bands")),
              (std::vector<std::int64_t>{2, 5}));
    EXPECT_EQ(std::get<double>(screen.params.at("t_const")), 3.5);
    // By role: the same rows.
    tree.stages["window_screen"].params["bands"] = ccd::config::roles({"red", "swir2"});
    EXPECT_EQ(golden::resolved(tree).stages, resolved);

    // 30 observations, 16 days apart, two of them far off in band 5.
    const ccd::index_t n = 30, bands = 7;
    std::vector<std::int64_t> dates(n);
    std::vector<double> spectral(bands * n);
    for (ccd::index_t t = 0; t < n; ++t)
    {
        dates[t] = 730000 + 16 * static_cast<std::int64_t>(t);
        for (ccd::index_t b = 0; b < bands; ++b)
            spectral[b * n + t] = 1000.0 + 50.0 * std::sin(0.3 * t + b) +
                                  ((t == 7 || t == 19) && b == 5 ? 900.0 : 0.0);
    }
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n});
    const auto x = ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {bands, n});
    const ccd::BandNoise variogram{std::vector<double>(bands, 20.0)};

    const ccd::ProcessingMask got = s.window_screen->screen(d, x, variogram);
    const ccd::ProcessingMask want = ccd::tmask(d, x, variogram, ccd::BandList{2, 5}, 3.5);
    ASSERT_EQ(got.size(), want.size());
    for (ccd::index_t i = 0; i < n; ++i)
        EXPECT_EQ(got[i], want[i]) << i;
    EXPECT_TRUE(got[7]);
    EXPECT_TRUE(got[19]);
}

TEST(StartSearch, PyccdIsMeowSizeAndDayDelta)
{
    ccd::config::Config tree = golden::ccdc();
    tree.shared["min_observations"] = ccd::config::integer(10);
    tree.stages["start_search"].params["day_delta"] = ccd::config::integer(200);
    EXPECT_EQ(tree.stages.at("start_search").strategy, "pyccd");
    const auto s = golden::strategies_of(tree);
    const ccd::StartSearch& search = *s.start_search;

    EXPECT_TRUE(search.too_few(10));
    EXPECT_FALSE(search.too_few(11));
    EXPECT_EQ(search.first(7).start, 7u);
    EXPECT_EQ(search.first(7).stop, 17u);
    EXPECT_EQ(search.min_window(), 10u);

    // room: stop + 10 <= n; exhausted unless stop + 10 < n.
    const ccd::Window w(5, 15);
    EXPECT_TRUE(search.room(w, 25));
    EXPECT_FALSE(search.room(w, 24));
    EXPECT_TRUE(search.exhausted(w, 25));
    EXPECT_FALSE(search.exhausted(w, 26));
    // Fewer observations than min_observations: no room, exhausted (no unsigned wrap).
    EXPECT_FALSE(search.room(ccd::Window(0, 10), 5));
    EXPECT_TRUE(search.exhausted(ccd::Window(0, 10), 5));

    // short_span: under 200 days; usable: 200 days and 10 observations.
    std::vector<std::int64_t> days(10);
    for (std::size_t i = 0; i < days.size(); ++i)
        days[i] = 730000 + 22 * static_cast<std::int64_t>(i);   // spans 198 days
    const auto d = [&](std::size_t n)
    { return ccd::ArrayView<const std::int64_t, 1>::contiguous(days.data(), {n}); };
    EXPECT_TRUE(search.short_span(d(10)));
    EXPECT_FALSE(search.usable(d(10)));
    days[9] = 730200;
    EXPECT_FALSE(search.short_span(d(10)));
    EXPECT_TRUE(search.usable(d(10)));
    EXPECT_FALSE(search.usable(d(9)));   // 9 observations

    tree.shared["min_observations"] = ccd::config::integer(0);
    expect_refused([&] { golden::strategies_of(tree); },
                   "shared parameter 'min_observations' is 0; it must be at least 1");

    // The start search's window is the shared min_observations, and so is
    // the whole-series curve's least: a stage sets neither.
    ccd::config::Config own = golden::ccdc();
    own.stages["start_search"].params["min_observations"] = ccd::config::integer(10);
    expect_refused([&] { golden::strategies_of(own); },
                   "start_search 'pyccd' parameter 'min_observations' is the shared value "
                   "'min_observations'; set shared 'min_observations' instead");
}

// The preset's Lookback judges at the change test's peek, with the change
// test's threshold and probability and the outlier rule's threshold: the
// shared ones.
TEST(Lookback, PyccdTakesItsThresholdsFromTheSharedValues)
{
    const auto out = golden::with_shared(golden::ccdc(), "outlier_probability",
                                         ccd::config::real(0.999));
    const ccd::config::ConfigTree tree = golden::resolved(out).stages;
    const auto& lookback = tree.at("lookback");
    const auto& test = tree.at("change_test");
    EXPECT_EQ(lookback.strategy, "pyccd");
    EXPECT_EQ(std::get<std::string>(lookback.params.at("peek")), "change_test");   // the test's
    EXPECT_EQ(std::get<double>(lookback.params.at("change_threshold")),
              std::get<double>(test.params.at("change_threshold")));
    EXPECT_EQ(std::get<double>(lookback.params.at("change_probability")),
              std::get<double>(test.params.at("change_probability")));
    EXPECT_EQ(std::get<double>(lookback.params.at("outlier_threshold")),
              ccd::chi2_quantile(0.999, 5));
    EXPECT_EQ(std::get<double>(lookback.params.at("outlier_threshold")),
              std::get<double>(tree.at("outlier").params.at("threshold")));
    EXPECT_NO_THROW(golden::strategies_of(out));

    auto own = golden::ccdc();
    own.stages["lookback"].params["outlier_threshold"] = ccd::config::real(20.0);
    expect_refused([&] { golden::strategies_of(own); },
                   "lookback 'pyccd' parameter 'outlier_threshold' is derived from the shared "
                   "value 'outlier_probability'");
}

TEST(Lookback, PyccdsThresholdIsAdjustedForALongerPeek)
{
    // A window 15 observations after the break, so a peek of 10 is far from
    // it: one statistic, between the default threshold of a 6-observation
    // peek and the lower one of a 10-observation peek, is a change only for
    // the longer peek.
    const ccd::strategies::PyccdLookback::Params p;
    const ccd::strategies::PyccdLookback lookback(p);
    const auto scratch = lookback.make_scratch();
    ccd::ChangeThresholdMemo memo;
    const double at6 = memo(6, 6, p.change_threshold, 5, p.change_probability);
    const double at10 = memo(10, 6, p.change_threshold, 5, p.change_probability);
    EXPECT_EQ(at6, p.change_threshold);
    EXPECT_LT(at10, at6);

    // A stability fit of 0.0 everywhere and residuals of 3.5 per band against
    // an RMSE of 1 give every statistic 5 * 3.5^2 = 61.25 over 5 bands; with
    // an RMSE of sqrt(61.25 / t) each statistic is t.
    class Flat final : public ccd::Model
    {
    public:
        explicit Flat(double rmse) : rmse_(rmse) {}
        bool fit(ccd::ArrayView<const std::int64_t, 1>, ccd::ArrayView<const double, 2>,
                 ccd::index_t, ccd::index_t, const ccd::BandList&, ccd::KeepResiduals) override
        {
            return true;
        }
        void peek_residuals(ccd::ArrayView<const std::int64_t, 1> dates,
                            ccd::ArrayView<const double, 2>, const ccd::BandList& bands,
                            std::vector<std::vector<double>>& residuals) override
        {
            for (const ccd::index_t b : bands)
                residuals[b].assign(dates.size(), 3.5);
        }
        double rmse(ccd::index_t) const override { return rmse_; }
        const std::vector<double>& residuals(ccd::index_t) const override { return none_; }
        ccd::BandFit band_fit(ccd::index_t) const override { return {}; }

    private:
        double rmse_;
        std::vector<double> none_;
    };

    const ccd::index_t n = 30, bands = 5;
    std::vector<std::int64_t> dates(n);
    for (ccd::index_t t = 0; t < n; ++t)
        dates[t] = 730000 + 16 * static_cast<std::int64_t>(t);
    std::vector<double> spectral(bands * n, 0.0);
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n});
    const auto x = ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {bands, n});
    const ccd::BandNoise variogram{std::vector<double>(bands, 0.0)};
    const ccd::BandList detection{0, 1, 2, 3, 4};
    const double statistic = (at6 + at10) / 2;   // between the two thresholds
    Flat fit(std::sqrt(61.25 / statistic));
    const ccd::Window window(15, 27);

    lookback.prepare(ccd::PeekPlan{6, true, 0, 6}, bands, *scratch);
    // Below 15.09 and below the outlier threshold: the observation joins.
    EXPECT_EQ(lookback.judge(fit, d, x, window, 0, variogram, detection, *scratch).verdict,
              ccd::LookbackVerdict::Include);
    lookback.prepare(ccd::PeekPlan{10, true, 0, 6}, bands, *scratch);
    const auto judged = lookback.judge(fit, d, x, window, 0, variogram, detection, *scratch);
    EXPECT_EQ(judged.verdict, ccd::LookbackVerdict::Change);
    EXPECT_EQ(judged.observation, 14u);
}

TEST(EdgeSegments, LassoIsTheCatchSegmentAndWholeSeriesKernels)
{
    ccd::config::Config config = golden::ccdc();
    config.shared["coefficients.min"] = ccd::config::integer(6);
    config.shared["min_observations"] = ccd::config::integer(15);
    config.shared["lasso.alpha"] = ccd::config::real(0.5);
    ccd::DataSpec reflectance;
    reflectance.reflectance_scale = 1.0;
    const ccd::config::ConfigTree tree = golden::resolved(config, reflectance).stages;
    const auto& edges = tree.at("edge_segments");
    EXPECT_EQ(edges.strategy, "lasso");
    EXPECT_EQ(std::get<std::int64_t>(edges.params.at("coefficients.min")), 6);
    EXPECT_EQ(std::get<std::int64_t>(edges.params.at("min_observations")), 15);
    EXPECT_EQ(golden::parameter<std::int64_t>(tree, "start_search", "min_observations"), 15);
    EXPECT_EQ(std::get<double>(edges.params.at("lasso.alpha")), 0.5);
    EXPECT_EQ(tree.at("segment_report").strategy, "model-fit");
    EXPECT_EQ(tree.at("mode").strategy, "pyccd");

    const auto s = golden::strategies_of(config, reflectance);
    const ccd::index_t n = 40, bands = 3;
    std::vector<std::int64_t> dates(n);
    std::vector<double> spectral(bands * n);
    for (ccd::index_t t = 0; t < n; ++t)
    {
        dates[t] = 730000 + 16 * static_cast<std::int64_t>(t);
        for (ccd::index_t b = 0; b < bands; ++b)
            spectral[b * n + t] = 900.0 + 100.0 * std::sin(0.2 * t + b) + 7.0 * ((t + b) % 3);
    }
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n});
    const auto x = ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {bands, n});
    const ccd::Window window(3, 31);

    const auto scratch = s.edge_segments->make_scratch();
    const ccd::ChangeModel got =
        s.edge_segments->catch_window(d, x, window, ccd::CurveQA::End, *scratch);
    // The kernels' solver with edge_segments' parameters.
    ccd::LassoOptions l;
    l.alpha = 0.5;
    ccd::LassoWorkspace workspace(0);
    ccd::LassoSolver solver(l);
    const ccd::ChangeModel want =
        ccd::catch_segment(d, x, window, 6, ccd::CurveQA::End, workspace, solver);
    ASSERT_EQ(got.bands.size(), want.bands.size());
    for (std::size_t b = 0; b < got.bands.size(); ++b)
    {
        EXPECT_EQ(got.bands[b].coefficients, want.bands[b].coefficients) << b;
        EXPECT_EQ(got.bands[b].rmse, want.bands[b].rmse) << b;
    }
    EXPECT_EQ(got.break_day, want.break_day);
    EXPECT_EQ(got.curve_qa, ccd::CurveQA::End);
}

TEST(Noise, VariogramIsTheAdjustedVariogram)
{
    const auto tree =
        golden::with(golden::ccdc(), "noise", "min_lag_days", ccd::config::integer(45));
    const auto s = golden::strategies_of(tree);
    const ccd::index_t n = 50, bands = 3;
    std::vector<std::int64_t> dates(n);
    std::vector<double> spectral(bands * n);
    for (ccd::index_t t = 0; t < n; ++t)
    {
        dates[t] = 730000 + 16 * static_cast<std::int64_t>(t) + (t % 3);
        for (ccd::index_t b = 0; b < bands; ++b)
            spectral[b * n + t] = 500.0 + 30.0 * std::sin(0.4 * t + b) + ((t * 5 + b) % 7);
    }
    const auto d = ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {n});
    const auto x = ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {bands, n});
    EXPECT_EQ(s.noise->estimate(d, x), ccd::adjusted_variogram(d, x, 45));
    EXPECT_EQ(golden::parameter<std::int64_t>(golden::resolved(tree).stages, "noise",
                                              "min_lag_days"),
              45);
}
