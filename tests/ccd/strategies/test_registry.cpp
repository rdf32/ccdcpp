//==============================================================================
//
// The registry and the strategy parameters (config/registry.hpp,
// config/params.hpp), on the library's strategies:
//
//   * the registry lists each stage's strategies, and builds one by name with
//     its parameters checked against its schema (unknown names, wrong types,
//     values out of range all refused, naming the stage, strategy and field);
//   * the configuration the options describe builds the strategies a run
//     uses, and the refit schedule decides as Extend's inline test did;
//   * the lasso monitoring model fits, predicts and scores as the shared
//     kernels do, bit for bit, with the solver options it was configured with;
//   * a band list parameter takes indices 0 to CCD_MAX_BANDS - 1, and the
//     tmask window screen is tmask() of its bands and t_const;
//   * the pyccd stability rule is pyccd's check of the stability fit, against
//     its threshold for the pixel's peek;
//   * a configuration whose strategy needs what no earlier stage provides is
//     refused, naming both; the floored outlier rule judges the change
//     test's statistic against its threshold;
//   * the chi2 change test's peek is the adjusted peek of its parameters, and
//     the threshold memo is adjust_change_threshold() for any arguments.
//
//==============================================================================

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/config/registry.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/procedure/kernels.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/refit/span_factor.hpp"
#include "ccd/config/from_options.hpp"
#include "ccd/strategies/register_core.hpp"

namespace
{

using ccd::config::StageConfig;
using ccd::config::Values;

template<class F>
void expect_refused(F make, const std::string& fragment)
{
    try
    {
        make();
        ADD_FAILURE() << "accepted; expected a refusal containing '" << fragment << "'";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

} // namespace

TEST(Registry, ListsTheCoreStrategies)
{
    const auto names = ccd::strategies::core_registry().names();
    ASSERT_EQ(names.count("window_screen"), 1u);
    EXPECT_EQ(names.at("window_screen"), (std::vector<std::string>{"tmask"}));
    ASSERT_EQ(names.count("stability_fit"), 1u);
    EXPECT_EQ(names.at("stability_fit"), (std::vector<std::string>{"lasso"}));
    ASSERT_EQ(names.count("stability_rule"), 1u);
    EXPECT_EQ(names.at("stability_rule"), (std::vector<std::string>{"pyccd"}));
    ASSERT_EQ(names.count("model"), 1u);
    EXPECT_EQ(names.at("model"), (std::vector<std::string>{"lasso"}));
    ASSERT_EQ(names.count("refit"), 1u);
    EXPECT_EQ(names.at("refit"), (std::vector<std::string>{"span-factor"}));
    ASSERT_EQ(names.count("change_test"), 1u);
    EXPECT_EQ(names.at("change_test"), (std::vector<std::string>{"chi2"}));
    ASSERT_EQ(names.count("outlier"), 1u);
    EXPECT_EQ(names.at("outlier"), (std::vector<std::string>{"floored"}));
}

TEST(Registry, BuildsByNameAndChecksTheParameters)
{
    using ccd::RefitSchedule;
    using ccd::config::flag;
    using ccd::config::integer;
    using ccd::config::real;
    const auto& r = ccd::strategies::core_registry();
    const auto refit = [&](Values values)
    { return r.make<RefitSchedule>({"span-factor", std::move(values)}); };

    // Defaults, and set values.
    const auto pyccd = refit({});
    EXPECT_TRUE(pyccd->due(24, 100.0, 100.0));
    EXPECT_FALSE(pyccd->due(25, 132.0, 100.0));
    EXPECT_TRUE(pyccd->due(25, 133.0, 100.0));
    const auto every = refit({{"min_obs", integer(1000)}, {"factor", real(1.0)}});
    EXPECT_TRUE(every->due(999, 1.0, 100.0));

    expect_refused([&] { r.make<RefitSchedule>({"every-n", {}}); },
                   "refit has no strategy 'every-n'; it has span-factor");
    expect_refused([&] { refit({{"minobs", integer(3)}}); },
                   "refit 'span-factor' has no parameter 'minobs'; it has min_obs, factor");
    expect_refused([&] { refit({{"min_obs", real(2.5)}}); },
                   "parameter 'min_obs' is a whole number");
    expect_refused([&] { refit({{"min_obs", integer(-1)}}); },
                   "parameter 'min_obs' is -1; it must be at least 0");
    expect_refused([&] { refit({{"min_obs", integer(0)}}); },
                   "parameter 'min_obs' is 0; it must be at least 1");
    expect_refused([&] { refit({{"factor", real(0.0)}}); },
                   "parameter 'factor' is 0; it must be greater than 0");
    expect_refused([&] { refit({{"factor", real(-1e-9)}}); },
                   "parameter 'factor' is -1e-09; it must be greater than 0");
    expect_refused([&] { refit({{"factor", real(NAN)}}); }, "parameter 'factor' is nan");
    expect_refused([&] { refit({{"factor", flag(true)}}); }, "parameter 'factor' is a number");
    // An integer is accepted for a real.
    EXPECT_NO_THROW(refit({{"factor", integer(2)}}));
}

TEST(Params, AFlagIsTrueOrFalse)
{
    struct P
    {
        bool on = false;
    };
    const std::vector<ccd::config::Field<P>> schema = {
        {"on", &P::on, ccd::config::Unit::None, ccd::config::any(), "a switch"}};
    P p;
    ccd::config::apply(schema, {{"on", ccd::config::flag(true)}}, p, "test");
    EXPECT_TRUE(p.on);
    expect_refused(
        [&] { ccd::config::apply(schema, {{"on", ccd::config::integer(1)}}, p, "test"); },
        "test parameter 'on' is a flag (true or false)");
    EXPECT_EQ(std::get<bool>(ccd::config::to_values(schema, p).at("on")), true);
}

TEST(Registry, RefusesAStrategyRegisteredTwice)
{
    ccd::config::Registry r;
    r.add<ccd::RefitSchedule, ccd::strategies::SpanFactor>("span-factor");
    EXPECT_THROW((r.add<ccd::RefitSchedule, ccd::strategies::SpanFactor>("span-factor")),
                 std::logic_error);
}

TEST(Registry, TheOptionsConfigurationIsTheirs)
{
    ccd::HarmonicOptions h;
    h.RETRAIN_MIN_OBS = 30;
    h.RETRAIN_SPAN_FACTOR = 1.5;
    const auto tree = ccd::config::from_options(h, ccd::LassoOptions{});
    const StageConfig& refit = tree.at("refit");
    EXPECT_EQ(refit.strategy, "span-factor");
    const auto s = ccd::config::strategies_for(h, ccd::LassoOptions{});
    EXPECT_TRUE(s.refit->due(30, 1.0, 1.0));
    EXPECT_FALSE(s.refit->due(31, 149.0, 100.0));
    EXPECT_TRUE(s.refit->due(31, 150.0, 100.0));

    // The schema gives the parameters back as set.
    const auto values = ccd::config::to_values(ccd::strategies::SpanFactor::schema(),
                                               ccd::strategies::SpanFactor::Params{30, 1.5});
    EXPECT_EQ(std::get<std::int64_t>(values.at("min_obs")), 30);
    EXPECT_EQ(std::get<double>(values.at("factor")), 1.5);

    // RETRAIN_MIN_OBS past what a configuration holds: refit at every step.
    h.RETRAIN_MIN_OBS = std::numeric_limits<ccd::index_t>::max();
    EXPECT_TRUE(ccd::config::strategies_for(h, ccd::LassoOptions{}).refit->due(1u << 20, 1.0, 100.0));

    expect_refused(
        [&] { ccd::strategies::build_strategies(ccd::strategies::core_registry(), {}); },
        "the configuration has no 'window_screen' stage");
}

TEST(MonitoringModel, TheLassoIsTheKernelsFit)
{
    ccd::LassoOptions l;
    l.alpha = 0.5;
    l.max_iter = 300;
    const auto tree = ccd::config::from_options(ccd::HarmonicOptions{}, l);
    EXPECT_EQ(tree.at("model").strategy, "lasso");
    const auto s = ccd::config::strategies_for(ccd::HarmonicOptions{}, l);
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

    const auto& r = ccd::strategies::core_registry();
    ccd::HarmonicOptions nan_threshold;
    nan_threshold.CHANGE_THRESHOLD = NAN;
    expect_refused([&] { ccd::config::strategies_for(nan_threshold, l); },
                   "stability_rule 'pyccd' parameter 'change_threshold' is nan");
    expect_refused(
        [&] { r.make<ccd::MonitoringModel>({"lasso", {{"alpha", ccd::config::real(-1)}}}); },
        "model 'lasso' parameter 'alpha' is -1; it must be at least 0");
}

// chi2's peek is adjust_peek() of its parameters (the options' PEEK_SIZE
// and PEEK_REFERENCE_DAYS) at the reference cadence (no adjustment) and at
// denser ones; the threshold memo gives adjust_change_threshold()'s value
// for every sequence of arguments.
TEST(ChangeTest, Chi2PeekIsTheAdjustedPeek)
{
    const ccd::HarmonicOptions h;
    const auto s = ccd::config::strategies_for(h, ccd::LassoOptions{});
    const auto scratch = s.change_test->make_scratch();
    for (const std::int64_t gap : {16, 8, 5})
    {
        std::vector<std::int64_t> dates(100);
        for (std::size_t i = 0; i < dates.size(); ++i)
            dates[i] = 730000 + gap * static_cast<std::int64_t>(i);
        const auto view = ccd::ArrayView<const std::int64_t, 1>::contiguous(
            dates.data(), {dates.size()});

        const ccd::index_t peek =
            s.change_test->prepare(view, h.DETECTION_BANDS.size(), *scratch);
        EXPECT_EQ(peek, ccd::adjust_peek(view, h.PEEK_SIZE, h.PEEK_REFERENCE_DAYS)) << gap;
        if (gap == 16)
            EXPECT_EQ(peek, h.PEEK_SIZE);
        else
            EXPECT_GT(peek, h.PEEK_SIZE) << gap;
    }

    // Fewer than two dates: the default peek.
    const std::int64_t one = 730000;
    EXPECT_EQ(s.change_test->prepare(ccd::ArrayView<const std::int64_t, 1>::contiguous(&one, {1}),
                                     h.DETECTION_BANDS.size(), *scratch),
              h.PEEK_SIZE);

    ccd::ChangeThresholdMemo memo;
    for (const ccd::index_t peek : {6u, 9u, 9u, 6u, 12u, 12u, 9u})
        for (const double probability : {0.99, 0.95, 0.99})
            EXPECT_EQ(memo(peek, 6, h.CHANGE_THRESHOLD, 7, probability),
                      ccd::adjust_change_threshold(peek, 6, h.CHANGE_THRESHOLD, 7, probability))
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
    ccd::index_t prepare(ccd::ArrayView<const std::int64_t, 1>, ccd::index_t,
                         ccd::Scratch&) const override
    {
        return 6;
    }
    ccd::TestOutcome test(const ccd::Peek&, ccd::Scratch&, ccd::PeekResult&) const override
    {
        return ccd::TestOutcome::NoChange;
    }
    ccd::index_t break_at(const ccd::Peek& peek) const override { return peek.peek.start; }
    double magnitude(const ccd::Peek&, ccd::index_t) const override { return 0; }
};

} // namespace

TEST(Capabilities, AStrategyNeedingWhatNoEarlierStageProvidesIsRefused)
{
    ccd::config::Registry r;
    ccd::strategies::register_core_strategies(r);
    r.add<ccd::MonitoringModel, Opaque>("opaque");
    auto tree = ccd::config::from_options(ccd::HarmonicOptions{}, ccd::LassoOptions{});
    EXPECT_NO_THROW(ccd::strategies::build_strategies(r, tree));
    tree["model"] = {"opaque", {}};
    expect_refused([&] { ccd::strategies::build_strategies(r, tree); },
                   "change_test 'chi2' needs peek_residuals, model_rmse, in_window_residuals, "
                   "which model 'opaque' does not provide");

    // The outlier rule's needs, from the model and the change test together.
    r.add<ccd::ChangeTest, Silent>("silent");
    tree["model"] = {"lasso", {}};
    tree["change_test"] = {"silent", {}};
    expect_refused([&] { ccd::strategies::build_strategies(r, tree); },
                   "outlier 'floored' needs peek_statistic, which model 'lasso' with "
                   "change_test 'silent' does not provide");

    // The stability rule's needs, from the stability fit; Lookback's too.
    r.add<ccd::StabilityFit, OpaqueFit>("opaque");
    tree = ccd::config::from_options(ccd::HarmonicOptions{}, ccd::LassoOptions{});
    tree["stability_fit"] = {"opaque", {}};
    expect_refused([&] { ccd::strategies::build_strategies(r, tree); },
                   "stability_rule 'pyccd' needs model_rmse, in_window_residuals, which "
                   "stability_fit 'opaque' does not provide");

    // A configuration without an outlier stage.
    tree = ccd::config::from_options(ccd::HarmonicOptions{}, ccd::LassoOptions{});
    tree.erase("outlier");
    expect_refused([&] { ccd::strategies::build_strategies(r, tree); },
                   "the configuration has no 'outlier' stage");
}

TEST(OutlierRule, FlooredJudgesTheStatisticAgainstItsThreshold)
{
    ccd::HarmonicOptions h;
    h.OUTLIER_THRESHOLD = 10.0;
    const auto tree = ccd::config::from_options(h, ccd::LassoOptions{});
    EXPECT_EQ(tree.at("outlier").strategy, "floored");
    const auto s = ccd::config::strategies_for(h, ccd::LassoOptions{});

    const std::int64_t day = 730000;
    const std::vector<std::vector<double>> residuals;
    const ccd::Variogram variogram;
    const ccd::BandList bands;
    const auto model = ccd::strategies::core_registry()
                           .make<ccd::MonitoringModel>({"lasso", {}})
                           ->make_model();
    const ccd::Peek peek{*model, ccd::ArrayView<const std::int64_t, 1>::contiguous(&day, {1}),
                         {}, {}, 0, {}, residuals, variogram, bands};
    ccd::PeekResult result;
    result.statistic = {10.0, 99.0};
    EXPECT_EQ(s.outlier->judge(peek, result), ccd::OutlierOutcome::Keep);
    result.statistic = {10.5, 0.0};
    EXPECT_EQ(s.outlier->judge(peek, result), ccd::OutlierOutcome::Drop);

    h.OUTLIER_THRESHOLD = NAN;
    expect_refused([&] { ccd::config::strategies_for(h, ccd::LassoOptions{}); },
                   "outlier 'floored' parameter 'threshold' is nan");
}

TEST(StabilityRule, PyccdIsTheSumOfSquaresBelowTheThreshold)
{
    const ccd::HarmonicOptions h;
    const auto s = ccd::config::strategies_for(h, ccd::LassoOptions{});
    EXPECT_EQ(ccd::config::from_options(h, ccd::LassoOptions{}).at("stability_rule").strategy,
              "pyccd");

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
    const ccd::Variogram variogram{std::vector<double>(bands, 20.0)};
    const ccd::BandList detection = h.DETECTION_BANDS;

    const auto scratch = s.stability_rule->make_scratch();
    s.stability_rule->prepare(h.PEEK_SIZE, detection.size(), *scratch);
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
                 h.COEFFICIENT_MIN, h.COEFFICIENT_MAX, detection, ccd::KeepResiduals::Yes);
        EXPECT_EQ(s.stability_rule->stable(*fit, d, variogram, detection, *scratch),
                  check(*fit) < h.CHANGE_THRESHOLD);
    }
    fit->fit(d, ccd::ArrayView<const double, 2>::contiguous(flat.data(), {bands, n}),
             h.COEFFICIENT_MIN, h.COEFFICIENT_MAX, detection, ccd::KeepResiduals::Yes);
    EXPECT_TRUE(s.stability_rule->stable(*fit, d, variogram, detection, *scratch));
    fit->fit(d, ccd::ArrayView<const double, 2>::contiguous(steep.data(), {bands, n}),
             h.COEFFICIENT_MIN, h.COEFFICIENT_MAX, detection, ccd::KeepResiduals::Yes);
    EXPECT_FALSE(s.stability_rule->stable(*fit, d, variogram, detection, *scratch));
}

TEST(Params, ABandListTakesBandIndices)
{
    struct P
    {
        ccd::BandList bands = {1};
    };
    const std::vector<ccd::config::Field<P>> schema = {
        {"bands", &P::bands, ccd::config::Unit::Bands, ccd::config::any(), "some bands"}};
    P p;
    ccd::config::apply(schema, {{"bands", ccd::config::bands({3, 0, 2})}}, p, "test");
    EXPECT_EQ(std::vector<ccd::index_t>(p.bands.begin(), p.bands.end()),
              (std::vector<ccd::index_t>{3, 0, 2}));
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(ccd::config::to_values(schema, p).at("bands")),
              (std::vector<std::int64_t>{3, 0, 2}));
    using List = std::vector<std::int64_t>;
    expect_refused(
        [&] { ccd::config::apply(schema, {{"bands", ccd::config::integer(1)}}, p, "test"); },
        "test parameter 'bands' is a list of band indices");
    expect_refused(
        [&] { ccd::config::apply(schema, {{"bands", ccd::config::Value(List{1, -1})}}, p, "test"); },
        "test parameter 'bands' holds band -1; a band index is at least 0");
    expect_refused(
        [&]
        {
            ccd::config::apply(
                schema, {{"bands", ccd::config::Value(List{1, ccd::CCD_MAX_BANDS})}}, p, "test");
        },
        "test parameter 'bands': band index");
}

TEST(WindowScreen, TmaskIsTmaskOfItsBands)
{
    ccd::HarmonicOptions h;
    h.TMASK_BANDS = {2, 5};
    h.T_CONST = 3.5;
    const auto s = ccd::config::strategies_for(h, ccd::LassoOptions{});
    const auto tree = ccd::config::from_options(h, ccd::LassoOptions{});
    const auto& screen = tree.at("window_screen");
    EXPECT_EQ(screen.strategy, "tmask");
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(screen.params.at("bands")),
              (std::vector<std::int64_t>{2, 5}));
    EXPECT_EQ(std::get<double>(screen.params.at("t_const")), 3.5);

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
    const ccd::Variogram variogram{std::vector<double>(bands, 20.0)};

    const ccd::ProcessingMask got = s.window_screen->screen(d, x, variogram);
    const ccd::ProcessingMask want = ccd::tmask(d, x, variogram, h.TMASK_BANDS, h.T_CONST);
    ASSERT_EQ(got.size(), want.size());
    for (ccd::index_t i = 0; i < n; ++i)
        EXPECT_EQ(got[i], want[i]) << i;
    EXPECT_TRUE(got[7]);
    EXPECT_TRUE(got[19]);
}
