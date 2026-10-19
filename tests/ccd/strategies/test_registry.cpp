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
        "the configuration has no 'model' stage");
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
                   "change_test 'chi2' parameter 'change_threshold' is nan");
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

// Test-only strategies: a monitoring model that provides nothing, and a
// change test that provides nothing but needs no more than chi2.
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
    std::unique_ptr<ccd::TestScratch> make_scratch() const override { return nullptr; }
    ccd::index_t prepare(ccd::ArrayView<const std::int64_t, 1>, ccd::index_t,
                         ccd::TestScratch&) const override
    {
        return 6;
    }
    ccd::TestOutcome test(const ccd::Peek&, ccd::TestScratch&, ccd::PeekResult&) const override
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
