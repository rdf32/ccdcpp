//==============================================================================
//
// The registry and the strategy parameters (config/registry.hpp,
// config/params.hpp), on the library's strategies:
//
//   * the registry lists each stage's strategies, and builds one by name with
//     its parameters checked against its schema (unknown names, wrong types,
//     values out of range all refused, naming the stage, strategy and field);
//   * the "ccdc" preset, written in physical terms, resolves on the default
//     input to pyccd's tree bit for bit; a configuration resolves against an
//     input (reflectance times its unit, roles as its rows, the shared
//     values and the chi-squared thresholds into the stages that read
//     them), and a stage setting a shared value is refused;
//   * the "ccdc" preset, edited by stage and parameter, builds the
//     strategies a run uses, and the refit schedule decides as Extend's
//     inline test did;
//   * the lasso monitoring model fits, predicts and scores as the shared
//     kernels do, bit for bit, with the solver options it was configured with;
//   * the pyccd start search's decisions are pyccd's comparisons with
//     meow_size (the shared min_observations) and day_delta;
//   * a band list parameter takes indices 0 to CCD_MAX_BANDS - 1, and the
//     tmask window screen is tmask() of its bands and t_const;
//   * the pyccd stability rule is pyccd's check of the stability fit, against
//     its threshold for the pixel's peek;
//   * a configuration whose strategy needs what no earlier stage provides is
//     refused, naming both; the floored outlier rule judges the change
//     test's statistic against its threshold;
//   * the chi2 change test's peek is the adjusted peek of its parameters, and
//     the threshold memo is adjust_change_threshold() for any arguments;
//   * a configuration's preprocessing resolves in the input's units and
//     codes, a bad one is refused when its Algorithm is built, and a
//     registry adds a step type of its own.
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
#include "ccd/input/pixel_series.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/machine/kernels.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/lookback/pyccd.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/refit/span_factor.hpp"
#include "../golden/specs.hpp"
#include "../golden/golden.hpp"
#include "../golden/synthetic.hpp"
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
    ASSERT_EQ(names.count("noise_floor"), 1u);
    EXPECT_EQ(names.at("noise_floor"), (std::vector<std::string>{"variogram"}));
    // The library's registry holds every core strategy (and its extensions').
    const auto library = ccd::strategies::library_registry().names();
    for (const auto& [stage, strategies] : names)
        for (const auto& strategy : strategies)
        {
            ASSERT_EQ(library.count(stage), 1u) << stage;
            const auto& in = library.at(stage);
            EXPECT_NE(std::find(in.begin(), in.end(), strategy), in.end()) << stage << " " << strategy;
        }
    ASSERT_EQ(names.count("mode"), 1u);
    EXPECT_EQ(names.at("mode"), (std::vector<std::string>{"pyccd"}));
    ASSERT_EQ(names.count("segment_fit"), 1u);
    EXPECT_EQ(names.at("segment_fit"), (std::vector<std::string>{"model-fit"}));
    ASSERT_EQ(names.count("edge_fit"), 1u);
    EXPECT_EQ(names.at("edge_fit"), (std::vector<std::string>{"lasso"}));
    ASSERT_EQ(names.count("start_search"), 1u);
    EXPECT_EQ(names.at("start_search"), (std::vector<std::string>{"pyccd"}));
    ASSERT_EQ(names.count("window_screen"), 1u);
    EXPECT_EQ(names.at("window_screen"), (std::vector<std::string>{"tmask"}));
    ASSERT_EQ(names.count("stability_fit"), 1u);
    EXPECT_EQ(names.at("stability_fit"), (std::vector<std::string>{"lasso"}));
    ASSERT_EQ(names.count("stability_rule"), 1u);
    EXPECT_EQ(names.at("stability_rule"), (std::vector<std::string>{"pyccd"}));
    ASSERT_EQ(names.count("lookback"), 1u);
    EXPECT_EQ(names.at("lookback"), (std::vector<std::string>{"pyccd"}));
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
    EXPECT_TRUE(pyccd->due(24, 100.0, 12, 100.0));
    EXPECT_FALSE(pyccd->due(25, 132.0, 12, 100.0));
    EXPECT_TRUE(pyccd->due(25, 133.0, 12, 100.0));
    const auto every = refit({{"min_obs", integer(1000)}, {"factor", real(1.0)}});
    EXPECT_TRUE(every->due(999, 1.0, 12, 100.0));

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

TEST(Registry, RefusesAPresetRegisteredTwice)
{
    ccd::config::Registry r;
    r.add_preset("mine", {});
    EXPECT_THROW(r.add_preset("mine", {}), std::logic_error);
}

// The "ccdc" preset is written in physical terms; on the default Input
// (Collection-1 DN, the Landsat rows) it resolves to pyccd's tree, every
// value bit for bit as pyccd holds it in DN. Its statistics cutoff is none,
// where pyccd's is 2017-12-31.
TEST(Registry, CcdcOnTheDefaultInputIsPyccdsTree)
{
    using ccd::config::integer;
    using ccd::config::real;
    using List = std::vector<std::int64_t>;
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    const ccd::config::Config& preset = r.preset("ccdc");

    // Physical: alpha in reflectance, probabilities, roles, the cutoff.
    EXPECT_EQ(preset.shared.at("alpha"), real(1e-4));
    EXPECT_EQ(preset.shared.at("coef_floor"), real(1e-4));
    EXPECT_EQ(preset.preprocess.at("insufficient_clear").steps.at(4).params.at("offset"),
              real(0.04));
    EXPECT_EQ(preset.shared.at("change_probability"), real(0.99));
    EXPECT_EQ(preset.shared.at("outlier_probability"), real(1.0 - 1e-6));
    EXPECT_EQ(preset.shared.at("stat_ord"), integer(ccd::config::NO_STAT_CUTOFF));
    EXPECT_EQ(preset.shared.at("detection_bands"),
              ccd::config::roles({"green", "red", "nir", "swir1", "swir2"}));
    EXPECT_EQ(preset.stages.at("window_screen").params.at("bands"),
              ccd::config::roles({"green", "swir1"}));
    EXPECT_EQ(preset.stages.at("model").params.count("alpha"), 0u);
    EXPECT_EQ(preset.stages.at("change_test").params.count("change_threshold"), 0u);

    // The tree the preset was in Collection-1 DN, every parameter.
    const ccd::config::ConfigTree want = {
        {"change_test", {"chi2", {{"change_probability", real(0x1.fae147ae147aep-1)},
                                  {"change_threshold", real(0x1.e2c2be7b52448p+3)},
                                  {"model_rmse_up_to", integer(24)},
                                  {"peek_reference_days", real(16.0)},
                                  {"peek_size", integer(6)},
                                  {"seasonal_count", integer(24)},
                                  {"seasonal_divisor", real(4.0)}}}},
        {"edge_fit", {"lasso", {{"alpha", real(1.0)},
                                {"coef_floor", real(1.0)},
                                {"coefficients", integer(4)},
                                {"max_iter", integer(1000)},
                                {"min_observations", integer(12)},
                                {"tolerance", real(0x1.a36e2eb1c432dp-14)}}}},
        {"lookback", {"pyccd", {{"change_probability", real(0x1.fae147ae147aep-1)},
                                {"change_threshold", real(0x1.e2c2be7b52448p+3)},
                                {"outlier_threshold", real(0x1.1f1b01b9054dfp+5)},
                                {"peek", integer(0)}}}},
        {"mode", {"pyccd", {{"clear_pct_threshold", real(0.25)},
                            {"snow_pct_threshold", real(0.75)}}}},
        {"model", {"lasso", {{"alpha", real(1.0)},
                             {"coef_floor", real(1.0)},
                             {"coefficient_max", integer(8)},
                             {"coefficient_mid", integer(6)},
                             {"coefficient_min", integer(4)},
                             {"max_iter", integer(1000)},
                             {"num_obs_factor", integer(3)},
                             {"tolerance", real(0x1.a36e2eb1c432dp-14)}}}},
        {"noise_floor", {"variogram", {{"min_lag_days", integer(30)}}}},
        {"outlier", {"floored", {{"statistic", ccd::config::Value(std::string("change_magnitude"))},
                                 {"threshold", real(0x1.1f1b01b9054dfp+5)}}}},
        {"refit", {"span-factor", {{"factor", real(0x1.547ae147ae148p+0)},
                                   {"min_obs", integer(24)}}}},
        {"segment_fit", {"model-fit", {}}},
        {"stability_fit", {"lasso", {{"alpha", real(1.0)},
                                     {"coef_floor", real(1.0)},
                                     {"coefficient_max", integer(8)},
                                     {"coefficient_mid", integer(6)},
                                     {"coefficient_min", integer(4)},
                                     {"max_iter", integer(1000)},
                                     {"num_obs_factor", integer(3)},
                                     {"tolerance", real(0x1.a36e2eb1c432dp-14)}}}},
        {"stability_rule", {"pyccd", {{"change_probability", real(0x1.fae147ae147aep-1)},
                                      {"change_threshold", real(0x1.e2c2be7b52448p+3)},
                                      {"peek", integer(0)}}}},
        {"start_search", {"pyccd", {{"day_delta", integer(365)}, {"meow_size", integer(12)}}}},
        {"window_screen", {"tmask", {{"bands", ccd::config::Value(List{1, 4})},
                                     {"t_const", real(0x1.38f5c28f5c28fp+2)}}}},
    };
    const ccd::config::Resolved got = r.resolve(preset, ccd::Input{});
    ASSERT_EQ(got.stages.size(), want.size());
    for (const auto& [stage, chosen] : want)
    {
        ASSERT_EQ(got.stages.count(stage), 1u) << stage;
        EXPECT_EQ(got.stages.at(stage).strategy, chosen.strategy) << stage;
        for (const auto& [name, value] : chosen.params)
            EXPECT_EQ(got.stages.at(stage).params.at(name), value) << stage << " " << name;
        EXPECT_EQ(got.stages.at(stage).params.size(), chosen.params.size()) << stage;
    }

    // The shared values, as the options held them: the detection rows and
    // the cutoff; the green median's offset in DN and its scope.
    const ccd::config::Shared& shared = got.shared;
    EXPECT_EQ(shared.detection_bands, (ccd::BandList{1, 2, 3, 4, 5}));
    EXPECT_EQ(shared.stat_ord, ccd::config::NO_STAT_CUTOFF);
    const ccd::config::Values& green = got.preprocess.at("insufficient_clear").steps.at(4).params;
    EXPECT_EQ(green.at("offset"), real(400.0));
    EXPECT_EQ(green.at("scope"), ccd::config::Value(std::string("stat_window")));
    EXPECT_EQ(shared.alpha, 1.0);
    EXPECT_EQ(shared.coef_floor, 1.0);
    EXPECT_EQ(shared, ccd::config::Shared{});
}

// resolve() spells out every parameter, so a partial configuration and the
// full one are the same run (and digest).
TEST(Registry, ResolveFillsTheDefaults)
{
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    const ccd::config::Config& ccdc = r.preset("ccdc");
    const ccd::Input input;

    ccd::config::Config partial = ccdc;
    partial.stages["refit"].params = {{"factor", ccd::config::real(1.5)}};
    partial.shared = {};
    ccd::config::Config full = ccdc;
    full.stages["refit"].params["factor"] = ccd::config::real(1.5);
    EXPECT_NE(ccd::config::digest(partial), ccd::config::digest(full));
    EXPECT_EQ(r.resolve(partial, input).stages, r.resolve(full, input).stages);
    EXPECT_EQ(r.resolve(partial, input).digest, r.resolve(full, input).digest);

    // A whole number for a real parameter is the real.
    partial.stages["refit"].params = {{"factor", ccd::config::integer(2)}};
    full.stages["refit"].params["factor"] = ccd::config::real(2.0);
    EXPECT_EQ(r.resolve(partial, input).stages, r.resolve(full, input).stages);

    partial.stages["refit"].params = {{"nope", ccd::config::integer(2)}};
    EXPECT_THROW(r.resolve(partial, input), std::invalid_argument);
    EXPECT_THROW(r.check(partial), std::invalid_argument);
    partial.stages["refit"].strategy = "none";
    EXPECT_THROW(r.resolve(partial, input), std::invalid_argument);
    partial = ccdc;
    partial.shared["nope"] = ccd::config::integer(1);
    expect_refused([&] { r.check(partial); }, "shared has no value 'nope'; it has detection_bands");
}

// The shared values reach every stage that reads them, resolved against the
// input; a stage cannot hold its own copy.
TEST(Registry, SharedValuesReachTheirStages)
{
    using ccd::config::real;
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    ccd::Input reflectance;
    reflectance.reflectance_unit = 1.0;

    // Reflectance values times the input's unit, in every Lasso stage.
    const auto lasso = golden::with_shared(golden::ccdc(), "alpha", real(0.5));
    for (const char* stage : {"model", "stability_fit", "edge_fit"})
    {
        EXPECT_EQ(golden::parameter<double>(r.resolve(lasso, reflectance).stages, stage, "alpha"),
                  0.5)
            << stage;
        EXPECT_EQ(golden::parameter<double>(r.resolve(lasso, ccd::Input{}).stages, stage, "alpha"),
                  0.5 * 10000.0)
            << stage;
    }
    EXPECT_EQ(r.resolve(golden::ccdc(), reflectance).plan.insufficient_clear.chain[4].green_offset,
              0.04);
    EXPECT_EQ(r.resolve(golden::ccdc(), reflectance).shared.alpha, 1e-4);

    // A probability, and the thresholds of its quantile for the count of
    // detection bands.
    auto probable = golden::with_shared(golden::ccdc(), "change_probability", real(0.95));
    probable = golden::with_shared(probable, "detection_bands",
                                   ccd::config::roles({"red", "nir", "swir1", "swir2"}));
    const ccd::config::ConfigTree tree = r.resolve(probable, ccd::Input{}).stages;
    for (const char* stage : {"change_test", "stability_rule", "lookback"})
    {
        EXPECT_EQ(golden::parameter<double>(tree, stage, "change_probability"), 0.95) << stage;
        EXPECT_EQ(golden::parameter<double>(tree, stage, "change_threshold"),
                  ccd::chi2_quantile(0.95, 4))
            << stage;
    }
    EXPECT_EQ(golden::parameter<double>(tree, "outlier", "threshold"),
              ccd::chi2_quantile(1.0 - 1e-6, 4));
    EXPECT_EQ(golden::parameter<double>(tree, "lookback", "outlier_threshold"),
              ccd::chi2_quantile(1.0 - 1e-6, 4));
    EXPECT_EQ(r.resolve(probable, ccd::Input{}).shared.detection_bands,
              (ccd::BandList{2, 3, 4, 5}));

    // edge_fit's coefficients are the coefficient rule's minimum.
    const auto six = golden::with_shared(golden::ccdc(), "coefficient_min", ccd::config::integer(6));
    EXPECT_EQ(golden::parameter<std::int64_t>(r.resolve(six, ccd::Input{}).stages, "edge_fit",
                                              "coefficients"),
              6);

    // A stage setting a shared or derived value is refused, naming it.
    auto copy = golden::ccdc();
    copy.stages["model"].params["alpha"] = real(1.0);
    expect_refused([&] { r.check(copy); },
                   "model 'lasso' parameter 'alpha' is the shared value 'alpha'; set shared "
                   "'alpha' instead");
    copy = golden::ccdc();
    copy.stages["outlier"].params["threshold"] = real(30.0);
    expect_refused([&] { r.resolve(copy, ccd::Input{}); },
                   "outlier 'floored' parameter 'threshold' is derived from the shared value "
                   "'outlier_probability'; set shared 'outlier_probability' instead");

    // Out of bounds as written, and the scope's names.
    expect_refused(
        [&] { r.check(golden::with_shared(golden::ccdc(), "change_probability", real(1.0))); },
        "shared parameter 'change_probability' is 1; it must be in (0, 1)");
    expect_refused(
        [&]
        {
            r.check(golden::with_steps(golden::ccdc(), "green_median", "scope",
                                       ccd::config::Value(std::string("some"))));
        },
        "preprocess insufficient_clear step 4 'green_median' parameter 'scope' is 'some'; "
        "it is all, stat_window or off");

    // The listing names each parameter's source.
    for (const auto& p : r.schemas().at("model").at("lasso"))
        if (p.name == "alpha")
        {
            EXPECT_EQ(p.source, ccd::config::Source::Shared);
            EXPECT_EQ(p.from, "alpha");
            EXPECT_FALSE(p.default_value.has_value());
        }
    EXPECT_EQ(r.defaults("model", "lasso").size(), 0u);
}

// Roles are rows of the input, in any band order; an unknown role, or one the
// input lacks, is refused, naming both.
TEST(Registry, RolesAreTheInputsRows)
{
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    ccd::Input reordered;
    reordered.roles = {"swir1", "swir2", "nir", "red", "green", "blue", "thermal"};
    const ccd::config::Resolved got = r.resolve(golden::ccdc(), reordered);
    EXPECT_EQ(got.shared.detection_bands, (ccd::BandList{4, 3, 2, 0, 1}));
    EXPECT_EQ(golden::parameter<std::vector<std::int64_t>>(got.stages, "window_screen", "bands"),
              (std::vector<std::int64_t>{4, 0}));
    EXPECT_NE(got.digest, r.resolve(golden::ccdc(), ccd::Input{}).digest);

    // band_range's "reflective": every row but thermal and other, in row
    // order; a list by role keeps its own order.
    ccd::Input mixed;
    mixed.roles = {"thermal", "swir1", "other", "green", "red", "nir", "swir2"};
    const auto band_rows = [&](const ccd::config::Config& c)
    { return golden::plan_of(c, mixed).standard.chain[2].bands; };
    EXPECT_EQ(band_rows(golden::ccdc()), (ccd::BandList{1, 3, 4, 5, 6}));
    EXPECT_EQ(band_rows(golden::with_steps(golden::ccdc(), "band_range", "bands",
                                           ccd::config::roles({"red", "swir1"}))),
              (ccd::BandList{4, 1}));

    ccd::Input four;
    four.roles = {"blue", "green", "red", "nir"};
    expect_refused([&] { r.resolve(golden::ccdc(), four); },
                   "shared value 'detection_bands' holds 'swir1', which the input does not have; "
                   "its rows are blue, green, red, nir");
    const auto unknown = golden::with(golden::ccdc(), "window_screen", "bands",
                                      ccd::config::roles({"green", "swir3"}));
    expect_refused([&] { r.check(unknown); },
                   "window_screen 'tmask' parameter 'bands' holds 'swir3', which is not a role; "
                   "the roles are blue, green");
    const auto other = golden::with(golden::ccdc(), "window_screen", "bands",
                                    ccd::config::roles({"other"}));
    expect_refused([&] { r.check(other); }, "holds 'other', which no band list names");
}

TEST(Registry, AnEditedPresetBuildsItsStrategies)
{
    ccd::config::Config tree = golden::ccdc();
    tree.stages["refit"].params["min_obs"] = ccd::config::integer(30);
    tree.stages["refit"].params["factor"] = ccd::config::real(1.5);
    const StageConfig& refit = tree.stages.at("refit");
    EXPECT_EQ(refit.strategy, "span-factor");
    const auto s = golden::strategies_of(tree);
    EXPECT_TRUE(s.refit->due(30, 1.0, 12, 1.0));
    EXPECT_FALSE(s.refit->due(31, 149.0, 12, 100.0));
    EXPECT_TRUE(s.refit->due(31, 150.0, 12, 100.0));

    // The schema gives the parameters back as set.
    const auto values = ccd::config::to_values(ccd::strategies::SpanFactor::schema(),
                                               ccd::strategies::SpanFactor::Params{30, 1.5});
    EXPECT_EQ(std::get<std::int64_t>(values.at("min_obs")), 30);
    EXPECT_EQ(std::get<double>(values.at("factor")), 1.5);

    // A min_obs (and the change test's model_rmse_up_to) as large as a
    // configuration holds: refit at every step.
    const ccd::config::Value most = ccd::config::integer(std::numeric_limits<std::int64_t>::max());
    tree.stages["refit"].params["min_obs"] = most;
    tree.stages["change_test"].params["model_rmse_up_to"] = most;
    EXPECT_TRUE(golden::strategies_of(tree).refit->due(1u << 20, 1.0, 12, 100.0));

    expect_refused([&] { ccd::strategies::core_registry().build({}); },
                   "the configuration has no 'mode' stage");
}

TEST(MonitoringModel, TheLassoIsTheKernelsFit)
{
    ccd::config::Config tree = golden::ccdc();
    tree.shared["alpha"] = ccd::config::real(0.5);
    tree.shared["max_iter"] = ccd::config::integer(300);
    EXPECT_EQ(tree.stages.at("model").strategy, "lasso");
    ccd::Input reflectance;
    reflectance.reflectance_unit = 1.0;
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

    const auto& r = ccd::strategies::core_registry();
    expect_refused([&] { golden::strategies_of(golden::with_shared(tree, "alpha",
                                                                   ccd::config::real(NAN))); },
                   "shared parameter 'alpha' is nan");
    expect_refused(
        [&] { r.make<ccd::MonitoringModel>({"lasso", {{"alpha", ccd::config::real(-1)}}}); },
        "model 'lasso' parameter 'alpha' is -1; it must be at least 0");
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
    const ccd::Input input;
    EXPECT_NO_THROW(r.resolve(tree, input));
    tree.stages["model"] = {"opaque", {}};
    expect_refused([&] { r.resolve(tree, input); },
                   "change_test 'chi2' needs peek_residuals, model_rmse, in_window_residuals, "
                   "which model 'opaque' does not provide");

    // The outlier rule's statistic, by name, from the change test.
    r.add<ccd::ChangeTest, Silent>("silent");
    tree.stages["model"] = {"lasso", {}};
    tree.stages["change_test"] = {"silent", {}};
    expect_refused([&] { r.check(tree); },
                   "outlier 'floored' reads the statistic 'change_magnitude', which "
                   "change_test 'silent' does not provide");
    tree = golden::ccdc();
    tree.stages["outlier"].params["statistic"] = ccd::config::Value(std::string("other"));
    expect_refused([&] { r.check(tree); },
                   "outlier 'floored' reads the statistic 'other', which change_test 'chi2' "
                   "does not provide; it provides change_magnitude");

    // The stability rule's needs, from the stability fit; Lookback's too.
    r.add<ccd::StabilityFit, OpaqueFit>("opaque");
    tree = golden::ccdc();
    tree.stages["stability_fit"] = {"opaque", {}};
    expect_refused([&] { r.resolve(tree, input); },
                   "stability_rule 'pyccd' needs model_rmse, in_window_residuals, which "
                   "stability_fit 'opaque' does not provide");

    r.add<ccd::StabilityFit, NoPeekFit>("no-peek");
    tree.stages["stability_fit"] = {"no-peek", {}};
    expect_refused([&] { r.resolve(tree, input); },
                   "lookback 'pyccd' needs peek_residuals, which stability_fit 'no-peek' does not "
                   "provide");

    // A configuration without an outlier stage.
    tree = golden::ccdc();
    tree.stages.erase("outlier");
    expect_refused([&] { r.check(tree); },
                   "the configuration has no 'outlier' stage");
}

TEST(OutlierRule, FlooredJudgesTheStatisticAgainstItsThreshold)
{
    // A resolved stage's threshold, as the registry builds it.
    const auto outlier = ccd::strategies::core_registry().make<ccd::OutlierRule>(
        {"floored", {{"threshold", ccd::config::real(10.0)}}});

    const std::int64_t day = 730000;
    const std::vector<std::vector<double>> residuals;
    const ccd::Variogram variogram;
    const ccd::BandList bands;
    const auto model = ccd::strategies::core_registry()
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
            ccd::strategies::core_registry().make<ccd::OutlierRule>(
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
    const ccd::index_t coefficient_min = count("stability_fit", "coefficient_min");
    const ccd::index_t coefficient_max = count("stability_fit", "coefficient_max");
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
    const ccd::Variogram variogram{std::vector<double>(bands, 20.0)};
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
        "test parameter 'bands' is a list of band roles or indices");
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
    const ccd::Variogram variogram{std::vector<double>(bands, 20.0)};

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
    // Fewer observations than meow_size: no room, exhausted (no unsigned wrap).
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
    own.stages["start_search"].params["meow_size"] = ccd::config::integer(10);
    expect_refused([&] { golden::strategies_of(own); },
                   "start_search 'pyccd' parameter 'meow_size' is the shared value "
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
    EXPECT_EQ(std::get<std::int64_t>(lookback.params.at("peek")), 0);   // the test's
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
    const ccd::Variogram variogram{std::vector<double>(bands, 0.0)};
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

TEST(EdgeFit, LassoIsTheCatchSegmentAndWholeSeriesKernels)
{
    ccd::config::Config config = golden::ccdc();
    config.shared["coefficient_min"] = ccd::config::integer(6);
    config.shared["min_observations"] = ccd::config::integer(15);
    config.shared["alpha"] = ccd::config::real(0.5);
    ccd::Input reflectance;
    reflectance.reflectance_unit = 1.0;
    const ccd::config::ConfigTree tree = golden::resolved(config, reflectance).stages;
    const auto& edges = tree.at("edge_fit");
    EXPECT_EQ(edges.strategy, "lasso");
    EXPECT_EQ(std::get<std::int64_t>(edges.params.at("coefficients")), 6);
    EXPECT_EQ(std::get<std::int64_t>(edges.params.at("min_observations")), 15);
    EXPECT_EQ(golden::parameter<std::int64_t>(tree, "start_search", "meow_size"), 15);
    EXPECT_EQ(std::get<double>(edges.params.at("alpha")), 0.5);
    EXPECT_EQ(tree.at("segment_fit").strategy, "model-fit");
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

    const auto scratch = s.edge_fit->make_scratch();
    const ccd::ChangeModel got =
        s.edge_fit->catch_window(d, x, window, ccd::CurveQA::End, *scratch);
    // The kernels' solver with the edge fit's parameters.
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

TEST(NoiseFloor, VariogramIsTheAdjustedVariogram)
{
    const auto tree =
        golden::with(golden::ccdc(), "noise_floor", "min_lag_days", ccd::config::integer(45));
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
    EXPECT_EQ(s.noise_floor->estimate(d, x), ccd::adjusted_variogram(d, x, 45));
    EXPECT_EQ(golden::parameter<std::int64_t>(golden::resolved(tree).stages, "noise_floor",
                                              "min_lag_days"),
              45);
}

TEST(ConfigDigest, EqualConfigurationsAndOnlyThem)
{
    const auto a = golden::ccdc();
    auto b = a;
    EXPECT_EQ(ccd::config::digest(a), ccd::config::digest(b));
    b.stages["refit"].params["factor"] = ccd::config::real(1.34);
    EXPECT_NE(ccd::config::digest(a), ccd::config::digest(b));
    b = a;
    b.stages["refit"].params["factor"] = ccd::config::integer(1);   // another type
    auto c = a;
    c.stages["refit"].params["factor"] = ccd::config::real(1.0);
    EXPECT_NE(ccd::config::digest(b), ccd::config::digest(c));
    b = golden::with_shared(a, "alpha", ccd::config::real(2e-4));
    EXPECT_NE(ccd::config::digest(a), ccd::config::digest(b));
    EXPECT_NE(golden::resolved(a).digest, golden::resolved(b).digest);
    // The resolved digest covers the input.
    ccd::Input other;
    other.qa_snow = 7;
    EXPECT_NE(golden::resolved(a).digest, golden::resolved(a, other).digest);
}

TEST(Registry, BandListsAreCheckedAgainstTheInput)
{
    const auto& r = ccd::strategies::core_registry();
    auto tree = golden::with(golden::ccdc(), "window_screen", "bands", ccd::config::bands({1, 4}));
    EXPECT_NO_THROW(r.resolve(tree, ccd::Input{}));

    ccd::Input four;
    four.roles = {"blue", "green", "red", "nir"};
    tree.shared["detection_bands"] = ccd::config::bands({1, 2, 3});
    expect_refused([&] { r.resolve(tree, four); },
                   "window_screen 'tmask' parameter 'bands' holds band 4; the input has 4 bands");
    tree.stages["window_screen"].params["bands"] = ccd::config::bands({2, 2});
    expect_refused([&] { r.resolve(tree, ccd::Input{}); }, "holds band 2 twice");
    tree = golden::with_shared(golden::ccdc(), "detection_bands", ccd::config::bands({1, 9}));
    expect_refused([&] { r.resolve(tree, ccd::Input{}); },
                   "shared value 'detection_bands' holds band 9; the input has 7 bands");
    tree = golden::with_shared(golden::ccdc(), "detection_bands", ccd::config::bands({}));
    expect_refused([&] { r.check(tree); }, "shared value 'detection_bands' is empty");
}

//------------------------------------------------------------------------------
// The preprocessing section: conversions, refusals, and the registration of
// a step type beyond the library's.
//------------------------------------------------------------------------------

// Degrees Celsius resolve to the thermal row's unit (hundredths of a degree
// times reflectance_unit / 10000), reflectance to reflectance_unit, QA code
// names to the input's codes; codes given as numbers stay as given.
TEST(Preprocess, StepValuesResolveInTheInputsUnitsAndCodes)
{
    using ccd::config::names;
    using ccd::config::real;
    const ccd::config::Registry& r = ccd::strategies::library_registry();

    ccd::Input renumbered;
    renumbered.qa_clear = 20;
    renumbered.qa_water = 21;
    renumbered.qa_snow = 23;
    renumbered.qa_cloud = 24;

    ccd::config::Config config = golden::with_step(golden::ccdc(), "standard", 0, "codes",
                                                   names({"clear", "water", "cloud"}));
    config = golden::with_step(config, "standard", 1, "min", real(-50.0));
    config = golden::with_step(config, "standard", 2, "max", real(0.5));
    const ccd::config::Resolved got = r.resolve(config, renumbered);
    const auto& standard = got.preprocess.at("standard").steps;
    EXPECT_EQ(standard.at(0).params.at("codes"),
              ccd::config::Value(std::vector<std::int64_t>{20, 21, 24}));
    EXPECT_EQ(standard.at(1).params.at("min"), real(-5000.0));
    EXPECT_EQ(standard.at(2).params.at("max"), real(5000.0));
    EXPECT_TRUE(got.plan.standard.chain[0].codes.test(24));
    EXPECT_FALSE(got.plan.standard.chain[0].codes.test(0));
    EXPECT_EQ(got.plan.snow.chain[4].codes, ccd::preprocess::qa_set({23}));

    ccd::Input reflectance = golden::physical_input();
    const ccd::config::Resolved phys = r.resolve(config, reflectance);
    EXPECT_EQ(phys.plan.standard.chain[1].range.lo.value, -0.5);
    EXPECT_EQ(phys.plan.standard.chain[2].range.hi.value, 0.5);

    // Codes by number, as given.
    const ccd::config::Config numbered = golden::with_step(
        golden::ccdc(), "standard", 0, "codes",
        ccd::config::Value(std::vector<std::int64_t>{0, 7}));
    EXPECT_EQ(r.resolve(numbered, renumbered).plan.standard.chain[0].codes,
              ccd::preprocess::qa_set({0, 7}));

    expect_refused(
        [&]
        {
            r.check(golden::with_step(golden::ccdc(), "standard", 0, "codes",
                                      names({"clear", "haze"})));
        },
        "preprocess standard step 0 'keep_qa' parameter 'codes' holds 'haze', which is not a QA "
        "code name; the names are fill, clear, water, shadow, snow, cloud, cirrus1, cirrus2, "
        "occlusion");
    expect_refused(
        [&]
        {
            r.check(golden::with_step(golden::ccdc(), "standard", 0, "codes",
                                      ccd::config::Value(std::vector<std::int64_t>{256})));
        },
        "parameter 'codes' holds 256; a code must be in [0, 255]");
}

// An Algorithm resolves an input once and keeps the CACHE_CAPACITY inputs
// used last; one more evicts the least recently used, which resolves again
// to an equal Resolved.
TEST(Algorithm, CachesTheInputsUsedLast)
{
    const ccd::Algorithm algorithm(golden::ccdc());
    const auto input = [](int i)
    {
        ccd::Input h;
        h.qa_occlusion = static_cast<std::uint8_t>(20 + i);
        return h;
    };
    const auto first = algorithm.for_input(input(0));
    EXPECT_EQ(algorithm.for_input(input(0)), first);

    // Inputs 1..CAPACITY - 1 fill the cache; input 0, used again, stays.
    const int capacity = static_cast<int>(ccd::Algorithm::CACHE_CAPACITY);
    std::vector<std::shared_ptr<const ccd::config::Resolved>> held;
    for (int i = 1; i < capacity; ++i)
        held.push_back(algorithm.for_input(input(i)));
    EXPECT_EQ(algorithm.for_input(input(0)), first);
    EXPECT_EQ(algorithm.for_input(input(1)), held[0]);

    // One more evicts the least recently used: input 2.
    const auto extra = algorithm.for_input(input(capacity));
    EXPECT_EQ(algorithm.for_input(input(0)), first);
    const auto again = algorithm.for_input(input(2));
    EXPECT_NE(again, held[1]);
    EXPECT_EQ(again->digest, held[1]->digest);
    EXPECT_NE(extra->digest, first->digest);
}

// What an Algorithm refuses in the preprocessing when it is built, before
// any input.
TEST(Preprocess, AnAlgorithmRefusesABadSectionWhenBuilt)
{
    const auto build = [](const ccd::config::Config& config)
    { return ccd::Algorithm(config); };

    ccd::config::Config unknown = golden::ccdc();
    unknown.preprocess.at("snow").steps.at(4).step = "keep_haze";
    expect_refused([&] { build(unknown); },
                   "preprocess snow step 4 is 'keep_haze', which is not a preprocessing step; "
                   "the steps are add_qa, band_range, dedupe, green_median, keep_qa, "
                   "thermal_range");

    ccd::config::Config missing = golden::ccdc();
    missing.preprocess.erase("snow");
    expect_refused([&] { build(missing); },
                   "the configuration's preprocess has no 'snow' procedure");

    ccd::config::Config extra = golden::ccdc();
    extra.preprocess["water"] = extra.preprocess.at("snow");
    expect_refused([&] { build(extra); },
                   "preprocess has no procedure 'water'; the procedures are standard, snow, "
                   "insufficient_clear");

    ccd::config::Config long_chain = golden::ccdc();
    auto& steps = long_chain.preprocess.at("standard").steps;
    while (steps.size() <= static_cast<std::size_t>(ccd::preprocess::CHAIN_CAPACITY))
        steps.push_back({"dedupe", {}});
    expect_refused([&] { build(long_chain); },
                   "preprocess standard has 9 steps; a chain holds at most 8");

    expect_refused(
        [&]
        {
            build(golden::with_steps(golden::ccdc(), "green_median", "scope",
                                     ccd::config::Value(std::string("clouds"))));
        },
        "preprocess insufficient_clear step 4 'green_median' parameter 'scope' is 'clouds'; "
        "it is all, stat_window or off");

    ccd::config::Config dated = golden::ccdc();
    dated.preprocess.at("insufficient_clear").steps.at(4).params["max_date"] =
        ccd::config::integer(736694);
    expect_refused([&] { build(dated); },
                   "preprocess insufficient_clear step 4 'green_median' parameter 'max_date' is "
                   "the shared value 'stat_ord'; set shared 'stat_ord' instead");

    ccd::config::Config typo = golden::ccdc();
    typo.preprocess.at("standard").steps.at(2).params["maximum"] = ccd::config::real(1.0);
    expect_refused([&] { build(typo); },
                   "preprocess standard step 2 'band_range' has no parameter 'maximum'");
}

namespace
{

// A rule of a step type a registry adds: drop the observations dated `day`,
// reading the row `band` (so validate() has something to refuse).
class DropDay final : public ccd::preprocess::CustomStep
{
public:
    DropDay(std::int64_t day, ccd::index_t band) : day_(day), band_(band) {}

    void apply(ccd::ArrayView<const std::int64_t, 1> dates, ccd::ArrayView<const double, 2>,
               ccd::ArrayView<const std::uint8_t, 1>, ccd::ProcessingMask& mask) const override
    {
        for (ccd::index_t t = 0; t < dates.size(); ++t)
            if (dates(t) == day_)
                mask.set(t, false);
    }

    void validate(ccd::index_t bands) const override
    {
        if (band_ >= bands)
            throw std::invalid_argument("band " + std::to_string(band_) + " is not below " +
                                        std::to_string(bands));
    }

    std::string describe() const override { return "DropDay " + std::to_string(day_); }

private:
    std::int64_t day_;
    ccd::index_t band_;
};

// The step type: its parameter, and its step on an input.
struct DropDayStep
{
    struct Params
    {
        std::int64_t day = 0;
    };

    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s = {
            {"day", &Params::day, ccd::config::Unit::None, ccd::config::any(),
             "the date dropped, an ordinal day"},
        };
        return s;
    }

    static std::optional<ccd::preprocess::Step> make(const Params& params, const ccd::Input& input)
    {
        return ccd::preprocess::Step::of(std::make_shared<const DropDay>(params.day,
                                                                       input.index("green")));
    }
};

} // namespace

// A registry adds a step type to the library's (Registry::add_step()): a
// configuration's chain names it, it resolves into a Custom step of the plan,
// and a run applies it.
TEST(Preprocess, ARegistryAddsAStepType)
{
    ccd::config::Registry r;
    ccd::strategies::register_core_strategies(r);
    r.add_step<DropDayStep>("drop_day");
    EXPECT_THROW(r.add_step<DropDayStep>("drop_day"), std::logic_error);
    EXPECT_EQ(r.step_defaults("drop_day"), (Values{{"day", ccd::config::integer(0)}}));
    EXPECT_EQ(r.step_schemas().at("drop_day").size(), 1u);

    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 1, 1);
    const std::vector<std::int64_t>& dates = c.dates;
    const std::vector<double>& spectral = c.spectral;
    const std::vector<std::uint8_t>& qas = c.qas;
    const ccd::index_t T = c.T;

    const auto run = [&](const ccd::config::Config& config)
    {
        return ccd::run_pixel(
                   golden::spec_of(ccd::Input{}, std::make_shared<const ccd::Algorithm>(config, r)),
                   ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
                   ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {7, T}),
                   ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}))
            .result;
    };
    const ccd::FitResult plain = run(golden::pyccd());

    // A clear observation the Standard mask keeps, dropped by the step.
    ccd::index_t kept = 0;
    while (!plain.mask[kept])
        ++kept;
    ccd::config::Config config = golden::pyccd();
    auto& steps = config.preprocess.at("standard").steps;
    steps.insert(steps.begin() + 1, {"drop_day", {{"day", ccd::config::integer(dates[kept])}}});

    const ccd::config::Resolved resolved = r.resolve(config, ccd::Input{});
    const ccd::preprocess::Step& step = resolved.plan.standard.chain[1];
    ASSERT_EQ(step.kind, ccd::preprocess::StepKind::Custom);
    EXPECT_EQ(step.custom->describe(), "DropDay " + std::to_string(dates[kept]));
    EXPECT_NE(resolved.digest, r.resolve(golden::pyccd(), ccd::Input{}).digest);

    // validate() asks the step.
    try
    {
        ccd::preprocess::validate(resolved.plan, 1);
        ADD_FAILURE() << "a plan reading band 1 accepted for one band";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("standard step 1 (Custom): band 1 is not below 1"),
                  std::string::npos)
            << e.what();
    }

    const ccd::FitResult dropped = run(config);
    EXPECT_FALSE(dropped.mask[kept]);
    for (ccd::index_t t = 0; t < T; ++t)
        if (t != kept)
            EXPECT_EQ(dropped.mask[t], plain.mask[t]) << t;

    // The library's registry has no such step.
    expect_refused([&] { ccd::Algorithm{config}; },
                   "preprocess standard step 1 is 'drop_day', which is not a preprocessing step");
}
