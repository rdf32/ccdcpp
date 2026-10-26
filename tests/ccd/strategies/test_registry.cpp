//==============================================================================
//
// The registry and the strategy parameters (config/registry.hpp,
// config/params.hpp), on the library's strategies:
//
//   * the registry lists each stage's strategies, and builds one by name with
//     its parameters checked against its schema (unknown names, wrong types,
//     values out of range all refused, naming the stage, strategy and field);
//   * the "ccdc" preset, written in physical terms, resolves on the default
//     DataSpec to pyccd's tree bit for bit; a configuration resolves against a
//     data spec (reflectance times its unit, roles as its rows, the shared
//     values and the chi-squared thresholds into the stages that read
//     them), and a stage setting a shared value is refused;
//   * the "ccdc" preset, edited by stage and parameter, builds the
//     strategies a run uses;
//   * a band list parameter takes indices 0 to CCD_MAX_BANDS - 1, and band
//     lists are checked against the spec;
//   * a configuration's digest is equal for equal configurations only.
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

using ccd::config::StageConfig;
using ccd::config::Values;
using registry_test::expect_refused;

} // namespace

TEST(Registry, ListsTheCoreStrategies)
{
    const auto names = golden::core_registry().names();
    ASSERT_TRUE(names.contains("noise"));
    EXPECT_EQ(names.at("noise"), (std::vector<std::string>{"variogram"}));
    // The library's registry holds every core strategy (and its extensions').
    const auto library = ccd::strategies::library_registry().names();
    for (const auto& [stage, strategies] : names)
        for (const auto& strategy : strategies)
        {
            ASSERT_TRUE(library.contains(stage)) << stage;
            const auto& in = library.at(stage);
            EXPECT_NE(std::find(in.begin(), in.end(), strategy), in.end()) << stage << " " << strategy;
        }
    ASSERT_TRUE(names.contains("mode"));
    EXPECT_EQ(names.at("mode"), (std::vector<std::string>{"pyccd"}));
    ASSERT_TRUE(names.contains("segment_report"));
    EXPECT_EQ(names.at("segment_report"), (std::vector<std::string>{"model-fit"}));
    ASSERT_TRUE(names.contains("edge_segments"));
    EXPECT_EQ(names.at("edge_segments"), (std::vector<std::string>{"lasso"}));
    ASSERT_TRUE(names.contains("start_search"));
    EXPECT_EQ(names.at("start_search"), (std::vector<std::string>{"pyccd"}));
    ASSERT_TRUE(names.contains("window_screen"));
    EXPECT_EQ(names.at("window_screen"), (std::vector<std::string>{"tmask"}));
    ASSERT_TRUE(names.contains("stability_fit"));
    EXPECT_EQ(names.at("stability_fit"), (std::vector<std::string>{"lasso"}));
    ASSERT_TRUE(names.contains("stability_rule"));
    EXPECT_EQ(names.at("stability_rule"), (std::vector<std::string>{"pyccd"}));
    ASSERT_TRUE(names.contains("lookback"));
    EXPECT_EQ(names.at("lookback"), (std::vector<std::string>{"pyccd"}));
    ASSERT_TRUE(names.contains("model"));
    EXPECT_EQ(names.at("model"), (std::vector<std::string>{"lasso"}));
    ASSERT_TRUE(names.contains("refit"));
    EXPECT_EQ(names.at("refit"), (std::vector<std::string>{"span-factor"}));
    ASSERT_TRUE(names.contains("change_test"));
    EXPECT_EQ(names.at("change_test"), (std::vector<std::string>{"chi2"}));
    ASSERT_TRUE(names.contains("outlier"));
    EXPECT_EQ(names.at("outlier"), (std::vector<std::string>{"floored"}));
}

TEST(Registry, BuildsByNameAndChecksTheParameters)
{
    using ccd::RefitSchedule;
    using ccd::config::flag;
    using ccd::config::integer;
    using ccd::config::real;
    const auto& r = golden::core_registry();
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

// The "ccdc" preset is written in physical terms; on the default DataSpec
// (Collection-1 DN, the Landsat rows) it resolves to pyccd's tree, every
// value bit for bit as pyccd holds it in DN. Its statistics cutoff is none,
// where pyccd's is 2017-12-31.
TEST(Registry, CcdcOnTheDefaultSpecIsPyccdsTree)
{
    using ccd::config::integer;
    using ccd::config::real;
    using List = std::vector<std::int64_t>;
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    const ccd::config::Config& preset = r.preset("ccdc");

    // Physical: alpha in reflectance, probabilities, roles, the cutoff.
    EXPECT_EQ(preset.shared.at("lasso.alpha"), real(1e-4));
    EXPECT_EQ(preset.shared.at("lasso.coef_floor"), real(1e-4));
    EXPECT_EQ(preset.preprocess.at("insufficient_clear").steps.at(4).params.at("offset"),
              real(0.04));
    EXPECT_EQ(preset.shared.at("change_probability"), real(0.99));
    EXPECT_EQ(preset.shared.at("outlier_probability"), real(1.0 - 1e-6));
    EXPECT_EQ(preset.shared.at("statistics_end"), ccd::config::Value(std::string("none")));
    EXPECT_EQ(preset.shared.at("detection_bands"),
              ccd::config::roles({"green", "red", "nir", "swir1", "swir2"}));
    EXPECT_EQ(preset.stages.at("window_screen").params.at("bands"),
              ccd::config::roles({"green", "swir1"}));
    EXPECT_FALSE(preset.stages.at("model").params.contains("lasso.alpha"));
    EXPECT_FALSE(preset.stages.at("change_test").params.contains("change_threshold"));

    // The tree the preset was in Collection-1 DN, every parameter.
    const ccd::config::ConfigTree want = {
        {"change_test", {"chi2", {{"change_probability", real(0x1.fae147ae147aep-1)},
                                  {"change_threshold", real(0x1.e2c2be7b52448p+3)},
                                  {"model_rmse_up_to", integer(24)},
                                  {"peek_reference_days", real(16.0)},
                                  {"peek_size", integer(6)},
                                  {"seasonal_count", integer(24)},
                                  {"seasonal_divisor", real(4.0)}}}},
        {"edge_segments", {"lasso", {{"coefficients.min", integer(4)},
                                {"lasso.alpha", real(1.0)},
                                {"lasso.coef_floor", real(1.0)},
                                {"lasso.max_iter", integer(1000)},
                                {"lasso.tolerance", real(0x1.a36e2eb1c432dp-14)},
                                {"min_observations", integer(12)}}}},
        {"lookback", {"pyccd", {{"change_probability", real(0x1.fae147ae147aep-1)},
                                {"change_threshold", real(0x1.e2c2be7b52448p+3)},
                                {"outlier_threshold", real(0x1.1f1b01b9054dfp+5)},
                                {"peek", ccd::config::Value(std::string("change_test"))}}}},
        {"mode", {"pyccd", {{"clear_pct_threshold", real(0.25)},
                            {"snow_pct_threshold", real(0.75)}}}},
        {"model", {"lasso", {{"coefficients.max", integer(8)},
                             {"coefficients.mid", integer(6)},
                             {"coefficients.min", integer(4)},
                             {"coefficients.per", integer(3)},
                             {"lasso.alpha", real(1.0)},
                             {"lasso.coef_floor", real(1.0)},
                             {"lasso.max_iter", integer(1000)},
                             {"lasso.tolerance", real(0x1.a36e2eb1c432dp-14)}}}},
        {"noise", {"variogram", {{"min_lag_days", integer(30)}}}},
        {"outlier", {"floored", {{"statistic", ccd::config::Value(std::string("change_magnitude"))},
                                 {"threshold", real(0x1.1f1b01b9054dfp+5)}}}},
        {"refit", {"span-factor", {{"factor", real(0x1.547ae147ae148p+0)},
                                   {"min_obs", integer(24)}}}},
        {"segment_report", {"model-fit", {}}},
        {"stability_fit", {"lasso", {{"coefficients.max", integer(8)},
                                     {"coefficients.mid", integer(6)},
                                     {"coefficients.min", integer(4)},
                                     {"coefficients.per", integer(3)},
                                     {"lasso.alpha", real(1.0)},
                                     {"lasso.coef_floor", real(1.0)},
                                     {"lasso.max_iter", integer(1000)},
                                     {"lasso.tolerance", real(0x1.a36e2eb1c432dp-14)}}}},
        {"stability_rule", {"pyccd", {{"change_probability", real(0x1.fae147ae147aep-1)},
                                      {"change_threshold", real(0x1.e2c2be7b52448p+3)},
                                      {"peek", ccd::config::Value(std::string("change_test"))}}}},
        {"start_search", {"pyccd", {{"day_delta", integer(365)}, {"min_observations", integer(12)}}}},
        {"window_screen", {"tmask", {{"bands", ccd::config::Value(List{1, 4})},
                                     {"t_const", real(0x1.38f5c28f5c28fp+2)}}}},
    };
    const ccd::config::Resolved got = ccd::config::resolve(r, preset, ccd::DataSpec{});
    ASSERT_EQ(got.stages.size(), want.size());
    for (const auto& [stage, chosen] : want)
    {
        ASSERT_TRUE(got.stages.contains(stage)) << stage;
        EXPECT_EQ(got.stages.at(stage).strategy, chosen.strategy) << stage;
        for (const auto& [name, value] : chosen.params)
            EXPECT_EQ(got.stages.at(stage).params.at(name), value) << stage << " " << name;
        EXPECT_EQ(got.stages.at(stage).params.size(), chosen.params.size()) << stage;
    }

    // The shared values, as the options held them: the detection rows and
    // the cutoff; the green median's offset in DN and its scope.
    const ccd::config::Shared& shared = got.shared;
    EXPECT_EQ(shared.detection_bands, (ccd::BandList{1, 2, 3, 4, 5}));
    EXPECT_EQ(shared.statistics_end, ccd::config::NO_STATISTICS_END);
    const ccd::config::Values& green = got.preprocess.at("insufficient_clear").steps.at(4).params;
    EXPECT_EQ(green.at("offset"), real(400.0));
    EXPECT_EQ(green.at("scope"), ccd::config::Value(std::string("stat_window")));
    EXPECT_EQ(shared.lasso.alpha, 1.0);
    EXPECT_EQ(shared.lasso.coef_floor, 1.0);
    EXPECT_EQ(shared, ccd::config::Shared{});
}

// resolve() spells out every parameter, so a partial configuration and the
// full one are the same run (and digest).
TEST(Registry, ResolveFillsTheDefaults)
{
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    const ccd::config::Config& ccdc = r.preset("ccdc");
    const ccd::DataSpec spec;

    ccd::config::Config partial = ccdc;
    partial.stages["refit"].params = {{"factor", ccd::config::real(1.5)}};
    partial.shared = {};
    ccd::config::Config full = ccdc;
    full.stages["refit"].params["factor"] = ccd::config::real(1.5);
    EXPECT_NE(ccd::config::digest(partial), ccd::config::digest(full));
    EXPECT_EQ(ccd::config::resolve(r, partial, spec).stages, ccd::config::resolve(r, full, spec).stages);
    EXPECT_EQ(ccd::config::resolve(r, partial, spec).digest, ccd::config::resolve(r, full, spec).digest);

    // A whole number for a real parameter is the real.
    partial.stages["refit"].params = {{"factor", ccd::config::integer(2)}};
    full.stages["refit"].params["factor"] = ccd::config::real(2.0);
    EXPECT_EQ(ccd::config::resolve(r, partial, spec).stages, ccd::config::resolve(r, full, spec).stages);

    partial.stages["refit"].params = {{"nope", ccd::config::integer(2)}};
    EXPECT_THROW(ccd::config::resolve(r, partial, spec), std::invalid_argument);
    EXPECT_THROW(ccd::config::check(r, partial), std::invalid_argument);
    partial.stages["refit"].strategy = "none";
    EXPECT_THROW(ccd::config::resolve(r, partial, spec), std::invalid_argument);
    partial = ccdc;
    partial.shared["nope"] = ccd::config::integer(1);
    expect_refused([&] { ccd::config::check(r, partial); }, "shared has no value 'nope'; it has detection_bands");
}

// A named value stands for the one value its field means something
// particular by, both ways: statistics_end "none" and a lookback's peek
// "change_test". A date is its ordinal, as Python's date.toordinal().
TEST(Registry, NamedValuesAndDates)
{
    using ccd::config::Value;
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    const auto end_of = [&](Value end)
    {
        return ccd::config::resolve(r, golden::with_shared(golden::ccdc(), "statistics_end",
                                                           std::move(end)),
                                    ccd::DataSpec{})
            .shared.statistics_end;
    };
    EXPECT_EQ(end_of(Value(std::string("none"))), ccd::config::NO_STATISTICS_END);
    for (const auto& [date, ordinal] : std::vector<std::pair<std::string, std::int64_t>>{
             {"0001-01-01", 1}, {"1970-01-01", 719163}, {"2000-02-29", 730179},
             {"2017-12-31", 736694}, {"2100-03-01", 766704}, {"9999-12-31", 3652059}})
        EXPECT_EQ(end_of(Value(date)), ordinal) << date;
    EXPECT_EQ(end_of(ccd::config::integer(736694)), 736694);
    for (const char* bad : {"2017-02-29", "2017-13-01", "0000-01-01"})
        expect_refused([&] { end_of(Value(std::string(bad))); },
                       "shared value 'statistics_end' is '" + std::string(bad) +
                           "', which is not a date");
    expect_refused([&] { end_of(Value(std::string("never"))); },
                   "shared parameter 'statistics_end' is 'never'; it is a number or \"none\"");

    // The resolved values give the name back; the green median reads it too.
    const ccd::config::Resolved resolved = golden::resolved(golden::ccdc());
    EXPECT_EQ(resolved.stages.at("lookback").params.at("peek"), Value(std::string("change_test")));
    EXPECT_EQ(resolved.preprocess.at("insufficient_clear").steps.at(4).params.at("statistics_end"),
              Value(std::string("none")));
    EXPECT_EQ(golden::parameter<std::int64_t>(
                  golden::resolved(golden::with(golden::ccdc(), "lookback", "peek",
                                                ccd::config::integer(8)))
                      .stages,
                  "lookback", "peek"),
              8);
    expect_refused([&] { golden::resolved(golden::with(golden::ccdc(), "lookback", "peek",
                                                       ccd::config::integer(0))); },
                   "lookback 'pyccd' parameter 'peek' is 0; it must be at least 1");
}

// canonical() writes a configuration out in full, in its own terms: the
// defaults it leaves out spelled out, a whole number for a real as the real.
// The Algorithm holds that form.
TEST(Registry, CanonicalSpellsOutTheConfiguration)
{
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    const ccd::config::Config& ccdc = r.preset("ccdc");
    EXPECT_EQ(ccd::config::canonical(r, ccdc), ccdc);

    ccd::config::Config partial = ccdc;
    partial.shared = {{"change_probability", ccd::config::real(0.95)}};
    partial.stages["refit"].params = {{"factor", ccd::config::integer(2)}};
    for (auto& step : partial.preprocess.at("snow").steps)
        step.params.clear();
    ccd::config::Config full = ccdc;
    full.shared["change_probability"] = ccd::config::real(0.95);
    full.stages["refit"].params["factor"] = ccd::config::real(2.0);
    EXPECT_EQ(ccd::config::canonical(r, partial), full);
    EXPECT_EQ(ccd::config::digest(ccd::config::canonical(r, partial)), ccd::config::digest(full));
    EXPECT_EQ(ccd::Algorithm(partial).config(), full);

    partial.stages["refit"].params = {{"nope", ccd::config::integer(2)}};
    expect_refused([&] { ccd::config::canonical(r, partial); },
                   "refit 'span-factor' has no parameter 'nope'");
}

// The shared values reach every stage that reads them, resolved against the
// spec; a stage cannot hold its own copy.
TEST(Registry, SharedValuesReachTheirStages)
{
    using ccd::config::real;
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    ccd::DataSpec reflectance;
    reflectance.reflectance_scale = 1.0;

    // Reflectance values times the spec's unit, in every Lasso stage.
    const auto lasso = golden::with_shared(golden::ccdc(), "lasso.alpha", real(0.5));
    for (const char* stage : {"model", "stability_fit", "edge_segments"})
    {
        EXPECT_EQ(golden::parameter<double>(ccd::config::resolve(r, lasso, reflectance).stages, stage, "lasso.alpha"),
                  0.5)
            << stage;
        EXPECT_EQ(golden::parameter<double>(ccd::config::resolve(r, lasso, ccd::DataSpec{}).stages, stage, "lasso.alpha"),
                  0.5 * 10000.0)
            << stage;
    }
    EXPECT_EQ(ccd::config::resolve(r, golden::ccdc(), reflectance).plan.insufficient_clear.chain[4].green_offset,
              0.04);
    EXPECT_EQ(ccd::config::resolve(r, golden::ccdc(), reflectance).shared.lasso.alpha, 1e-4);

    // A probability, and the thresholds of its quantile for the count of
    // detection bands.
    auto probable = golden::with_shared(golden::ccdc(), "change_probability", real(0.95));
    probable = golden::with_shared(probable, "detection_bands",
                                   ccd::config::roles({"red", "nir", "swir1", "swir2"}));
    const ccd::config::ConfigTree tree = ccd::config::resolve(r, probable, ccd::DataSpec{}).stages;
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
    EXPECT_EQ(ccd::config::resolve(r, probable, ccd::DataSpec{}).shared.detection_bands,
              (ccd::BandList{2, 3, 4, 5}));

    // edge_segments' coefficients.min is the coefficient rule's minimum.
    const auto six = golden::with_shared(golden::ccdc(), "coefficients.min", ccd::config::integer(6));
    EXPECT_EQ(golden::parameter<std::int64_t>(ccd::config::resolve(r, six, ccd::DataSpec{}).stages, "edge_segments",
                                              "coefficients.min"),
              6);

    // A stage setting a shared or derived value is refused, naming it.
    auto copy = golden::ccdc();
    copy.stages["model"].params["lasso.alpha"] = real(1.0);
    expect_refused([&] { ccd::config::check(r, copy); },
                   "model 'lasso' parameter 'lasso.alpha' is the shared value 'lasso.alpha'; set "
                   "shared 'lasso.alpha' instead");
    copy = golden::ccdc();
    copy.stages["outlier"].params["threshold"] = real(30.0);
    expect_refused([&] { ccd::config::resolve(r, copy, ccd::DataSpec{}); },
                   "outlier 'floored' parameter 'threshold' is derived from the shared value "
                   "'outlier_probability'; set shared 'outlier_probability' instead");

    // Out of bounds as written, and the scope's names.
    expect_refused(
        [&] { ccd::config::check(r, golden::with_shared(golden::ccdc(), "change_probability", real(1.0))); },
        "shared parameter 'change_probability' is 1; it must be in (0, 1)");
    expect_refused(
        [&]
        {
            ccd::config::check(r, golden::with_steps(golden::ccdc(), "green_median", "scope",
                                       ccd::config::Value(std::string("some"))));
        },
        "preprocess insufficient_clear step 4 'green_median' parameter 'scope' is 'some'; "
        "it is all, stat_window or off");

    // The listing names each parameter's source.
    const auto schemas = r.schemas();
    for (const auto& p : schemas.at("model").at("lasso"))
        if (p.name == "lasso.alpha")
        {
            EXPECT_EQ(p.source, ccd::config::Source::Shared);
            EXPECT_EQ(p.from, "lasso.alpha");
            EXPECT_FALSE(p.default_value.has_value());
        }
    EXPECT_EQ(r.defaults("model", "lasso").size(), 0u);
}

// Roles are rows of the spec, in any band order; an unknown role, or one the
// spec lacks, is refused, naming both.
TEST(Registry, RolesAreTheSpecsRows)
{
    const ccd::config::Registry& r = ccd::strategies::library_registry();
    ccd::DataSpec reordered;
    reordered.roles = {"swir1", "swir2", "nir", "red", "green", "blue", "thermal"};
    const ccd::config::Resolved got = ccd::config::resolve(r, golden::ccdc(), reordered);
    EXPECT_EQ(got.shared.detection_bands, (ccd::BandList{4, 3, 2, 0, 1}));
    EXPECT_EQ(golden::parameter<std::vector<std::int64_t>>(got.stages, "window_screen", "bands"),
              (std::vector<std::int64_t>{4, 0}));
    EXPECT_NE(got.digest, ccd::config::resolve(r, golden::ccdc(), ccd::DataSpec{}).digest);

    // band_range's list by role keeps its own order: ccdc's blue to swir2,
    // whatever rows they are; a listed role the spec lacks is refused.
    ccd::DataSpec mixed;
    mixed.roles = {"thermal", "swir1", "blue", "other", "green", "red", "nir", "swir2"};
    const auto band_rows = [&](const ccd::config::Config& c)
    { return golden::plan_of(c, mixed).standard.chain[2].bands; };
    EXPECT_EQ(band_rows(golden::ccdc()), (ccd::BandList{2, 4, 5, 6, 1, 7}));
    EXPECT_EQ(band_rows(golden::with_steps(golden::ccdc(), "band_range", "bands",
                                           ccd::config::roles({"red", "swir1"}))),
              (ccd::BandList{5, 1}));
    ccd::DataSpec no_blue;
    no_blue.roles = {"coastal", "green", "red", "nir", "swir1", "swir2", "thermal"};
    expect_refused([&] { ccd::config::resolve(r, golden::ccdc(), no_blue); },
                   "preprocess standard step 2 'band_range' parameter 'bands' holds 'blue', "
                   "which the data spec does not have; its rows are coastal, green, red, nir, "
                   "swir1, swir2, thermal");

    ccd::DataSpec four;
    four.roles = {"blue", "green", "red", "nir"};
    expect_refused([&] { ccd::config::resolve(r, golden::ccdc(), four); },
                   "shared value 'detection_bands' holds 'swir1', which the data spec does not have; "
                   "its rows are blue, green, red, nir");
    const auto unknown = golden::with(golden::ccdc(), "window_screen", "bands",
                                      ccd::config::roles({"green", "swir3"}));
    expect_refused([&] { ccd::config::check(r, unknown); },
                   "window_screen 'tmask' parameter 'bands' holds 'swir3', which is not a role; "
                   "the roles are blue, green");
    const auto other = golden::with(golden::ccdc(), "window_screen", "bands",
                                    ccd::config::roles({"other"}));
    expect_refused([&] { ccd::config::check(r, other); }, "holds 'other', which no band list names");
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
                                               ccd::strategies::SpanFactor::Params{.min_obs = 30, .factor = 1.5});
    EXPECT_EQ(std::get<std::int64_t>(values.at("min_obs")), 30);
    EXPECT_EQ(std::get<double>(values.at("factor")), 1.5);

    // A min_obs (and the change test's model_rmse_up_to) as large as a
    // configuration holds: refit at every step.
    const ccd::config::Value most = ccd::config::integer(std::numeric_limits<std::int64_t>::max());
    tree.stages["refit"].params["min_obs"] = most;
    tree.stages["change_test"].params["model_rmse_up_to"] = most;
    EXPECT_TRUE(golden::strategies_of(tree).refit->due(1u << 20, 1.0, 12, 100.0));

    expect_refused([&] { ccd::config::build_strategies(golden::core_registry(), {}); },
                   "the configuration has no 'mode' stage");
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
    b = golden::with_shared(a, "lasso.alpha", ccd::config::real(2e-4));
    EXPECT_NE(ccd::config::digest(a), ccd::config::digest(b));
    EXPECT_NE(golden::resolved(a).digest, golden::resolved(b).digest);
    // The resolved digest covers the data spec.
    ccd::DataSpec other;
    other.qa_snow = 7;
    EXPECT_NE(golden::resolved(a).digest, golden::resolved(a, other).digest);
}

TEST(Registry, BandListsAreCheckedAgainstTheSpec)
{
    const auto& r = golden::core_registry();
    auto tree = golden::with(golden::ccdc(), "window_screen", "bands", ccd::config::bands({1, 4}));
    EXPECT_NO_THROW(ccd::config::resolve(r, tree, ccd::DataSpec{}));

    ccd::DataSpec four;
    four.roles = {"blue", "green", "red", "nir"};
    tree.shared["detection_bands"] = ccd::config::bands({1, 2, 3});
    expect_refused([&] { ccd::config::resolve(r, tree, four); },
                   "window_screen 'tmask' parameter 'bands' holds band 4; the data spec has 4 bands");
    tree.stages["window_screen"].params["bands"] = ccd::config::bands({2, 2});
    expect_refused([&] { ccd::config::resolve(r, tree, ccd::DataSpec{}); }, "holds band 2 twice");
    tree = golden::with_shared(golden::ccdc(), "detection_bands", ccd::config::bands({1, 9}));
    expect_refused([&] { ccd::config::resolve(r, tree, ccd::DataSpec{}); },
                   "shared value 'detection_bands' holds band 9; the data spec has 7 bands");
    tree = golden::with_shared(golden::ccdc(), "detection_bands", ccd::config::bands({}));
    expect_refused([&] { ccd::config::check(r, tree); }, "shared value 'detection_bands' is empty");
}
