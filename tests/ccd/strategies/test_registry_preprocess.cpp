//==============================================================================
//
// The configuration's preprocessing section (config/resolve.hpp): it
// resolves in the input's units and codes, an Algorithm resolves an input
// once and keeps the inputs used last, a bad section is refused when its
// Algorithm is built, and a registry adds a step type of its own.
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

using ccd::config::Values;
using registry_test::expect_refused;

} // namespace

//------------------------------------------------------------------------------
// The preprocessing section: conversions, refusals, and the registration of
// a step type beyond the library's.
//------------------------------------------------------------------------------

// Degrees Celsius resolve to the thermal row's unit (hundredths of a degree
// times reflectance_scale / 10000), reflectance to reflectance_scale, QA code
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
    config = golden::with_step(config, "standard", 1, "celsius", ccd::config::reals({-50.0, 70.7}));
    config = golden::with_step(config, "standard", 2, "reflectance", ccd::config::reals({0.0, 0.5}));
    const ccd::config::Resolved got = ccd::config::resolve(r, config, renumbered);
    const auto& standard = got.preprocess.at("standard").steps;
    EXPECT_EQ(standard.at(0).params.at("codes"),
              ccd::config::Value(std::vector<std::int64_t>{20, 21, 24}));
    EXPECT_EQ(std::get<std::vector<double>>(standard.at(1).params.at("celsius")).at(0), -5000.0);
    EXPECT_EQ(std::get<std::vector<double>>(standard.at(2).params.at("reflectance")).at(1), 5000.0);
    EXPECT_TRUE(got.plan.standard.chain[0].codes.test(24));
    EXPECT_FALSE(got.plan.standard.chain[0].codes.test(0));
    EXPECT_EQ(got.plan.snow.chain[4].codes, ccd::preprocess::qa_set({23}));

    ccd::Input reflectance = golden::physical_input();
    const ccd::config::Resolved phys = ccd::config::resolve(r, config, reflectance);
    EXPECT_EQ(phys.plan.standard.chain[1].range.lo.value, -0.5);
    EXPECT_EQ(phys.plan.standard.chain[2].range.hi.value, 0.5);

    // Codes by number, as given.
    const ccd::config::Config numbered = golden::with_step(
        golden::ccdc(), "standard", 0, "codes",
        ccd::config::Value(std::vector<std::int64_t>{0, 7}));
    EXPECT_EQ(ccd::config::resolve(r, numbered, renumbered).plan.standard.chain[0].codes,
              ccd::preprocess::qa_set({0, 7}));

    expect_refused(
        [&]
        {
            ccd::config::check(r, golden::with_step(golden::ccdc(), "standard", 0, "codes",
                                      names({"clear", "haze"})));
        },
        "preprocess standard step 0 'keep_qa' parameter 'codes' holds 'haze', which is not a QA "
        "code name; the names are fill, clear, water, shadow, snow, cloud, cirrus1, cirrus2, "
        "occlusion");
    expect_refused(
        [&]
        {
            ccd::config::check(r, golden::with_step(golden::ccdc(), "standard", 0, "codes",
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

    // The message lists the registered steps: the library's, and an
    // extension's if it registers more.
    ccd::config::Config unknown = golden::ccdc();
    unknown.preprocess.at("snow").steps.at(4).step = "keep_haze";
    expect_refused([&] { build(unknown); },
                   "preprocess snow step 4 is 'keep_haze', which is not a preprocessing step; "
                   "the steps are ");
    try
    {
        build(unknown);
    }
    catch (const std::invalid_argument& e)
    {
        const std::string message = e.what();
        const std::string listed = message.substr(message.find("the steps are "));
        for (const char* step : {"add_qa", "band_range", "one_per_date", "green_median", "keep_qa",
                                 "thermal_range"})
            EXPECT_NE(listed.find(step), std::string::npos) << step << ": " << message;
    }

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
        steps.push_back({"one_per_date", {}});
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
    dated.preprocess.at("insufficient_clear").steps.at(4).params["statistics_end"] =
        ccd::config::integer(736694);
    expect_refused([&] { build(dated); },
                   "preprocess insufficient_clear step 4 'green_median' parameter 'statistics_end' "
                   "is the shared value 'statistics_end'; set shared 'statistics_end' instead");

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

    // Every step's context, as make() was handed it; a negative day is
    // refused.
    static std::vector<ccd::config::StepContext>& contexts()
    {
        static std::vector<ccd::config::StepContext> seen;
        return seen;
    }

    static std::optional<ccd::preprocess::Step> make(const Params& params, const ccd::Input& input,
                                                     const ccd::config::StepContext& context)
    {
        contexts().push_back(context);
        if (params.day < 0)
            throw std::invalid_argument("parameter 'day' is negative");
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
        return ccd::detect(
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

    DropDayStep::contexts().clear();
    const ccd::config::Resolved resolved = ccd::config::resolve(r, config, ccd::Input{});
    const ccd::preprocess::Step& step = resolved.plan.standard.chain[1];
    ASSERT_EQ(step.kind, ccd::preprocess::StepKind::Custom);

    // make() is handed where the step is: its procedure, how that procedure
    // reads the thermal row, its position.
    ASSERT_EQ(DropDayStep::contexts().size(), 1u);
    EXPECT_EQ(DropDayStep::contexts()[0].procedure, "standard");
    EXPECT_EQ(DropDayStep::contexts()[0].thermal, ccd::config::Thermal::Celsius);
    EXPECT_EQ(DropDayStep::contexts()[0].position, 1u);
    {
        ccd::config::Config snow = golden::pyccd();
        auto& chain = snow.preprocess.at("snow").steps;
        chain.insert(chain.begin() + 3, {"drop_day", {{"day", ccd::config::integer(dates[kept])}}});
        DropDayStep::contexts().clear();
        ccd::config::resolve(r, snow, ccd::Input{});
        ASSERT_EQ(DropDayStep::contexts().size(), 1u);
        EXPECT_EQ(DropDayStep::contexts()[0].procedure, "snow");
        EXPECT_EQ(DropDayStep::contexts()[0].thermal, ccd::config::Thermal::Stored);
        EXPECT_EQ(DropDayStep::contexts()[0].position, 3u);

        // A refusal of make() names the step.
        chain[3].params["day"] = ccd::config::integer(-1);
        expect_refused([&] { ccd::config::resolve(r, snow, ccd::Input{}); },
                       "preprocess snow step 3 'drop_day': parameter 'day' is negative");
    }
    EXPECT_EQ(step.custom->describe(), "DropDay " + std::to_string(dates[kept]));
    EXPECT_NE(resolved.digest, ccd::config::resolve(r, golden::pyccd(), ccd::Input{}).digest);

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
