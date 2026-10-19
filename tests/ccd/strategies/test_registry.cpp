//==============================================================================
//
// The registry and the strategy parameters (config/registry.hpp,
// config/params.hpp), on the library's strategies:
//
//   * the registry lists each stage's strategies, and builds one by name with
//     its parameters checked against its schema (unknown names, wrong types,
//     values out of range all refused, naming the stage, strategy and field);
//   * the configuration HarmonicOptions describe builds the strategies a run
//     uses, and the refit schedule decides as Extend's inline test did.
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
#include "ccd/strategies/refit/span_factor.hpp"
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
    ASSERT_EQ(names.count("refit"), 1u);
    EXPECT_EQ(names.at("refit"), (std::vector<std::string>{"span-factor"}));
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
    const auto tree = ccd::strategies::config_from_options(h);
    const StageConfig& refit = tree.at("refit");
    EXPECT_EQ(refit.strategy, "span-factor");
    const auto s = ccd::strategies::strategies_for(h);
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
    EXPECT_TRUE(ccd::strategies::strategies_for(h).refit->due(1u << 20, 1.0, 100.0));

    expect_refused(
        [&] { ccd::strategies::build_strategies(ccd::strategies::core_registry(), {}); },
        "the configuration has no 'refit' stage");
}
