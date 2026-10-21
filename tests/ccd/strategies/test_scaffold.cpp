//==============================================================================
//
// tools/new_strategy.py's scaffold, of every stage: the build generates one
// strategy per stage from include/ccd/strategies/stages.hpp ("scaffold-<stage>",
// CMakeLists.txt) and compiles it with this test, so the scaffold follows the
// interfaces. Each registers, builds from its empty parameters, and throws
// from a method until it is written.
//
//==============================================================================

#include <algorithm>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

// Every stage, as CMakeLists.txt scaffolds them.
#define CCD_SCAFFOLDED(X)                                                                         \
    X(mode) X(noise) X(start_search) X(window_screen) X(stability_fit) X(stability_rule)          \
    X(lookback) X(refit) X(model) X(change_test) X(outlier) X(segment_report) X(edge_segments)

namespace ccd
{
namespace strategies
{

#define CCD_DECLARE(stage) void register_##stage##_scaffold_##stage(config::Registry& registry);
CCD_SCAFFOLDED(CCD_DECLARE)
#undef CCD_DECLARE

} // namespace strategies
} // namespace ccd

TEST(Scaffold, EveryStagesScaffoldRegistersAndBuilds)
{
    ccd::config::Registry r;
#define CCD_REGISTER(stage) ccd::strategies::register_##stage##_scaffold_##stage(r);
    CCD_SCAFFOLDED(CCD_REGISTER)
#undef CCD_REGISTER

    const auto names = r.names();
    EXPECT_EQ(names.size(), ccd::STAGES.size()) << "a stage CMakeLists.txt does not scaffold";
    for (const ccd::StageInfo& stage : ccd::STAGES)
    {
        std::string name = std::string("scaffold-") + stage.name;
        std::replace(name.begin(), name.end(), '_', '-');
        ASSERT_EQ(names.count(stage.name), 1u) << stage.name;
        EXPECT_EQ(names.at(stage.name), std::vector<std::string>{name});
        EXPECT_TRUE(r.strategy(stage.name, name).make({})) << name;
        EXPECT_TRUE(r.defaults(stage.name, name).empty()) << name;
    }

    const auto outlier = r.make<ccd::OutlierRule>({"scaffold-outlier", {}});
    try
    {
        outlier->needs();
        ADD_FAILURE() << "a scaffold's method returned";
    }
    catch (const std::logic_error& e)
    {
        EXPECT_EQ(std::string(e.what()), "outlier 'scaffold-outlier': needs() is not written yet");
    }
}
