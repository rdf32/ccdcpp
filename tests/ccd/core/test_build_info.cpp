// ccd::build_info() (ccd/core/build_info.hpp): each field is this build's, as
// the test, compiled with the library's flags, sees them.

#include "ccd/core/build_info.hpp"

#include <gtest/gtest.h>

#include <regex>

TEST(BuildInfo, SaysWhatThisBuildIs)
{
    const ccd::BuildInfo info = ccd::build_info();
    EXPECT_EQ(info.version, CCD_PROJECT_VERSION);
    // The hash, not a tag name: what a downstream build pins.
    EXPECT_TRUE(std::regex_match(info.commit, std::regex("[0-9a-f]{40}(-dirty)?|unknown")))
        << info.commit;
    EXPECT_FALSE(info.describe.empty());
#ifdef _OPENMP
    EXPECT_TRUE(info.openmp);
    EXPECT_EQ(info.openmp_version, _OPENMP);
#else
    EXPECT_FALSE(info.openmp);
    EXPECT_EQ(info.openmp_version, 0);
#endif
    EXPECT_FALSE(info.compiler.empty());
    EXPECT_GE(info.cxx_standard, 202002L);
    EXPECT_EQ(info.observe, static_cast<bool>(CCD_OBSERVE));
}
