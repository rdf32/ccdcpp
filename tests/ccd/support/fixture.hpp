#pragma once

//==============================================================================
//
// A dump's comparison with its committed fixture, the check of test_golden.cpp,
// test_bands.cpp and test_transitions.cpp. A fixture's first line names the
// platform that wrote it (platform(), dump.hpp): exact equality is only claimed
// there, so on another platform a mismatch is reported as a skip that says so,
// not as a failure.
//
//==============================================================================

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "dump.hpp"

namespace golden
{

// `dump` cut into its pixels, each from its "P " line up to the next.
inline std::vector<std::string> split_pixels(const std::string& dump)
{
    std::vector<std::string> out;
    std::istringstream in(dump);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("P ", 0) == 0 || out.empty())
            out.emplace_back();
        out.back() += line;
        out.back() += '\n';
    }
    return out;
}

// `dump` equals the fixture at `path` byte for byte, after its platform line.
// With the environment variable `write_env` set, `dump` is written as the
// fixture instead, and the test skips. A failure shows the first three pixels
// that differ.
inline void expect_matches_fixture(const std::string& dump, const char* path, const char* write_env)
{
    const std::string header = "# platform: " + platform() + "\n";

    if (std::getenv(write_env))
    {
        std::ofstream out(path, std::ios::binary);
        out << header << dump;
        GTEST_SKIP() << "fixture written to " << path;
    }

    std::ifstream in(path, std::ios::binary);
    ASSERT_TRUE(in) << "missing fixture " << path;

    std::string fixture_header;
    std::getline(in, fixture_header);
    fixture_header += '\n';
    const std::string expected(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    if (dump == expected)
        return;

    if (fixture_header != header)
    {
        GTEST_SKIP()
            << "output differs from a fixture written on another platform\n"
            << "  fixture: " << fixture_header
            << "  this:    " << header
            << "exact equality is only claimed on the fixture's platform";
    }

    const auto actual_px = split_pixels(dump);
    const auto expected_px = split_pixels(expected);

    ASSERT_EQ(actual_px.size(), expected_px.size()) << "pixel count differs";

    std::size_t differing = 0;
    for (std::size_t i = 0; i < actual_px.size(); ++i)
    {
        if (actual_px[i] == expected_px[i])
            continue;

        if (++differing <= 3)
        {
            ADD_FAILURE()
                << "expected:\n" << expected_px[i]
                << "actual:\n" << actual_px[i];
        }
    }
    FAIL() << differing << " of " << actual_px.size() << " pixels differ from the fixture";
}

} // namespace golden
