#pragma once

// The refusal check test_registry's sources share. Not part of the library.

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace registry_test
{

template<class F>
inline void expect_refused(F make, const std::string& fragment)
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

} // namespace registry_test
