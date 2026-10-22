// The ArrayView printer (ccd/core/array_print.hpp), a development toolkit no
// library code includes: included on its own, each of its printers on a small
// (2, 3) view, so that it keeps compiling.

#include "ccd/core/array_print.hpp"

#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

const std::vector<int> values = {1, 2, 3, 4, 5, 6};
const auto view = ccd::ArrayView<const int, 2>::contiguous(values.data(), {2, 3});

} // namespace

TEST(ArrayPrint, StreamsAViewByRows)
{
    std::ostringstream os;
    os << view;
    EXPECT_EQ(os.str(), "[\n  [1, 2, 3],\n  [4, 5, 6]\n]\n");
}

TEST(ArrayPrint, PrintsItsValuesAndLayout)
{
    std::ostringstream flat, memory, info;
    ccd::print_flat(view, 4, flat);
    EXPECT_EQ(flat.str(), "[1, 2, 3, 4, ...]\n");
    ccd::print_memory(view, 100, memory);
    EXPECT_EQ(memory.str(), "[1, 2, 3, 4, 5, 6]\n");
    ccd::print_info(view, info);
    EXPECT_NE(info.str().find("Shape   : (2, 3)"), std::string::npos) << info.str();
    EXPECT_NE(info.str().find("Contiguous : true"), std::string::npos) << info.str();
}
