#pragma once

#include <vector>      // std::vector
#include <algorithm>   // std::nth_element, std::max_element


namespace ccd
{

template<typename T>
static T median(
    std::vector<T>& values
) {
    if (values.empty()) {
        return static_cast<T>(0);
    }

    const auto middle = 
        values.begin() + values.size() / 2;

    std::nth_element(
        values.begin(),
        middle,
        values.end()
    );

    T result = *middle;
    // even number of elements
    if (values.size() % 2 == 0) {
        const auto lower = 
            std::max_element(
                values.begin(),
                middle
            );
        result = (*lower + result) * 0.5;
    }
    return result;
}

}