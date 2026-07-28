#pragma once

#include <vector>      // std::vector
#include <cmath>
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
        result = (*lower + result) / static_cast<T>(2);
    }
    return result;
}

template<typename T>
static T absolute(
    T value
)
{
    return std::abs(value);
}

template<typename T>
static std::vector<T> absolute(
    const std::vector<T>& values
)
{
    std::vector<T> result(values.size());

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        result[i] = std::abs(values[i]);
    }

    return result;
}

template<typename T>
static T median_absolute(
    const std::vector<T>& values
)
{
    auto temp = absolute(values);
    return median(temp);
}

}