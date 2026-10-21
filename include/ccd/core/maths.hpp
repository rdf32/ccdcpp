#pragma once

#include <vector>      // std::vector
#include <cmath>
#include <algorithm>   // std::nth_element, std::max_element
#include <type_traits> // std::is_floating_point

#include "ccd/core/types.hpp"


namespace ccd
{

template<typename T>
static T median(
    std::vector<T>& values
) {
    // The mean of the two middle values of an even count is not a T for an
    // integral T.
    static_assert(std::is_floating_point<T>::value, "median() of an integral type truncates");

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

// The chi-squared quantile at probability p with df degrees of freedom: the
// distribution pyccd's change thresholds are read from
// (adjust_change_threshold(), strategies/pyccd/peek_statistics.hpp, and a
// configuration's Chi2 fields).
scalar_t chi2_quantile(scalar_t p, index_t df);

}
