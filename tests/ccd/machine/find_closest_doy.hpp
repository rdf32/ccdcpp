#pragma once

// pyccd's find_closest_doy() with a stable sort: the reference the seasonal
// RMSE's selection (SeasonalOrder) is tested against. Not part of the
// library.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "ccd/core/array_view.hpp"
#include "ccd/core/types.hpp"
#include "ccd/core/window.hpp"

namespace reference
{

// The `count` positions of `window` nearest dates(date_index) in day of
// year, by |round(delta / 365.25) * 365.25 - delta| for delta the days
// between them, nearest first, ties in window order.
inline std::vector<ccd::index_t> find_closest_doy(
    ccd::ArrayView<const std::int64_t, 1> dates,
    ccd::index_t date_index,
    const ccd::Window& window,
    ccd::index_t count
)
{
    //----------------------------------------------------------
    // The seasonal distance of a day from the target is
    // |round(delta / 365.25) * 365.25 - delta|, delta the days
    // between them: a multiple of a quarter day. In quarter
    // days it is the integer |1461 k - 4 |delta||, k the
    // integer nearest 4 |delta| / 1461. That quotient is never
    // a half (8 |delta| is even, 1461 (2k + 1) odd) and is at
    // least 1 / 2922 from one, far beyond the double
    // quotient's rounding, so k is the double formula's and
    // the integer distance orders as the double one does. It
    // is at most 730.
    //----------------------------------------------------------
    constexpr std::int64_t QUARTERS_PER_YEAR = 1461;
    constexpr std::size_t MAX_DISTANCE = QUARTERS_PER_YEAR / 2;

    const std::int64_t target = dates(date_index);
    const ccd::index_t n = window.size();
    count = std::min(count, n);

    std::vector<std::uint16_t> distance(n);
    std::array<ccd::index_t, MAX_DISTANCE + 2> at{};   // a count, then an offset
    for (ccd::index_t i = 0; i < n; ++i)
    {
        const std::int64_t quarters = 4 * std::abs(dates(window.start + i) - target);
        const std::int64_t k = (2 * quarters + QUARTERS_PER_YEAR) / (2 * QUARTERS_PER_YEAR);
        const auto d = static_cast<std::uint16_t>(std::abs(QUARTERS_PER_YEAR * k - quarters));
        distance[i] = d;
        ++at[d + 1];
    }

    //----------------------------------------------------------
    // The `count` nearest, nearest first, ties in window order:
    // a counting sort by distance, which keeps window order
    // within a distance, up to the distance the count reaches.
    //----------------------------------------------------------
    std::size_t last = 0;
    for (ccd::index_t seen = 0; last <= MAX_DISTANCE && seen < count; ++last)
        seen += at[last + 1];
    for (std::size_t d = 1; d <= last; ++d)
        at[d] += at[d - 1];

    std::vector<ccd::index_t> indices(at[last]);
    for (ccd::index_t i = 0; i < n; ++i)
        if (distance[i] < last)
            indices[at[distance[i]]++] = i;
    indices.resize(count);

    return indices;
}

} // namespace reference
