#include "ccd/core/window.hpp"

namespace ccd
{

index_t count_through(ArrayView<const std::int64_t, 1> dates, std::int64_t day)
{
    index_t lo = 0, hi = dates.size();
    while (lo < hi)
    {
        const index_t mid = lo + (hi - lo) / 2;
        if (dates(mid) <= day)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

} // namespace ccd
