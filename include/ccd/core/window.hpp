#pragma once

#include <cassert>
#include <cstdint>

#include "ccd/core/array_view.hpp"
#include "ccd/core/types.hpp"

namespace ccd
{

// A window of consecutive positions of a masked series, [start, stop).
struct Window
{
    index_t start = 0;
    index_t stop  = 0;

    Window() = default;

    Window(index_t start_, index_t stop_)
        : start(start_), stop(stop_){}

    index_t size() const
    {
        return stop - start;
    }

    void grow()
    {
        ++stop;
    }

    void shift()
    {
        ++start;
        ++stop;
    }
};

// The days `window` spans in `dates`: its last date minus its first.
inline scalar_t span(
    ArrayView<const std::int64_t, 1> dates,
    const Window& window
)
{
    assert(window.stop > window.start);

    return static_cast<scalar_t>(
        dates(window.stop - 1) -
        dates(window.start)
    );
}

// The observations of `dates` dated up to `day`: dates is sorted.
index_t count_through(ArrayView<const std::int64_t, 1> dates, std::int64_t day);

} // namespace ccd
