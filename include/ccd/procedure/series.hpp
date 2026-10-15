#pragma once

#include <cassert>
#include <cstdint>
#include <utility>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/pmask.hpp"

#include "ccd/harmonic/harmonic.hpp"
#include "ccd/procedure/fit.hpp"

namespace ccd
{

//==============================================================================
//
// Series
//
// One pixel's processing mask and the observations it keeps, owned together.
// Shared by the change-detection procedures.
//
// After reset() and after every drop:
//
//     size() == mask().count()
//     dates(), spectral() == apply_mask(workspace, mask())
//
// that is, the same values in the same order. A position is a masked
// coordinate: an index into dates() and spectral(). Dropping the observation
// at position i moves every later observation down one position. The two
// drops below are the only writes, and each moves the caller's window with
// the observations, so that rule is applied here and nowhere else:
//
//     a window bound above the dropped position moves down by one.
//
// The buffers keep their capacity across reset(), so a Series reused across
// pixels stops allocating.
//
//==============================================================================

class Series
{
public:

    // The observations of `workspace` that `mask` keeps.
    void reset(const HarmonicWorkspace& workspace, ProcessingMask mask);

    index_t size() const
    {
        return kept_.observations;
    }

    ArrayView<const std::int64_t, 1> dates() const
    {
        return ArrayView<const std::int64_t, 1>::contiguous(
            kept_.dates.data(),
            {kept_.observations}
        );
    }

    ArrayView<const scalar_t, 2> spectral() const
    {
        return ArrayView<const scalar_t, 2>::contiguous(
            kept_.spectral.data(),
            {kept_.bands, kept_.observations}
        );
    }

    const ProcessingMask& mask() const
    {
        return mask_;
    }

    // The number of leading observations dated on or before `day`. Dates are
    // non-decreasing, so these are exactly the observations dated <= day.
    index_t count_until(std::int64_t day) const;

    // Drops the observation at `position` and moves `window` with the
    // observations after it.
    void drop(index_t position, Window& window);

    // Drops the observations `outliers` flags in `window`, if what is left of
    // the window passes `accept`.
    //
    // outliers[j] flags position window.start + j, so every flagged position
    // is inside the window (outliers.size() == window.size()). The series
    // without them is built aside first, and accept(dates) is asked with the
    // dates of the shortened window. On true the mask, the observations and
    // the window are updated (window.start stays, window.stop moves down by
    // the number dropped) and drop_outliers_if returns true; on false nothing
    // changes.
    template<typename Accept>
    bool drop_outliers_if(
        const ProcessingMask& outliers,
        Window& window,
        Accept&& accept
    )
    {
        assert(outliers.size() == window.size());

        build_without(outliers, window.start);

        const Window shortened(window.start, window.stop - outliers.count());
        const auto shortened_dates =
            ArrayView<const std::int64_t, 1>::contiguous(
                candidate_.dates.data(),
                {candidate_.observations}
            ).slice(range(shortened.start, shortened.stop));

        if (!accept(shortened_dates))
            return false;

        commit_without(outliers, window.start);
        window = shortened;
        return true;
    }

    // Moves the mask out. The series is empty until the next reset().
    ProcessingMask take_mask();

private:

    // kept_ without the positions offset + j that outliers[j] flags, into
    // candidate_.
    void build_without(const ProcessingMask& outliers, index_t offset);

    // Clears those positions in mask_ and makes candidate_ the kept data.
    void commit_without(const ProcessingMask& outliers, index_t offset);

    ProcessingMask mask_;
    MaskedData kept_;        // apply_mask(workspace, mask_)
    MaskedData candidate_;   // drop_outliers_if()'s tentative removal
};

} // namespace ccd
