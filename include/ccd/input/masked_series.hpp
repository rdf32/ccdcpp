#pragma once

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/input/processing_mask.hpp"

#include "ccd/input/pixel_series.hpp"
#include "ccd/core/window.hpp"

namespace ccd
{

//==============================================================================
//
// MaskedData
//
// The masked copy of a series' dates and (B, n) values: the observations a
// mask keeps, in order (apply_mask()). A MaskedSeries holds its observations
// in one; the whole-series fit takes one per pixel.
//
//     MaskedData kept = apply_mask(pixel, mask);   // kept.observations ==
//                                                  // mask.count()
//
//==============================================================================

struct MaskedData
{
    std::vector<std::int64_t> dates;
    std::vector<scalar_t> spectral;

    index_t bands;
    index_t observations;

    ArrayView<std::int64_t, 1> dates_view()
    {
        return ArrayView<std::int64_t,1>::contiguous(
            dates.data(),
            {observations}
        );
    }

    ArrayView<scalar_t, 2> spectral_view()
    {
        return ArrayView<scalar_t,2>::contiguous(
            spectral.data(),
            {bands, observations}
        );
    }

    //----------------------------------------------------------------------
    // Remove the observation at masked position `index`, in place.
    //
    // The result is what apply_mask() gives for the mask with that
    // observation cleared: the same values, in the same order.
    //----------------------------------------------------------------------
    void erase(index_t index)
    {
        assert(index < observations);

        dates.erase(dates.begin() + index);

        // Compacting forward never overwrites an unread value: the write
        // position trails the read position.
        index_t out = 0;
        for (index_t b = 0; b < bands; ++b)
        {
            for (index_t i = 0; i < observations; ++i)
            {
                if (i != index)
                    spectral[out++] = spectral[b * observations + i];
            }
        }

        --observations;
        spectral.resize(bands * observations);
    }
};

MaskedData apply_mask(
    const PixelSeries& series,
    const ProcessingMask& mask
);

// apply_mask() into `out`, reusing its storage.
void apply_mask(
    const PixelSeries& series,
    const ProcessingMask& mask,
    MaskedData& out
);

//==============================================================================
//
// MaskedSeries
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
// With reason codes (reset()'s `reasons`), each drop also writes why at the
// dropped observation's index in the pixel's series: they stay Reason::Kept
// exactly where mask() keeps.
//
// The buffers keep their capacity across reset(), so a MaskedSeries reused
// across pixels stops allocating.
//
//==============================================================================

class MaskedSeries
{
public:

    // The observations of `pixel` that `mask` keeps, and with `reasons` (T
    // codes, Reason::Kept where `mask` keeps) the codes the drops write; none
    // when it is empty.
    void reset(const PixelSeries& pixel, ProcessingMask mask,
               std::vector<std::uint8_t> reasons = {});

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

    // The reason codes (reset()), empty without them.
    const std::vector<std::uint8_t>& reasons() const
    {
        return reasons_;
    }

    // The number of leading observations dated on or before `day`. Dates are
    // non-decreasing, so these are exactly the observations dated <= day.
    index_t count_until(std::int64_t day) const;

    // Drops the observation at `position`, for `why`, and moves `window`
    // with the observations after it.
    void drop(index_t position, Window& window, Reason why);

    // Drops the observations `outliers` flags in `window`, for `why`, if what
    // is left of the window passes `accept`.
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
        Reason why,
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

        commit_without(outliers, window.start, why);
        window = shortened;
        return true;
    }

    // Moves the mask out. The series is empty until the next reset().
    ProcessingMask take_mask();

    // Moves the reason codes out (empty without them).
    std::vector<std::uint8_t> take_reasons();

private:

    // kept_ without the positions offset + j that outliers[j] flags, into
    // candidate_.
    void build_without(const ProcessingMask& outliers, index_t offset);

    // Clears those positions in mask_, writing `why` at them in reasons_,
    // and makes candidate_ the kept data.
    void commit_without(const ProcessingMask& outliers, index_t offset, Reason why);

    ProcessingMask mask_;
    std::vector<std::uint8_t> reasons_;   // T codes, or none
    MaskedData kept_;        // apply_mask(workspace, mask_)
    MaskedData candidate_;   // drop_outliers_if()'s tentative removal
};

} // namespace ccd
