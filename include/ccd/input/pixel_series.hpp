#pragma once

#include <cassert>
#include <cstdint>

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"

namespace ccd
{

//==============================================================================
//
// PixelSeries
//
// One pixel's data, as a run reads it: its dates, its (B, T) values (the
// run's per-pixel copy, or the caller's array when it needs no conversion)
// and its QA codes, all views of storage that outlives the series. It
// computes nothing: the masks, the harmonic bases, the QA statistics and the
// fits are the procedures'.
//
//     const PixelSeries pixel(dates, values, qas);
//     MaskedData kept = apply_mask(pixel, mask);   // input/masked_series.hpp
//
//==============================================================================

class PixelSeries
{
public:

    // The three views; `spectral` is contiguous, and all three have one
    // timestep count. Date order is checked once per call at the entry
    // points (validate_dates()), not per pixel here.
    PixelSeries(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        ArrayView<const std::uint8_t, 1> qas
    )
        :
        dates_(dates),
        spectral_(spectral),
        qas_(qas)
    {
        assert(dates_.size() == spectral_.extent(1));
        assert(dates_.size() == qas_.size());
        assert(spectral_.is_contiguous());
    }

    ArrayView<const std::int64_t, 1> dates() const
    {
        return dates_;
    }

    ArrayView<const scalar_t, 2> spectral() const
    {
        return spectral_;
    }

    ArrayView<const std::uint8_t, 1> qas() const
    {
        return qas_;
    }

private:

    ArrayView<const std::int64_t, 1> dates_;
    ArrayView<const scalar_t, 2> spectral_;
    ArrayView<const std::uint8_t, 1> qas_;
};

} // namespace ccd
