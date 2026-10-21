#pragma once

#include <vector>
#include <cassert>
#include <algorithm>
#include <string>

#include "ccd/core/types.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/pmask.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/array_view.hpp"

namespace ccd
{

//==============================================================================
//
// PixelSeries
//
// One pixel's data, as a run reads it: its dates, its (B, T) values (the
// run's per-pixel copy, or the caller's array when it needs no conversion)
// and its QA codes, all views, and the run's Input and resolved shared values
// by reference. It holds no copy of either and computes nothing: the masks,
// the harmonic bases, the QA statistics and the fits are the procedures'.
//
//==============================================================================

class PixelSeries
{
public:

    //----------------------------------------------------------------------
    // Constructor
    //----------------------------------------------------------------------

    // The series refers to `input` and `shared` (the run's resolved shared
    // values), as it refers to the arrays, and never writes them: what a
    // procedure derives per pixel (the Standard procedure's adjusted peek and
    // change threshold) it keeps in its own records. Both must outlive the
    // series.
    PixelSeries(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        ArrayView<const std::uint8_t, 1> qas,
        const Input& input,
        const config::Shared& shared
    )
        :
        dates_(dates),
        spectral_(spectral),
        qas_(qas),
        input_(input),
        shared_(shared)
    {
        validate();
    }

    // A temporary would not outlive the series.
    PixelSeries(ArrayView<const std::int64_t, 1>, ArrayView<const scalar_t, 2>,
                      ArrayView<const std::uint8_t, 1>, Input&&, const config::Shared&) = delete;
    PixelSeries(ArrayView<const std::int64_t, 1>, ArrayView<const scalar_t, 2>,
                      ArrayView<const std::uint8_t, 1>, const Input&, config::Shared&&) = delete;

    //----------------------------------------------------------------------
    // Validation
    //----------------------------------------------------------------------

    void validate() const
    {
        assert(dates_.size() == spectral_.extent(1));
        assert(dates_.size() == qas_.size());

        assert(spectral_.is_contiguous());

        // Date order is checked once per call at the entry points
        // (ccd::validate_dates), not per pixel here.
    }

    //----------------------------------------------------------------------
    // Input data
    //----------------------------------------------------------------------

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

    const Input& input() const
    {
        return input_;
    }

    const config::Shared& shared() const
    {
        return shared_;
    }

private:

    ArrayView<const std::int64_t, 1> dates_;
    ArrayView<const scalar_t, 2> spectral_;
    ArrayView<const std::uint8_t, 1> qas_;

    const Input& input_;
    const config::Shared& shared_;

};

} // namespace ccd