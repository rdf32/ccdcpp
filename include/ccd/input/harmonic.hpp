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
// HarmonicWorkspace
//
// Immutable workspace describing one pixel.
//
// Represents a single pixel time series with multiple bands / features/

// Stores immutable observations together with reusable scratch
// buffers used during harmonic fitting. The workspace itself
// contains no fitting logic.
//
// Responsibilities:
//
//  • Validate input arrays
//  • Store dates
//  • Store spectral observations
//  • Store QA observations
//
// Everything derived from these inputs (processing masks, harmonic basis,
// quality statistics, etc.) is computed elsewhere.
//
//==============================================================================

class HarmonicWorkspace
{
public:

    //----------------------------------------------------------------------
    // Constructor
    //----------------------------------------------------------------------

    // The workspace refers to `input` and `shared` (the run's resolved shared
    // values), as it refers to the arrays, and never writes them: what a
    // procedure derives per pixel (the Standard procedure's adjusted peek and
    // change threshold) it keeps in its own records. Both must outlive the
    // workspace.
    HarmonicWorkspace(
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

    // A temporary would not outlive the workspace.
    HarmonicWorkspace(ArrayView<const std::int64_t, 1>, ArrayView<const scalar_t, 2>,
                      ArrayView<const std::uint8_t, 1>, Input&&, const config::Shared&) = delete;
    HarmonicWorkspace(ArrayView<const std::int64_t, 1>, ArrayView<const scalar_t, 2>,
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
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask
);

// apply_mask() into `out`, reusing its storage.
void apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask,
    MaskedData& out
);

} // namespace ccd