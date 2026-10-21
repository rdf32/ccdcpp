#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "ccd/core/types.hpp"


namespace ccd
{
//==============================================================================
//
// ProcessingMask
//
// Boolean mask describing which observations are active during processing.
//
// Used throughout CCD for:
//
//      - clear observation selection
//      - snow masking
//      - recovery procedures
//      - filtered harmonic fitting
//
// The mask owns storage.
//
// Example:
//
//      ProcessingMask clear_mask(num_dates);
//
//      clear_mask.set(i, true);
//
//      if(clear_mask[i])
//      {
//          use_observation(i);
//      }
//
//
//
// Storage uses uint8_t rather than vector<bool>.
//
// vector<bool> is a specialized packed-bit representation that returns proxy
// references instead of bool&, which complicates interoperability and views.
//
// For CCD workloads, predictable memory access is preferred.
//
//==============================================================================
class ProcessingMask
{
public:


    //--------------------------------------------------------------------------
    // Construction
    //--------------------------------------------------------------------------
    ProcessingMask() = default;
    
    explicit ProcessingMask(
        index_t size
    )
        :
        mask_(
            size,
            static_cast<std::uint8_t>(0)
        )
    {}

    //--------------------------------------------------------------------------
    // Element access
    //--------------------------------------------------------------------------
    // Mutable access.
    //
    // Allows:
    //
    //      mask[i] = 1;
    //
    std::uint8_t& operator[](
        index_t i
    )
    {
        assert(i < mask_.size());
        return mask_[i];
    }

    //
    // Const access.
    //
    // Returns logical boolean value.
    //
    bool operator[](
        index_t i
    ) const
    {
        assert(i < mask_.size());
        return mask_[i] != 0;
    }

    //--------------------------------------------------------------------------
    // Explicit operations
    //--------------------------------------------------------------------------
    void set(
        index_t i,
        bool value
    )
    {
        assert(i < mask_.size());

        mask_[i] =
            value
                ? static_cast<std::uint8_t>(1)
                : static_cast<std::uint8_t>(0);
    }

    bool test(
        index_t i
    ) const
    {
        assert(i < mask_.size());

        return mask_[i] != 0;
    }
    //--------------------------------------------------------------------------
    // Information
    //--------------------------------------------------------------------------
    index_t size() const noexcept
    {
        return mask_.size();
    }

    index_t count() const
    {
        return std::count(
            mask_.begin(),
            mask_.end(),
            static_cast<std::uint8_t>(1)
        );
    }

    bool any() const noexcept
    {
        return std::any_of(
            mask_.begin(),
            mask_.end(),
            [](std::uint8_t value)
            {
                return value != 0;
            }
        );
    }

    bool none() const noexcept
    {
        return !any();
    }

    bool empty() const noexcept
    {
        return mask_.empty();
    }

    //--------------------------------------------------------------------------
    // Modification
    //--------------------------------------------------------------------------
    void clear()
    {
        std::fill(
            mask_.begin(),
            mask_.end(),
            static_cast<std::uint8_t>(0)
        );
    }

    void fill()
    {
        std::fill(
            mask_.begin(),
            mask_.end(),
            static_cast<std::uint8_t>(1)
        );
    }

    //--------------------------------------------------------------------------
    // Raw access
    //--------------------------------------------------------------------------
    const std::vector<std::uint8_t>& data() const noexcept
    {
        return mask_;
    }


    std::vector<std::uint8_t>& data() noexcept
    {
        return mask_;
    }

    std::vector<int> as_ints() const
    {
        std::vector<int> result;
        result.reserve(mask_.size());

        for (std::uint8_t value : mask_)
        {
            result.push_back(static_cast<int>(value));
        }

        return result;
    }
private:

    std::vector<std::uint8_t> mask_;

};

//==============================================================================
//
// Reason
//
// Why an observation is where a run left it, one byte per observation of a
// pixel (RunSpec::reasons). The first three are kept by the processing mask,
// the others are not: the mask is reasons < FIRST_DROP. A preprocessing
// step's code is the last step that removed the observation (an add_qa can
// take one back); the machine's, the stage that dropped it.
//
//==============================================================================

enum class Reason : std::uint8_t
{
    // Kept by the processing mask:
    Kept,             // in a segment
    NoFit,            // after the last segment of a pixel whose model could
                      // not be fitted on its window (machine::Exit::NoFit)
    NoSegment,        // in no segment otherwise: an edge segment not due, the
                      // gap a Lookback stopped by a change leaves, a pixel
                      // with too few observations or a NaN variogram

    // Removed by the procedure's preprocessing (its chain, then the mode's
    // extra steps):
    Qa,               // keep_qa: its QA code is not kept
    Thermal,          // thermal_range
    BandRange,        // band_range
    Green,            // green_median
    DuplicateDate,    // one_per_date: a later observation of its date
    CustomStep,       // a step a registry adds (preprocess::StepKind::Custom)

    // Dropped by the machine:
    WindowScreen,     // Initialize's window screen (Tmask)
    LookbackOutlier,  // Lookback's outlier, before the stable window
    Outlier,          // Extend's outlier rule, the first observation of a peek
    BeforeChange,     // the peek's observations before the change test's
                      // break (ChangeTest::break_at() past the peek's first)
    Stored            // a drop of the run this one resumed: the record keeps
                      // its day, not why
};

// The first code the processing mask does not keep.
inline constexpr Reason FIRST_DROP = Reason::Qa;

// The number of codes: every Reason is below it.
inline constexpr std::uint8_t REASON_COUNT = static_cast<std::uint8_t>(Reason::Stored) + 1;

// "kept", "no_fit", "no_segment", "qa", ..., "stored": the code's name, as
// the Python package's REASONS lists them.
const char* name(Reason reason);

} // namespace ccd
