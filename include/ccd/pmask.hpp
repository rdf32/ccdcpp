#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "ccd/types.hpp"


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

private:

    std::vector<std::uint8_t> mask_;

};

} // namespace ccd
