#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd/types.hpp"
#include "ccd/constants.hpp"

namespace ccd
{

//==============================================================================
//
// BandList
//
// An ordered list of band indices -- rows of the spectral array -- with a
// fixed capacity of CCD_MAX_BANDS, stored inline. HarmonicOptions holds three
// of them and is copied once per pixel, so the list allocates nothing.
//
// The stored order is the order the procedures iterate in, and it is never
// changed here: the detection-band sums are accumulated in this order, so two
// lists with the same bands in a different order can give results that differ
// in the last bits.
//
// Every index is below CCD_MAX_BANDS; adding one that is not, or adding to a
// full list, throws std::invalid_argument. Whether an index is below the
// actual band count, and whether a list repeats a band, is checked per call
// by ccd::validate().
//
//==============================================================================

class BandList
{
public:

    using value_type = std::uint8_t;
    using const_iterator = const value_type*;

    BandList() = default;

    BandList(std::initializer_list<index_t> bands)
    {
        for (const index_t band : bands)
            push_back(band);
    }

    explicit BandList(const std::vector<index_t>& bands)
    {
        for (const index_t band : bands)
            push_back(band);
    }

    void push_back(index_t band)
    {
        if (band >= CCD_MAX_BANDS)
        {
            throw std::invalid_argument(
                "band index " + std::to_string(band) +
                " is not below CCD_MAX_BANDS (" +
                std::to_string(CCD_MAX_BANDS) + ")"
            );
        }

        if (count_ == CCD_MAX_BANDS)
        {
            throw std::invalid_argument(
                "a band list holds at most " +
                std::to_string(CCD_MAX_BANDS) + " bands"
            );
        }

        bands_[count_++] = static_cast<value_type>(band);
    }

    index_t size() const
    {
        return count_;
    }

    bool empty() const
    {
        return count_ == 0;
    }

    index_t operator[](index_t i) const
    {
        return bands_[i];
    }

    index_t front() const
    {
        return bands_[0];
    }

    const_iterator begin() const
    {
        return bands_.data();
    }

    const_iterator end() const
    {
        return bands_.data() + count_;
    }

    bool contains(index_t band) const
    {
        for (const index_t b : *this)
        {
            if (b == band)
                return true;
        }
        return false;
    }

    std::vector<index_t> to_vector() const
    {
        return std::vector<index_t>(begin(), end());
    }

private:

    std::array<value_type, CCD_MAX_BANDS> bands_{};
    std::uint8_t count_ = 0;
};

} // namespace ccd
