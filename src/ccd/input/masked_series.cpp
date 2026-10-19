#include "ccd/input/masked_series.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace ccd
{

namespace
{

//------------------------------------------------------------------------------
// Clears in `mask` the entry of each masked position offset + j that
// outliers[j] flags.
//
// Walks the kept observations, counting masked positions. Clearing the
// current entry after testing it leaves the count of the rest unchanged.
//------------------------------------------------------------------------------
void clear_outliers(
    ProcessingMask& mask,
    const ProcessingMask& outliers,
    index_t offset
)
{
    index_t masked = 0;
    for (index_t i = 0; i < mask.size(); ++i)
    {
        if (!mask[i])
            continue;

        if (masked >= offset &&
            masked - offset < outliers.size() &&
            outliers[masked - offset])
        {
            mask.set(i, false);
        }
        ++masked;
    }
}

//------------------------------------------------------------------------------
// Clears in `mask` the entry of masked position `position`.
//------------------------------------------------------------------------------
void clear_position(
    ProcessingMask& mask,
    index_t position
)
{
    index_t masked = 0;
    for (index_t i = 0; i < mask.size(); ++i)
    {
        if (!mask[i])
            continue;

        if (masked == position)
        {
            mask.set(i, false);
            return;
        }
        ++masked;
    }
    assert(false && "position out of range");
}

} // namespace

void MaskedSeries::reset(const HarmonicWorkspace& workspace, ProcessingMask mask)
{
    mask_ = std::move(mask);
    apply_mask(workspace, mask_, kept_);
}

index_t MaskedSeries::count_until(std::int64_t day) const
{
    const std::int64_t* first = kept_.dates.data();
    const std::int64_t* last = first + kept_.observations;

    return static_cast<index_t>(std::upper_bound(first, last, day) - first);
}

void MaskedSeries::drop(index_t position, Window& window)
{
    assert(position < size());

    clear_position(mask_, position);
    kept_.erase(position);

    if (window.start > position)
        --window.start;
    if (window.stop > position)
        --window.stop;

    assert(size() == mask_.count());
}

void MaskedSeries::build_without(const ProcessingMask& outliers, index_t offset)
{
    const index_t n = kept_.observations;

    auto dropped = [&](index_t i)
    {
        return i >= offset &&
            i - offset < outliers.size() &&
            outliers[i - offset];
    };

    candidate_.bands = kept_.bands;
    candidate_.dates.clear();
    candidate_.spectral.clear();

    for (index_t i = 0; i < n; ++i)
    {
        if (!dropped(i))
            candidate_.dates.push_back(kept_.dates[i]);
    }

    candidate_.observations = candidate_.dates.size();

    for (index_t b = 0; b < kept_.bands; ++b)
    {
        for (index_t i = 0; i < n; ++i)
        {
            if (!dropped(i))
                candidate_.spectral.push_back(kept_.spectral[b * n + i]);
        }
    }
}

void MaskedSeries::commit_without(const ProcessingMask& outliers, index_t offset)
{
    clear_outliers(mask_, outliers, offset);
    std::swap(kept_, candidate_);

    assert(size() == mask_.count());
}

ProcessingMask MaskedSeries::take_mask()
{
    kept_.dates.clear();
    kept_.spectral.clear();
    kept_.observations = 0;

    return std::move(mask_);
}

} // namespace ccd
