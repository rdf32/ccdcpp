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
// outliers[j] flags, writing `why` there in `reasons` when it holds codes.
//
// Walks the kept observations, counting masked positions. Clearing the
// current entry after testing it leaves the count of the rest unchanged.
//------------------------------------------------------------------------------
void clear_outliers(
    ProcessingMask& mask,
    std::vector<std::uint8_t>& reasons,
    const ProcessingMask& outliers,
    index_t offset,
    Reason why
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
            if (!reasons.empty())
                reasons[i] = static_cast<std::uint8_t>(why);
        }
        ++masked;
    }
}

//------------------------------------------------------------------------------
// Clears in `mask` the entry of masked position `position`, writing `why`
// there in `reasons` when it holds codes.
//------------------------------------------------------------------------------
void clear_position(
    ProcessingMask& mask,
    std::vector<std::uint8_t>& reasons,
    index_t position,
    Reason why
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
            if (!reasons.empty())
                reasons[i] = static_cast<std::uint8_t>(why);
            return;
        }
        ++masked;
    }
    assert(false && "position out of range");
}

} // namespace

MaskedData apply_mask(
    const PixelSeries& series,
    const ProcessingMask& mask
)
{
    MaskedData out;
    apply_mask(series, mask, out);
    return out;
}

void apply_mask(
    const PixelSeries& series,
    const ProcessingMask& mask,
    MaskedData& out
)
{
    const auto dates    = series.dates();
    const auto spectral = series.spectral();

    const index_t count = mask.count();
    const index_t bands = spectral.extent(0);

    out.bands = bands;
    out.observations = count;

    out.dates.clear();
    out.spectral.clear();

    out.dates.reserve(count);
    out.spectral.reserve(bands * count);

    for(index_t i = 0; i < dates.size(); ++i)
    {
        if(mask.test(i))
            out.dates.push_back(dates(i));
    }

    for(index_t b = 0; b < bands; ++b)
    {
        for(index_t i = 0; i < dates.size(); ++i)
        {
            if(mask.test(i))
                out.spectral.push_back(
                    spectral(b, i)
                );
        }
    }
}

void MaskedSeries::reset(const PixelSeries& pixel, ProcessingMask mask,
                         std::vector<std::uint8_t> reasons)
{
    assert(reasons.empty() || reasons.size() == mask.size());
    mask_ = std::move(mask);
    reasons_ = std::move(reasons);
    apply_mask(pixel, mask_, kept_);
}

index_t MaskedSeries::count_until(std::int64_t day) const
{
    const std::int64_t* first = kept_.dates.data();
    const std::int64_t* last = first + kept_.observations;

    return static_cast<index_t>(std::upper_bound(first, last, day) - first);
}

void MaskedSeries::drop(index_t position, Window& window, Reason why)
{
    assert(position < size());

    clear_position(mask_, reasons_, position, why);
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

void MaskedSeries::commit_without(const ProcessingMask& outliers, index_t offset, Reason why)
{
    clear_outliers(mask_, reasons_, outliers, offset, why);
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

std::vector<std::uint8_t> MaskedSeries::take_reasons()
{
    return std::move(reasons_);
}

} // namespace ccd
