#include "ccd/input/pixel_series.hpp"

#include <stdexcept>


namespace ccd
{

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


} // namespace ccd