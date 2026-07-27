#include "ccd/harmonic/harmonic.hpp"


namespace ccd
{

MaskedData apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask
)
{   
    const auto dates    = workspace.dates();
    const auto spectral = workspace.spectral();

    const index_t count = mask.count();
    const index_t bands = spectral.extent(0);

    MaskedData out;

    out.bands = bands;
    out.observations = count;

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

    return out;
}


} // namespace ccd