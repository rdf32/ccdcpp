#include "ccd/input/harmonic.hpp"

#include <stdexcept>


namespace ccd
{

const char* method_name(ChangeMethod method)
{
    switch (method)
    {
        case ChangeMethod::Ccdc: return "ccdc";
    }
    return "?";
}

ChangeMethod method_from_name(const std::string& name)
{
    for (const ChangeMethod m : {ChangeMethod::Ccdc})
        if (name == method_name(m))
            return m;
    throw std::invalid_argument(
        "unknown change detection method \"" + name + "\": ccdc");
}

HarmonicOptions with_method(HarmonicOptions options, ChangeMethod method)
{
    options.METHOD = method;
    return options;
}

MaskedData apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask
)
{   
    MaskedData out;
    apply_mask(workspace, mask, out);
    return out;
}

void apply_mask(
    const HarmonicWorkspace& workspace,
    const ProcessingMask& mask,
    MaskedData& out
)
{   
    const auto dates    = workspace.dates();
    const auto spectral = workspace.spectral();

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