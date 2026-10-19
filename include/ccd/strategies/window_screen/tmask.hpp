#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// window_screen "tmask": pyccd's Tmask. Fits `bands` of the window with a
// robust annual harmonic (tmask(), procedure/fit.hpp); an observation is an
// outlier when its residual in any of them exceeds t_const variograms.
//------------------------------------------------------------------------------
struct Tmask final : WindowScreen
{
    struct Params
    {
        BandList bands = {1, 4};
        scalar_t t_const = 4.89;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit Tmask(const Params& p) : p_(p) {}

    ProcessingMask screen(ArrayView<const std::int64_t, 1> dates,
                          ArrayView<const scalar_t, 2> spectral,
                          const Variogram& variogram) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
