#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{

// pyccd's Tmask over one window (dates_window (n), spect_window (B, n)):
// each band of `bands` fitted with a robust annual harmonic; an observation
// is set in the returned mask when its residual in any of them exceeds
// t_const times the band's variogram.
ProcessingMask tmask(
    ArrayView<const std::int64_t, 1> dates_window,
    ArrayView<const scalar_t, 2> spect_window,
    const BandNoise& variogram,
    const BandList& bands,
    scalar_t t_const
);

namespace strategies
{

//------------------------------------------------------------------------------
// window_screen "tmask": pyccd's Tmask. Fits `bands` of the window with a
// robust annual harmonic (tmask()); an observation is an outlier when its
// residual in any of them exceeds t_const variograms.
//------------------------------------------------------------------------------
struct Tmask final : WindowScreen
{
    struct Params
    {
        BandList bands = {1, 4};
        scalar_t t_const = 4.89;
    };

    static const std::vector<config::Field<Params>>& schema();

    // Refuses an empty band list.
    explicit Tmask(const Params& p);

    ProcessingMask screen(ArrayView<const std::int64_t, 1> dates,
                          ArrayView<const scalar_t, 2> spectral,
                          const BandNoise& variogram) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
