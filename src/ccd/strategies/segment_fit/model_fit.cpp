#include "ccd/strategies/segment_fit/model_fit.hpp"

#include <algorithm>

#include "ccd/strategies/model.hpp"

namespace ccd
{
namespace strategies
{

namespace
{

// `model` fitted on `window` with q coefficients, every band.
void refit(Model& model, ArrayView<const std::int64_t, 1> dates,
           ArrayView<const scalar_t, 2> spectral, const Window& window, index_t q,
           index_t num_coef)
{
    model.fit(dates.slice(range(window.start, window.stop)),
              spectral.slice(all(), range(window.start, window.stop)), q,
              std::max(q, num_coef), every_band(spectral.extent(0)), KeepResiduals::No);
}

void band_fits(const Model& model, index_t num_bands, std::vector<BandFit>& bands)
{
    bands.clear();
    bands.reserve(num_bands);
    for (index_t band = 0; band < num_bands; ++band)
        bands.push_back(model.band_fit(band));
}

} // namespace

const std::vector<config::Field<ModelFit::Params>>& ModelFit::schema()
{
    static const std::vector<config::Field<Params>> s;
    return s;
}

std::unique_ptr<Scratch> ModelFit::make_scratch() const
{
    return std::make_unique<Scratch>();
}

index_t ModelFit::segment(Model& model, ArrayView<const std::int64_t, 1> dates,
                          ArrayView<const scalar_t, 2> spectral, const Window& window,
                          index_t num_coef, index_t max_coefficients, Scratch&,
                          std::vector<BandFit>& bands) const
{
    const index_t q = capped_coefficients(num_coef, max_coefficients);
    if (q != num_coef)
        refit(model, dates, spectral, window, q, num_coef);
    band_fits(model, spectral.extent(0), bands);
    return q;
}

index_t ModelFit::unfitted(Model& model, ArrayView<const std::int64_t, 1> dates,
                           ArrayView<const scalar_t, 2> spectral, const Window& window,
                           index_t num_coef, index_t max_coefficients, Scratch&,
                           std::vector<BandFit>& bands) const
{
    const index_t q = capped_coefficients(num_coef, max_coefficients);
    refit(model, dates, spectral, window, q, num_coef);
    band_fits(model, spectral.extent(0), bands);
    return q;
}

} // namespace strategies
} // namespace ccd
