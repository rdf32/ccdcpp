#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// segment_fit "model-fit": pyccd's. A segment reports the monitoring
// model's fits, as its last refit left them (Model::band_fit()), with the
// count of that refit. Under a mode's cap below it, and for a forced start
// the series ended before Extend fitted, the model is fitted on the
// segment's window with the capped count first.
//------------------------------------------------------------------------------
// num_coef under a mode's cap max_coefficients (0: none).
inline index_t capped_coefficients(index_t num_coef, index_t max_coefficients)
{
    return max_coefficients > 0 && max_coefficients < num_coef ? max_coefficients : num_coef;
}

struct ModelFit final : SegmentFit
{
    struct Params
    {
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit ModelFit(const Params&) {}

    std::unique_ptr<Scratch> make_scratch() const override;
    index_t segment(Model& model, ArrayView<const std::int64_t, 1> dates,
                    ArrayView<const scalar_t, 2> spectral, const Window& window, index_t num_coef,
                    index_t max_coefficients, Scratch& scratch,
                    std::vector<BandFit>& bands) const override;
    index_t unfitted(Model& model, ArrayView<const std::int64_t, 1> dates,
                     ArrayView<const scalar_t, 2> spectral, const Window& window, index_t num_coef,
                     index_t max_coefficients, Scratch& scratch,
                     std::vector<BandFit>& bands) const override;
};

} // namespace strategies
} // namespace ccd
