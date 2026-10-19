#include "ccd/strategies/segment_fit/model_fit.hpp"

#include "ccd/strategies/model.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<ModelFit::Params>>& ModelFit::schema()
{
    static const std::vector<config::Field<Params>> s;
    return s;
}

BandFit ModelFit::band(const Model& model, index_t band) const
{
    return model.band_fit(band);
}

} // namespace strategies
} // namespace ccd
