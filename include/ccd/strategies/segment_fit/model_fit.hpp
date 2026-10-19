#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// segment_fit "model-fit": pyccd's. A segment reports the monitoring
// model's fits, as its last refit left them (Model::band_fit()).
//------------------------------------------------------------------------------
struct ModelFit final : SegmentFit
{
    struct Params
    {
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit ModelFit(const Params&) {}

    BandFit band(const Model& model, index_t band) const override;
};

} // namespace strategies
} // namespace ccd
