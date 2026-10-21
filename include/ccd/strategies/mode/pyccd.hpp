#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// mode "pyccd": pyccd's choice. Standard when the clear fraction reaches
// clear_pct_threshold; otherwise PermanentSnow when the snow fraction
// reaches snow_pct_threshold; otherwise InsufficientClear.
//------------------------------------------------------------------------------
struct PyccdMode final : ModeRule
{
    struct Params
    {
        scalar_t clear_pct_threshold = 0.25;
        scalar_t snow_pct_threshold = 0.75;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit PyccdMode(const Params& p) : p_(p) {}

    PixelMode choose(const QaStatistics& statistics) const override;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
