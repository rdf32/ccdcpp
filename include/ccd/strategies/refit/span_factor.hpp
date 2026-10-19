#pragma once

#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// refit "span-factor": pyccd's schedule. Refit while the window holds at most
// `min_obs` observations, and after that whenever the window spans at least
// `factor` times the span of the last fit.
//------------------------------------------------------------------------------
struct SpanFactor final : RefitSchedule
{
    struct Params
    {
        index_t min_obs = 24;
        scalar_t factor = 1.33;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit SpanFactor(const Params& p) : p_(p) {}

    bool due(index_t window_size, scalar_t window_span, scalar_t fit_span) const override
    {
        return window_size <= p_.min_obs || window_span >= p_.factor * fit_span;
    }

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
