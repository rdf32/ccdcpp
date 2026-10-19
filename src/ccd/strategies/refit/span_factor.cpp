#include "ccd/strategies/refit/span_factor.hpp"

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<SpanFactor::Params>>& SpanFactor::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"min_obs", &Params::min_obs, Unit::Observations, config::at_least(1),
         "observations up to which every test refits"},
        {"factor", &Params::factor, Unit::None, config::greater_than(0),
         "refit once the window spans this many times the last fit's span"},
    };
    return s;
}

} // namespace strategies
} // namespace ccd
