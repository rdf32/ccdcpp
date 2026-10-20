#include "ccd/strategies/start_search/pyccd.hpp"

#include <stdexcept>

namespace ccd
{
namespace strategies
{

const std::vector<config::Field<PyccdSearch::Params>>& PyccdSearch::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"meow_size", &Params::meow_size, Unit::Observations, config::at_least(1),
         "observations a candidate start window holds, and needs after it",
         config::Source::Shared, "min_observations"},
        {"day_delta", &Params::day_delta, Unit::Days, config::any(),
         "days a candidate spans at least"},
    };
    return s;
}

bool PyccdSearch::too_few(index_t observations) const
{
    return observations <= p_.meow_size;
}

Window PyccdSearch::first(index_t at) const
{
    return Window(at, at + p_.meow_size);
}

bool PyccdSearch::room(const Window& window, index_t observations) const
{
    // window.stop <= observations - meow_size, written without the
    // subtraction: index_t is unsigned, and observations < meow_size would
    // wrap it to a huge bound.
    return window.stop + p_.meow_size <= observations;
}

bool PyccdSearch::exhausted(const Window& window, index_t observations) const
{
    return !(window.stop + p_.meow_size < observations);
}

Window PyccdSearch::forced(ArrayView<const std::int64_t, 1>, index_t, std::int64_t) const
{
    throw std::logic_error("start_search \"pyccd\" forces no start");
}

bool PyccdSearch::short_span(ArrayView<const std::int64_t, 1> dates) const
{
    return !enough_time(dates, p_.day_delta);
}

bool PyccdSearch::usable(ArrayView<const std::int64_t, 1> dates) const
{
    return enough_time(dates, p_.day_delta) && enough_samples(dates, p_.meow_size);
}

} // namespace strategies
} // namespace ccd
