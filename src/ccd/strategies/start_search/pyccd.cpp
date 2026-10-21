#include "ccd/strategies/start_search/pyccd.hpp"

#include <stdexcept>

namespace ccd
{
namespace strategies
{

namespace
{

// Whether the dates of a window span at least day_delta days.
bool enough_time(
    ArrayView<const std::int64_t, 1> dates_window,
    std::int64_t day_delta
) {
    const auto first_date = dates_window(0);
    const auto last_date = dates_window(dates_window.size() - 1);

    return (last_date - first_date) >= day_delta;
}

// Whether a window holds at least min_observations observations.
bool enough_samples(
    ArrayView<const std::int64_t, 1> dates_window,
    index_t min_observations
) {
    return dates_window.size() >= min_observations;
}

} // namespace

const std::vector<config::Field<PyccdSearch::Params>>& PyccdSearch::schema()
{
    using config::Unit;
    static const std::vector<config::Field<Params>> s = {
        {"min_observations", &Params::min_observations, Unit::Observations, config::at_least(1),
         "observations a candidate start window holds, and needs after it",
         config::Source::Shared, "min_observations"},
        {"day_delta", &Params::day_delta, Unit::Days, config::any(),
         "days a candidate spans at least"},
    };
    return s;
}

bool PyccdSearch::too_few(index_t observations) const
{
    return observations <= p_.min_observations;
}

Window PyccdSearch::first(index_t at) const
{
    return Window(at, at + p_.min_observations);
}

bool PyccdSearch::room(const Window& window, index_t observations) const
{
    // window.stop <= observations - min_observations, written without the
    // subtraction: index_t is unsigned, and observations < min_observations
    // would wrap it to a huge bound.
    return window.stop + p_.min_observations <= observations;
}

bool PyccdSearch::exhausted(const Window& window, index_t observations) const
{
    return !(window.stop + p_.min_observations < observations);
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
    return enough_time(dates, p_.day_delta) && enough_samples(dates, p_.min_observations);
}

} // namespace strategies
} // namespace ccd
