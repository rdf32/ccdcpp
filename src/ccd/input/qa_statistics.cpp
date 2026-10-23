#include "ccd/input/qa_statistics.hpp"

namespace ccd
{

namespace
{

// One observation's QA code counted.
void count(QaStatistics& stats, std::uint8_t value, const DataSpec& spec)
{
    if(value == spec.qa_fill)
    {
        ++stats.fill_count;
    }
    else if(value == spec.qa_clear)
    {
        ++stats.clear_count;
    }
    else if(value == spec.qa_water)
    {
        ++stats.water_count;
    }
    else if(value == spec.qa_snow)
    {
        ++stats.snow_count;
    }
    else if(value == spec.qa_cloud)
    {
        ++stats.cloud_count;
    }

    ++stats.total_count;
}

// The fractions, from the counts.
void set_probabilities(QaStatistics& stats)
{
    const scalar_t clear_water =
        static_cast<scalar_t>(
            stats.clear_count + stats.water_count
        );

    const scalar_t valid =
        static_cast<scalar_t>(
            stats.total_count - stats.fill_count
        );

    stats.cloud_probability =
        valid > 0
            ? static_cast<scalar_t>(stats.cloud_count) / valid
            : 0.0;

    stats.snow_probability =
        static_cast<scalar_t>(stats.snow_count) /
        (clear_water + stats.snow_count + 0.01);

    stats.water_probability =
        static_cast<scalar_t>(stats.water_count) /
        (clear_water + 0.01);

    stats.clear_probability =
        valid > 0
            ? clear_water / valid
            : 0.0;
}

} // namespace

QaStatistics qa_statistics(
    ArrayView<const std::uint8_t, 1> qas,
    ArrayView<const std::int64_t, 1> dates,
    const DataSpec& spec,
    std::int64_t statistics_end
)
{
    QaStatistics stats;

    for(index_t i = 0; i < qas.size(); ++i)
    {
        if(dates(i) <= statistics_end)
        {
            count(stats, qas(i), spec);
        }
    }

    set_probabilities(stats);

    return stats;
}

} // namespace ccd
