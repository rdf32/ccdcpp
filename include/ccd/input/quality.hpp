#pragma once

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/input/input.hpp"


namespace ccd
{
//==============================================================================
//
// Quality
//
// Computes QA statistics for a single pixel time series.
//
//==============================================================================
class Quality
{
public:

    struct Statistics
    {
        index_t total_count = 0;

        index_t fill_count  = 0;
        index_t clear_count = 0;
        index_t water_count = 0;
        index_t snow_count  = 0;
        index_t cloud_count = 0;

        scalar_t cloud_probability = 0.0;
        scalar_t snow_probability  = 0.0;
        scalar_t water_probability = 0.0;
        scalar_t clear_probability = 0.0;

        void finalize()
        {
            const scalar_t clear_water =
                static_cast<scalar_t>(
                    clear_count + water_count
                );

            const scalar_t valid =
                static_cast<scalar_t>(
                    total_count - fill_count
                );

            cloud_probability =
                valid > 0
                    ? static_cast<scalar_t>(cloud_count) / valid
                    : 0.0;

            snow_probability =
                static_cast<scalar_t>(snow_count) /
                (clear_water + snow_count + 0.01);

            water_probability =
                static_cast<scalar_t>(water_count) /
                (clear_water + 0.01);

            clear_probability =
                valid > 0
                    ? clear_water / valid
                    : 0.0;
        }
        bool enough_clear(scalar_t threshold) const {
            return clear_probability >= threshold;
        }

        bool enough_snow(scalar_t threshold) const {
            return snow_probability >= threshold;
        }
    };

public:
    // The statistics of every observation, by `input`'s QA codes.
    static Statistics compute(
        ArrayView<const std::uint8_t, 1> qas,
        const Input& input
    )
    {
        Statistics stats;

        for(index_t i = 0; i < qas.size(); ++i)
        {
            update(stats, qas(i), input);
        }

        stats.finalize();

        return stats;
    }

    // The statistics of the observations dated <= statistics_end.
    static Statistics compute_until(
        ArrayView<const std::uint8_t,1> qas,
        ArrayView<const std::int64_t,1> dates,
        const Input& input,
        std::int64_t statistics_end
    )
    {
        Statistics stats;

        for(index_t i = 0; i < qas.size(); ++i)
        {
            if(dates(i) <= statistics_end)
            {
                update(stats, qas(i), input);
            }
        }

        stats.finalize();

        return stats;
    }

private:

    static void update(
        Statistics& stats,
        std::uint8_t value,
        const Input& input
    )
    {
        if(value == input.qa_fill)
        {
            ++stats.fill_count;
        }
        else if(value == input.qa_clear)
        {
            ++stats.clear_count;
        }
        else if(value == input.qa_water)
        {
            ++stats.water_count;
        }
        else if(value == input.qa_snow)
        {
            ++stats.snow_count;
        }
        else if(value == input.qa_cloud)
        {
            ++stats.cloud_count;
        }

        ++stats.total_count;
    }
};

} // namespace ccd