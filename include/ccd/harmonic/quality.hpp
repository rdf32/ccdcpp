#pragma once

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/harmonic/harmonic.hpp"


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
        const bool enough_clear(scalar_t threshold) {
            return clear_probability >= threshold;
        }

        const bool enough_snow(scalar_t threshold) {
            return snow_probability >= threshold;
        }
    };

public:
    static Statistics compute(
        ArrayView<const std::uint8_t, 1> qas,
        ArrayView<const std::int64_t, 1> dates,
        const HarmonicOptions& options
    )
    {
        Statistics stats;

        for(index_t i = 0; i < qas.size(); ++i)
        {
            update(stats, qas(i), options);
        }

        stats.finalize();

        return stats;
    }

    static Statistics compute_until(
        ArrayView<const std::uint8_t,1> qas,
        ArrayView<const std::int64_t,1> dates,
        const HarmonicOptions& options
    )
    {
        Statistics stats;

        for(index_t i = 0; i < qas.size(); ++i)
        {
            if(dates(i) <= options.STAT_ORD)
            {
                update(stats, qas(i), options);
            }
        }

        stats.finalize();

        return stats;
    }

private:

    static void update(
        Statistics& stats,
        std::uint8_t value,
        const HarmonicOptions& options
    )
    {
        if(value == options.QA_FILL)
        {
            ++stats.fill_count;
        }
        else if(value == options.QA_CLEAR)
        {
            ++stats.clear_count;
        }
        else if(value == options.QA_WATER)
        {
            ++stats.water_count;
        }
        else if(value == options.QA_SNOW)
        {
            ++stats.snow_count;
        }
        else if(value == options.QA_CLOUD)
        {
            ++stats.cloud_count;
        }

        ++stats.total_count;
    }
};

} // namespace ccd