#pragma once

#include <cstdint>

#include "ccd/core/array_view.hpp"
#include "ccd/core/types.hpp"
#include "ccd/input/data_spec.hpp"

namespace ccd
{

//------------------------------------------------------------------------------
//
// One pixel's QA statistics: how many of its observations hold each QA code,
// and the fractions the mode stage chooses the pixel's procedure by
// (ModeRule::choose, strategies/stages.hpp).
//
//   clear_probability  (clear + water) / (total - fill), 0 when every
//                      observation is fill
//   snow_probability   snow / (clear + water + snow + 0.01)
//   water_probability  water / (clear + water + 0.01)
//   cloud_probability  cloud / (total - fill), 0 when every observation is
//                      fill
//
//------------------------------------------------------------------------------
struct QaStatistics
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

    bool enough_clear(scalar_t threshold) const { return clear_probability >= threshold; }

    bool enough_snow(scalar_t threshold) const { return snow_probability >= threshold; }
};

// The statistics of the observations dated <= statistics_end, by `spec`'s
// QA codes (`qas` and `dates` of one length).
QaStatistics qa_statistics(
    ArrayView<const std::uint8_t, 1> qas,
    ArrayView<const std::int64_t, 1> dates,
    const DataSpec& spec,
    std::int64_t statistics_end
);

} // namespace ccd
