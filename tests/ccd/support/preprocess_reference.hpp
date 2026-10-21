#pragma once

//==============================================================================
//
// The "ccdc" preprocessing chains written out by hand from the filter
// kernels (input/mask/filter.hpp), with the bounds converted here by the rule
// input.hpp states, independently of the registry's resolution: the reference
// test_preprocess.cpp holds plan_of()'s chains to, mask for mask, and
// tools/bench/bench_preprocess.cpp times them against.
//
//==============================================================================

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "ccd/ccd.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/input/mask/filter.hpp"

#include "config.hpp"
#include "series.hpp"
#include "units.hpp"

namespace golden::chains
{

namespace pp = ccd::preprocess;

using ccd::index_t;
using ccd::scalar_t;

// An input and the configuration whose preprocessing it runs.
struct Options
{
    ccd::Input input;
    ccd::config::Config config = golden::pyccd();
};

// The physical-unit input's.
inline Options physical_options()
{
    return {golden::physical_input(), golden::pyccd()};
}

inline pp::PreprocessPlan plan_of(const Options& options)
{
    return golden::plan_of(options.config, options.input);
}

inline double number(const ccd::config::Values& values, const char* name)
{
    return std::get<double>(values.at(name));
}

// End `end` (0: min, 1: max) of the range `name`, and whether it is kept.
inline double end_of(const ccd::config::Values& values, const char* name, std::size_t end)
{
    return std::get<std::vector<double>>(values.at(name)).at(end);
}

inline bool inclusive(const ccd::config::Values& values, std::size_t end)
{
    return std::get<std::vector<bool>>(values.at("inclusive")).at(end);
}

// The values the reference chains read: the configuration's bounds, offset
// and scope as written (physical), converted here to the input's units by
// the rule input.hpp states -- reflectance times reflectance_scale, degrees
// Celsius as hundredths of a degree times thermal_at_100c / 10000 --
// independently of the registry's resolution.
struct Bounds
{
    ccd::BandList bands;   // band_range's rows
    pp::Range reflectance;
    pp::Range thermal;
    scalar_t green_offset;
    ccd::GreenFilterScope scope;
    std::int64_t statistics_end;
};

// band_range's bands as rows of `input`: roles are their rows, rows as
// given.
inline ccd::BandList band_rows(const ccd::config::Value& bands, const ccd::Input& input)
{
    ccd::BandList rows;
    if (const auto* roles = std::get_if<std::vector<std::string>>(&bands))
    {
        for (const std::string& role : *roles)
            rows.push_back(input.index(role));
    }
    else
    {
        for (const std::int64_t r : std::get<std::vector<std::int64_t>>(bands))
            rows.push_back(static_cast<index_t>(r));
    }
    return rows;
}

inline Bounds bounds_of(const Options& options)
{
    const auto& standard = options.config.preprocess.at("standard").steps;
    const ccd::config::Values& thermal = standard.at(1).params;
    const ccd::config::Values& band = standard.at(2).params;
    const ccd::config::Values& green =
        options.config.preprocess.at("insufficient_clear").steps.at(4).params;
    const scalar_t u = options.input.reflectance_scale;
    const scalar_t tu = options.input.thermal_at_100c;
    const auto celsius = [tu](double c) { return c * 100.0 * tu / 10000.0; };
    return {
        band_rows(band.at("bands"), options.input),
        {{end_of(band, "reflectance", 0) * u, inclusive(band, 0)},
         {end_of(band, "reflectance", 1) * u, inclusive(band, 1)}},
        {{celsius(end_of(thermal, "celsius", 0)), inclusive(thermal, 0)},
         {celsius(end_of(thermal, "celsius", 1)), inclusive(thermal, 1)}},
        number(green, "offset") * u,
        *ccd::green_filter_scope(std::get<std::string>(green.at("scope"))),
        std::get<std::int64_t>(options.config.shared.at("statistics_end")),
    };
}

//------------------------------------------------------------------------------
// The three chains written out by hand from the filter kernels: the
// reference plan_of() is held to.
//------------------------------------------------------------------------------

// The thermal range when the input has a thermal row. Standard's chain is
// handed its thermal row transformed, so `transformed` tests the row as
// given; the other chains read the stored row, which is tested as the
// input's transform makes each value, against the same bounds in degrees
// Celsius.
inline void reference_thermal(
    const golden::PixelView& px,
    const Options& options,
    bool transformed,
    ccd::ProcessingMask& mask
)
{
    const index_t thermal = options.input.index("thermal");
    if (thermal == ccd::NO_BAND)
        return;
    const ccd::filter::Range range = bounds_of(options).thermal;
    for (index_t i = 0; i < px.dates.size(); ++i)
    {
        const double stored = px.spectral(thermal, i);
        const double value = transformed ? stored
                                         : stored * options.input.thermal_scale +
                                               options.input.thermal_offset;
        if (!ccd::filter::in_range(value, range))
            mask.set(i, false);
    }
}

// Clear and water kept, the thermal range (reference_thermal), the
// reflectance range, one per date.
inline ccd::ProcessingMask reference_common(
    const golden::PixelView& px,
    const Options& options,
    bool transformed
)
{
    ccd::ProcessingMask mask(px.dates.size());

    for (index_t i = 0; i < px.dates.size(); ++i)
    {
        const auto qa = px.qas(i);
        if (qa == options.input.qa_water || qa == options.input.qa_clear)
            mask.set(i, true);
    }

    reference_thermal(px, options, transformed, mask);

    ccd::filter::apply_band_range(
        px.spectral,
        bounds_of(options).bands,
        bounds_of(options).reflectance,
        mask
    );

    ccd::filter::apply_one_per_date(px.dates, mask);

    return mask;
}

inline ccd::ProcessingMask reference_standard(
    const golden::PixelView& px,
    const Options& options
)
{
    return reference_common(px, options, true);
}

inline ccd::ProcessingMask reference_snow(
    const golden::PixelView& px,
    const Options& options
)
{
    // The thermal range again after the snow is back, over it too.
    ccd::ProcessingMask mask = reference_common(px, options, false);

    for (index_t i = 0; i < px.qas.size(); ++i)
    {
        if (px.qas(i) == options.input.qa_snow)
            mask.set(i, true);
    }

    ccd::filter::apply_one_per_date(px.dates, mask);
    reference_thermal(px, options, false, mask);

    return mask;
}

inline ccd::ProcessingMask reference_insufficient_clear(
    const golden::PixelView& px,
    const Options& options
)
{
    ccd::ProcessingMask mask = reference_common(px, options, false);
    const Bounds bounds = bounds_of(options);

    ccd::filter::apply_green_median(
        px.dates,
        px.spectral,
        options.input.index("green"),
        bounds.statistics_end,
        bounds.green_offset,
        bounds.scope,
        mask
    );

    ccd::filter::apply_one_per_date(px.dates, mask);

    return mask;
}

inline ccd::ProcessingMask run_chain(const pp::Chain& chain, const golden::PixelView& px)
{
    ccd::ProcessingMask mask;
    pp::apply(chain, px.dates, px.spectral, px.qas, mask);
    return mask;
}

} // namespace golden::chains
