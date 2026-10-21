//==============================================================================
//
// The preprocessing plan (preprocess/preprocess.hpp).
//
//   CcdcPlan.*        the "ccdc" preprocessing resolved against an input
//                     (plan_of()) against the chains written out by hand from
//                     the filter kernels (reference_* below), the bounds
//                     converted here by the rule input.hpp states, mask for
//                     mask: on every fixture input (the reference pixels, the
//                     Landsat cubes of the golden fixture, the band profiles
//                     of the bands fixture) and on randomized series with NaN,
//                     values on and beside the bounds, repeated dates, snow
//                     and observations dated after statistics_end, in DN and in
//                     physical units, under every GreenFilterScope and every
//                     combination of the four inclusive flags.
//   CustomPlan.*      configurations whose chains change what detect()
//                     does: a Standard chain that keeps snow, and the thermal
//                     transform on every procedure; the chains written out
//                     step by step are the preset's.
//   PlanValidation.*  every plan validate() rejects, and that a configuration
//                     whose plan it rejects is refused before any pixel runs.
//   Microbench.*      apply(ccdc chain) against the hand-written chain on
//                     the 443-observation pyccd reference pixel. Reported,
//                     not gated.
//
//==============================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../golden/specs.hpp"

#include "ccd/ccd.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/preprocess.hpp"

#include "../golden/golden.hpp"
#include "../golden/synthetic.hpp"
#include "../golden/synthetic_bands.hpp"
#include "../golden/band_profiles.hpp"

namespace
{

namespace pp = ccd::preprocess;

using ccd::index_t;
using ccd::scalar_t;

constexpr scalar_t NaN = std::numeric_limits<scalar_t>::quiet_NaN();

//------------------------------------------------------------------------------
// One pixel's views.
//------------------------------------------------------------------------------

struct PixelView
{
    ccd::ArrayView<const std::int64_t, 1> dates;
    ccd::ArrayView<const scalar_t, 2> spectral;
    ccd::ArrayView<const std::uint8_t, 1> qas;
};

PixelView view(
    const std::vector<std::int64_t>& dates,
    const double* spectral,
    index_t B,
    const std::uint8_t* qas
)
{
    const index_t T = dates.size();
    return {
        ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
        ccd::ArrayView<const scalar_t, 2>::contiguous(spectral, {B, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas, {T}),
    };
}

// An input and the configuration whose preprocessing it runs.
struct Options
{
    ccd::Input input;
    ccd::config::Config config = golden::pyccd();
};

// The physical-unit input's.
Options physical_options()
{
    return {golden::physical_input(), golden::pyccd()};
}

pp::PreprocessPlan plan_of(const Options& options)
{
    return golden::plan_of(options.config, options.input);
}

double number(const ccd::config::Values& values, const char* name)
{
    return std::get<double>(values.at(name));
}

bool flag(const ccd::config::Values& values, const char* name)
{
    return std::get<bool>(values.at(name));
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

// band_range's bands as rows of `input`: "reflective" is every row whose
// role is not thermal or other, in row order; roles are their rows.
ccd::BandList band_rows(const ccd::config::Value& bands, const ccd::Input& input)
{
    ccd::BandList rows;
    if (std::holds_alternative<std::string>(bands))
    {
        for (index_t r = 0; r < input.bands(); ++r)
            if (input.roles[r] != "thermal" && input.roles[r] != "other")
                rows.push_back(r);
    }
    else if (const auto* roles = std::get_if<std::vector<std::string>>(&bands))
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

Bounds bounds_of(const Options& options)
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
        {{number(band, "min") * u, flag(band, "min_inclusive")},
         {number(band, "max") * u, flag(band, "max_inclusive")}},
        {{celsius(number(thermal, "min")), flag(thermal, "min_inclusive")},
         {celsius(number(thermal, "max")), flag(thermal, "max_inclusive")}},
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
void reference_thermal(
    const PixelView& px,
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
// reflectance range, deduped.
ccd::ProcessingMask reference_common(
    const PixelView& px,
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

    ccd::filter::apply_dedupe(px.dates, mask);

    return mask;
}

ccd::ProcessingMask reference_standard(
    const PixelView& px,
    const Options& options
)
{
    return reference_common(px, options, true);
}

ccd::ProcessingMask reference_snow(
    const PixelView& px,
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

    ccd::filter::apply_dedupe(px.dates, mask);
    reference_thermal(px, options, false, mask);

    return mask;
}

ccd::ProcessingMask reference_insufficient_clear(
    const PixelView& px,
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

    ccd::filter::apply_dedupe(px.dates, mask);

    return mask;
}

ccd::ProcessingMask run_chain(const pp::Chain& chain, const PixelView& px)
{
    ccd::ProcessingMask mask;
    pp::apply(chain, px.dates, px.spectral, px.qas, mask);
    return mask;
}

//------------------------------------------------------------------------------
// Every comparison of one pixel under one set of options. Returns the number
// of procedures whose mask kept at least one observation, so a caller can
// check the inputs reached the filters at all.
//------------------------------------------------------------------------------

struct Tally
{
    std::size_t masks = 0;
    std::size_t nonempty = 0;
    std::size_t snow_differs = 0;     // snow mask != standard mask
    std::size_t green_differs = 0;    // insufficient_clear mask != standard
};

void check_pixel(
    const PixelView& px,
    const Options& options,
    const std::string& label,
    Tally& tally
)
{
    const pp::PreprocessPlan plan = plan_of(options);
    ASSERT_NO_THROW(pp::validate(plan, px.spectral.extent(0))) << label;

    const ccd::ProcessingMask standard = reference_standard(px, options);
    const ccd::ProcessingMask snow = reference_snow(px, options);
    const ccd::ProcessingMask clear = reference_insufficient_clear(px, options);

    EXPECT_EQ(run_chain(plan.standard.chain, px).data(), standard.data())
        << "standard " << label;
    EXPECT_EQ(run_chain(plan.snow.chain, px).data(), snow.data())
        << "snow " << label;
    EXPECT_EQ(run_chain(plan.insufficient_clear.chain, px).data(), clear.data())
        << "insufficient_clear " << label;

    tally.masks += 3;
    tally.nonempty += standard.any() + snow.any() + clear.any();
    tally.snow_differs += snow.data() != standard.data();
    tally.green_differs += clear.data() != standard.data();
}

//------------------------------------------------------------------------------
// Option variations: every scope and every combination of the four flags.
//------------------------------------------------------------------------------

std::vector<Options> variations(const Options& base)
{
    std::vector<Options> out;
    for (const char* scope : {"all", "stat_window", "off"})
    {
        for (unsigned flags = 0; flags < 16; ++flags)
        {
            using ccd::config::flag;
            Options h = base;
            ccd::config::Config& c = h.config;
            c = golden::with_steps(c, "green_median", "scope", ccd::config::Value(std::string(scope)));
            c = golden::with_steps(c, "band_range", "min_inclusive", flag(flags & 1u));
            c = golden::with_steps(c, "band_range", "max_inclusive", flag(flags & 2u));
            c = golden::with_steps(c, "thermal_range", "min_inclusive", flag(flags & 4u));
            c = golden::with_steps(c, "thermal_range", "max_inclusive", flag(flags & 8u));
            out.push_back(h);
        }
    }
    return out;
}

std::string variation_label(const Options& h)
{
    const Bounds b = bounds_of(h);
    return std::string("scope ") + ccd::name(b.scope) +
           " refl " + std::to_string(b.reflectance.lo.inclusive) +
           std::to_string(b.reflectance.hi.inclusive) +
           " therm " + std::to_string(b.thermal.lo.inclusive) +
           std::to_string(b.thermal.hi.inclusive);
}

//------------------------------------------------------------------------------
// Fixture inputs
//------------------------------------------------------------------------------

struct Pixel
{
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (7, T)
    std::vector<std::uint8_t> qas;
};

Pixel read_csv(const std::string& filename)
{
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);
    if (!file)
        throw std::runtime_error("Unable to open " + filename);

    std::array<std::vector<double>, 7> bands;
    Pixel out;

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        out.dates.push_back(static_cast<std::int64_t>(std::stod(value)));

        for (std::size_t b = 0; b < 7; ++b)
        {
            std::getline(ss, value, ',');
            bands[b].push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        out.qas.push_back(static_cast<std::uint8_t>(std::stod(value)));
    }

    const std::size_t T = out.dates.size();
    out.spectral.resize(7 * T);
    for (std::size_t b = 0; b < 7; ++b)
        std::copy(bands[b].begin(), bands[b].end(), out.spectral.begin() + b * T);

    return out;
}

// A cube as flat arrays, B rows per pixel.
struct Series
{
    std::string name;
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;   // (N, B, T)
    std::vector<std::uint8_t> qas;  // (N, T)
    std::size_t N = 0;
    std::size_t B = 0;
    std::size_t T = 0;
    Options h;

    PixelView pixel(std::size_t p) const
    {
        return view(dates, spectral.data() + p * B * T, B, qas.data() + p * T);
    }
};

struct GoldenCube
{
    const char* name;
    std::uint64_t seed;
    golden::Regime regime;
    std::size_t H;
    std::size_t W;
};

// The cubes of test_golden.cpp, with its seeds.
const GoldenCube GOLDEN_CUBES[] = {
    {"c16",    11, golden::Regime::Cadence16,  10, 16},
    {"c8",     12, golden::Regime::Cadence8,    6,  8},
    {"dup",    13, golden::Regime::Duplicates,  5,  8},
    {"short",  14, golden::Regime::Short,       4,  5},
    {"medium", 15, golden::Regime::Medium,      5,  8},
};

Series from_cube(const char* name, const golden::Cube& c, const Options& h)
{
    return {name, c.dates, c.spectral, c.qas, c.H * c.W, 7, c.T, h};
}

Series to_physical(Series s)
{
    for (std::size_t p = 0; p < s.N; ++p)
        golden::to_physical(s.spectral.data() + p * s.B * s.T, s.T);
    s.name += ":phys";
    s.h = physical_options();
    return s;
}

Series from_pixel(const std::string& name, const Pixel& p, const Options& h)
{
    return {name, p.dates, p.spectral, p.qas, 1, 7, p.dates.size(), h};
}

// Every fixture input with the options its fixture runs it under.
std::vector<Series> fixture_inputs()
{
    std::vector<Series> out;

    {
        const Pixel p = read_csv("test_3657_3610_observations.csv");
        out.push_back(from_pixel("pyccd", p, Options{}));
        out.push_back(to_physical(out.back()));

        Pixel bounds = p;
        golden::put_bound_values(bounds.spectral.data(), bounds.qas.data(), bounds.dates.size());
        out.push_back(from_pixel("bounds", bounds, Options{}));
        out.push_back(to_physical(out.back()));

        out.push_back(from_pixel("c2", read_csv("test002_c2.csv"),
                                 Options{golden::c2_input(), golden::c2_config()}));
    }

    for (const auto& spec : GOLDEN_CUBES)
    {
        out.push_back(from_cube(
            spec.name,
            golden::make_cube(spec.seed, spec.regime, spec.H, spec.W),
            Options{}));
        out.push_back(to_physical(out.back()));
    }

    const auto profiles = golden::bands::profiles();
    for (std::size_t pi = 0; pi < profiles.size(); ++pi)
    {
        for (const auto& spec : golden::bands::CUBES)
        {
            const golden::BandCube c = golden::make_band_cube(
                100 * (pi + 1) + spec.seed, spec.regime, spec.H, spec.W,
                profiles[pi].specs);
            out.push_back({
                std::string(profiles[pi].name) + ":" + spec.name,
                c.dates, c.spectral, c.qas, c.H * c.W, c.B, c.T,
                {profiles[pi].h, profiles[pi].config}});
        }
    }

    return out;
}

//------------------------------------------------------------------------------
// Randomized series
//
// Each value is drawn from the cases the filters distinguish: either bound
// itself, just inside and just outside each, mid-range, far outside, and
// NaN. Dates repeat, and statistics_end falls before, inside or after the series.
//------------------------------------------------------------------------------

// The bounds of "ccdc" in the data's units, and the input's reflectance_scale
// that resolves them.
struct Unit
{
    const char* name;
    scalar_t refl_lo, refl_hi;
    scalar_t therm_lo, therm_hi;
    scalar_t green_filter;
    scalar_t reflectance_scale;
    scalar_t thermal_at_100c;
};

const Unit DN = {"dn", 0.0, 10000.0, ccd::MIN_CELSIUS, ccd::MAX_CELSIUS, 400.0, 10000.0, 10000.0};
const Unit PHYSICAL = {"phys", 0.0, 1.0,
    ccd::MIN_CELSIUS / 10000.0, ccd::MAX_CELSIUS / 10000.0, 0.04, 1.0, 1.0};

scalar_t draw_value(golden::Rng& rng, scalar_t lo, scalar_t hi)
{
    const scalar_t span = hi - lo;
    const scalar_t eps = span * 1e-9;
    switch (rng.below(10))
    {
        case 0: return lo;
        case 1: return hi;
        case 2: return lo + eps;
        case 3: return hi - eps;
        case 4: return lo - eps;
        case 5: return hi + eps;
        case 6: return NaN;
        case 7: return rng.below(2) ? lo - span : hi + span;
        default: return rng.uniform(lo, hi);
    }
}

// A random (7, T) series in `unit`, Landsat layout.
Series random_series(golden::Rng& rng, const Unit& unit)
{
    const std::size_t T = 1 + rng.below(200);

    Series s;
    s.name = unit.name;
    s.N = 1;
    s.B = 7;
    s.T = T;
    s.dates.resize(T);
    s.spectral.resize(7 * T);
    s.qas.resize(T);

    std::int64_t date = 730000 + static_cast<std::int64_t>(rng.below(100));
    for (std::size_t t = 0; t < T; ++t)
    {
        if (t > 0 && rng.uniform() >= 0.15)
            date += 1 + static_cast<std::int64_t>(rng.below(32));
        s.dates[t] = date;
    }

    const std::uint8_t codes[] = {0, 0, 0, 1, 2, 3, 3, 4, 255, 8};
    for (std::size_t t = 0; t < T; ++t)
        s.qas[t] = codes[rng.below(sizeof(codes))];

    // Green (band 1) mostly mid-range with spikes, so the median filter has
    // something to drop.
    const scalar_t green_mid = unit.refl_lo + 0.1 * (unit.refl_hi - unit.refl_lo);
    for (std::size_t b = 0; b < 7; ++b)
    {
        for (std::size_t t = 0; t < T; ++t)
        {
            scalar_t v;
            if (b == 6)
                v = draw_value(rng, unit.therm_lo, unit.therm_hi);
            else if (b == 1 && rng.uniform() < 0.7)
            {
                // Two draws, one statement each: in one expression their
                // order would be the compiler's.
                const scalar_t scale = rng.uniform() < 0.2 ? 3.0 : 0.5;
                v = green_mid + scale * unit.green_filter * rng.uniform(-1.0, 3.0);
            }
            else if (rng.uniform() < 0.6)
                v = rng.uniform(unit.refl_lo, unit.refl_hi);
            else
                v = draw_value(rng, unit.refl_lo, unit.refl_hi);
            s.spectral[b * T + t] = v;
        }
    }

    ccd::Input& h = s.h.input;
    h.reflectance_scale = unit.reflectance_scale;
    h.thermal_at_100c = unit.thermal_at_100c;

    // statistics_end before the series, inside it (often on an observation's
    // date), or after it.
    std::int64_t statistics_end = 0;
    switch (rng.below(4))
    {
        case 0: statistics_end = s.dates.front() - 1; break;
        case 1: statistics_end = s.dates.back() + 1; break;
        default: statistics_end = s.dates[rng.below(T)]; break;
    }
    s.h.config = golden::with_shared(s.h.config, "statistics_end", ccd::config::integer(statistics_end));

    // Some series have no thermal band to test, or band_range on two bands,
    // out of row order.
    if (rng.uniform() < 0.1)
        h.roles[6] = "other";
    if (rng.uniform() < 0.05)
        s.h.config = golden::with_steps(s.h.config, "band_range", "bands",
                                        ccd::config::roles({"swir2", "blue"}));

    return s;
}

std::string dump(const ccd::FitResult& r)
{
    std::string out;
    golden::dump_pixel(out, "px", r);
    return out;
}

// A cube pixel's dump, its mask as the cube holds it.
std::string dump(const ccd::CubeOutput& cube, const ccd::CubePixel& px)
{
    std::string out;
    golden::dump_pixel(out, "px", cube, px);
    return out;
}

// detect() under `config`.
ccd::FitResult run_detect(
    const PixelView& px,
    const ccd::Input& h,
    const ccd::config::Config& config = golden::pyccd()
)
{
    return ccd::detect(golden::spec_of(h, config), px.dates, px.spectral, px.qas).result;
}

} // namespace

//==============================================================================
// CcdcPlan
//==============================================================================

// The steps of each procedure, in order, and the transform on Standard only.
TEST(CcdcPlan, Shape)
{
    using K = pp::StepKind;
    const std::vector<K> common = {K::KeepQa, K::ThermalRange, K::BandRange, K::Dedupe};

    auto kinds = [](const pp::Chain& chain)
    {
        std::vector<K> out;
        for (const auto& step : chain)
            out.push_back(step.kind);
        return out;
    };

    const ccd::Input h;
    const ccd::config::Config config = golden::pyccd();
    const ccd::config::Shared shared = golden::shared_of(config, h);
    const pp::PreprocessPlan plan = golden::plan_of(config, h);

    EXPECT_EQ(kinds(plan.standard.chain), common);

    std::vector<K> snow = common;
    snow.insert(snow.end(), {K::AddQa, K::Dedupe, K::ThermalRange});
    EXPECT_EQ(kinds(plan.snow.chain), snow);

    std::vector<K> clear = common;
    clear.insert(clear.end(), {K::GreenMedian, K::Dedupe});
    EXPECT_EQ(kinds(plan.insufficient_clear.chain), clear);

    ASSERT_TRUE(plan.standard.transform.has_value());
    EXPECT_EQ(plan.standard.transform->band, h.index("thermal"));
    EXPECT_EQ(plan.standard.transform->scale, h.thermal_scale);
    EXPECT_EQ(plan.standard.transform->offset, h.thermal_offset);
    EXPECT_FALSE(plan.snow.transform.has_value());
    EXPECT_FALSE(plan.insufficient_clear.transform.has_value());

    const pp::Step& keep = plan.standard.chain[0];
    EXPECT_EQ(keep.codes, pp::qa_set({h.qa_clear, h.qa_water}));
    EXPECT_EQ(plan.snow.chain[4].codes, pp::qa_set({h.qa_snow}));

    const pp::Step& green = plan.insufficient_clear.chain[4];
    EXPECT_EQ(green.band, h.index("green"));
    EXPECT_EQ(green.green_offset, 400.0);
    EXPECT_EQ(green.max_date, shared.statistics_end);
    EXPECT_EQ(green.scope, ccd::GreenFilterScope::StatWindow);

    // The identity transform, and an input without thermal, have none; the
    // latter also has no ThermalRange step.
    ccd::Input identity = h;
    identity.thermal_scale = 1.0;
    identity.thermal_offset = 0.0;
    EXPECT_FALSE(golden::plan_of(config, identity).standard.transform.has_value());

    ccd::Input no_thermal = h;
    no_thermal.roles[6] = "other";
    const pp::PreprocessPlan nt = golden::plan_of(config, no_thermal);
    EXPECT_FALSE(nt.standard.transform.has_value());
    EXPECT_EQ(kinds(nt.standard.chain), (std::vector<K>{K::KeepQa, K::BandRange, K::Dedupe}));
}

namespace
{

// Every field of a plan, the reals as hex floats: equal for equal plans.
std::string fields(const pp::PreprocessPlan& plan)
{
    const auto hex = [](double v)
    {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%a", v);
        return std::string(buffer);
    };
    std::string out;
    for (const pp::ProcedurePlan* procedure : {&plan.standard, &plan.snow, &plan.insufficient_clear})
    {
        out += "procedure";
        if (procedure->transform)
            out += " transform " + std::to_string(procedure->transform->band) + " " +
                   hex(procedure->transform->scale) + " " + hex(procedure->transform->offset);
        out += "\n";
        for (const pp::Step& step : procedure->chain)
        {
            out += std::string(pp::name(step.kind)) + " codes " + step.codes.to_string().substr(240) +
                   " bands";
            for (const index_t b : step.bands)
                out += " " + std::to_string(b);
            out += " band " + std::to_string(step.band) + " range " + hex(step.range.lo.value) +
                   (step.range.lo.inclusive ? "i " : "x ") + hex(step.range.hi.value) +
                   (step.range.hi.inclusive ? "i" : "x") + " green " + hex(step.green_offset) +
                   " " + std::to_string(step.max_date) + " " + ccd::name(step.scope) + "\n";
        }
    }
    return out;
}

// The "ccdc" plan of `input` written out with the Step constructors: the
// thermal range on the thermal row when it has one, (therm_lo, therm_hi) in
// Standard and (stored_lo, stored_hi) in the other two (the snow chain's
// again at its end), the reflectance range (0, refl_hi) on its reflective rows, the green
// median `green` above, up to statistics_end 736694, Standard transformed by
// `transform`.
pp::PreprocessPlan ccdc_plan(const ccd::Input& input, double therm_lo, double therm_hi,
                             double stored_lo, double stored_hi, double refl_hi, double green,
                             std::optional<pp::Affine> transform)
{
    const auto thermal = [&](pp::Chain& out, double lo, double hi)
    {
        if (input.index("thermal") != ccd::NO_BAND)
            out.push_back(pp::Step::thermal_range(input.index("thermal"), {{lo, false}, {hi, false}}));
    };
    const auto chain = [&](double lo, double hi)
    {
        pp::Chain out;
        out.push_back(pp::Step::keep_qa(pp::qa_set({input.qa_clear, input.qa_water})));
        thermal(out, lo, hi);
        out.push_back(pp::Step::band_range(
            band_rows(ccd::config::Value(std::string("reflective")), input),
            {{0.0, false}, {refl_hi, false}}));
        out.push_back(pp::Step::dedupe());
        return out;
    };
    const pp::Chain common = chain(stored_lo, stored_hi);

    pp::PreprocessPlan plan;
    plan.standard.transform = transform;
    plan.standard.chain = chain(therm_lo, therm_hi);
    plan.snow.chain = common;
    plan.snow.chain.push_back(pp::Step::add_qa(pp::qa_set({input.qa_snow})));
    plan.snow.chain.push_back(pp::Step::dedupe());
    thermal(plan.snow.chain, stored_lo, stored_hi);
    plan.insufficient_clear.chain = common;
    plan.insufficient_clear.chain.push_back(pp::Step::green_median(
        input.index("green"), green, golden::PYCCD_STATISTICS_END, ccd::GreenFilterScope::StatWindow));
    plan.insufficient_clear.chain.push_back(pp::Step::dedupe());
    return plan;
}

} // namespace

// The "ccdc" preprocessing resolved against an input, bit for bit: on
// Collection-1 DN -9320 < thermal < 7070 in Standard, after its thermal x 10
// - 27315, and 1799.5 < thermal < 3438.5 on the stored row of the other two
// (the same degrees Celsius), 0 < reflectance < 10000, the green median 400
// above; on the physical-unit input the same divided by 10000 (-0.932,
// 0.707, 1, 0.04) in every chain, no transform; without a thermal row, no
// thermal range and no transform. pyccd's plan but for the stored rows'
// thermal range, which pyccd tests against -9320 and 7070 untransformed.
TEST(CcdcPlan, IsWrittenOutStepByStep)
{
    const ccd::Input dn;
    EXPECT_EQ(fields(ccd::strategies::library_registry().resolve(golden::pyccd(), dn).plan),
              fields(ccdc_plan(dn, -9320.0, 7070.0, 1799.5, 3438.5, 10000.0, 400.0,
                               pp::Affine{6, 10.0, -27315.0})));

    const ccd::Input phys = golden::physical_input();
    EXPECT_EQ(fields(golden::plan_of(golden::pyccd(), phys)),
              fields(ccdc_plan(phys, -0.932, 0.707, -0.932, 0.707, 1.0, 0.04, std::nullopt)));

    ccd::Input no_thermal;
    no_thermal.roles[6] = "other";
    const pp::PreprocessPlan nt = golden::plan_of(golden::pyccd(), no_thermal);
    EXPECT_EQ(fields(nt),
              fields(ccdc_plan(no_thermal, 0.0, 0.0, 0.0, 0.0, 10000.0, 400.0, std::nullopt)));
    for (const pp::ProcedurePlan* procedure : {&nt.standard, &nt.snow, &nt.insufficient_clear})
        for (const pp::Step& step : procedure->chain)
            EXPECT_NE(step.kind, pp::StepKind::ThermalRange);
}

// Every fixture input, under its fixture's options and every scope and flag
// combination.
TEST(CcdcPlan, MatchesReferenceOnFixtureInputs)
{
    Tally tally;
    for (const Series& s : fixture_inputs())
    {
        const auto vars = variations(s.h);
        for (std::size_t p = 0; p < s.N; ++p)
        {
            const PixelView px = s.pixel(p);
            check_pixel(px, s.h, s.name + ":" + std::to_string(p), tally);
            for (const auto& h : vars)
                check_pixel(px, h,
                    s.name + ":" + std::to_string(p) + " " + variation_label(h),
                    tally);
        }
    }

    EXPECT_GT(tally.nonempty, tally.masks / 2);
    EXPECT_GT(tally.snow_differs, 0u);
    EXPECT_GT(tally.green_differs, 0u);
}

// Randomized series, DN and physical units, every scope and flag combination.
TEST(CcdcPlan, MatchesReferenceOnRandomSeries)
{
    Tally tally;
    golden::Rng rng(20261013);

    for (const Unit& unit : {DN, PHYSICAL})
    {
        for (int n = 0; n < 400; ++n)
        {
            const Series s = random_series(rng, unit);
            const PixelView px = s.pixel(0);
            for (const auto& h : variations(s.h))
            {
                check_pixel(px, h,
                    std::string(unit.name) + " series " + std::to_string(n) +
                    " " + variation_label(h),
                    tally);
            }
        }
    }

    EXPECT_GT(tally.nonempty, tally.masks / 2);
    EXPECT_GT(tally.snow_differs, 0u);
    EXPECT_GT(tally.green_differs, 0u);
}

// apply() resizes and resets the mask it is given.
TEST(CcdcPlan, ApplyResetsTheMask)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const PixelView px = view(p.dates, p.spectral.data(), 7, p.qas.data());
    const pp::PreprocessPlan plan = plan_of(Options{});

    ccd::ProcessingMask fresh;
    pp::apply(plan.snow.chain, px.dates, px.spectral, px.qas, fresh);

    ccd::ProcessingMask reused(3);
    pp::apply(plan.standard.chain, px.dates, px.spectral, px.qas, reused);
    pp::apply(plan.snow.chain, px.dates, px.spectral, px.qas, reused);

    EXPECT_EQ(reused.data(), fresh.data());
}

//==============================================================================
// CustomPlan
//==============================================================================

// The "ccdc" chains written out step by step, every parameter spelled out
// in physical terms, are the preset's: the same plan, and the same output
// through detect() and detect_cube().
TEST(CustomPlan, TheChainsWrittenOutAreThePresets)
{
    using ccd::config::flag;
    using ccd::config::names;
    using ccd::config::real;
    const ccd::config::Values thermal = {{"min", real(-93.2)}, {"max", real(70.7)},
                                         {"min_inclusive", flag(false)},
                                         {"max_inclusive", flag(false)}};
    const ccd::config::Values band = {{"bands", ccd::config::Value(std::string("reflective"))},
                                      {"min", real(0.0)}, {"max", real(1.0)},
                                      {"min_inclusive", flag(false)},
                                      {"max_inclusive", flag(false)}};
    const auto chain = [&](bool transform, std::vector<ccd::config::StepConfig> after)
    {
        ccd::config::ProcedureConfig procedure{transform,
                                               {{"keep_qa", {{"codes", names({"clear", "water"})}}},
                                                {"thermal_range", thermal},
                                                {"band_range", band},
                                                {"dedupe", {}}}};
        procedure.steps.insert(procedure.steps.end(), after.begin(), after.end());
        return procedure;
    };
    ccd::config::Config written = golden::pyccd();
    written.preprocess = {
        {"standard", chain(true, {})},
        {"snow", chain(false, {{"add_qa", {{"codes", names({"snow"})}}}, {"dedupe", {}},
                               {"thermal_range", thermal}})},
        {"insufficient_clear",
         chain(false, {{"green_median", {{"offset", real(0.04)},
                                         {"scope", ccd::config::Value(std::string("stat_window"))}}},
                       {"dedupe", {}}})},
    };

    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 4, 6);
    const ccd::Input h;
    const std::size_t T = c.T;
    EXPECT_EQ(golden::resolved(written, h).digest, golden::resolved(golden::pyccd(), h).digest);

    for (std::size_t p = 0; p < c.H * c.W; ++p)
    {
        const PixelView px = view(c.dates, c.spectral.data() + p * 7 * T, 7, c.qas.data() + p * T);
        EXPECT_EQ(dump(run_detect(px, h)), dump(run_detect(px, h, written))) << "pixel " << p;
    }

    auto cube = [&](const ccd::config::Config& config)
    {
        const ccd::CubeOutput r = ccd::detect_cube(
            golden::spec_of(h, config),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
            ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
            2);
        std::string out;
        for (const auto& px : r.pixels)
            out += dump(r, px);
        return out;
    };

    EXPECT_EQ(cube(golden::pyccd()), cube(written));
}

// A Standard chain that keeps snow: the default Standard mask never holds a
// snow observation; this one does, and the fit changes.
TEST(CustomPlan, StandardKeepsSnow)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const PixelView px = view(p.dates, p.spectral.data(), 7, p.qas.data());
    const ccd::Input h;

    const ccd::config::Config snowy_config = golden::with_step(
        golden::pyccd(), "standard", 0, "codes", ccd::config::names({"clear", "water", "snow"}));

    const ccd::FitResult base = run_detect(px, h);
    const ccd::FitResult snowy = run_detect(px, h, snowy_config);

    // Both are Standard fits.
    ASSERT_FALSE(base.models.empty());
    ASSERT_FALSE(snowy.models.empty());
    EXPECT_NE(base.models.front().curve_qa, ccd::CurveQA::PersistentSnow);
    EXPECT_NE(snowy.models.front().curve_qa, ccd::CurveQA::PersistentSnow);

    auto snow_kept = [&](const ccd::ProcessingMask& mask)
    {
        std::size_t n = 0;
        for (index_t t = 0; t < mask.size(); ++t)
            n += mask.test(t) && p.qas[t] == h.qa_snow;
        return n;
    };

    EXPECT_EQ(snow_kept(base.mask), 0u);
    EXPECT_GT(snow_kept(snowy.mask), 0u);
    EXPECT_NE(dump(base), dump(snowy));
}

// The thermal transform on every procedure. That is the "ccdc" chains run
// on an input whose thermal band the caller transformed beforehand, with the
// identity transform in the input -- for every procedure, Standard
// included. Thermal values of 1000 (Kelvin x 10: -173 degrees Celsius) at
// some clear observations fail the thermal range of every procedure, the
// transformed row's and, in the default PermanentSnow and InsufficientClear
// chains, the stored row's against the bounds taken back through the
// transform: the runs keep the same observations, pixel for pixel, and the
// two that transform everywhere are one run. (pyccd's untransformed test,
// -9320 < 1000 < 7070, keeps them.)
TEST(CustomPlan, TransformOnEveryProcedure)
{
    golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 10, 16);
    const std::size_t T = c.T;
    const std::size_t N = c.H * c.W;

    for (std::size_t p = 0; p < N; ++p)
    {
        std::size_t clear = 0;
        for (std::size_t t = 0; t < T; ++t)
        {
            if (c.qas[p * T + t] != 0)
                continue;
            if (clear++ % 5 == 0)
                c.spectral[p * 7 * T + 6 * T + t] = 1000.0;
        }
    }

    const ccd::Input h;

    ccd::config::Config everywhere = golden::pyccd();
    everywhere.preprocess.at("snow").thermal_transform = true;
    everywhere.preprocess.at("insufficient_clear").thermal_transform = true;

    // Kelvin x 10 integers times 10 less 27315: exact in double.
    std::vector<double> converted = c.spectral;
    for (std::size_t p = 0; p < N; ++p)
        for (std::size_t t = 0; t < T; ++t)
        {
            double& v = converted[p * 7 * T + 6 * T + t];
            v = v * h.thermal_scale + h.thermal_offset;
        }

    ccd::Input identity = h;
    identity.thermal_scale = 1.0;
    identity.thermal_offset = 0.0;

    // Pixels of the other procedures, and their clear observations of 1000
    // the mask drops.
    std::size_t other = 0;
    std::size_t dropped = 0;
    std::string per_pixel;

    for (std::size_t p = 0; p < N; ++p)
    {
        const PixelView raw = view(c.dates, c.spectral.data() + p * 7 * T, 7, c.qas.data() + p * T);
        const PixelView pre = view(c.dates, converted.data() + p * 7 * T, 7, c.qas.data() + p * T);

        const ccd::FitResult with_plan = run_detect(raw, h, everywhere);
        EXPECT_EQ(dump(with_plan), dump(run_detect(pre, identity))) << "pixel " << p;
        // The default keeps the same observations; it fits thermal as stored.
        EXPECT_EQ(with_plan.mask.data(), run_detect(raw, h).mask.data()) << "pixel " << p;
        per_pixel += dump(with_plan);

        if (!with_plan.models.empty() &&
            with_plan.models.front().curve_qa >= ccd::CurveQA::InsufficientClear)
        {
            ++other;
            for (std::size_t t = 0; t < T; ++t)
                dropped += c.qas[p * T + t] == 0 && c.spectral[p * 7 * T + 6 * T + t] == 1000.0 &&
                           !with_plan.mask[t];
        }
    }

    EXPECT_GT(other, 0u);
    EXPECT_GT(dropped, 0u);

    const ccd::CubeOutput cube = ccd::detect_cube(
        golden::spec_of(h, everywhere),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
        4);
    std::string from_cube;
    for (const auto& px : cube.pixels)
        from_cube += dump(cube, px);
    EXPECT_EQ(from_cube, per_pixel);
}

//==============================================================================
// PlanValidation
//==============================================================================

namespace
{

// validate() throws invalid_argument with `fragment` in its message.
void expect_rejected(const pp::PreprocessPlan& plan, index_t bands, const std::string& fragment)
{
    try
    {
        pp::validate(plan, bands);
        ADD_FAILURE() << "accepted; expected: " << fragment;
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

} // namespace

TEST(PlanValidation, CcdcPlanIsValid)
{
    EXPECT_NO_THROW(pp::validate(plan_of(Options{}), 7));
    EXPECT_NO_THROW(pp::validate(plan_of(physical_options()), 7));
    for (const auto& profile : golden::bands::profiles())
        EXPECT_NO_THROW(pp::validate(golden::plan_of(profile.config, profile.h), profile.h.bands()))
            << profile.name;
}

TEST(PlanValidation, BandIndexNotBelowBandCount)
{
    const Options h;
    const pp::Range r{{0.0, false}, {1.0, false}};

    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.snow.chain[6] = pp::Step::thermal_range(7, r);
        expect_rejected(plan, 7, "snow step 6 (ThermalRange): band 7");
    }
    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.standard.chain[2] = pp::Step::band_range({0, 1, 9}, r);
        expect_rejected(plan, 7, "standard step 2 (BandRange): band 9");
    }
    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.insufficient_clear.chain[4].band = 8;
        expect_rejected(plan, 7, "insufficient_clear step 4 (GreenMedian): band 8");
    }
    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.standard.transform->band = 7;
        expect_rejected(plan, 7, "standard transform: band 7");
    }
    // The plan is valid for 7 bands and not for 6: thermal is band 6.
    expect_rejected(plan_of(h), 6, "band 6 is not below the band count 6");
}

TEST(PlanValidation, GreenMedianWithoutBand)
{
    pp::PreprocessPlan plan = plan_of(Options{});
    plan.insufficient_clear.chain[4].band = ccd::NO_BAND;
    expect_rejected(plan, 7, "insufficient_clear step 4 (GreenMedian): needs a band");
}

TEST(PlanValidation, TransformBandInBandRange)
{
    pp::PreprocessPlan plan = plan_of(Options{});
    plan.snow.transform = pp::Affine{3, 2.0, 1.0};
    expect_rejected(plan, 7, "snow transform: band 3 is also tested by step 2 (BandRange)");

    // The same band outside every BandRange is accepted.
    plan.snow.transform = pp::Affine{6, 2.0, 1.0};
    EXPECT_NO_THROW(pp::validate(plan, 7));
}

TEST(PlanValidation, AddQaNotFollowedByDedupe)
{
    const Options options;
    const ccd::Input& h = options.input;
    {
        // AddQa last.
        pp::PreprocessPlan plan = plan_of(options);
        plan.standard.chain.push_back(pp::Step::add_qa(pp::qa_set({h.qa_snow})));
        expect_rejected(plan, 7, "standard step 4 (AddQa): must be followed by a Dedupe");
    }
    {
        // AddQa followed by another step.
        pp::PreprocessPlan plan = plan_of(options);
        plan.snow.chain[5] = pp::Step::keep_qa(pp::qa_set({h.qa_clear}));
        expect_rejected(plan, 7, "snow step 4 (AddQa): must be followed by a Dedupe");
    }
}

TEST(PlanValidation, ChainOverCapacity)
{
    pp::Chain chain;
    for (index_t i = 0; i < pp::CHAIN_CAPACITY; ++i)
        chain.push_back(pp::Step::dedupe());
    EXPECT_THROW(chain.push_back(pp::Step::dedupe()), std::invalid_argument);

    pp::PreprocessPlan plan = plan_of(Options{});
    plan.standard.chain.count = pp::CHAIN_CAPACITY + 1;
    expect_rejected(plan, 7, "standard chain: holds 9 steps; at most 8");
}

// A configuration whose plan validate() refuses (an add_qa without its
// dedupe) is refused when it is resolved, before any pixel: detect() and
// detect_cube() throw rather than landing in CubeOutput::failures.
TEST(PlanValidation, EntryPointsReject)
{
    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 2, 2);
    const std::size_t T = c.T;
    const ccd::Input h;

    ccd::config::Config bad = golden::pyccd();
    auto& snow_steps = bad.preprocess.at("snow").steps;
    snow_steps.erase(snow_steps.begin() + 5);   // the dedupe after add_qa

    const PixelView px = view(c.dates, c.spectral.data(), 7, c.qas.data());
    EXPECT_THROW(run_detect(px, h, bad), std::invalid_argument);

    EXPECT_THROW(
        ccd::detect_cube(
            golden::spec_of(h, bad),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
            ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
            1),
        std::invalid_argument);
}

//==============================================================================
// Microbench
//==============================================================================

// apply(ccdc chain), with the plan built once, against the hand-written
// chain, on the pyccd reference pixel. Prints ns per mask; asserts only that
// the masks agree.
TEST(Microbench, ApplyCcdcChain)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const PixelView px = view(p.dates, p.spectral.data(), 7, p.qas.data());
    const Options h;
    const pp::PreprocessPlan plan = plan_of(h);

    constexpr int REPS = 20000;
    using clock = std::chrono::steady_clock;

    auto time_ns = [&](auto&& build)
    {
        std::size_t sink = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int round = 0; round < 5; ++round)
        {
            const auto start = clock::now();
            for (int i = 0; i < REPS; ++i)
                sink += build().count();
            const auto stop = clock::now();
            best = std::min(best,
                std::chrono::duration<double, std::nano>(stop - start).count() / REPS);
        }
        EXPECT_GT(sink, 0u);
        return best;
    };

    struct Row
    {
        const char* name;
        double reference, chain;
    };

    const Row rows[] = {
        {"standard",
            time_ns([&] { return reference_standard(px, h); }),
            time_ns([&] { return run_chain(plan.standard.chain, px); })},
        {"snow",
            time_ns([&] { return reference_snow(px, h); }),
            time_ns([&] { return run_chain(plan.snow.chain, px); })},
        {"insufficient_clear",
            time_ns([&] { return reference_insufficient_clear(px, h); }),
            time_ns([&] { return run_chain(plan.insufficient_clear.chain, px); })},
    };

    std::cout << "[ microbench ] T = " << p.dates.size()
              << ", best of 5 x " << REPS << ", ns per mask\n";
    for (const Row& r : rows)
    {
        std::cout << "[ microbench ] " << r.name
                  << ": hand-written " << r.reference
                  << ", apply(ccdc chain) " << r.chain
                  << " (" << r.chain / r.reference << "x)\n";
    }

    EXPECT_EQ(run_chain(plan.standard.chain, px).data(), reference_standard(px, h).data());
}
