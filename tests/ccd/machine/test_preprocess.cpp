//==============================================================================
//
// The preprocessing plan (preprocess/preprocess.hpp).
//
//   DefaultPlan.*     default_plan() against the chains written out by hand
//                     from the filter kernels (reference_* below), mask for
//                     mask: on every fixture input (the reference pixels, the
//                     Landsat cubes of the golden fixture, the band profiles
//                     of the bands fixture) and on randomized series with NaN,
//                     values on and beside the bounds, repeated dates, snow
//                     and observations dated after stat_ord, in DN and in
//                     physical units, under every GreenFilterScope and every
//                     combination of the four inclusive flags. The
//                     filter::standard/snow/insufficientclear wrappers are
//                     held to the same references.
//   CustomPlan.*      plans that change what run_pixel() does: a Standard chain
//                     that keeps snow, and the thermal transform on every
//                     procedure; a null plan and an explicit default plan
//                     agree.
//   PlanValidation.*  every plan validate() rejects, and that run_pixel() and
//                     run_cube() reject them before any pixel runs.
//   Microbench.*      apply(default chain) against the hand-written chain on
//                     the 443-observation pyccd reference pixel. Reported,
//                     not gated.
//
//==============================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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

// An input and the shared values a run resolves for it: what default_plan()
// reads.
struct Options
{
    ccd::Input input;
    ccd::config::Shared shared = golden::shared_of();
};

// The physical-unit input's.
Options physical_options()
{
    const ccd::Input input = golden::physical_input();
    return {input, golden::shared_of(golden::pyccd(), input)};
}

pp::PreprocessPlan default_plan(const Options& options)
{
    return pp::default_plan(options.input, options.shared);
}

//------------------------------------------------------------------------------
// The three chains written out by hand from the filter kernels: the
// reference default_plan() is held to.
//------------------------------------------------------------------------------

ccd::ProcessingMask reference_standard(
    const PixelView& px,
    const Options& options
)
{
    ccd::ProcessingMask mask(px.dates.size());

    for (index_t i = 0; i < px.dates.size(); ++i)
    {
        const auto qa = px.qas(i);
        if (qa == options.input.qa_water || qa == options.input.qa_clear)
            mask.set(i, true);
    }

    if (options.input.index("thermal") != ccd::NO_BAND)
    {
        ccd::filter::apply_thermal_filter(
            px.spectral,
            options.input.index("thermal"),
            ccd::filter::thermal_range(options.input),
            mask
        );
    }

    ccd::filter::apply_saturation_filter(
        px.spectral,
        options.input.saturation_bands,
        ccd::filter::reflectance_range(options.input),
        mask
    );

    ccd::filter::remove_duplicate_dates(px.dates, mask);

    return mask;
}

ccd::ProcessingMask reference_snow(
    const PixelView& px,
    const Options& options
)
{
    ccd::ProcessingMask mask = reference_standard(px, options);

    for (index_t i = 0; i < px.qas.size(); ++i)
    {
        if (px.qas(i) == options.input.qa_snow)
            mask.set(i, true);
    }

    ccd::filter::remove_duplicate_dates(px.dates, mask);

    return mask;
}

ccd::ProcessingMask reference_insufficient_clear(
    const PixelView& px,
    const Options& options
)
{
    ccd::ProcessingMask mask = reference_standard(px, options);

    ccd::filter::apply_green_median_filter(
        px.dates,
        px.spectral,
        options.input.index("green"),
        options.shared.stat_ord,
        options.shared.green_filter,
        *ccd::green_filter_scope(options.shared.green_filter_scope),
        mask
    );

    ccd::filter::remove_duplicate_dates(px.dates, mask);

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
    const pp::PreprocessPlan plan = default_plan(options);
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

    const ccd::HarmonicWorkspace workspace(px.dates, px.spectral, px.qas, options.input,
                                           options.shared);
    EXPECT_EQ(ccd::filter::standard(workspace).data(), standard.data())
        << "filter::standard " << label;
    EXPECT_EQ(ccd::filter::snow(workspace).data(), snow.data())
        << "filter::snow " << label;
    EXPECT_EQ(ccd::filter::insufficientclear(workspace).data(), clear.data())
        << "filter::insufficientclear " << label;

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
            Options h = base;
            h.shared.green_filter_scope = scope;
            h.input.refl_min_inclusive  = flags & 1u;
            h.input.refl_max_inclusive  = flags & 2u;
            h.input.therm_min_inclusive = flags & 4u;
            h.input.therm_max_inclusive = flags & 8u;
            out.push_back(h);
        }
    }
    return out;
}

std::string variation_label(const Options& h)
{
    return "scope " + h.shared.green_filter_scope +
           " refl " + std::to_string(h.input.refl_min_inclusive) +
           std::to_string(h.input.refl_max_inclusive) +
           " therm " + std::to_string(h.input.therm_min_inclusive) +
           std::to_string(h.input.therm_max_inclusive);
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

        out.push_back(from_pixel("c2", read_csv("test002_c2.csv"), Options{golden::c2_input(), {}}));
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
                {profiles[pi].h, golden::shared_of(profiles[pi].config, profiles[pi].h)}});
        }
    }

    return out;
}

//------------------------------------------------------------------------------
// Randomized series
//
// Each value is drawn from the cases the filters distinguish: either bound
// itself, just inside and just outside each, mid-range, far outside, and
// NaN. Dates repeat, and stat_ord falls before, inside or after the series.
//------------------------------------------------------------------------------

struct Unit
{
    const char* name;
    scalar_t refl_lo, refl_hi;
    scalar_t therm_lo, therm_hi;
    scalar_t green_filter;
};

const Unit DN = {"dn", 0.0, 10000.0, ccd::MIN_CELSIUS, ccd::MAX_CELSIUS, 400.0};
const Unit PHYSICAL = {"phys", 0.0, 1.0,
    ccd::MIN_CELSIUS / 10000.0, ccd::MAX_CELSIUS / 10000.0, 0.04};

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
    h.refl_min = unit.refl_lo;
    h.refl_max = unit.refl_hi;
    h.therm_min = unit.therm_lo;
    h.therm_max = unit.therm_hi;
    s.h.shared.green_filter = unit.green_filter;

    // stat_ord before the series, inside it (often on an observation's
    // date), or after it.
    switch (rng.below(4))
    {
        case 0: s.h.shared.stat_ord = s.dates.front() - 1; break;
        case 1: s.h.shared.stat_ord = s.dates.back() + 1; break;
        default: s.h.shared.stat_ord = s.dates[rng.below(T)]; break;
    }

    // Some series have no thermal band to test, or no saturation bands.
    if (rng.uniform() < 0.1)
        h.roles[6] = "other";
    if (rng.uniform() < 0.05)
        h.saturation_bands = ccd::BandList{};

    return s;
}

std::string dump(const ccd::FitResult& r)
{
    std::string out;
    golden::dump_pixel(out, "px", r);
    return out;
}

// run_pixel() under "ccdc".
ccd::FitResult run_detect(
    const PixelView& px,
    const ccd::Input& h,
    const ccd::PreprocessPlanPtr& plan
)
{
    return ccd::run_pixel(golden::spec_of(h, golden::pyccd_algorithm(), plan),
                          px.dates, px.spectral, px.qas).result;
}

} // namespace

//==============================================================================
// DefaultPlan
//==============================================================================

// The steps of each procedure, in order, and the transform on Standard only.
TEST(DefaultPlan, Shape)
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
    const ccd::config::Shared shared;
    const pp::PreprocessPlan plan = pp::default_plan(h, shared);

    EXPECT_EQ(kinds(plan.standard.chain), common);

    std::vector<K> snow = common;
    snow.insert(snow.end(), {K::AddQa, K::Dedupe});
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
    EXPECT_EQ(green.green_offset, shared.green_filter);
    EXPECT_EQ(green.green_offset, 400.0);
    EXPECT_EQ(green.max_date, shared.stat_ord);
    EXPECT_EQ(green.scope, ccd::GreenFilterScope::StatWindow);

    // The identity transform, and an input without thermal, have none; the
    // latter also has no ThermalRange step.
    ccd::Input identity = h;
    identity.thermal_scale = 1.0;
    identity.thermal_offset = 0.0;
    EXPECT_FALSE(pp::default_plan(identity, shared).standard.transform.has_value());

    ccd::Input no_thermal = h;
    no_thermal.roles[6] = "other";
    const pp::PreprocessPlan nt = pp::default_plan(no_thermal, shared);
    EXPECT_FALSE(nt.standard.transform.has_value());
    EXPECT_EQ(kinds(nt.standard.chain), (std::vector<K>{K::KeepQa, K::BandRange, K::Dedupe}));
}

// Every fixture input, under its fixture's options and every scope and flag
// combination.
TEST(DefaultPlan, MatchesReferenceOnFixtureInputs)
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
TEST(DefaultPlan, MatchesReferenceOnRandomSeries)
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
TEST(DefaultPlan, ApplyResetsTheMask)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const PixelView px = view(p.dates, p.spectral.data(), 7, p.qas.data());
    const pp::PreprocessPlan plan = default_plan(Options{});

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

// A null plan and an explicit default plan give the same output, through
// run_pixel() and run_cube().
TEST(CustomPlan, NullIsTheDefaultPlan)
{
    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 4, 6);
    const ccd::Input h;
    const auto plan = std::make_shared<const pp::PreprocessPlan>(default_plan(Options{}));
    const std::size_t T = c.T;

    for (std::size_t p = 0; p < c.H * c.W; ++p)
    {
        const PixelView px = view(c.dates, c.spectral.data() + p * 7 * T, 7, c.qas.data() + p * T);
        EXPECT_EQ(dump(run_detect(px, h, nullptr)), dump(run_detect(px, h, plan)))
            << "pixel " << p;
    }

    auto cube = [&](const ccd::PreprocessPlanPtr& given)
    {
        const ccd::CubeResult r = ccd::run_cube(
            golden::spec_of(h, golden::pyccd_algorithm(), given),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
            ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
            2);
        std::string out;
        for (const auto& px : r.pixels)
            out += dump(px.result);
        return out;
    };

    EXPECT_EQ(cube(nullptr), cube(plan));
}

// A Standard chain that keeps snow: the default Standard mask never holds a
// snow observation; this one does, and the fit changes.
TEST(CustomPlan, StandardKeepsSnow)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const PixelView px = view(p.dates, p.spectral.data(), 7, p.qas.data());
    const ccd::Input h;

    pp::PreprocessPlan custom = default_plan(Options{});
    custom.standard.chain[0].codes.set(h.qa_snow);
    const auto plan = std::make_shared<const pp::PreprocessPlan>(custom);

    const ccd::FitResult base = run_detect(px, h, nullptr);
    const ccd::FitResult snowy = run_detect(px, h, plan);

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

// The thermal transform on every procedure. That is the default plan run
// on an input whose thermal band the caller transformed beforehand, with the
// identity transform in the input -- for every procedure, Standard
// included. Thermal values of 1000 (Kelvin x 10) at some clear observations
// pass the untransformed test of the default PermanentSnow and
// InsufficientClear chains, and fail once transformed, so those pixels
// change.
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

    pp::PreprocessPlan everywhere = default_plan(Options{});
    everywhere.snow.transform = everywhere.standard.transform;
    everywhere.insufficient_clear.transform = everywhere.standard.transform;
    const auto plan = std::make_shared<const pp::PreprocessPlan>(everywhere);

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

    // Pixels whose processing mask the plan changed, by procedure.
    std::size_t standard_changed = 0;
    std::size_t other_changed = 0;
    std::string per_pixel;

    for (std::size_t p = 0; p < N; ++p)
    {
        const PixelView raw = view(c.dates, c.spectral.data() + p * 7 * T, 7, c.qas.data() + p * T);
        const PixelView pre = view(c.dates, converted.data() + p * 7 * T, 7, c.qas.data() + p * T);

        const ccd::FitResult with_plan = run_detect(raw, h, plan);
        EXPECT_EQ(dump(with_plan), dump(run_detect(pre, identity, nullptr))) << "pixel " << p;
        per_pixel += dump(with_plan);

        const ccd::FitResult base = run_detect(raw, h, nullptr);
        if (base.mask.data() != with_plan.mask.data())
        {
            const bool standard =
                base.models.empty() ||
                base.models.front().curve_qa < ccd::CurveQA::InsufficientClear;
            ++(standard ? standard_changed : other_changed);
        }
    }

    EXPECT_GT(other_changed, 0u);
    EXPECT_EQ(standard_changed, 0u);

    const ccd::CubeResult cube = ccd::run_cube(
        golden::spec_of(h, golden::pyccd_algorithm(), plan),
        ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
        4);
    std::string from_cube;
    for (const auto& px : cube.pixels)
        from_cube += dump(px.result);
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

TEST(PlanValidation, DefaultPlanIsValid)
{
    EXPECT_NO_THROW(pp::validate(default_plan(Options{}), 7));
    EXPECT_NO_THROW(pp::validate(default_plan(physical_options()), 7));
    for (const auto& profile : golden::bands::profiles())
        EXPECT_NO_THROW(pp::validate(
                            pp::default_plan(profile.h, golden::shared_of(profile.config, profile.h)),
                            profile.h.bands()))
            << profile.name;
}

TEST(PlanValidation, BandIndexNotBelowBandCount)
{
    const Options h;
    const pp::Range r{{0.0, false}, {1.0, false}};

    {
        pp::PreprocessPlan plan = default_plan(h);
        plan.snow.chain[1] = pp::Step::thermal_range(7, r);
        expect_rejected(plan, 7, "snow step 1 (ThermalRange): band 7");
    }
    {
        pp::PreprocessPlan plan = default_plan(h);
        plan.standard.chain[2] = pp::Step::band_range({0, 1, 9}, r);
        expect_rejected(plan, 7, "standard step 2 (BandRange): band 9");
    }
    {
        pp::PreprocessPlan plan = default_plan(h);
        plan.insufficient_clear.chain[4].band = 8;
        expect_rejected(plan, 7, "insufficient_clear step 4 (GreenMedian): band 8");
    }
    {
        pp::PreprocessPlan plan = default_plan(h);
        plan.standard.transform->band = 7;
        expect_rejected(plan, 7, "standard transform: band 7");
    }
    // The default plan is valid for 7 bands and not for 6: thermal is band 6.
    expect_rejected(default_plan(h), 6, "band 6 is not below the band count 6");
}

TEST(PlanValidation, GreenMedianWithoutBand)
{
    pp::PreprocessPlan plan = default_plan(Options{});
    plan.insufficient_clear.chain[4].band = ccd::NO_BAND;
    expect_rejected(plan, 7, "insufficient_clear step 4 (GreenMedian): needs a band");
}

TEST(PlanValidation, TransformBandInBandRange)
{
    pp::PreprocessPlan plan = default_plan(Options{});
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
        pp::PreprocessPlan plan = default_plan(options);
        plan.standard.chain.push_back(pp::Step::add_qa(pp::qa_set({h.qa_snow})));
        expect_rejected(plan, 7, "standard step 4 (AddQa): must be followed by a Dedupe");
    }
    {
        // AddQa followed by another step.
        pp::PreprocessPlan plan = default_plan(options);
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

    pp::PreprocessPlan plan = default_plan(Options{});
    plan.standard.chain.count = pp::CHAIN_CAPACITY + 1;
    expect_rejected(plan, 7, "standard chain: holds 9 steps; at most 8");
}

// run_pixel() and run_cube() run validate() once per call, before any pixel:
// an invalid plan throws rather than landing in CubeResult::failures.
TEST(PlanValidation, EntryPointsReject)
{
    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 2, 2);
    const std::size_t T = c.T;
    const ccd::Input h;

    pp::PreprocessPlan bad = default_plan(Options{});
    bad.insufficient_clear.chain[4].band = ccd::NO_BAND;
    const auto plan = std::make_shared<const pp::PreprocessPlan>(bad);

    const PixelView px = view(c.dates, c.spectral.data(), 7, c.qas.data());
    EXPECT_THROW(run_detect(px, h, plan), std::invalid_argument);

    EXPECT_THROW(
        ccd::run_cube(
            golden::spec_of(h, golden::pyccd_algorithm(), plan),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
            ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
            1),
        std::invalid_argument);
}

//==============================================================================
// Microbench
//==============================================================================

// apply(default chain), with the plan built once, against the hand-written
// chain and against the filter:: wrappers (which build the plan per call),
// on the pyccd reference pixel. Prints ns per mask; asserts only that the
// masks agree.
TEST(Microbench, ApplyDefaultChain)
{
    const Pixel p = read_csv("test_3657_3610_observations.csv");
    const PixelView px = view(p.dates, p.spectral.data(), 7, p.qas.data());
    const Options h;
    const pp::PreprocessPlan plan = default_plan(h);
    const ccd::HarmonicWorkspace workspace(px.dates, px.spectral, px.qas, h.input, h.shared);

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
        double reference, chain, wrapper;
    };

    const Row rows[] = {
        {"standard",
            time_ns([&] { return reference_standard(px, h); }),
            time_ns([&] { return run_chain(plan.standard.chain, px); }),
            time_ns([&] { return ccd::filter::standard(workspace); })},
        {"snow",
            time_ns([&] { return reference_snow(px, h); }),
            time_ns([&] { return run_chain(plan.snow.chain, px); }),
            time_ns([&] { return ccd::filter::snow(workspace); })},
        {"insufficient_clear",
            time_ns([&] { return reference_insufficient_clear(px, h); }),
            time_ns([&] { return run_chain(plan.insufficient_clear.chain, px); }),
            time_ns([&] { return ccd::filter::insufficientclear(workspace); })},
    };

    std::cout << "[ microbench ] T = " << p.dates.size()
              << ", best of 5 x " << REPS << ", ns per mask\n";
    for (const Row& r : rows)
    {
        std::cout << "[ microbench ] " << r.name
                  << ": hand-written " << r.reference
                  << ", apply(default chain) " << r.chain
                  << " (" << r.chain / r.reference << "x)"
                  << ", filter:: wrapper " << r.wrapper << "\n";
    }

    EXPECT_EQ(run_chain(plan.standard.chain, px).data(), reference_standard(px, h).data());
}
