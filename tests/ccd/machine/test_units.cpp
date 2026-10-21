//==============================================================================
//
// The unit-convention guard.
//
// The Input carries the statements about what unit the caller's `spectral`
// array is in: reflectance_scale (what one unit of reflectance is in the
// data), thermal_scale and thermal_offset. The configuration's values are
// physical -- the shared alpha and coef_floor, band_range's bounds and
// green_median's offset in reflectance, multiplied by reflectance_scale;
// thermal_range's bounds in degrees Celsius, converted to the thermal row's
// unit -- when a run resolves it. The Input defaults are Collection-1 DN, and
// the regression tests in test_standard.cpp pin those.
//
// This file pins the OTHER end: that dividing every band of the input by
// 10,000, with reflectance_scale 1 and the identity thermal transform, gives
// the identical fit, scaled. That is what makes
// the algorithm unit-agnostic rather than merely parameterised, and it is
// the property a future edit is most likely to break silently -- every
// failure mode here returns numbers, none of them raise.
//
// Seven tests, the property twice and then in order of how hard the failure
// is to notice:
//
//   PhysicalUnitsAreEquivalent  the property itself
//   PhysicalUnitsAreEquivalentWithExclusiveBounds
//                               the property with observations on the
//                               reflectance bounds, which the default
//                               exclusive bounds drop and inclusive ones keep:
//                               the *_inclusive flags are not unit constants
//   ReflectanceUnitIsTheUnit    reflectance_scale left at 10000 on reflectance
//                               data: alpha and coef_floor at their DN values
//   AlphaIsAUnitConstant        alpha at its DN value: 89 % of coefficients
//                               zeroed and a different segment count -- and
//                               it runs FASTER, so wall clock calls it an
//                               improvement
//   CoefFloorIsAUnitConstant    coef_floor at its DN value: right sparsity,
//                               right segments, coefficients wrong by 160 %.
//                               The one a nonzero-count check would miss.
//   BoundsAreUnitConstants      the two filter ranges, in the direction that
//                               bites -- see below, it is not the obvious one
//   GreenMedianFilterIsAUnitConstant   the green filter, which `detect` on
//                               this pixel cannot reach at all
//
// and Collection2DnResolvesReflectanceZeroToOne, the input with an
// offset: reflectance levels take it, differences do not.
//
// Input is the same pyccd reference pixel test_standard.cpp uses, so a failure
// here and a failure there point at the same 443 observations.
//
//------------------------------------------------------------------------------
// What the equivalence test does NOT guard, measured rather than assumed
//
// PhysicalUnitsAreEquivalent passing does not mean every value was
// converted. Putting each one back to its DN value one at a time, with the rest
// of the physical run correct:
//
//     alpha               = 1.0       -> 4 segments instead of 5     CAUGHT
//     coef_floor          = 1.0       -> coefficients out by 163 %   CAUGHT
//     band_range bounds               -> no effect whatsoever
//     thermal_range bounds            -> no effect whatsoever
//     green_median offset = 400.0     -> no effect whatsoever
//
// The bounds are invisible because the physical run makes them NARROWER, so
// leaving them at the DN value only ever loosens a screen, and a screen that
// rejects nothing is indistinguishable from a correct screen on clean data.
// The direction that bites is the reverse one -- a DN run with the physical
// bound, the reflectance number reaching the data unconverted -- and it is
// deafening: every observation falls outside the range, the mask empties,
// and the pixel returns 0 segments instead of 5. That is
// BoundsAreUnitConstants.
//
// The green filter is invisible for a different and more interesting reason:
// "ccdc" runs the green median filter only in the InsufficientClear
// chain, and this pixel takes the Standard procedure, so the constant is never
// read. Even
// setting it to 0.0 -- which would drop every observation at or above the
// median green -- changes nothing through detect(). Its guard therefore has to
// fit the InsufficientClear procedure directly (ccd::fit_whole_series), the way
// test_clear.cpp does. That is
// GreenMedianFilterIsAUnitConstant, and without it this file would have looked
// like it covered every value while missing one.
//
//==============================================================================

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.hpp"

#include "ccd/ccd.hpp"
#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/results/segment.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"
#include "ccd/input/procedure.hpp"
#include "ccd/input/mask/filter.hpp"

namespace {

// The one divisor. Every band of the input and every absolute threshold.
constexpr ccd::scalar_t SCALE = 10000.0;

struct Observations
{
    std::vector<std::int64_t> dates;
    std::vector<std::uint8_t> qas;
    std::vector<ccd::scalar_t> spectral;   // (7, T) row-major, as numpy gives
    ccd::index_t T = 0;
};

Observations read_observations(const std::string& filename)
{
    std::ifstream file(std::string(CCD_TEST_DATA_DIR "/") + filename);

    if (!file)
        throw std::runtime_error("Unable to open " + filename);

    std::vector<std::int64_t> dates;
    std::vector<std::uint8_t> qas;
    std::array<std::vector<ccd::scalar_t>, 7> bands;

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        dates.push_back(std::stoll(value));

        for (ccd::index_t b = 0; b < 7; ++b)
        {
            std::getline(ss, value, ',');
            bands[b].push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        qas.push_back(static_cast<std::uint8_t>(std::stoi(value)));
    }

    Observations out;
    out.dates = dates;
    out.qas = qas;
    out.T = static_cast<ccd::index_t>(dates.size());
    out.spectral.resize(7 * out.T);

    for (ccd::index_t b = 0; b < 7; ++b)
        for (ccd::index_t t = 0; t < out.T; ++t)
            out.spectral[b * out.T + t] = bands[b][t];

    return out;
}

//------------------------------------------------------------------------------
// Collection-1 DN -> the preset's units.
//
// The thermal row applies the transform detect() would otherwise have applied
// itself, THEN divides by the same 10,000 as every other band. That second
// division is the whole trick: it puts thermal on the same scale factor as the
// optical bands, which is what lets one `alpha` serve all seven fits. Leaving
// thermal in plain Celsius -- the obvious choice -- makes it a 100x factor
// against the optical bands' 10,000x and breaks thermal alone.
//------------------------------------------------------------------------------

std::vector<ccd::scalar_t> to_physical(
    const std::vector<ccd::scalar_t>& dn,
    ccd::index_t T
)
{
    std::vector<ccd::scalar_t> out(dn.size());

    for (std::size_t i = 0; i < dn.size(); ++i)
        out[i] = dn[i] / SCALE;

    const ccd::index_t thermal = 6;

    for (ccd::index_t t = 0; t < T; ++t)
    {
        out[thermal * T + t] =
            (dn[thermal * T + t] * ccd::KELVIN_SCALE - ccd::KELVIN_OFFSET)
            / SCALE;
    }

    return out;
}

// The physical-unit input: reflectance its unit, the thermal transform the
// identity (the caller already applied it). The configuration's bounds
// resolve against it to the DN ones divided by SCALE.
ccd::Input physical_input()
{
    ccd::Input h;

    h.reflectance_scale = 1.0;

    h.thermal_scale = 1.0;
    h.thermal_offset = 0.0;
    h.thermal_at_100c = 1.0;

    return h;
}

// "ccdc" with every step of type `step` given bounds `lo` and `hi` and the
// inclusive flags `inclusive`.
ccd::config::Config with_bounds(const std::string& step, double lo, double hi,
                                bool inclusive = false)
{
    const char* range = step == "thermal_range" ? "celsius" : "reflectance";
    const ccd::config::Config c =
        golden::with_steps(golden::pyccd(), step, range, ccd::config::reals({lo, hi}));
    return golden::with_steps(c, step, "inclusive", ccd::config::flags({inclusive, inclusive}));
}

// "ccdc" with the shared alpha and coef_floor at `alpha` and `coef_floor`
// (in reflectance), which every Lasso stage reads.
ccd::config::Config with_lasso(double alpha, double coef_floor)
{
    return golden::with_shared(golden::with_shared(golden::pyccd(), "lasso.alpha",
                                                   ccd::config::real(alpha)),
                               "lasso.coef_floor", ccd::config::real(coef_floor));
}

// The kernels' solver with the same constants, for fit_whole_series().
ccd::LassoOptions physical_solver()
{
    ccd::LassoOptions lo;

    lo.alpha = 1.0 / SCALE;
    lo.coef_floor = 1.0 / SCALE;

    return lo;
}

ccd::FitResult run(
    const Observations& obs,
    std::vector<ccd::scalar_t> spectral,
    const ccd::Input& h,
    const ccd::config::Config& config = golden::pyccd()
)
{
    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        obs.dates.data(), {obs.T});

    auto spec = ccd::ArrayView<ccd::scalar_t, 2>::contiguous(
        spectral.data(), {7, obs.T});

    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(
        obs.qas.data(), {obs.T});

    return ccd::detect(golden::spec_of(h, config), dates, spec, qas).result;
}

//------------------------------------------------------------------------------
// Fit the InsufficientClear procedure directly (ccd::fit_whole_series), as
// test_clear.cpp does.
//
// Needed because the green filter is unreachable through detect() on this
// pixel: "ccdc" runs the green median filter only in the InsufficientClear
// chain, and detect() sends this pixel to the Standard procedure.
//
// This procedure applies NO thermal transform -- that is Standard-only, both
// here and in pyccd -- so the exact scale relation for it is a plain divisor on
// all seven bands, thermal included and left in K x 10. That is what
// plain_divide() builds, and it is deliberately NOT the same input as
// to_physical(): the unit a band arrives in is a property of the procedure that
// reads it, which is the whole reason these are the input's.
//------------------------------------------------------------------------------

std::vector<ccd::scalar_t> plain_divide(const std::vector<ccd::scalar_t>& dn)
{
    std::vector<ccd::scalar_t> out(dn.size());

    for (std::size_t i = 0; i < dn.size(); ++i)
        out[i] = dn[i] / SCALE;

    return out;
}

ccd::FitResult run_insufficient_clear(
    const Observations& obs,
    const std::vector<ccd::scalar_t>& spectral,   // const: nothing writes to it
    const ccd::Input& h,
    const ccd::config::Config& config,
    const ccd::LassoOptions& lo
)
{
    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        obs.dates.data(), {obs.T});

    auto spec = ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
        spectral.data(), {7, obs.T});

    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(
        obs.qas.data(), {obs.T});

    const ccd::config::Shared shared = golden::shared_of(config, h);
    ccd::PixelSeries series(dates, spec, qas);
    const ccd::config::ConfigTree tree = golden::resolved().stages;

    ccd::LassoSolver solver(lo);
    ccd::LassoWorkspace lworkspace(obs.T);

    return ccd::fit_whole_series(
        ccd::PixelProcedure::InsufficientClear,
        series,
        golden::chain_mask(series, h, "insufficient_clear", config),
        static_cast<ccd::index_t>(
            golden::parameter<std::int64_t>(tree, "edge_segments", "min_observations")),
        static_cast<ccd::index_t>(
            golden::parameter<std::int64_t>(tree, "edge_segments", "coefficients.min")),
        lworkspace,
        solver
    );
}

std::size_t count_nonzero(const ccd::FitResult& fit)
{
    std::size_t n = 0;

    for (const auto& model : fit.models)
        for (const auto& band : model.bands)
            for (const auto& c : band.coefficients)
                if (c != 0.0)
                    ++n;

    return n;
}

// Worst relative difference between a DN quantity and the same quantity fitted
// in preset units and multiplied back up. Relative, because the claim is a
// multiplicative one -- an absolute tolerance would pass trivially on the
// reflectance side. Values at roundoff level are skipped, since a relative
// measure is meaningless there.
double worst_relative(const ccd::FitResult& dn, const ccd::FitResult& phys)
{
    double worst = 0.0;

    const auto compare = [&worst](double a, double b_scaled) {
        if (std::abs(a) <= 1e-9)
            return;

        worst = std::max(worst, std::abs(a - b_scaled) / std::abs(a));
    };

    for (std::size_t m = 0; m < dn.models.size(); ++m)
    {
        for (std::size_t b = 0; b < 7; ++b)
        {
            const auto& d = dn.models[m].bands[b];
            const auto& p = phys.models[m].bands[b];

            compare(d.intercept, p.intercept * SCALE);
            compare(d.rmse, p.rmse * SCALE);
            compare(d.magnitude, p.magnitude * SCALE);

            const auto& dc = d.coefficients;
            const auto& pc = p.coefficients;

            for (std::size_t c = 0; c < dc.size(); ++c)
                compare(dc[c], pc[c] * SCALE);
        }
    }

    return worst;
}

const char* OBSERVATIONS = "test_3657_3610_observations.csv";

} // namespace


//------------------------------------------------------------------------------
// The property.
//------------------------------------------------------------------------------

TEST(Units, PhysicalUnitsAreEquivalent)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(
        obs, obs.spectral,
        ccd::Input{}
    );

    const auto phys = run(
        obs, to_physical(obs.spectral, obs.T),
        physical_input()
    );

    // Structure first: if the segmentation moved, comparing coefficients is
    // comparing different models and the tolerance below means nothing.
    ASSERT_EQ(dn.models.size(), phys.models.size());
    ASSERT_EQ(dn.mask.size(), phys.mask.size());

    for (ccd::index_t i = 0; i < static_cast<ccd::index_t>(dn.mask.size()); ++i)
        EXPECT_EQ(dn.mask[i], phys.mask[i]) << "processing mask differs at " << i;

    for (std::size_t m = 0; m < dn.models.size(); ++m)
    {
        EXPECT_EQ(dn.models[m].start_day, phys.models[m].start_day) << "model " << m;
        EXPECT_EQ(dn.models[m].end_day, phys.models[m].end_day) << "model " << m;
        EXPECT_EQ(dn.models[m].break_day, phys.models[m].break_day) << "model " << m;
        EXPECT_EQ(dn.models[m].observation_count,
                  phys.models[m].observation_count) << "model " << m;
        EXPECT_EQ(dn.models[m].curve_qa, phys.models[m].curve_qa) << "model " << m;
        EXPECT_DOUBLE_EQ(dn.models[m].change_probability,
                         phys.models[m].change_probability) << "model " << m;
    }

    // Sparsity next. Identical nonzero counts are what says the Lasso reached
    // the same solution rather than a more heavily regularised one.
    EXPECT_EQ(count_nonzero(dn), count_nonzero(phys));

    // Then the numbers. 1e-9 is far looser than the ~4e-13 this actually
    // achieves; the gap is deliberate headroom for compiler and platform
    // reassociation, since the two runs execute different arithmetic.
    const double worst = worst_relative(dn, phys);

    EXPECT_LT(worst, 1e-9)
        << "worst relative difference after x" << SCALE << " is " << worst
        << " -- an absolute threshold has been missed somewhere; see the"
           " guards in docs/units.md";
}

//------------------------------------------------------------------------------
// The property again, with the reflectance bounds exclusive and observations
// sitting exactly on them.
//
// The *_inclusive flags are not unit constants: 0 and 10000 in DN are 0.0 and
// 1.0 in reflectance, so an exclusive bound rejects the same observations in
// both units. Five clear observations are moved onto a bound first, so the
// flags have something to decide: under the default exclusive bounds the
// filter drops them, under inclusive ones it keeps them, and the DN and
// physical runs agree.
//------------------------------------------------------------------------------

TEST(Units, PhysicalUnitsAreEquivalentWithExclusiveBounds)
{
    auto obs = read_observations(OBSERVATIONS);
    const auto original = run(obs, obs.spectral, ccd::Input{});

    std::vector<ccd::index_t> moved;
    for (ccd::index_t t = 0; t < obs.T && moved.size() < 5; t += 37)
    {
        if (!original.mask[t])
            continue;
        // Alternate the band and the bound.
        const ccd::index_t band = moved.size() % 6;
        obs.spectral[band * obs.T + t] = moved.size() % 2 ? 10000.0 : 0.0;
        moved.push_back(t);
    }
    ASSERT_EQ(moved.size(), 5u);

    const ccd::Input dn_h;
    const ccd::Input phys_h = physical_input();
    const ccd::config::Config inclusive = with_bounds("band_range", 0.0, 1.0, true);

    const auto dn_inclusive = run(obs, obs.spectral, dn_h, inclusive);

    const auto dn = run(obs, obs.spectral, dn_h);
    const auto phys = run(obs, to_physical(obs.spectral, obs.T), phys_h);

    // The filter mask, before any fitting: the inclusive bound keeps the
    // moved observations, the exclusive one drops them, in both units.
    const auto filter_mask = [&obs](const std::vector<ccd::scalar_t>& spectral,
                                    const ccd::Input& h, const ccd::config::Config& config) {
        const ccd::config::Shared shared = golden::shared_of(config, h);
        const ccd::PixelSeries workspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(obs.dates.data(), {obs.T}),
            ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(spectral.data(), {7, obs.T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(obs.qas.data(), {obs.T}));
        return golden::chain_mask(workspace, h, "standard", config);
    };

    const auto physical = to_physical(obs.spectral, obs.T);
    const auto inclusive_mask = filter_mask(obs.spectral, dn_h, inclusive);
    const auto exclusive_mask = filter_mask(obs.spectral, dn_h, golden::pyccd());
    const auto physical_exclusive_mask = filter_mask(physical, phys_h, golden::pyccd());

    for (const auto t : moved)
    {
        EXPECT_TRUE(inclusive_mask.test(t)) << "inclusive bound dropped " << t;
        EXPECT_FALSE(exclusive_mask.test(t)) << "exclusive bound kept " << t;
        EXPECT_FALSE(dn.mask[t]) << "exclusive bound kept " << t;
    }

    for (ccd::index_t i = 0; i < obs.T; ++i)
        EXPECT_EQ(exclusive_mask.test(i), physical_exclusive_mask.test(i))
            << "filter mask differs at " << i;

    EXPECT_NE(dn_inclusive.mask.count(), dn.mask.count());

    ASSERT_EQ(dn.models.size(), phys.models.size());
    ASSERT_GT(dn.models.size(), 0u);
    ASSERT_EQ(dn.mask.size(), phys.mask.size());

    for (ccd::index_t i = 0; i < static_cast<ccd::index_t>(dn.mask.size()); ++i)
        EXPECT_EQ(dn.mask[i], phys.mask[i]) << "processing mask differs at " << i;

    for (std::size_t m = 0; m < dn.models.size(); ++m)
    {
        EXPECT_EQ(dn.models[m].start_day, phys.models[m].start_day) << "model " << m;
        EXPECT_EQ(dn.models[m].end_day, phys.models[m].end_day) << "model " << m;
        EXPECT_EQ(dn.models[m].break_day, phys.models[m].break_day) << "model " << m;
    }

    EXPECT_EQ(count_nonzero(dn), count_nonzero(phys));
    EXPECT_LT(worst_relative(dn, phys), 1e-9);
}

//------------------------------------------------------------------------------
// Negative control 0 -- the unit itself left behind: reflectance data
// described as DN, so alpha and coef_floor resolve to their DN values.
//------------------------------------------------------------------------------

TEST(Units, ReflectanceUnitIsTheUnit)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(obs, obs.spectral, ccd::Input{});

    ccd::Input h = physical_input();
    h.reflectance_scale = 10000.0;
    const auto bad = run(obs, to_physical(obs.spectral, obs.T), h);

    ASSERT_GT(count_nonzero(dn), 0u);
    EXPECT_LT(count_nonzero(bad) * 4, count_nonzero(dn));
    EXPECT_NE(dn.models.size(), bad.models.size());
}

//------------------------------------------------------------------------------
// Negative control 1 -- the loud one, which is still not loud.
//------------------------------------------------------------------------------

TEST(Units, AlphaIsAUnitConstant)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(
        obs, obs.spectral,
        ccd::Input{}
    );

    // alpha written at the DN value, which every Lasso stage reads.
    const ccd::config::Config config = with_lasso(1.0, 1.0 / SCALE);

    const auto bad = run(
        obs, to_physical(obs.spectral, obs.T),
        physical_input(), config
    );

    const std::size_t reference = count_nonzero(dn);
    const std::size_t got = count_nonzero(bad);

    ASSERT_GT(reference, 0u);

    // Most of the fit collapses -- but not all of it, which is why "check for
    // all zeros" is not a sufficient guard.
    EXPECT_LT(got * 4, reference)
        << "alpha=1.0 at reflectance scale kept " << got << " of "
        << reference << " nonzero coefficients; it is supposed to collapse";

    EXPECT_GT(got, 0u)
        << "if this ever becomes 0 the failure got LOUDER, which is fine --"
           " update the comment in LassoOptions::alpha";

    // And the segmentation moves, because collapsed fits change the residuals
    // that change detection thresholds against.
    EXPECT_NE(dn.models.size(), bad.models.size());
}

//------------------------------------------------------------------------------
// Negative control 2 -- the quiet one. This is the reason coef_floor exists.
//------------------------------------------------------------------------------

TEST(Units, CoefFloorIsAUnitConstant)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(
        obs, obs.spectral,
        ccd::Input{}
    );

    // coef_floor written at the DN value, which every Lasso stage reads.
    const ccd::config::Config config = with_lasso(1.0 / SCALE, 1.0);

    const auto quiet = run(
        obs, to_physical(obs.spectral, obs.T),
        physical_input(), config
    );

    ASSERT_EQ(dn.models.size(), quiet.models.size());

    // Every signal a cheap check would look at is unchanged...
    EXPECT_EQ(count_nonzero(dn), count_nonzero(quiet));

    for (std::size_t m = 0; m < dn.models.size(); ++m)
    {
        EXPECT_EQ(dn.models[m].curve_qa, quiet.models[m].curve_qa);
        EXPECT_EQ(dn.models[m].observation_count,
                  quiet.models[m].observation_count);
    }

    // ...and the coefficients are still wrong, by a lot.
    const double worst = worst_relative(dn, quiet);

    EXPECT_GT(worst, 1e-3)
        << "coef_floor=1.0 at reflectance scale produced a fit within "
        << worst << " relative of the DN fit. If the solver's stopping rule"
           " changed so this no longer matters, coef_floor can go -- but"
           " verify against Units.PhysicalUnitsAreEquivalent first";
}

//------------------------------------------------------------------------------
// Negative control 3 -- the two filter ranges, in the direction that bites.
//
// Backwards from the other two on purpose. A bound left at its DN value
// while feeding reflectance only widens the screen, and a screen that rejects
// nothing looks exactly like a correct one. So this asserts the opposite: the
// physical number reaching DN data unconverted must reject everything. The
// configuration writes each bound so that it resolves, on the DN input, to
// the physical number itself -- reflectance 0..1 / SCALE resolves to 0..1,
// thermal (-93.2, 70.7) degrees / SCALE to -0.932..0.707 -- which is what a
// resolution that skipped the conversion would hand the filter.
//------------------------------------------------------------------------------

TEST(Units, BoundsAreUnitConstants)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(
        obs, obs.spectral,
        ccd::Input{}
    );

    ASSERT_GT(dn.models.size(), 0u);

    {
        // DN input, the physical saturation bound.
        const auto out =
            run(obs, obs.spectral, ccd::Input{}, with_bounds("band_range", 0.0, 1.0 / SCALE));

        EXPECT_EQ(out.models.size(), 0u)
            << "reflectance bounds against DN input left "
            << out.models.size() << " segments; every observation is supposed"
               " to be screened out, which is what makes band_range's bounds"
               " observable at all";
    }

    {
        // DN input, the physical thermal bound.
        const auto out = run(obs, obs.spectral, ccd::Input{},
                             with_bounds("thermal_range", -93.2 / SCALE, 70.7 / SCALE));

        EXPECT_EQ(out.models.size(), 0u)
            << "thermal bounds against DN input left "
            << out.models.size() << " segments";
    }
}

//------------------------------------------------------------------------------
// Negative control 4 -- the constant detect() cannot see.
//
// Through the InsufficientClear procedure, because that is the only caller of
// apply_green_median. Two assertions: the plain-divisor equivalence holds
// for this procedure too, and the green filter at its DN value, 400.0, breaks
// it.
//------------------------------------------------------------------------------

// Collection-2 DN describes itself with a scale and an offset. Its surface
// reflectance is r = DN x 2.75e-5 - 0.2, so DN = r / 2.75e-5 + 0.2 /
// 2.75e-5: reflectance_scale 1 / 2.75e-5 (36363.63...) and
// reflectance_offset 0.2 / 2.75e-5 (7272.72...), each one double division.
// band_range's levels 0 and 1 resolve to 0 x unit + offset, the offset
// itself, and 1 x unit + offset, 43636.36...: on integer DN the range keeps
// 7273 to 43636, the values whose reflectance is in (0, 1) (DN 7273 is
// 7.5e-6, DN 43636 is 0.99999). The differences -- alpha, coef_floor, the
// green median's offset -- are reflectance_scale times their value, with no
// offset.
TEST(Units, Collection2DnResolvesReflectanceZeroToOne)
{
    ccd::Input h;
    h.reflectance_scale = 1.0 / 2.75e-5;
    h.reflectance_offset = 0.2 / 2.75e-5;

    const ccd::preprocess::PreprocessPlan plan = golden::plan_of(golden::pyccd(), h);
    const ccd::preprocess::Step& band = plan.standard.chain[2];
    ASSERT_EQ(band.kind, ccd::preprocess::StepKind::BandRange);
    EXPECT_EQ(band.range.lo.value, h.reflectance_offset);
    const double one = 1.0 * h.reflectance_scale;
    EXPECT_EQ(band.range.hi.value, one + h.reflectance_offset);
    EXPECT_GT(band.range.lo.value, 7272.0);
    EXPECT_LT(band.range.lo.value, 7273.0);
    EXPECT_GT(band.range.hi.value, 43636.0);
    EXPECT_LT(band.range.hi.value, 43637.0);
    for (const double dn : {7273.0, 43636.0})
        EXPECT_TRUE(ccd::filter::in_range(dn, band.range)) << dn;
    for (const double dn : {7272.0, 43637.0})
        EXPECT_FALSE(ccd::filter::in_range(dn, band.range)) << dn;

    // The differences carry no offset.
    EXPECT_EQ(plan.insufficient_clear.chain[4].green_offset, 0.04 * h.reflectance_scale);
    const ccd::config::Resolved r = golden::resolved(golden::pyccd(), h);
    EXPECT_EQ(r.shared.lasso.alpha, 1e-4 * h.reflectance_scale);
    EXPECT_EQ(r.shared.lasso.coef_floor, 1e-4 * h.reflectance_scale);
}

TEST(Units, GreenMedianFilterIsAUnitConstant)
{
    const auto obs = read_observations(OBSERVATIONS);
    const auto scaled = plain_divide(obs.spectral);

    const auto dn = run_insufficient_clear(
        obs, obs.spectral,
        ccd::Input{}, golden::pyccd(), ccd::LassoOptions{}
    );

    // Thermal is K x 10 / SCALE here, not Celsius -- see plain_divide's
    // comment -- inside the thermal bounds, -0.932..0.707, as K x 10 is
    // inside -9320..7070.
    const ccd::Input h = physical_input();
    ASSERT_EQ(golden::plan_of(golden::pyccd(), h).insufficient_clear.chain[4].green_offset, 0.04);

    const auto phys =
        run_insufficient_clear(obs, scaled, h, golden::pyccd(), physical_solver());

    ASSERT_EQ(dn.models.size(), phys.models.size());
    ASSERT_GT(dn.models.size(), 0u);

    EXPECT_EQ(count_nonzero(dn), count_nonzero(phys));
    EXPECT_LT(worst_relative(dn, phys), 1e-9);

    // Now the constant on its own.
    // The DN value, left behind.
    const ccd::config::Config bad =
        golden::with_steps(golden::pyccd(), "green_median", "offset", ccd::config::real(400.0));

    const auto out = run_insufficient_clear(
        obs, scaled, h, bad, physical_solver()
    );

    // Compared against the DN fit, not against `phys` -- worst_relative()
    // scales its second argument up by SCALE, so both arguments have to be the
    // (DN, preset) pair it was written for.
    const bool moved =
        out.models.size() != dn.models.size() ||
        worst_relative(dn, out) > 1e-9 ||
        count_nonzero(out) != count_nonzero(dn);

    EXPECT_TRUE(moved)
        << "green_median offset=400.0 at reflectance scale changed nothing"
           " even through the procedure that reads it. If that is now true,"
           " this value has no guard anywhere -- say so in docs/units.md"
           " rather than deleting the test";
}
