//==============================================================================
//
// The unit-convention guard.
//
// HarmonicOptions and LassoOptions carry seven absolute thresholds that are
// really statements about what unit the caller's `spectral` array is in
// (THERMAL_SCALE, THERMAL_OFFSET, REFL_MIN, REFL_MAX, THERM_MIN, THERM_MAX,
// MEDIAN_GREEN_FILTER, plus LassoOptions::alpha and coef_floor). Their defaults
// are Collection-1 DN, and the regression tests in test_standard.cpp pin those.
//
// This file pins the OTHER end: that dividing every band of the input by
// 10,000 and dividing every one of those thresholds by the same 10,000 gives
// the identical fit, scaled. That is what makes the algorithm unit-agnostic
// rather than merely parameterised, and it is the property a future edit is
// most likely to break silently -- every failure mode here returns numbers,
// none of them raise.
//
// Five tests, in order of how hard the failure is to notice:
//
//   PhysicalUnitsAreEquivalent  the property itself
//   AlphaIsAUnitConstant        alpha left behind: 89 % of coefficients zeroed
//                               and a different segment count -- and it runs
//                               FASTER, so wall clock calls it an improvement
//   CoefFloorIsAUnitConstant    coef_floor left behind: right sparsity, right
//                               segments, coefficients wrong by 160 %. The one
//                               a nonzero-count check would miss.
//   BoundsAreUnitConstants      the two filter ranges, in the direction that
//                               bites -- see below, it is not the obvious one
//   GreenMedianFilterIsAUnitConstant   the fifth constant, which `detect` on
//                               this pixel cannot reach at all
//
// Input is the same pyccd reference pixel test_standard.cpp uses, so a failure
// here and a failure there point at the same 443 observations.
//
//------------------------------------------------------------------------------
// What the equivalence test does NOT guard, measured rather than assumed
//
// PhysicalUnitsAreEquivalent passing does not mean all nine constants were
// converted. Putting each one back to its DN value one at a time, with the rest
// of the preset correct:
//
//     alpha               = 1.0       -> 4 segments instead of 5     CAUGHT
//     coef_floor          = 1.0       -> coefficients out by 163 %   CAUGHT
//     REFL_MIN / REFL_MAX             -> no effect whatsoever
//     THERM_MIN / THERM_MAX           -> no effect whatsoever
//     MEDIAN_GREEN_FILTER = 400.0     -> no effect whatsoever
//
// The three bounds are invisible because the preset makes them NARROWER, so
// leaving them at the DN value only ever loosens a screen, and a screen that
// rejects nothing is indistinguishable from a correct screen on clean data.
// The direction that bites is the reverse one -- DN input with the physical
// bound -- and it is deafening: every observation falls outside the range, the
// mask empties, and the pixel returns 0 segments instead of 5. That is
// BoundsAreUnitConstants.
//
// MEDIAN_GREEN_FILTER is invisible for a different and more interesting reason:
// apply_green_median_filter is called only from filter::insufficientclear, and
// this pixel takes the Standard procedure, so the constant is never read. Even
// setting it to 0.0 -- which would drop every observation at or above the
// median green -- changes nothing through detect(). Its guard therefore has to
// drive ccd::InsufficientClear directly, the way test_clear.cpp does. That is
// GreenMedianFilterIsAUnitConstant, and without it this file would have looked
// like it covered five constants while covering four.
//
//==============================================================================

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/insufficient_clear.hpp"

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
    std::ifstream file(filename);

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

ccd::HarmonicOptions physical_harmonic()
{
    ccd::HarmonicOptions h;

    // Identity: the caller already applied the transform, and the identity
    // also stops detect() writing into the caller's buffer.
    h.THERMAL_SCALE = 1.0;
    h.THERMAL_OFFSET = 0.0;

    h.REFL_MIN = 0.0 / SCALE;
    h.REFL_MAX = 10000.0 / SCALE;

    h.THERM_MIN = ccd::MIN_CELSIUS / SCALE;
    h.THERM_MAX = ccd::MAX_CELSIUS / SCALE;

    h.MEDIAN_GREEN_FILTER = 400.0 / SCALE;

    return h;
}

ccd::LassoOptions physical_lasso()
{
    ccd::LassoOptions lo;

    lo.alpha = 1.0 / SCALE;
    lo.coef_floor = 1.0 / SCALE;

    return lo;
}

ccd::FitResult run(
    const Observations& obs,
    std::vector<ccd::scalar_t> spectral,      // by value: detect() may write
    const ccd::HarmonicOptions& h,
    const ccd::LassoOptions& lo
)
{
    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        obs.dates.data(), {obs.T});

    auto spec = ccd::ArrayView<ccd::scalar_t, 2>::contiguous(
        spectral.data(), {7, obs.T});

    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(
        obs.qas.data(), {obs.T});

    return ccd::detect(dates, spec, qas, h, lo);
}

//------------------------------------------------------------------------------
// Drive ccd::InsufficientClear directly, as test_clear.cpp does.
//
// Needed because MEDIAN_GREEN_FILTER is unreachable through detect() on this
// pixel: apply_green_median_filter is called from filter::insufficientclear
// only, and detect() sends this pixel to the Standard procedure.
//
// This procedure applies NO thermal transform -- that is Standard-only, both
// here and in pyccd -- so the exact scale relation for it is a plain divisor on
// all seven bands, thermal included and left in K x 10. That is what
// plain_divide() builds, and it is deliberately NOT the same input as
// to_physical(): the unit a band arrives in is a property of the procedure that
// reads it, which is the whole reason these are options.
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
    ccd::HarmonicOptions h,                       // by value: the workspace
                                                  // binds a non-const reference
    const ccd::LassoOptions& lo
)
{
    auto dates = ccd::ArrayView<const std::int64_t, 1>::contiguous(
        obs.dates.data(), {obs.T});

    auto spec = ccd::ArrayView<const ccd::scalar_t, 2>::contiguous(
        spectral.data(), {7, obs.T});

    auto qas = ccd::ArrayView<const std::uint8_t, 1>::contiguous(
        obs.qas.data(), {obs.T});

    ccd::HarmonicWorkspace hworkspace(dates, spec, qas, h);

    ccd::LassoSolver solver(lo);
    ccd::LassoWorkspace lworkspace(obs.T);

    ccd::InsufficientClear procedure;

    return procedure.run(hworkspace, lworkspace, solver);
}

std::size_t count_nonzero(const ccd::FitResult& fit)
{
    std::size_t n = 0;

    for (const auto& model : fit.models)
        for (const auto& band : model.bands)
            for (const auto& c : band.model.coefficients())
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

            compare(d.model.intercept(), p.model.intercept() * SCALE);
            compare(d.score.rmse, p.score.rmse * SCALE);
            compare(d.score.magn, p.score.magn * SCALE);

            const auto& dc = d.model.coefficients();
            const auto& pc = p.model.coefficients();

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
        ccd::HarmonicOptions{}, ccd::LassoOptions{}
    );

    const auto phys = run(
        obs, to_physical(obs.spectral, obs.T),
        physical_harmonic(), physical_lasso()
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
           " \"Input unit convention\" block in harmonic.hpp";
}

//------------------------------------------------------------------------------
// Negative control 1 -- the loud one, which is still not loud.
//------------------------------------------------------------------------------

TEST(Units, AlphaIsAUnitConstant)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(
        obs, obs.spectral,
        ccd::HarmonicOptions{}, ccd::LassoOptions{}
    );

    ccd::LassoOptions lo = physical_lasso();
    lo.alpha = 1.0;                       // the DN value, left behind

    const auto bad = run(
        obs, to_physical(obs.spectral, obs.T),
        physical_harmonic(), lo
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
        ccd::HarmonicOptions{}, ccd::LassoOptions{}
    );

    ccd::LassoOptions lo = physical_lasso();
    lo.coef_floor = 1.0;                  // the DN value, left behind

    const auto quiet = run(
        obs, to_physical(obs.spectral, obs.T),
        physical_harmonic(), lo
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
// Backwards from the other two on purpose. Leaving a bound at its DN value
// while feeding reflectance only widens the screen, and a screen that rejects
// nothing looks exactly like a correct one. So this asserts the opposite: the
// physical bound applied to DN input must reject everything.
//------------------------------------------------------------------------------

TEST(Units, BoundsAreUnitConstants)
{
    const auto obs = read_observations(OBSERVATIONS);

    const auto dn = run(
        obs, obs.spectral,
        ccd::HarmonicOptions{}, ccd::LassoOptions{}
    );

    ASSERT_GT(dn.models.size(), 0u);

    {
        ccd::HarmonicOptions h;                 // DN input...
        h.REFL_MIN = 0.0 / SCALE;               // ...physical saturation bound
        h.REFL_MAX = 10000.0 / SCALE;

        const auto out = run(obs, obs.spectral, h, ccd::LassoOptions{});

        EXPECT_EQ(out.models.size(), 0u)
            << "reflectance bounds against DN input left "
            << out.models.size() << " segments; every observation is supposed"
               " to be screened out, which is what makes REFL_MIN/REFL_MAX"
               " observable at all";
    }

    {
        ccd::HarmonicOptions h;                 // DN input...
        h.THERM_MIN = ccd::MIN_CELSIUS / SCALE; // ...physical thermal bound
        h.THERM_MAX = ccd::MAX_CELSIUS / SCALE;

        const auto out = run(obs, obs.spectral, h, ccd::LassoOptions{});

        EXPECT_EQ(out.models.size(), 0u)
            << "thermal bounds against DN input left "
            << out.models.size() << " segments";
    }
}

//------------------------------------------------------------------------------
// Negative control 4 -- the constant detect() cannot see.
//
// Through the InsufficientClear procedure, because that is the only caller of
// apply_green_median_filter. Two assertions: the plain-divisor equivalence holds
// for this procedure too, and MEDIAN_GREEN_FILTER left at 400.0 breaks it.
//------------------------------------------------------------------------------

TEST(Units, GreenMedianFilterIsAUnitConstant)
{
    const auto obs = read_observations(OBSERVATIONS);
    const auto scaled = plain_divide(obs.spectral);

    const auto dn = run_insufficient_clear(
        obs, obs.spectral,
        ccd::HarmonicOptions{}, ccd::LassoOptions{}
    );

    ccd::HarmonicOptions h = physical_harmonic();
    h.THERM_MIN = ccd::MIN_CELSIUS / SCALE;     // thermal is K x 10 / SCALE
    h.THERM_MAX = ccd::MAX_CELSIUS / SCALE;     // here, not Celsius -- see
                                                // plain_divide's comment

    const auto phys = run_insufficient_clear(obs, scaled, h, physical_lasso());

    ASSERT_EQ(dn.models.size(), phys.models.size());
    ASSERT_GT(dn.models.size(), 0u);

    EXPECT_EQ(count_nonzero(dn), count_nonzero(phys));
    EXPECT_LT(worst_relative(dn, phys), 1e-9);

    // Now the constant on its own.
    ccd::HarmonicOptions bad = h;
    bad.MEDIAN_GREEN_FILTER = 400.0;            // the DN value, left behind

    const auto out = run_insufficient_clear(
        obs, scaled, bad, physical_lasso()
    );

    // Compared against the DN fit, not against `phys` -- worst_relative()
    // scales its second argument up by SCALE, so both arguments have to be the
    // (DN, preset) pair it was written for.
    const bool moved =
        out.models.size() != dn.models.size() ||
        worst_relative(dn, out) > 1e-9 ||
        count_nonzero(out) != count_nonzero(dn);

    EXPECT_TRUE(moved)
        << "MEDIAN_GREEN_FILTER=400.0 at reflectance scale changed nothing"
           " even through the procedure that reads it. If that is now true,"
           " this constant has no guard anywhere -- say so in harmonic.hpp"
           " rather than deleting the test";
}
