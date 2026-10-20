//==============================================================================
//
// The resume record (include/ccd/results/record.hpp): what run_pixel() with RunSpec::record writes.
//
//   * Record.*: the result is run_pixel()'s, bit for bit; the header names the
//     run; each pixel's suspension agrees with its output (the segments
//     written before it, the provisional one after it, the window at the
//     series end) and its excluded days are the filter mask less the
//     result's.
//   * OptionsDigest.*: equal options, equal digests; a change to any group
//     of fields, the plan or the QA table, a different one.
//   * InputFingerprint.*: what the fingerprint reads and what it leaves out.
//
// That a record lets a longer run continue where this one stopped is
// resume()'s test; here, only what the record says.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../golden/specs.hpp"

#include "ccd/ccd.hpp"
#include "ccd/strategies/model/lasso.hpp"

#include "characterisation.hpp"
#include "golden.hpp"

namespace
{

using ccd::index_t;
using Exit = ccd::standard::Exit;

ccd::ArrayView<const std::int64_t, 1> dates_of(const golden::CharPixel& p)
{
    return ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {p.dates.size()});
}

ccd::ArrayView<const double, 2> spectral_of(const golden::CharPixel& p)
{
    return ccd::ArrayView<const double, 2>::contiguous(p.spectral.data(), {7, p.dates.size()});
}

ccd::ArrayView<const std::uint8_t, 1> qas_of(const golden::CharPixel& p)
{
    return ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {p.qas.size()});
}

ccd::PixelRun recorded(const golden::CharPixel& p)
{
    return ccd::run_pixel(
        golden::spec_of(p.h, p.config, true),
        dates_of(p),
        spectral_of(p),
        qas_of(p));
}

ccd::FitResult detected(const golden::CharPixel& p)
{
    return ccd::run_pixel(golden::spec_of(p.h, p.config), dates_of(p), spectral_of(p), qas_of(p)).result;
}

std::string dump(const ccd::FitResult& r)
{
    std::string out;
    golden::dump_pixel(out, "pixel", r);
    return out;
}

// The dates of the observations a mask keeps, in order: masked coordinates.
std::vector<std::int64_t> kept_dates(const golden::CharPixel& p, const ccd::ProcessingMask& mask)
{
    std::vector<std::int64_t> out;
    for (index_t t = 0; t < mask.size(); ++t)
        if (mask[t])
            out.push_back(p.dates[t]);
    return out;
}

// The leading segments no longer series can rewrite, read from the output
// as test_prefix.cpp reads them.
std::size_t final_count(const ccd::FitResult& r)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < r.models.size(); ++i)
        if (r.models[i].change_probability == 1.0)
            n = i + 1;
    if (n == 0 && !r.models.empty() && r.models[0].curve_qa == ccd::CurveQA::Start)
        n = 1;
    return n;
}

const golden::CharPixel& pixel(const std::vector<golden::CharPixel>& all, const std::string& name)
{
    for (const auto& p : all)
        if (p.name == name)
            return p;
    throw std::runtime_error("no characterization pixel " + name);
}

} // namespace

//------------------------------------------------------------------------------
// Record
//------------------------------------------------------------------------------

TEST(Record, TheResultIsDetects)
{
    for (const auto& p : golden::characterisation_pixels())
        EXPECT_EQ(dump(recorded(p).result), dump(detected(p))) << p.name;
}

TEST(Record, TheHeaderNamesTheRun)
{
    const auto all = golden::characterisation_pixels();
    const golden::CharPixel& p = all.front();
    const ccd::PixelRun r = recorded(p);

    EXPECT_EQ(r.header.format, ccd::RESUME_FORMAT);
    EXPECT_EQ(r.header.results_version, ccd::RESULTS_VERSION);
    EXPECT_EQ(r.header.series_end, p.dates.back());
    EXPECT_EQ(r.header.statistics_end, p.dates.back());
    const ccd::config::Resolved resolved = golden::resolved(p.config, p.h);
    EXPECT_EQ(r.header.options_digest, resolved.digest);
}

TEST(Record, EachExitAndItsSuspension)
{
    const auto all = golden::characterisation_pixels();
    const std::pair<const char*, Exit> expected[] = {
        {"standard_too_few", Exit::TooFewObservations},
        {"variogram_nan", Exit::VariogramNaN},
        {"standard_no_room", Exit::NoRoomToInitialize},
        {"constant_detection_band", Exit::InitializeExhausted},
        {"peek_guard_tail", Exit::PeekBeyondEnd},
        {"start_catch", Exit::SeriesEnd},
        {"series_end_after_outlier", Exit::SeriesEnd},
    };
    for (const auto& [name, exit] : expected)
        EXPECT_EQ(recorded(pixel(all, name)).record.standard.exit, exit) << name;

    // Not Standard: nothing for the machine to resume.
    const ccd::PixelRun snow = recorded(pixel(all, "select_snow_at_threshold"));
    EXPECT_EQ(snow.record.procedure, ccd::PixelProcedure::PermanentSnow);
    EXPECT_EQ(snow.record.standard.exit, Exit::None);
    EXPECT_TRUE(snow.record.excluded_days.empty());
}

TEST(Record, TheSuspensionAgreesWithTheOutput)
{
    for (const auto& p : golden::characterisation_pixels())
    {
        const ccd::PixelRun r = recorded(p);
        const ccd::standard::Suspended& s = r.record.standard;
        if (r.record.procedure != ccd::PixelProcedure::Standard)
            continue;

        // The segments written before the guard are the final ones.
        EXPECT_EQ(s.final_segments, final_count(r.result)) << p.name;

        if (s.exit != Exit::SeriesEnd)
        {
            EXPECT_FALSE(s.grow) << p.name;
            continue;
        }

        // The series end: one provisional segment after the final ones, over
        // the window, breaking at the observation after it; the fit window
        // starts with it and ends inside it.
        ASSERT_EQ(r.result.models.size(), s.final_segments + 1) << p.name;
        const ccd::ChangeModel& open = r.result.models[s.final_segments];
        const auto kept = kept_dates(p, r.result.mask);
        ASSERT_LT(s.window.stop, kept.size()) << p.name;
        EXPECT_EQ(open.start_day, kept[s.window.start]) << p.name;
        EXPECT_EQ(open.end_day, kept[s.window.stop - 1]) << p.name;
        EXPECT_EQ(open.break_day, kept[s.window.stop]) << p.name;
        EXPECT_GT(s.fit_stop, s.window.start) << p.name;
        EXPECT_LE(s.fit_stop, s.window.stop) << p.name;

        // The open segment's fits are those of its models: refitted from the
        // record (the fit window, its coefficient count), bit for bit.
        // The pixel as the Standard machine reads it: its plan's transform
        // applied.
        const ccd::config::Resolved resolved = golden::resolved(p.config, p.h);
        const ccd::config::Shared& shared = resolved.shared;
        const auto& plan = resolved.plan;
        const index_t T = p.dates.size();
        std::vector<double> block = p.spectral;
        if (plan.standard.transform)
        {
            const auto& a = *plan.standard.transform;
            for (index_t t = 0; t < T; ++t)
                block[a.band * T + t] = block[a.band * T + t] * a.scale + a.offset;
        }
        ccd::MaskedSeries series;
        const ccd::PixelSeries ws(
            dates_of(p), ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
            qas_of(p), p.h, shared);
        series.reset(ws, r.result.mask);
        // The model's solver, and its coefficient rule as configured.
        const ccd::Strategies strategies = golden::strategies_of(p.config);
        ccd::LassoWorkspace lasso(0);
        ccd::LassoSolver solver(
            static_cast<const ccd::strategies::Lasso&>(*strategies.model).solver_options());
        const auto model_count = [&](const char* name)
        {
            return static_cast<index_t>(
                golden::parameter<std::int64_t>(golden::resolved(p.config).stages, "model", name));
        };
        std::vector<ccd::LassoResult> fits;
        const ccd::Window fit(s.window.start, s.fit_stop);
        ccd::fit_bands(
            series.dates().slice(ccd::range(fit.start, fit.stop)),
            series.spectral().slice(ccd::all(), ccd::range(fit.start, fit.stop)),
            ccd::CoefficientRule{model_count("coefficient_min"), model_count("coefficient_mid"),
                                 model_count("coefficient_max"), model_count("num_obs_factor")}
                .count(fit.size()),
            ccd::KeepResiduals::No, lasso, solver, fits);
        for (std::size_t b = 0; b < fits.size(); ++b)
        {
            const ccd::BandFit refit = ccd::band_fit(fits[b]);
            EXPECT_EQ(refit.intercept, open.bands[b].intercept) << p.name << " band " << b;
            EXPECT_EQ(refit.coefficients, open.bands[b].coefficients) << p.name << " band " << b;
            EXPECT_EQ(refit.rmse, open.bands[b].rmse) << p.name << " band " << b;
        }
        EXPECT_EQ(s.magnitude.size(), open.bands.size()) << p.name;
        for (std::size_t b = 0; b < open.bands.size() && b < s.magnitude.size(); ++b)
            EXPECT_EQ(s.magnitude[b], open.bands[b].magnitude) << p.name << " band " << b;
        EXPECT_EQ(static_cast<int>(s.num_coef), static_cast<int>(open.curve_qa)) << p.name;
    }

    // prev_break is the last change's break, as a masked position.
    for (const auto& p : golden::characterisation_pixels())
    {
        const ccd::PixelRun r = recorded(p);
        const ccd::standard::Suspended& s = r.record.standard;
        if (r.record.procedure != ccd::PixelProcedure::Standard || s.final_segments == 0 ||
            r.result.models[s.final_segments - 1].change_probability != 1.0)
            continue;
        const auto kept = kept_dates(p, r.result.mask);
        ASSERT_LT(s.prev_break, kept.size()) << p.name;
        EXPECT_EQ(kept[s.prev_break], r.result.models[s.final_segments - 1].break_day) << p.name;
    }

    // After a grow the peek reached the last observation; after a drop it
    // fell short of it.
    const auto all = golden::characterisation_pixels();
    EXPECT_TRUE(recorded(pixel(all, "start_catch")).record.standard.grow);
    EXPECT_FALSE(recorded(pixel(all, "series_end_after_outlier")).record.standard.grow);
}

TEST(Record, TheExcludedDaysAreTheMachinesDrops)
{
    for (const auto& p : golden::characterisation_pixels())
    {
        const ccd::PixelRun r = recorded(p);
        if (r.record.procedure != ccd::PixelProcedure::Standard)
            continue;

        // The Standard filter chain's mask, as Run::pixel builds it.
        const ccd::config::Resolved resolved = golden::resolved(p.config, p.h);
        const ccd::config::Shared& shared = resolved.shared;
        const auto& plan = resolved.plan;
        const index_t T = p.dates.size();
        std::vector<double> block = p.spectral;
        if (plan.standard.transform)
        {
            const auto& a = *plan.standard.transform;
            for (index_t t = 0; t < T; ++t)
                block[a.band * T + t] = block[a.band * T + t] * a.scale + a.offset;
        }
        ccd::ProcessingMask filtered;
        ccd::preprocess::apply(
            plan.standard.chain,
            dates_of(p),
            ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
            qas_of(p),
            filtered);

        std::vector<std::int64_t> dropped;
        for (index_t t = 0; t < T; ++t)
            if (filtered[t] && !r.result.mask[t])
                dropped.push_back(p.dates[t]);
        EXPECT_EQ(r.record.excluded_days, dropped) << p.name;
    }

    // The pixels built to drop observations do.
    const auto all = golden::characterisation_pixels();
    EXPECT_FALSE(recorded(pixel(all, "series_end_after_outlier")).record.excluded_days.empty());
    EXPECT_FALSE(recorded(pixel(all, "lookback_near_break")).record.excluded_days.empty());
}

//------------------------------------------------------------------------------
// OptionsDigest
//------------------------------------------------------------------------------

namespace
{

std::uint64_t digest_of(
    const ccd::Input& h,
    const ccd::config::Config& config = golden::pyccd())
{
    return golden::resolved(config, h).digest;
}

} // namespace

TEST(OptionsDigest, EqualOptionsEqualDigests)
{
    EXPECT_EQ(digest_of(ccd::Input{}), digest_of(ccd::Input{}));
    EXPECT_EQ(digest_of(golden::physical_input()), digest_of(golden::physical_input()));
    // A configuration that leaves its defaults out is the same run.
    ccd::config::Config partial = golden::pyccd();
    partial.shared = {{"stat_ord", partial.shared.at("stat_ord")}};
    partial.stages["refit"].params.clear();
    EXPECT_EQ(digest_of(ccd::Input{}, partial), digest_of(ccd::Input{}));
}

// The configuration is in the digest: the preset's, and another's.
TEST(OptionsDigest, TheConfigurationMoves)
{
    const ccd::Input h;
    ccd::config::Config tree = golden::pyccd();
    const std::uint64_t base = golden::resolved(tree, h).digest;
    EXPECT_EQ(base, digest_of(h));
    tree.stages["lookback"].params["peek"] = ccd::config::integer(6);
    const std::uint64_t other = golden::resolved(tree, h).digest;
    EXPECT_NE(other, base);
    EXPECT_EQ(other, golden::resolved(tree, h).digest);
}

TEST(OptionsDigest, EveryGroupOfFieldsMoves)
{
    const std::uint64_t base = digest_of(ccd::Input{});

    // Every group of the input's fields.
    using Set = std::function<void(ccd::Input&)>;
    const std::pair<const char*, Set> input[] = {
        {"qa_snow", [](auto& h) { h.qa_snow = 7; }},
        {"qa_occlusion", [](auto& h) { h.qa_occlusion = 11; }},
        {"roles", [](auto& h) { h.roles[0] = "coastal"; }},
        {"reflectance_unit", [](auto& h) { h.reflectance_unit = 1.0; }},
        {"reflectance_offset", [](auto& h) { h.reflectance_offset = 0.5; }},
        {"saturated", [](auto& h) { h.saturated = 65535.0; }},
        {"saturated 20000", [](auto& h) { h.saturated = 20000.0; }},
        {"thermal_scale", [](auto& h) { h.thermal_scale = 1.0; }},
        {"thermal_offset", [](auto& h) { h.thermal_offset = -2731.0; }},
        {"thermal_unit", [](auto& h) { h.thermal_unit = 1.0; }},
    };
    for (const auto& [name, set] : input)
    {
        ccd::Input h;
        set(h);
        EXPECT_NE(digest_of(h), base) << name;
    }

    // Every shared value, to another it accepts.
    const ccd::config::Config preset = golden::pyccd();
    for (const auto& [name, value] : preset.shared)
    {
        const ccd::config::Value moved = std::visit(
            [](auto v) -> ccd::config::Value
            {
                using T = decltype(v);
                if constexpr (std::is_same_v<T, std::int64_t>)
                    return ccd::config::integer(v - 1);
                else if constexpr (std::is_same_v<T, double>)
                    return ccd::config::real(v * 0.5);
                else if constexpr (std::is_same_v<T, std::string>)
                    return std::string("all");
                else if constexpr (std::is_same_v<T, std::vector<std::string>>)
                {
                    v.pop_back();
                    return ccd::config::roles(v);
                }
                else
                    return v;
            },
            value);
        EXPECT_NE(digest_of(ccd::Input{}, golden::with_shared(preset, name, moved)), base) << name;
    }

    // Every parameter of every stage of the configuration, and every stage's
    // strategy name.
    std::size_t parameters = 0;
    for (const auto& [stage, sc] : preset.stages)
    {
        for (const auto& [name, value] : sc.params)
        {
            // The outlier rule's statistic has no other value: chi2 provides
            // one statistic only.
            if (stage == "outlier" && name == "statistic")
                continue;
            const ccd::config::Value moved = std::visit(
                [](auto v) -> ccd::config::Value
                {
                    using T = decltype(v);
                    if constexpr (std::is_same_v<T, bool>)
                        return ccd::config::flag(!v);
                    else if constexpr (std::is_same_v<T, std::int64_t>)
                        return ccd::config::integer(v + 1);
                    else if constexpr (std::is_same_v<T, double>)
                        return ccd::config::real(v + 0.5);
                    else if constexpr (std::is_same_v<T, std::string>)
                        return v + "-other";
                    else if constexpr (std::is_same_v<T, std::vector<std::string>>)
                    {
                        v.push_back("nir");
                        return ccd::config::roles(v);
                    }
                    else
                    {
                        v.push_back(5);
                        return v;
                    }
                },
                value);
            EXPECT_NE(digest_of(ccd::Input{}, golden::with(preset, stage, name, moved)), base)
                << stage << " " << name;
            ++parameters;
        }
        ccd::config::Config renamed = preset;
        renamed.stages[stage].strategy += "-other";
        EXPECT_NE(ccd::config::digest(renamed), ccd::config::digest(preset)) << stage;
    }
    EXPECT_GT(parameters, 15u);
}

TEST(OptionsDigest, ThePreprocessingAndTheQaTableMove)
{
    const ccd::Input h;
    const std::uint64_t base = digest_of(h);
    const ccd::config::Config preset = golden::pyccd();

    // Every parameter of every step of every procedure, and each procedure's
    // thermal transform.
    std::size_t parameters = 0;
    for (const auto& [procedure, chain] : preset.preprocess)
    {
        ccd::config::Config flipped = preset;
        flipped.preprocess[procedure].thermal_transform = !chain.thermal_transform;
        EXPECT_NE(digest_of(h, flipped), base) << procedure;
        for (std::size_t i = 0; i < chain.steps.size(); ++i)
            for (const auto& [name, value] : chain.steps[i].params)
            {
                if (name == "max_date")
                    continue;   // the shared stat_ord's, moved above
                const ccd::config::Value moved = std::visit(
                    [](auto v) -> ccd::config::Value
                    {
                        using T = decltype(v);
                        if constexpr (std::is_same_v<T, bool>)
                            return ccd::config::flag(!v);
                        else if constexpr (std::is_same_v<T, double>)
                            return ccd::config::real(v + 0.5);
                        else if constexpr (std::is_same_v<T, std::string>)
                        {
                            if (v == ccd::config::REFLECTIVE)
                                return ccd::config::roles({"blue", "green"});
                            return std::string(v == "all" ? "off" : "all");
                        }
                        else if constexpr (std::is_same_v<T, std::vector<std::string>>)
                        {
                            v.push_back("cloud");
                            return ccd::config::names(v);
                        }
                        else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
                        {
                            v.push_back(0);
                            return v;
                        }
                        else
                            return ccd::config::integer(v + 1);
                    },
                    value);
                EXPECT_NE(digest_of(h, golden::with_step(preset, procedure, i, name, moved)), base)
                    << procedure << " " << i << " " << name;
                ++parameters;
            }
        ccd::config::Config shorter = preset;
        shorter.preprocess[procedure].steps.erase(shorter.preprocess[procedure].steps.begin());
        EXPECT_NE(digest_of(h, shorter), base) << procedure;
    }
    EXPECT_GT(parameters, 20u);

    // The input's QA table, by its codes.
    const auto table = ccd::qa_tables::landsat_c2_qa_pixel(h);
    const std::uint64_t with_table = digest_of(golden::with_table(h, table));
    EXPECT_NE(with_table, base);
    EXPECT_EQ(digest_of(golden::with_table(h, ccd::qa_tables::landsat_c2_qa_pixel(h))),
              with_table);

    ccd::QaTable::Codes changed = table->code;
    changed[21824] = changed[21824] == 0 ? 1 : 0;
    EXPECT_NE(digest_of(golden::with_table(h, std::make_shared<const ccd::QaTable>(changed))),
              with_table);
}

// digest(Input) names every field of the input one by one, so a field added
// to it must be added to the digest too. Its size is the tripwire: on any
// LP64 platform this is the size the digest was written for, and a test that
// fails here is a reminder, not a platform difference. The configuration's
// part of the digest is of its values by name, preprocessing included, so no
// struct of the plan needs one.
TEST(OptionsDigest, TheInputIsTheOneItWasWrittenFor)
{
    EXPECT_EQ(sizeof(ccd::Input), 112u);
}

//------------------------------------------------------------------------------
// InputFingerprint
//------------------------------------------------------------------------------

namespace
{

std::uint64_t fingerprint(const golden::CharPixel& p, std::int64_t through)
{
    return ccd::input_fingerprint(dates_of(p), spectral_of(p), qas_of(p), p.h.qa_fill, through);
}

} // namespace

TEST(InputFingerprint, ReadsEveryKeptValueUpToTheDate)
{
    const auto all = golden::characterisation_pixels();
    const golden::CharPixel base = pixel(all, "start_catch");
    const std::int64_t through = base.dates[150];
    const std::uint64_t f = fingerprint(base, through);

    EXPECT_EQ(fingerprint(base, through), f);
    EXPECT_EQ(recorded(base).record.input_fingerprint, fingerprint(base, base.dates.back()));

    // One ulp of one value, one QA code, one date: different.
    {
        golden::CharPixel p = base;
        const index_t T = p.dates.size();
        p.spectral[3 * T + 40] = std::nextafter(p.spectral[3 * T + 40], 1e9);
        EXPECT_NE(fingerprint(p, through), f);
    }
    {
        golden::CharPixel p = base;
        p.qas[40] = p.h.qa_cloud;
        EXPECT_NE(fingerprint(p, through), f);
    }
    {
        golden::CharPixel p = base;
        p.dates[40] += 1;
        EXPECT_NE(fingerprint(p, through), f);
    }

    // After the date: the same.
    {
        golden::CharPixel p = base;
        const index_t T = p.dates.size();
        p.spectral[2 * T + 151] += 100.0;
        p.qas[160] = p.h.qa_cloud;
        EXPECT_EQ(fingerprint(p, through), f);
    }
}

TEST(InputFingerprint, LeavesOutFill)
{
    const auto all = golden::characterisation_pixels();
    golden::CharPixel base = pixel(all, "start_catch");
    const index_t T = base.dates.size();
    base.qas[40] = base.h.qa_fill;
    const std::int64_t through = base.dates[150];
    const std::uint64_t f = fingerprint(base, through);

    // A fill observation's values are not read.
    {
        golden::CharPixel p = base;
        for (index_t b = 0; b < 7; ++b)
            p.spectral[b * T + 40] = -9999.0;
        EXPECT_EQ(fingerprint(p, through), f);
    }

    // A fill observation added between two others (a scene that does not
    // cover the pixel) changes nothing.
    {
        golden::CharPixel p = base;
        const index_t at = 60;
        p.dates.insert(p.dates.begin() + at, p.dates[at - 1] + 1);
        p.qas.insert(p.qas.begin() + at, p.h.qa_fill);
        std::vector<double> spectral;
        for (index_t b = 0; b < 7; ++b)
        {
            spectral.insert(spectral.end(), base.spectral.begin() + b * T,
                            base.spectral.begin() + b * T + at);
            spectral.push_back(0.0);
            spectral.insert(spectral.end(), base.spectral.begin() + b * T + at,
                            base.spectral.begin() + (b + 1) * T);
        }
        p.spectral = spectral;
        EXPECT_EQ(fingerprint(p, through), f);
    }
}
