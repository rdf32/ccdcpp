//==============================================================================
//
// The resume record (include/ccd/record.hpp): what detect_recorded() writes.
//
//   * Record.*: the result is detect()'s, bit for bit; the header names the
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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

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

ccd::Recorded recorded(const golden::CharPixel& p)
{
    return ccd::detect_recorded(dates_of(p), spectral_of(p), qas_of(p), p.h, p.l);
}

ccd::FitResult detected(const golden::CharPixel& p)
{
    return ccd::detect(dates_of(p), spectral_of(p), qas_of(p), p.h, p.l);
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
    const ccd::Recorded r = recorded(p);

    EXPECT_EQ(r.header.format, ccd::RESUME_FORMAT);
    EXPECT_EQ(r.header.results_version, ccd::RESULTS_VERSION);
    EXPECT_EQ(r.header.series_end, p.dates.back());
    EXPECT_EQ(r.header.statistics_end, p.dates.back());
    EXPECT_EQ(r.header.options_digest,
              ccd::options_digest(p.h, p.l, ccd::preprocess::default_plan(p.h), nullptr));
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
    const ccd::Recorded snow = recorded(pixel(all, "select_snow_at_threshold"));
    EXPECT_EQ(snow.record.procedure, ccd::PixelProcedure::PermanentSnow);
    EXPECT_EQ(snow.record.standard.exit, Exit::None);
    EXPECT_TRUE(snow.record.excluded_days.empty());
}

TEST(Record, TheSuspensionAgreesWithTheOutput)
{
    for (const auto& p : golden::characterisation_pixels())
    {
        const ccd::Recorded r = recorded(p);
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
        const auto plan = ccd::preprocess::default_plan(p.h);
        const index_t T = p.dates.size();
        std::vector<double> block = p.spectral;
        if (plan.standard.transform)
        {
            const auto& a = *plan.standard.transform;
            for (index_t t = 0; t < T; ++t)
                block[a.band * T + t] = block[a.band * T + t] * a.scale + a.offset;
        }
        ccd::MaskedSeries series;
        const ccd::HarmonicWorkspace ws(
            dates_of(p), ccd::ArrayView<const double, 2>::contiguous(block.data(), {7, T}),
            qas_of(p), p.h);
        series.reset(ws, r.result.mask);
        ccd::LassoWorkspace lasso(0);
        ccd::LassoSolver solver(p.l);
        std::vector<ccd::LassoResult> fits;
        const ccd::Window fit(s.window.start, s.fit_stop);
        ccd::fit_bands(
            series.dates().slice(ccd::range(fit.start, fit.stop)),
            series.spectral().slice(ccd::all(), ccd::range(fit.start, fit.stop)),
            ccd::standard::coefficient_count(fit.size(), p.h),
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
        const ccd::Recorded r = recorded(p);
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
        const ccd::Recorded r = recorded(p);
        if (r.record.procedure != ccd::PixelProcedure::Standard)
            continue;

        // The Standard filter chain's mask, as detect_pixel builds it.
        const auto plan = ccd::preprocess::default_plan(p.h);
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
    const ccd::HarmonicOptions& h,
    const ccd::LassoOptions& l = ccd::LassoOptions{},
    const ccd::QaTable* table = nullptr)
{
    return ccd::options_digest(h, l, ccd::preprocess::default_plan(h), table);
}

} // namespace

TEST(OptionsDigest, EqualOptionsEqualDigests)
{
    EXPECT_EQ(digest_of(ccd::HarmonicOptions{}), digest_of(ccd::HarmonicOptions{}));
    EXPECT_EQ(digest_of(golden::physical_harmonic(), golden::physical_lasso()),
              digest_of(golden::physical_harmonic(), golden::physical_lasso()));
}

TEST(OptionsDigest, EveryGroupOfFieldsMoves)
{
    const std::uint64_t base = digest_of(ccd::HarmonicOptions{});

    using Set = std::function<void(ccd::HarmonicOptions&)>;
    const std::pair<const char*, Set> harmonic[] = {
        {"MEOW_SIZE", [](auto& h) { h.MEOW_SIZE = 13; }},
        {"PEEK_SIZE", [](auto& h) { h.PEEK_SIZE = 7; }},
        {"DAY_DELTA", [](auto& h) { h.DAY_DELTA = 366; }},
        {"QA_SNOW", [](auto& h) { h.QA_SNOW = 7; }},
        {"QA_OCCLUSION", [](auto& h) { h.QA_OCCLUSION = 11; }},
        {"CLEAR_PCT_THRESHOLD", [](auto& h) { h.CLEAR_PCT_THRESHOLD = 0.26; }},
        {"OUTLIER_THRESHOLD", [](auto& h) { h.OUTLIER_THRESHOLD = 36.0; }},
        {"T_CONST", [](auto& h) { h.T_CONST = 4.9; }},
        {"VARIOGRAM_MIN_LAG_DAYS", [](auto& h) { h.VARIOGRAM_MIN_LAG_DAYS = 31; }},
        {"RETRAIN_MIN_OBS", [](auto& h) { h.RETRAIN_MIN_OBS = 25; }},
        {"SEASONAL_RMSE_DIVISOR", [](auto& h) { h.SEASONAL_RMSE_DIVISOR = 4.5; }},
        {"THERMAL_OFFSET", [](auto& h) { h.THERMAL_OFFSET = -2731.0; }},
        {"REFL_MAX_INCLUSIVE", [](auto& h) { h.REFL_MAX_INCLUSIVE = true; }},
        {"THERM_MIN", [](auto& h) { h.THERM_MIN = -9000.0; }},
        {"GREEN_FILTER_SCOPE", [](auto& h) { h.GREEN_FILTER_SCOPE = ccd::GreenFilterScope::All; }},
        {"STAT_ORD", [](auto& h) { h.STAT_ORD = 736695; }},
        {"COEFFICIENT_MID", [](auto& h) { h.COEFFICIENT_MID = 7; }},
        {"NUM_OBS_FACTOR", [](auto& h) { h.NUM_OBS_FACTOR = 4; }},
        {"DETECTION_BANDS", [](auto& h) { h.DETECTION_BANDS = {1, 2, 3, 4}; }},
        {"TMASK_BANDS", [](auto& h) { h.TMASK_BANDS = {1, 5}; }},
        {"SATURATION_BANDS", [](auto& h) { h.SATURATION_BANDS = {0, 1, 2, 3, 4}; }},
        {"QA_IDX", [](auto& h) { h.QA_IDX = 8; }},
    };
    for (const auto& [name, set] : harmonic)
    {
        ccd::HarmonicOptions h;
        set(h);
        EXPECT_NE(digest_of(h), base) << name;
    }

    using SetLasso = std::function<void(ccd::LassoOptions&)>;
    const std::pair<const char*, SetLasso> lasso[] = {
        {"max_iter", [](auto& l) { l.max_iter = 999; }},
        {"alpha", [](auto& l) { l.alpha = 2.0; }},
        {"tolerance", [](auto& l) { l.tolerance = 1e-5; }},
        {"coef_floor", [](auto& l) { l.coef_floor = 2.0; }},
        {"fit_intercept", [](auto& l) { l.fit_intercept = false; }},
        {"warm_start", [](auto& l) { l.warm_start = true; }},
        {"unbiased_rmse", [](auto& l) { l.unbiased_rmse = false; }},
    };
    for (const auto& [name, set] : lasso)
    {
        ccd::LassoOptions l;
        set(l);
        EXPECT_NE(digest_of(ccd::HarmonicOptions{}, l), base) << name;
    }
}

TEST(OptionsDigest, ThePlanAndTheQaTableMove)
{
    const ccd::HarmonicOptions h;
    const ccd::LassoOptions l;
    const auto plan = ccd::preprocess::default_plan(h);
    const std::uint64_t base = ccd::options_digest(h, l, plan, nullptr);

    auto other = plan;
    other.insufficient_clear.chain.steps[0].codes.set(2);
    EXPECT_NE(ccd::options_digest(h, l, other, nullptr), base);

    other = plan;
    other.standard.transform.reset();
    EXPECT_NE(ccd::options_digest(h, l, other, nullptr), base);

    const auto table = ccd::qa_tables::landsat_c2_qa_pixel(h);
    const std::uint64_t with_table = ccd::options_digest(h, l, plan, table.get());
    EXPECT_NE(with_table, base);

    ccd::QaTable changed = *table;
    changed.code[21824] = changed.code[21824] == 0 ? 1 : 0;
    EXPECT_NE(ccd::options_digest(h, l, plan, &changed), with_table);
}

// options_digest() names every field of these structs one by one, so a
// field added to one must be added to the digest too. A struct's size is the
// tripwire: on any LP64 platform these are the sizes the digest was written
// for, and a test that fails here is a reminder, not a platform difference.
TEST(OptionsDigest, TheStructsAreTheOnesItWasWrittenFor)
{
    EXPECT_EQ(sizeof(ccd::HarmonicOptions), 448u);
    EXPECT_EQ(sizeof(ccd::LassoOptions), 40u);
    EXPECT_EQ(sizeof(ccd::preprocess::Step), 144u);
    EXPECT_EQ(sizeof(ccd::preprocess::Affine), 24u);
}

//------------------------------------------------------------------------------
// InputFingerprint
//------------------------------------------------------------------------------

namespace
{

std::uint64_t fingerprint(const golden::CharPixel& p, std::int64_t through)
{
    return ccd::input_fingerprint(dates_of(p), spectral_of(p), qas_of(p), p.h.QA_FILL, through);
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
        p.qas[40] = p.h.QA_CLOUD;
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
        p.qas[160] = p.h.QA_CLOUD;
        EXPECT_EQ(fingerprint(p, through), f);
    }
}

TEST(InputFingerprint, LeavesOutFill)
{
    const auto all = golden::characterisation_pixels();
    golden::CharPixel base = pixel(all, "start_catch");
    const index_t T = base.dates.size();
    base.qas[40] = base.h.QA_FILL;
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
        p.qas.insert(p.qas.begin() + at, p.h.QA_FILL);
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
