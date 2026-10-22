//==============================================================================
//
// What a run reports beside its output (RunSpec::reasons,
// RunSpec::diagnostics):
//
//   * Reasons.*: each observation's reason code says why it is where the run
//     left it. Asking for them changes nothing else; the codes below
//     FIRST_DROP are the processing mask's; every path that removes an
//     observation from the fit gives its code, on the golden pixels, through
//     detect() and detect_cube(), a registry's step and strategies, and a
//     resume.
//   * Diagnostics.*: each pixel's procedure, exit, counts and peek, and the
//     machine's work on each segment. Asking for them changes nothing else;
//     they agree with the reason codes and the record; each exit and the
//     counters are pinned on the golden pixels that take them, and the
//     drops of a registry's restoring search and forced start.
//
//==============================================================================

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"
#include "ccd/strategies/change_test/chi2.hpp"
#include "ccd/strategies/model/lasso.hpp"
#include "ccd/strategies/start_search/pyccd.hpp"

#include "characterization.hpp"
#include "config.hpp"
#include "dump.hpp"
#include "series.hpp"
#include "series_corpus.hpp"

namespace
{

using ccd::index_t;
using ccd::Reason;
using golden::Series;

// `spec` asking for the reason codes.
ccd::RunSpec with_reasons(ccd::RunSpec spec)
{
    spec.reasons = true;
    return spec;
}

ccd::PixelOutput run(const ccd::RunSpec& spec, const Series& s)
{
    return ccd::detect(spec, s.date_view(), s.spectral_view(), s.qa_view());
}

// The number of observations of each code.
std::map<Reason, std::size_t> counts(const std::vector<std::uint8_t>& reasons)
{
    std::map<Reason, std::size_t> out;
    for (const std::uint8_t code : reasons)
        ++out[static_cast<Reason>(code)];
    return out;
}

// The pixels the tests read: the golden corpus's, the transitions', the CSV
// pixels.
const std::vector<Series>& pixels()
{
    static const std::vector<Series> all = []
    {
        std::vector<Series> out = golden::synthetic_series();
        for (Series& s : golden::characterization_pixels())
            out.push_back(std::move(s));
        for (Series& s : golden::csv_series(CCD_TEST_DATA_DIR))
            out.push_back(std::move(s));
        return out;
    }();
    return all;
}

// What every result with reason codes holds: one code per observation, below
// REASON_COUNT; the mask keeps exactly those below FIRST_DROP; a kept
// observation is InSegment exactly when a segment's days hold it.
void expect_consistent(const ccd::PixelOutput& r, const Series& s)
{
    ASSERT_EQ(r.reasons.size(), s.size()) << s.name;
    for (std::size_t t = 0; t < s.size(); ++t)
    {
        const std::uint8_t code = r.reasons[t];
        ASSERT_LT(code, ccd::REASON_COUNT) << s.name << " " << t;
        EXPECT_EQ(code < static_cast<std::uint8_t>(ccd::FIRST_DROP), r.result.mask[t])
            << s.name << " " << t;
        bool in_segment = false;
        for (const ccd::ChangeModel& m : r.result.models)
            in_segment = in_segment || (m.start_day <= s.dates[t] && s.dates[t] <= m.end_day);
        if (r.result.mask[t])
            EXPECT_EQ(code == static_cast<std::uint8_t>(Reason::InSegment), in_segment)
                << s.name << " " << t;
    }
}

} // namespace

TEST(Reasons, EveryObservationHasOneAndTheRunIsTheSame)
{
    std::map<Reason, std::size_t> seen;
    for (const Series& s : pixels())
    {
        const ccd::RunSpec plain = golden::spec_of(s.h, s.config);
        const ccd::PixelOutput without = run(plain, s);
        const ccd::PixelOutput with = run(with_reasons(plain), s);
        EXPECT_TRUE(without.reasons.empty()) << s.name;
        EXPECT_EQ(golden::dump(without.result), golden::dump(with.result)) << s.name;
        expect_consistent(with, s);
        for (const auto& [code, n] : counts(with.reasons))
            seen[code] += n;
    }
    // The codes the corpus takes: all but those of the routes chi2, the
    // Lasso and the library's steps never take.
    for (const Reason code : {Reason::InSegment, Reason::NoSegment, Reason::Qa, Reason::Thermal,
                              Reason::BandRange, Reason::Green, Reason::DuplicateDate,
                              Reason::WindowScreen, Reason::LookbackOutlier, Reason::Outlier})
        EXPECT_GT(seen[code], 0u) << ccd::name(code);
    for (const Reason code : {Reason::NoFit, Reason::CustomStep, Reason::BeforeChange,
                              Reason::Stored})
        EXPECT_EQ(seen[code], 0u) << ccd::name(code);
}

// Each path through the preprocessing and the machine, on a golden pixel that
// takes it: the number of observations it gives its code.
TEST(Reasons, EachPathGivesItsCode)
{
    struct Pin
    {
        const char* pixel;
        Reason code;
        std::size_t count;
    };
    const Pin pins[] = {
        // The preprocessing: QA, thermal and band range, the insufficient
        // clear procedure's green median, the second observation of a date.
        {"c16:dn:1", Reason::Qa, 74},
        {"c16:dn:1", Reason::Thermal, 2},
        {"c16:dn:5", Reason::BandRange, 1},
        {"c16:dn:47", Reason::Green, 1},
        {"dup:dn:1", Reason::DuplicateDate, 3},
        // The machine: Initialize's window screen (Tmask commit), Lookback's
        // and Extend's outliers.
        {"lookback_near_break", Reason::WindowScreen, 1},
        {"lookback_far_peek_decides", Reason::LookbackOutlier, 5},
        {"series_end_after_outlier", Reason::Outlier, 1},
        // Kept in no segment: between a break and the next segment's
        // Lookback change, after the last full peek, an early exit's.
        {"lookback_near_break", Reason::NoSegment, 23},
        {"dense_series_end", Reason::NoSegment, 32},
        {"standard_too_few", Reason::NoSegment, 12},
        {"variogram_nan", Reason::NoSegment, 61},
        {"snow_too_few", Reason::NoSegment, 11},
        {"peek_guard_tail", Reason::InSegment, 400},
    };
    for (const Pin& pin : pins)
    {
        const Series& s = golden::named(pixels(), pin.pixel);
        const ccd::PixelOutput r = run(with_reasons(golden::spec_of(s.h, s.config)), s);
        EXPECT_EQ(counts(r.reasons)[pin.code], pin.count) << pin.pixel << " " << ccd::name(pin.code);
    }
}

// The observer's drops are the machine's codes, one event per observation,
// named as the code.
TEST(Reasons, TheObserversDropsAreTheMachinesCodes)
{
#if CCD_OBSERVE
    struct Drops final : ccd::Observer
    {
        std::map<std::int64_t, std::string> by_day;
        void event(index_t, const ccd::ObserverEvent& e) override
        {
            if (e.kind == ccd::ObserverEvent::Kind::Drop)
                by_day[e.day] = e.outcome;
        }
    };
    for (const char* pixel : {"lookback_near_break_decides", "lookback_far_peek_decides", "pyccd"})
    {
        const Series& s = golden::named(pixels(), pixel);
        auto drops = std::make_shared<Drops>();
        ccd::RunSpec spec = with_reasons(golden::spec_of(s.h, s.config));
        spec.observer = drops;
        const ccd::PixelOutput r = run(spec, s);
        std::map<std::int64_t, std::string> machine;
        for (std::size_t t = 0; t < s.size(); ++t)
            if (r.reasons[t] >= static_cast<std::uint8_t>(Reason::WindowScreen))
                machine[s.dates[t]] = ccd::name(static_cast<Reason>(r.reasons[t]));
        EXPECT_FALSE(machine.empty()) << pixel;
        EXPECT_EQ(drops->by_day, machine) << pixel;
    }
#else
    GTEST_SKIP() << "built with CCD_OBSERVE=OFF";
#endif
}

// A cube's codes are each pixel's detect() codes, in one (H, W, T) array.
TEST(Reasons, ACubesCodesAreEachPixels)
{
    const golden::Cube c = golden::make_cube(13, golden::Regime::Duplicates, 2, 3);
    const index_t T = c.T;
    ccd::RunSpec spec = with_reasons(golden::spec_of(ccd::Input{}, golden::pyccd()));
    spec.masks = false;
    const ccd::CubeOutput cube = ccd::detect_cube(
        spec, ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {2, 3, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {2, 3, T}), 2);
    ASSERT_EQ(cube.reasons.size(), 6 * T);
    EXPECT_TRUE(cube.masks.empty());
    EXPECT_TRUE(cube.diagnostics.empty());
    for (index_t p = 0; p < 6; ++p)
    {
        const ccd::PixelOutput one = ccd::detect(
            spec, ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(c.spectral.data() + p * 7 * T, {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data() + p * T, {T}));
        const auto codes = cube.reasons_of(p / 3, p % 3);
        EXPECT_EQ(std::vector<std::uint8_t>(&codes(0), &codes(0) + T), one.reasons) << p;
    }
}

namespace
{

//------------------------------------------------------------------------------
// A registry's step and strategies, each taking a route the library's do not
//------------------------------------------------------------------------------

// Drops the observations dated `day`.
class DropDay final : public ccd::preprocess::CustomStep
{
public:
    explicit DropDay(std::int64_t day) : day_(day) {}

    void apply(ccd::ArrayView<const std::int64_t, 1> dates, ccd::ArrayView<const double, 2>,
               ccd::ArrayView<const std::uint8_t, 1>, ccd::ProcessingMask& mask) const override
    {
        for (index_t t = 0; t < dates.size(); ++t)
            if (dates(t) == day_)
                mask.set(t, false);
    }
    void validate(index_t) const override {}
    std::string describe() const override { return "DropDay"; }

private:
    std::int64_t day_;
};

struct DropDayStep
{
    struct Params
    {
        std::int64_t day = 0;
    };
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        static const std::vector<ccd::config::Field<Params>> s = {
            {"day", &Params::day, ccd::config::Unit::None, ccd::config::any(), "the date dropped"},
        };
        return s;
    }
    static std::optional<ccd::preprocess::Step> make(const Params& params, const ccd::Input&,
                                                     const ccd::config::StepContext&)
    {
        return ccd::preprocess::Step::of(std::make_shared<const DropDay>(params.day));
    }
};

// chi2, but a change breaks one observation into the peek: the peek's first
// observation leaves the series before the change.
struct SkipChi2 final : ccd::ChangeTest
{
    using Params = ccd::strategies::Chi2::Params;
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        return ccd::strategies::Chi2::schema();
    }
    explicit SkipChi2(const Params& p) : chi2(p) {}

    ccd::Capabilities needs() const override { return chi2.needs(); }
    ccd::Capabilities provides() const override { return chi2.provides(); }
    std::vector<std::string> statistics() const override { return chi2.statistics(); }
    std::unique_ptr<ccd::Scratch> make_scratch() const override { return chi2.make_scratch(); }
    ccd::PeekPlan prepare(ccd::ArrayView<const std::int64_t, 1> dates, index_t bands,
                          ccd::Scratch& scratch) const override
    {
        return chi2.prepare(dates, bands, scratch);
    }
    ccd::TestOutcome test(const ccd::Peek& peek, ccd::Scratch& scratch,
                          ccd::PeekResult& result) const override
    {
        return chi2.test(peek, scratch, result);
    }
    index_t break_at(const ccd::Peek& peek, const ccd::Scratch& scratch) const override
    {
        return chi2.break_at(peek, scratch) + 1;
    }
    double magnitude(const ccd::Peek& peek, const ccd::Scratch& scratch,
                     index_t band) const override
    {
        return chi2.magnitude(peek, scratch, band);
    }

    ccd::strategies::Chi2 chi2;
};

// The Lasso, but no monitoring fit succeeds: Extend's first refit is NoFit.
struct NeverFits final : ccd::MonitoringModel
{
    class Fits final : public ccd::Model
    {
    public:
        bool fit(ccd::ArrayView<const std::int64_t, 1>, ccd::ArrayView<const double, 2>, index_t,
                 index_t, const ccd::BandList&, ccd::KeepResiduals) override
        {
            return false;
        }
        void peek_residuals(ccd::ArrayView<const std::int64_t, 1>, ccd::ArrayView<const double, 2>,
                            const ccd::BandList&, std::vector<std::vector<double>>&) override {}
        double rmse(index_t) const override { return 0; }
        const std::vector<double>& residuals(index_t) const override { return none_; }
        ccd::BandFit band_fit(index_t) const override { return {}; }

    private:
        std::vector<double> none_;
    };

    using Params = ccd::strategies::Lasso::Params;
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        return ccd::strategies::Lasso::schema();
    }
    explicit NeverFits(const Params& p) : lasso(p) {}

    ccd::Capabilities provides() const override { return lasso.provides(); }
    std::unique_ptr<ccd::Model> make_model() const override { return std::make_unique<Fits>(); }
    ccd::CoefficientRule coefficients() const override { return lasso.coefficients(); }

    ccd::strategies::Lasso lasso;
};

// pyccd's search, but an exhausted one restores the observations its window
// screen dropped (Exhaustion::Restore) or forces a start (Exhaustion::Force):
// the min_observations observations from where it began.
template<ccd::Exhaustion When>
struct ExhaustedSearch final : ccd::StartSearch
{
    using Params = ccd::strategies::PyccdSearch::Params;
    static const std::vector<ccd::config::Field<Params>>& schema()
    {
        return ccd::strategies::PyccdSearch::schema();
    }
    explicit ExhaustedSearch(const Params& p) : pyccd(p) {}

    bool too_few(index_t n) const override { return pyccd.too_few(n); }
    ccd::Window first(index_t at) const override { return pyccd.first(at); }
    bool room(const ccd::Window& w, index_t n) const override { return pyccd.room(w, n); }
    ccd::SearchPlan begin(ccd::ArrayView<const std::int64_t, 1>, index_t,
                          std::int64_t) const override
    {
        ccd::SearchPlan plan;
        plan.exhausted = When;
        return plan;
    }
    bool exhausted(const ccd::Window& w, index_t n) const override
    {
        return pyccd.exhausted(w, n);
    }
    ccd::Window forced(ccd::ArrayView<const std::int64_t, 1>, index_t at,
                       std::int64_t) const override
    {
        return ccd::Window(at, at + pyccd.min_window());
    }
    bool short_span(ccd::ArrayView<const std::int64_t, 1> d) const override
    {
        return pyccd.short_span(d);
    }
    bool usable(ccd::ArrayView<const std::int64_t, 1> d) const override
    {
        return pyccd.usable(d);
    }
    index_t min_window() const override { return pyccd.min_window(); }

    ccd::strategies::PyccdSearch pyccd;
};

const ccd::config::Registry& test_registry()
{
    static const ccd::config::Registry registry = []
    {
        ccd::config::Registry r;
        ccd::strategies::register_core_strategies(r);
        r.add_step<DropDayStep>("drop_day");
        r.add<ccd::ChangeTest, SkipChi2>("skip-chi2");
        r.add<ccd::MonitoringModel, NeverFits>("never-fits");
        r.add<ccd::StartSearch, ExhaustedSearch<ccd::Exhaustion::Restore>>("restoring");
        r.add<ccd::StartSearch, ExhaustedSearch<ccd::Exhaustion::Force>>("forcing");
        return r;
    }();
    return registry;
}

// `config` with stage `stage`'s strategy `strategy`, its parameters kept.
ccd::config::Config with_strategy(ccd::config::Config config, const std::string& stage,
                                  const std::string& strategy)
{
    config.stages.at(stage).strategy = strategy;
    return config;
}

ccd::PixelOutput run_in_test_registry(const Series& s, const ccd::config::Config& config)
{
    return run(with_reasons(golden::spec_of(
                   s.h, std::make_shared<const ccd::Algorithm>(config, test_registry()))),
               s);
}

} // namespace

TEST(Reasons, ARegistrysStepAndStrategiesGiveTheirCodes)
{
    const Series& s = golden::named(pixels(), "lookback_near_break");
    const ccd::PixelOutput plain = run(with_reasons(golden::spec_of(s.h, s.config)), s);
    ASSERT_EQ(plain.result.models.size(), 2u);

    // A step a registry adds: its code, CustomStep, where it removes one.
    std::size_t kept = 0;
    while (plain.reasons[kept] != static_cast<std::uint8_t>(Reason::InSegment))
        ++kept;
    ccd::config::Config config = s.config;
    auto& steps = config.preprocess.at("standard").steps;
    steps.insert(steps.begin() + 1, {"drop_day", {{"day", ccd::config::integer(s.dates[kept])}}});
    const ccd::PixelOutput stepped = run_in_test_registry(s, config);
    EXPECT_EQ(stepped.reasons[kept], static_cast<std::uint8_t>(Reason::CustomStep));
    expect_consistent(stepped, s);

    // A change test breaking past the peek's first observation: that one is
    // BeforeChange, the first observation after the first segment.
    const ccd::PixelOutput skipped =
        run_in_test_registry(s, with_strategy(s.config, "change_test", "skip-chi2"));
    ASSERT_GE(skipped.result.models.size(), 2u);
    ASSERT_EQ(counts(skipped.reasons)[Reason::BeforeChange], 1u);
    for (std::size_t t = 0; t < s.size(); ++t)
        if (skipped.reasons[t] == static_cast<std::uint8_t>(Reason::BeforeChange))
            EXPECT_GT(s.dates[t], skipped.result.models[0].end_day);
    expect_consistent(skipped, s);

    // A model that cannot fit: Extend's first refit is NoFit, and every kept
    // observation from its window on (there is no segment) is NoFit.
    const ccd::PixelOutput unfitted =
        run_in_test_registry(s, with_strategy(s.config, "model", "never-fits"));
    EXPECT_TRUE(unfitted.result.models.empty());
    const auto unfit = counts(unfitted.reasons);
    EXPECT_GT(unfit.at(Reason::NoFit), 0u);
    EXPECT_EQ(unfit.count(Reason::InSegment), 0u);
    expect_consistent(unfitted, s);
}

// A resumed pixel's codes are the full run's, but the stored drops, which
// the record keeps by day only: Stored.
TEST(Reasons, AResumeGivesTheStoredDropsTheirOwnCode)
{
    std::size_t with_stored = 0;
    for (const Series& s : golden::synthetic_series())
    {
        const Series cut = s.prefix(s.size() * 2 / 3);
        const ccd::PixelOutput stored = run(golden::spec_of(cut.h, cut.config, true), cut);
        if (stored.record.excluded_days.empty())
            continue;
        const ccd::RunSpec spec = with_reasons(golden::spec_of(s.h, s.config));
        const ccd::PixelOutput resumed = ccd::detect(
            spec, s.date_view(), s.spectral_view(), s.qa_view(),
            ccd::Resume{stored.header, {stored.record}, {stored.result.models}});
        if (resumed.status != ccd::ResumeStatus::Resumed)
            continue;
        ccd::RunSpec capped = spec;
        capped.algorithm = golden::algorithm(golden::with_shared(
            s.config, "statistics_end",
            ccd::config::integer(std::min(s.statistics_end(), stored.header.statistics_end))));
        const ccd::PixelOutput full = run(capped, s);
        ASSERT_EQ(golden::dump(resumed.result), golden::dump(full.result)) << s.name;

        std::vector<std::uint8_t> expected = full.reasons;
        for (std::size_t t = 0; t < s.size(); ++t)
            for (const std::int64_t day : stored.record.excluded_days)
                if (s.dates[t] == day && !full.result.mask[t])
                    expected[t] = static_cast<std::uint8_t>(Reason::Stored);
        EXPECT_EQ(resumed.reasons, expected) << s.name;
        ++with_stored;
        if (with_stored == 10)
            break;
    }
    EXPECT_EQ(with_stored, 10u);
}

namespace
{

ccd::RunSpec with_diagnostics(ccd::RunSpec spec)
{
    spec.diagnostics = true;
    return spec;
}

} // namespace

TEST(Diagnostics, AreTheRunsAndChangeNothing)
{
    const auto first_drop = static_cast<std::uint8_t>(ccd::FIRST_DROP);
    const auto machine_drop = static_cast<std::uint8_t>(Reason::WindowScreen);
    for (const Series& s : pixels())
    {
        const ccd::RunSpec plain = golden::spec_of(s.h, s.config, true);
        const ccd::PixelOutput without = run(plain, s);
        ccd::RunSpec spec = with_diagnostics(with_reasons(plain));
        const ccd::PixelOutput with = run(spec, s);
        EXPECT_EQ(golden::dump(without.result), golden::dump(with.result)) << s.name;

        const ccd::PixelDiagnostics& d = with.diagnostics;
        EXPECT_TRUE(without.diagnostics.segments.empty()) << s.name;
        EXPECT_EQ(d.procedure, with.record.procedure) << s.name;
        if (with.record.standard.exit != ccd::machine::Exit::None)
            EXPECT_EQ(d.exit, with.record.standard.exit) << s.name;
        if (d.procedure != ccd::PixelProcedure::Standard)
        {
            EXPECT_EQ(d.exit, ccd::machine::Exit::None) << s.name;
            EXPECT_EQ(d.peek, 0u) << s.name;
        }

        // The counts are the reason codes': the preprocessing's removals, the mask's.
        std::size_t removed = 0, kept = 0, dropped = 0;
        for (const std::uint8_t code : with.reasons)
        {
            removed += code >= first_drop && code < machine_drop;
            kept += code < first_drop;
            dropped += code >= machine_drop;
        }
        EXPECT_EQ(d.clear, s.size() - removed) << s.name;
        EXPECT_EQ(d.kept, kept) << s.name;

        // One per segment; Extend's own fitted once at least; the drops they count
        // are the machine's.
        ASSERT_EQ(d.segments.size(), with.result.models.size()) << s.name;
        std::size_t counted = 0;
        for (std::size_t i = 0; i < d.segments.size(); ++i)
        {
            const auto qa = static_cast<int>(with.result.models[i].curve_qa);
            if (d.procedure == ccd::PixelProcedure::Standard && qa <= 8)
                EXPECT_GE(d.segments[i].refits, 1u) << s.name << " " << i;
            if (qa == static_cast<int>(ccd::CurveQA::Start) || qa > 24)
                EXPECT_EQ(d.segments[i].initialize_steps + d.segments[i].tests, 0u)
                    << s.name << " " << i;
            counted += d.segments[i].dropped;
        }
        EXPECT_LE(counted, dropped) << s.name;
    }
}

// Each exit, the procedures and the counters, on golden pixels that take them.
TEST(Diagnostics, EachExitAndCounterOnAGoldenPixel)
{
    using ccd::machine::Exit;
    struct Pin
    {
        const char* pixel;
        ccd::PixelProcedure procedure;
        Exit exit;
        index_t clear, kept, peek;
    };
    const Pin pins[] = {
        {"select_clear_below_threshold", ccd::PixelProcedure::InsufficientClear, Exit::None, 24, 24, 0},
        {"select_snow_at_threshold", ccd::PixelProcedure::PermanentSnow, Exit::None, 81, 81, 0},
        {"standard_too_few", ccd::PixelProcedure::Standard, Exit::TooFewObservations, 12, 12, 0},
        {"variogram_nan", ccd::PixelProcedure::Standard, Exit::VariogramNaN, 61, 61, 6},
        {"standard_no_room", ccd::PixelProcedure::Standard, Exit::NoRoomToInitialize, 20, 20, 6},
        {"constant_detection_band", ccd::PixelProcedure::Standard, Exit::InitializeExhausted, 228, 228, 6},
        {"peek_guard_tail", ccd::PixelProcedure::Standard, Exit::PeekBeyondEnd, 400, 400, 19},
        {"series_end_after_outlier", ccd::PixelProcedure::Standard, Exit::SeriesEnd, 228, 227, 6},
        {"dense_series_end", ccd::PixelProcedure::Standard, Exit::SeriesEnd, 288, 288, 32},
        {"lookback_near_break_decides", ccd::PixelProcedure::Standard, Exit::SeriesEnd, 533, 493, 32},
    };
    for (const Pin& pin : pins)
    {
        const Series& s = golden::named(pixels(), pin.pixel);
        const ccd::PixelDiagnostics d =
            run(with_diagnostics(golden::spec_of(s.h, s.config)), s).diagnostics;
        EXPECT_EQ(d.procedure, pin.procedure) << pin.pixel;
        EXPECT_EQ(d.exit, pin.exit) << pin.pixel;
        EXPECT_EQ(d.clear, pin.clear) << pin.pixel;
        EXPECT_EQ(d.kept, pin.kept) << pin.pixel;
        EXPECT_EQ(d.peek, pin.peek) << pin.pixel;
    }

    // The counters of lookback_far_peek_decides' three curves: the second's
    // search, its 24 Lookback steps back to 127 observations and their 5
    // outliers among its 13 drops.
    const Series& s = golden::named(pixels(), "lookback_far_peek_decides");
    const ccd::PixelDiagnostics d =
        run(with_diagnostics(golden::spec_of(s.h, s.config)), s).diagnostics;
    ASSERT_EQ(d.segments.size(), 3u);
    const index_t expected[3][5] = {{13, 0, 77, 6, 1}, {46, 24, 86, 4, 13}, {13, 0, 38, 4, 1}};
    for (std::size_t i = 0; i < 3; ++i)
    {
        const ccd::SegmentDiagnostics& g = d.segments[i];
        EXPECT_EQ((std::vector<index_t>{g.initialize_steps, g.lookback_steps, g.tests, g.refits,
                                        g.dropped}),
                  std::vector<index_t>(expected[i], expected[i] + 5))
            << i;
    }

    // A Start segment holds none of the work: its search is the next curve's.
    const Series& start = golden::named(pixels(), "start_catch");
    const ccd::PixelDiagnostics sd =
        run(with_diagnostics(golden::spec_of(start.h, start.config)), start).diagnostics;
    ASSERT_EQ(sd.segments.size(), 2u);
    EXPECT_EQ(sd.segments[0].initialize_steps + sd.segments[0].lookback_steps, 0u);
    EXPECT_EQ(sd.segments[1].initialize_steps, 23u);
    EXPECT_EQ(sd.segments[1].lookback_steps, 1u);

    // NoFit: a model that cannot fit.
    const Series& unfit = golden::named(pixels(), "lookback_near_break");
    ccd::RunSpec spec = with_diagnostics(golden::spec_of(
        unfit.h, std::make_shared<const ccd::Algorithm>(
                     with_strategy(unfit.config, "model", "never-fits"), test_registry())));
    EXPECT_EQ(run(spec, unfit).diagnostics.exit, ccd::machine::Exit::NoFit);
}

// A cube's are each pixel's detect() diagnostics; a resume's carried
// segments hold zeros.
TEST(Diagnostics, ACubesAreEachPixelsAndAResumesCarriedSegmentsHoldNone)
{
    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 2, 3);
    const index_t T = c.T;
    const ccd::RunSpec spec = with_diagnostics(golden::spec_of(ccd::Input{}, golden::pyccd()));
    const ccd::CubeOutput cube = ccd::detect_cube(
        spec, ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
        ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {2, 3, 7, T}),
        ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {2, 3, T}), 2);
    ASSERT_EQ(cube.pixels.size(), 6u);
    for (index_t p = 0; p < 6; ++p)
    {
        const ccd::PixelDiagnostics one = ccd::detect(
            spec, ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(c.spectral.data() + p * 7 * T, {7, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(c.qas.data() + p * T, {T}))
            .diagnostics;
        const ccd::PixelDiagnostics& d = cube.diagnostics_of(p / 3, p % 3);
        EXPECT_EQ(d.procedure, one.procedure) << p;
        EXPECT_EQ(d.exit, one.exit) << p;
        EXPECT_EQ(d.clear, one.clear) << p;
        EXPECT_EQ(d.kept, one.kept) << p;
        EXPECT_EQ(d.peek, one.peek) << p;
        ASSERT_EQ(d.segments.size(), one.segments.size()) << p;
        for (std::size_t i = 0; i < d.segments.size(); ++i)
            EXPECT_EQ(d.segments[i].tests, one.segments[i].tests) << p << " " << i;
    }

    std::size_t carried = 0;
    for (const Series& s : golden::synthetic_series())
    {
        const Series cut = s.prefix(s.size() * 2 / 3);
        const ccd::PixelOutput stored = run(golden::spec_of(cut.h, cut.config, true), cut);
        const std::size_t final_segments = stored.record.standard.final_segments;
        if (final_segments == 0)
            continue;
        const ccd::PixelOutput resumed = ccd::detect(
            with_diagnostics(golden::spec_of(s.h, s.config)), s.date_view(), s.spectral_view(),
            s.qa_view(), ccd::Resume{stored.header, {stored.record}, {stored.result.models}});
        if (resumed.status != ccd::ResumeStatus::Resumed)
            continue;
        const ccd::PixelDiagnostics& d = resumed.diagnostics;
        ASSERT_EQ(d.segments.size(), resumed.result.models.size()) << s.name;
        for (std::size_t i = 0; i < final_segments; ++i)
            EXPECT_EQ(d.segments[i].initialize_steps + d.segments[i].tests +
                          d.segments[i].refits,
                      0u)
                << s.name << " " << i;
        if (++carried == 10)
            break;
    }
    EXPECT_EQ(carried, 10u);
}


// A search that restores its window screen's drops when exhausted takes
// them out of the count with the observations; a forced start counts its
// own. Everywhere the drops the segments count are the machine's codes.
TEST(Diagnostics, ARestoringSearchUncountsWhatItRestoresAndAForcedStartCountsItsDrops)
{
    // The (Initialize) transitions' outcomes and the codes and counts of a
    // pixel's run under `strategy`, the run's start search.
    struct Run
    {
        ccd::PixelOutput out;
        std::vector<std::string> outcomes;
        std::size_t machine = 0, counted = 0;
    };
    struct Outcomes final : ccd::Observer
    {
        std::vector<std::string> seen;
        void event(index_t, const ccd::ObserverEvent& e) override
        {
            if (e.kind == ccd::ObserverEvent::Kind::Transition)
                seen.push_back(e.outcome);
        }
    };
    const auto run_with = [](const Series& s, const char* strategy)
    {
        ccd::RunSpec spec = with_diagnostics(with_reasons(golden::spec_of(
            s.h, std::make_shared<const ccd::Algorithm>(
                     with_strategy(s.config, "start_search", strategy), test_registry()))));
        auto outcomes = std::make_shared<Outcomes>();
        spec.observer = outcomes;
        Run r{run(spec, s), {}, 0, 0};
        r.outcomes = outcomes->seen;
        for (const std::uint8_t code : r.out.reasons)
            r.machine += code >= static_cast<std::uint8_t>(Reason::WindowScreen);
        for (const ccd::SegmentDiagnostics& g : r.out.diagnostics.segments)
            r.counted += g.dropped;
        return r;
    };

    for (const char* strategy : {"restoring", "forcing"})
        for (const Series& s : pixels())
        {
            if (s.config.stages.at("start_search").strategy != "pyccd")
                continue;
            const Run r = run_with(s, strategy);
            expect_consistent(r.out, s);
            EXPECT_LE(r.counted, r.machine) << strategy << " " << s.name;
        }

    // c8:dn:26's last search screens four observations out, then finds no
    // stable window: restored, they are in no segment and not counted.
    const Series& s = golden::named(pixels(), "c8:dn:26");
    const Run plain = run_with(s, "pyccd");
    const Run restored = run_with(s, "restoring");
    ASSERT_EQ(plain.out.result.models.size(), 2u);
    ASSERT_EQ(restored.out.result.models.size(), 2u);
    EXPECT_EQ(counts(plain.out.reasons)[Reason::WindowScreen], 4u);
    EXPECT_EQ(counts(restored.out.reasons).count(Reason::WindowScreen), 0u);
    EXPECT_EQ(plain.counted, 12u);
    EXPECT_EQ(restored.counted, 8u);
    EXPECT_EQ(restored.counted, restored.machine);
#if CCD_OBSERVE
    EXPECT_EQ(std::count(restored.outcomes.begin(), restored.outcomes.end(),
                         "exhausted, restored"),
              1);
    EXPECT_EQ(std::count(plain.outcomes.begin(), plain.outcomes.end(), "exhausted, restored"),
              0);
#endif

    // Forced there, the start is a third segment, which counts its drops.
    const Run forced = run_with(s, "forcing");
    ASSERT_EQ(forced.out.result.models.size(), 3u);
    EXPECT_EQ(forced.out.diagnostics.exit, ccd::machine::Exit::SeriesEnd);
    EXPECT_EQ(forced.counted, 12u);
    EXPECT_EQ(forced.counted, forced.machine);
}
