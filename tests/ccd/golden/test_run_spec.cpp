//==============================================================================
//
// What a run's spec adds to the run (run/run.hpp):
//
//   * RunConfiguration.*: a configuration (RunSpec::config) runs its
//     strategies with their defaults filling the parameters it leaves out,
//     and its band list is checked against the spec's bands.
//   * Diagnostics.*: an observer (RunSpec::observer) is told every
//     transition, test, drop, segment and the exit, and changes nothing; a
//     resumed pixel's transitions chain as a fresh one's do.
//
//==============================================================================

#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>


#include "ccd/ccd.hpp"
#include "config.hpp"

#include "dump.hpp"
#include "series_corpus.hpp"

namespace
{

using golden::Series;
using golden::resume_on;

} // namespace

// A run under a configuration (RunSpec::config) runs its strategies with
// their defaults filling the parameters it leaves out: a configuration
// without the parameters at their defaults gives every pixel the output and
// the resume digest of the one that sets them all. A configuration's band
// list is checked against the spec's bands.
TEST(RunConfiguration, ParametersLeftOutRunAtTheirDefaults)
{
    const auto& registry = ccd::strategies::library_registry();
    std::size_t pixels = 0;
    for (const auto& s : golden::synthetic_series())
    {
        if (++pixels > 40)
            break;
        const ccd::config::Config spelled = s.config;
        ccd::config::Config left = spelled;
        for (auto& [stage, sc] : left.stages)
        {
            const ccd::config::Values defaults = registry.defaults(stage, sc.strategy);
            for (auto it = sc.params.begin(); it != sc.params.end();)
                it = defaults.at(it->first) == it->second ? sc.params.erase(it) : std::next(it);
        }
        const ccd::config::Values shared = ccd::config::shared_defaults();
        for (auto it = left.shared.begin(); it != left.shared.end();)
            it = shared.at(it->first) == it->second ? left.shared.erase(it) : std::next(it);
        ASSERT_NE(left, spelled) << s.name;
        const ccd::RunSpec spelled_out = golden::spec_of(s.h, spelled, true);
        const ccd::RunSpec left_out = golden::spec_of(s.h, left, true);
        const ccd::PixelOutput a =
            ccd::detect(spelled_out, s.date_view(), s.spectral_view(), s.qa_view());
        const ccd::PixelOutput b =
            ccd::detect(left_out, s.date_view(), s.spectral_view(), s.qa_view());

        EXPECT_EQ(golden::dump(a.result), golden::dump(b.result)) << s.name;
        EXPECT_EQ(a.header.digest, b.header.digest) << s.name;
    }

    const std::vector<Series> all = golden::synthetic_series();
    const Series& s = all.front();
    ccd::RunSpec spec = golden::spec_of(s.h, s.config);
    auto tree = s.config;
    tree.stages["window_screen"].params["bands"] = ccd::config::bands({1, 9});
    spec.algorithm = golden::algorithm(tree);
    try
    {
        ccd::detect(spec, s.date_view(), s.spectral_view(), s.qa_view());
        ADD_FAILURE() << "a band the spec lacks was accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find(
                      "window_screen 'tmask' parameter 'bands' holds band 9; the data spec has 7 bands"),
                  std::string::npos)
            << e.what();
    }
    tree.stages["window_screen"].params["bands"] = ccd::config::bands({1, 1});
    spec.algorithm = golden::algorithm(tree);
    EXPECT_THROW(ccd::detect(spec, s.date_view(), s.spectral_view(), s.qa_view()),
                 std::invalid_argument);
}

// An observer (RunSpec::observer) is told every transition, test, drop,
// segment and the exit, and changes nothing: the observed run's output is the
// plain run's, bit for bit. A cube run reports the observed pixels only.
namespace
{

struct Recording final : ccd::Observer
{
    std::mutex mutex;
    std::vector<std::pair<ccd::index_t, ccd::ObserverEvent>> events;
    void event(ccd::index_t pixel, const ccd::ObserverEvent& e) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        events.emplace_back(pixel, e);
    }
};

} // namespace

TEST(Diagnostics, AnObserverSeesTheRunAndChangesNothing)
{
    const std::vector<Series> all = golden::synthetic_series();
    std::size_t with_segments = 0;
    for (std::size_t i = 0; i < 20 && i < all.size(); ++i)
    {
        const Series& s = all[i];
        const ccd::PixelOutput plain =
            ccd::detect(golden::spec_of(s.h, s.config), s.date_view(), s.spectral_view(), s.qa_view());
        auto recording = std::make_shared<Recording>();
        ccd::RunSpec spec = golden::spec_of(s.h, s.config);
        spec.observer = recording;
        const ccd::PixelOutput observed =
            ccd::detect(spec, s.date_view(), s.spectral_view(), s.qa_view());
        EXPECT_EQ(golden::dump(plain.result), golden::dump(observed.result)) << s.name;

#if CCD_OBSERVE
        const auto& ev = recording->events;
        // The whole-series pixels (insufficient clear, snow) take no machine:
        // nothing to observe.
        if (plain.result.models.empty() ||
            plain.result.models.front().curve_qa == ccd::CurveQA::InsufficientClear ||
            plain.result.models.front().curve_qa == ccd::CurveQA::PersistentSnow)
            continue;
        ++with_segments;
        ASSERT_FALSE(ev.empty()) << s.name;
        EXPECT_EQ(ev.front().second.kind, ccd::ObserverEvent::Kind::Transition) << s.name;
        EXPECT_STREQ(ev.front().second.phase, "Prepare") << s.name;
        EXPECT_EQ(ev.back().second.kind, ccd::ObserverEvent::Kind::Exit) << s.name;
        std::size_t segments = 0;
        for (const auto& [pixel, e] : ev)
        {
            EXPECT_EQ(pixel, 0u);
            if (e.kind == ccd::ObserverEvent::Kind::Segment)
                ++segments;
            // A transition says what an Initialize, Lookback or Extend step
            // decided.
            if (e.kind == ccd::ObserverEvent::Kind::Transition)
            {
                const std::string from = e.phase;
                EXPECT_EQ(e.outcome.empty(),
                          from != "Initialize" && from != "Lookback" && from != "Extend")
                    << s.name << " " << from;
            }
        }
        EXPECT_EQ(segments, plain.result.models.size()) << s.name;
#endif
    }
#if CCD_OBSERVE
    EXPECT_GT(with_segments, 0u);
#endif

    // A cube of four pixels, two of them observed; the text observer's lines.
    const Series& s = all.front();
    const ccd::index_t T = s.size();
    std::vector<double> cube(4 * 7 * T);
    std::vector<std::uint8_t> qas(4 * T);
    for (std::size_t p = 0; p < 4; ++p)
    {
        std::copy(s.spectral.begin(), s.spectral.end(), cube.begin() + p * 7 * T);
        std::copy(s.qas.begin(), s.qas.end(), qas.begin() + p * T);
    }
    auto recording = std::make_shared<Recording>();
    ccd::RunSpec spec = golden::spec_of(s.h, s.config);
    spec.observer = recording;
    spec.observe = {1, 3};
    ccd::detect_cube(spec, s.date_view(),
                  ccd::ArrayView<const double, 4>::contiguous(cube.data(), {2, 2, 7, T}),
                  ccd::ArrayView<const std::uint8_t, 3>::contiguous(qas.data(), {2, 2, T}), 2);
    std::set<ccd::index_t> seen;
    for (const auto& [pixel, e] : recording->events)
        seen.insert(pixel);
#if CCD_OBSERVE
    EXPECT_EQ(seen, (std::set<ccd::index_t>{1, 3}));
#else
    EXPECT_TRUE(seen.empty());
#endif

    std::ostringstream text;
    ccd::TextObserver printer(text);
    for (const auto& [pixel, e] : recording->events)
        printer.event(pixel, e);
#if CCD_OBSERVE
    EXPECT_NE(text.str().find("pixel 1: transition Prepare -> "), std::string::npos) << text.str();
    EXPECT_NE(text.str().find("pixel 1: transition Initialize -> Extend (stable) window (0, 12)"),
              std::string::npos)
        << text.str();
#endif
}

// A resumed pixel's transitions chain as a fresh one's do: Prepare goes to the
// phase the stored run continues in, and each transition starts where the
// one before it went.
TEST(Diagnostics, AResumedRunTellsTheObserverWhereItContinues)
{
    const std::vector<Series> all = golden::synthetic_series();
    std::size_t continued = 0;
    for (std::size_t i = 0; i < 40 && i < all.size(); ++i)
    {
        const Series& s = all[i];
        const Series cut = s.prefix(s.size() * 2 / 3);
        const ccd::PixelOutput stored = ccd::detect(
            golden::spec_of(s.h, s.config, true), cut.date_view(), cut.spectral_view(),
            cut.qa_view());
        auto recording = std::make_shared<Recording>();
        ccd::RunSpec spec = golden::spec_of(s.h, s.config);
        spec.observer = recording;
        const ccd::PixelOutput resumed = ccd::detect(
            spec, s.date_view(), s.spectral_view(), s.qa_view(),
            ccd::Resume{stored.header, {stored.record}, {stored.result.models}, true});
        EXPECT_EQ(golden::dump(resumed.result), golden::dump(resume_on(s, stored).result)) << s.name;

#if CCD_OBSERVE
        const char* went = nullptr;
        for (const auto& [pixel, e] : recording->events)
        {
            if (e.kind != ccd::ObserverEvent::Kind::Transition)
                continue;
            if (went == nullptr)
                EXPECT_STREQ(e.phase, "Prepare") << s.name;
            else
                EXPECT_STREQ(e.phase, went) << s.name;
            went = e.next;
        }
        if (went != nullptr && stored.record.standard.exit == ccd::machine::Exit::SeriesEnd)
            ++continued;
#endif
    }
#if CCD_OBSERVE
    EXPECT_GT(continued, 0u);
#endif
}
