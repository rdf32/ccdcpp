//==============================================================================
//
// Characterization of the Standard procedure's transitions.
//
// The golden corpus (test_golden.cpp) is broad but random: it takes most of
// the Standard procedure's transitions many times and a few never. The pixels
// of characterisation.hpp are built for the transitions, guards and rules
// that the corpus misses, and for the boundaries of the procedure
// choice. Their detect() output is pinned, byte for byte, in
// transitions_fixture.txt. A change meant to keep the output reproduces it
// exactly; a change meant to move it moves only the pixels it is meant to,
// and says which.
//
// Everything here goes through ccd::detect(), the public entry point, and
// reads only its result, so the tests hold whatever the procedure's insides
// become, and stay valid beside a second change-detection procedure.
//
// Three tests:
//
//   * Transitions.InputsAreTheSameOnEveryPlatform: a fingerprint of each
//     pixel's input (dates, QA, every spectral value's bits), pinned. The
//     pixels are whole DN, so the fingerprints hold on any little-endian
//     platform whose libm moves no value across a .5 rounding (the Mac's
//     and Linux's agree); a generator whose output depends on the compiler
//     fails here first.
//
//   * Transitions.BitIdenticalToFixture: the dump against the fixture, as
//     test_golden.cpp does it (a different platform is a skip, not a failure).
//     To regenerate, from the build whose output is the new reference, once
//     its diff is known to move only the pins it should:
//     CCD_TRANSITIONS_WRITE=1 ./test_transitions
//
//   * Transitions.PathsAsDocumented: the structure of each result (segment
//     count, curve_qa, change probability, observation counts, the mask) and
//     the observable signature of the rule a pixel is for, and on the
//     pixels whose output a rule decides without moving that structure, the
//     values it decides, to a tolerance. Platform free.
//
// The rules these pixels pin, where ccdcpp keeps pyccd's intent over its
// literals, numbered as in the design notes:
//
//   1  far from the previous break, Lookback peeks the adjusted peek of
//      observations, as Extend does;
//   2  near it, Lookback peeks down to the previous break and never below,
//      into the previous segment;
//   3  Lookback compares against the stable start window's RMSE, refitting
//      nothing;
//   4  Initialize fits COEFFICIENT_MIN coefficients and both peek bases are
//      COEFFICIENT_MAX wide;
//   5  the early exits return the processing mask, as every exit does;
//   6  a curve ends at the series end once its peek reaches the last
//      observation, before the observation the peek starts at, and it is the
//      last curve: EndCatch follows, not Restart;
//   7  Extend refits while the window holds at most RETRAIN_MIN_OBS
//      observations, the same windows that compare against the model RMSE.
//
// Which pixel takes which transition. The states are the machine's
// (Select, the whole-series fit, Prepare, Initialize, Lookback, StartCatch,
// the peek guard, Extend, Restart, EndCatch); "corpus" names a pixel
// of the golden corpus that also takes it, as test_golden.cpp labels it.
// Every reachable transition is taken by at least one pixel of the two
// fixtures; this table is checked against a trace-instrumented build
// (MAX_LOG_LEVEL=0) of the code it describes.
//
//   Select (ccd::select_procedure, called by detect_pixel)
//     Standard                  select_clear_at_threshold (clear 25/100 =
//                               CLEAR_PCT_THRESHOLD, the >= side), every
//                               Standard pixel below; corpus pyccd:dn
//     PermanentSnow             select_snow_at_threshold (61/81.01 = 0.753),
//                               snow_too_few; corpus c16:dn:0,1
//     InsufficientClear         select_clear_below_threshold (24/100),
//                               select_snow_below_threshold (60/80.01 =
//                               0.7499); corpus c16:dn:1,0
//   Whole-series fit (PermanentSnow, InsufficientClear)
//     too few -> Done, the mask          snow_too_few (11 < MEOW_SIZE)
//     fit                                the three whole-series select_* pixels
//   Prepare
//     too few -> Done, the mask          standard_too_few (12 <= MEOW_SIZE)
//     peek not adjusted                  start_catch, series_end_after_outlier
//     peek adjusted 7-12                 corpus c8:dn:0,0
//     peek adjusted 13-25                lookback_near_break (19),
//                                        peek_guard_tail (19)
//     peek adjusted >= 26                dense_series_end (32)
//     variogram lag raised               start_catch and most Standard pixels
//                                        (not select_clear_at_threshold: its
//                                        gaps exceed VARIOGRAM_MIN_LAG_DAYS)
//     variogram NaN -> Done, the mask    variogram_nan
//     no room for Initialize -> EndCatch  standard_no_room
//   Initialize
//     grow: span < DAY_DELTA             start_catch, peek_guard_tail, ...
//     grow: Tmask flags all              corpus c8:dn:2,3
//     grow: candidate too short in time  lookback_near_break,
//                                        lookback_far_peek_decides
//     grow: candidate too few samples    corpus pyccd:dn
//     Tmask commit                       lookback_near_break,
//                                        lookback_far_peek_decides
//     shift: unstable                    start_catch, constant_detection_band
//                                        (red constant: 0/0 in stable(), so
//                                        never stable)
//     stable                             every pixel with a segment from
//                                        Extend
//     exhausted, no segment -> EndCatch  constant_detection_band
//     exhausted after a segment -> EndCatch  corpus c2:c2
//   Lookback
//     skipped (start == prev_break)      series_end_after_outlier, ...
//     peek far: the adjusted peek (rule 1)  start_catch,
//                                        lookback_far_peek_decides
//     peek near: down to prev_break
//       (rule 2), prev_break 0           peek_guard_tail
//     the same, prev_break > 0 and
//       window start <= peek             lookback_near_break,
//                                        lookback_near_break_decides
//     change, first segment              start_catch
//     change, later segment: the
//       observations between the break
//       and the window are in no segment lookback_near_break
//     outlier drop                       lookback_far_peek_decides; corpus
//                                        bounds:dn
//     include one                        lookback_near_break, peek_guard_tail
//     reached prev_break                 peek_guard_tail
//     compares against the start
//       window's RMSE (rule 3)           every Lookback step above
//   StartCatch
//     emits a Start segment              start_catch; corpus bounds:dn
//     off: gap <= peek                   every first segment but start_catch
//     off: later segment                 peek_guard_tail, lookback_near_break
//   Peek guard
//     pass                               every pixel with an Extend segment
//     fail -> EndCatch                   peek_guard_tail
//   Extend
//     refit: untrained / <= RETRAIN_MIN_OBS / span factor
//                                        lookback_near_break (all three)
//     refit at exactly RETRAIN_MIN_OBS
//       (rule 7)                         lookback_far_peek_decides;
//                                        corpus pyccd:dn
//     compare: model RMSE (<= 24 obs)    series_end_after_outlier
//     compare: seasonal RMSE             series_end_after_outlier
//     coefficients 4 / 6 / 8             select_clear_at_threshold (4, 6),
//                                        lookback_near_break (4, 8)
//     COEFFICIENT_MIN Initialize,
//       COEFFICIENT_MAX peek basis
//       (rule 4)                         every Extend step above; with
//                                        COEFFICIENT_MIN = 6,
//                                        coefficient_min_6
//     change                             lookback_near_break, peek_guard_tail
//     outlier drop                       series_end_after_outlier
//     grow                               every Extend
//     series end, the peek at the last
//       observation: end day < break
//       day (rule 6)                     lookback_near_break,
//                                        dense_series_end
//     series end after an outlier drop:
//       break day after the dropped one  series_end_after_outlier
//     curve_qa is the last step's count,
//       not the fit's                    corpus c2:c2
//   Restart
//     room -> Initialize                 lookback_near_break, peek_guard_tail
//     no room -> EndCatch                none of these pixels after a
//                                        change; test_machine's
//                                        sparse_random seed 12
//   Series end -> EndCatch, no Restart (rule 6)
//                                        every series-end pixel
//   EndCatch
//     End segment                        standard_no_room,
//                                        constant_detection_band,
//                                        peek_guard_tail
//     none (prev_break + peek >= n)      every series-end pixel
//   Early exits return the mask (rule 5)
//                                        snow_too_few, standard_too_few,
//                                        variogram_nan
//
//==============================================================================

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/ccd.hpp"

#include "characterisation.hpp"
#include "golden.hpp"

#ifndef CCD_TRANSITIONS_FIXTURE
#define CCD_TRANSITIONS_FIXTURE "transitions_fixture.txt"
#endif

namespace
{

ccd::FitResult run_detect(const golden::CharPixel& p)
{
    const std::size_t T = p.dates.size();

    // A private copy: the reference output is the one a fresh buffer gives.
    std::vector<double> block = p.spectral;

    return ccd::detect(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(p.dates.data(), {T}),
        ccd::ArrayView<double, 2>::contiguous(block.data(), {7, T}),
        ccd::ArrayView<const std::uint8_t, 1>::contiguous(p.qas.data(), {T}),
        p.h,
        p.l
    );
}

const golden::CharPixel& pixel(const std::vector<golden::CharPixel>& all, const std::string& name)
{
    for (const auto& p : all)
        if (p.name == name)
            return p;
    throw std::runtime_error("no characterization pixel " + name);
}

// The values rules 4 and 7 decide on pixels whose shapes they leave alone:
// each band's RMSE of one segment. Each moves every one of these by at least
// 8.8e-4 relative (rule 7 against pyccd's refit while fewer than
// RETRAIN_MIN_OBS: 8.8e-4 to 4.8e-2; rule 4 against pyccd's literal 4 and 8:
// 4.2e-3 to 1.0e-1); building with
// -ffp-contract=off, which stands in for another compiler's contraction,
// moves them by under 1e-13, and -ffp-contract=fast not at all. A relative
// 1e-6 sits between the two.
constexpr double DECIDED_RTOL = 1e-6;

void expect_rmse(
    const std::string& name,
    const ccd::FitResult& r,
    std::size_t segment,
    const std::vector<double>& expected
)
{
    ASSERT_LT(segment, r.models.size()) << name;
    const auto& bands = r.models[segment].bands;
    ASSERT_EQ(bands.size(), expected.size()) << name;
    for (std::size_t b = 0; b < expected.size(); ++b)
        EXPECT_NEAR(bands[b].rmse, expected[b], DECIDED_RTOL * expected[b])
            << name << " segment " << segment << " band " << b;
}

// One segment's shape: curve_qa, change probability, observation count.
struct Shape
{
    int curve_qa;
    double change_probability;
    std::size_t observations;
};

void expect_shapes(
    const std::string& name,
    const ccd::FitResult& r,
    const std::vector<Shape>& shapes
)
{
    ASSERT_EQ(r.models.size(), shapes.size()) << name;
    for (std::size_t i = 0; i < shapes.size(); ++i)
    {
        EXPECT_EQ(static_cast<int>(r.models[i].curve_qa), shapes[i].curve_qa) << name << " segment " << i;
        EXPECT_EQ(r.models[i].change_probability, shapes[i].change_probability) << name << " segment " << i;
        EXPECT_EQ(r.models[i].observation_count, shapes[i].observations) << name << " segment " << i;
    }
}

// FNV-1a over a pixel's dates, QA codes and spectral bit patterns.
std::uint64_t fingerprint(const golden::CharPixel& p)
{
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* data, std::size_t bytes)
    {
        const auto* c = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i)
            h = (h ^ c[i]) * 1099511628211ull;
    };
    mix(p.dates.data(), p.dates.size() * sizeof(std::int64_t));
    mix(p.qas.data(), p.qas.size());
    mix(p.spectral.data(), p.spectral.size() * sizeof(double));
    return h;
}

} // namespace

TEST(Transitions, InputsAreTheSameOnEveryPlatform)
{
    const std::pair<const char*, std::uint64_t> expected[] = {
        {"select_clear_at_threshold", 0x62f2d01d24261440ull},
        {"select_clear_below_threshold", 0x9152613b22cd12ecull},
        {"select_snow_at_threshold", 0xaab87ecbd49eab41ull},
        {"select_snow_below_threshold", 0x230cdeb343e527f8ull},
        {"snow_too_few", 0x4069e1afaf2d07ccull},
        {"standard_too_few", 0xb6c3c5743f25bfa1ull},
        {"standard_no_room", 0xdcfc9365923140ffull},
        {"variogram_nan", 0x9afc74b04cafacd8ull},
        {"constant_detection_band", 0x7fc72ef9b305ecfbull},
        {"start_catch", 0xf7ca2b6b44163a39ull},
        {"coefficient_min_6", 0xc6fe65d8c948ae77ull},
        {"lookback_near_break", 0xaa29a7010494c9b1ull},
        {"lookback_near_break_decides", 0x2de711abc976bdfdull},
        {"lookback_far_peek_decides", 0xcbd41435e2ee30deull},
        {"peek_guard_tail", 0xd019800e2ee6f191ull},
        {"series_end_after_outlier", 0x73e2bfe02ecda114ull},
        {"dense_series_end", 0x2b119249d8626b0full},
    };

    const auto all = golden::characterisation_pixels();
    ASSERT_EQ(all.size(), std::size(expected));
    for (std::size_t i = 0; i < all.size(); ++i)
    {
        EXPECT_EQ(all[i].name, expected[i].first);
        EXPECT_EQ(fingerprint(all[i]), expected[i].second)
            << all[i].name << ": 0x" << std::hex << fingerprint(all[i]);
    }
}

TEST(Transitions, BitIdenticalToFixture)
{
    std::string dump;
    for (const auto& p : golden::characterisation_pixels())
        golden::dump_pixel(dump, "transitions:" + p.name, run_detect(p));

    const std::string header = "# platform: " + golden::platform() + "\n";

    if (std::getenv("CCD_TRANSITIONS_WRITE"))
    {
        std::ofstream out(CCD_TRANSITIONS_FIXTURE, std::ios::binary);
        out << header << dump;
        GTEST_SKIP() << "fixture written to " << CCD_TRANSITIONS_FIXTURE;
    }

    std::ifstream in(CCD_TRANSITIONS_FIXTURE, std::ios::binary);
    ASSERT_TRUE(in) << "missing fixture " << CCD_TRANSITIONS_FIXTURE;

    std::string fixture_header;
    std::getline(in, fixture_header);
    fixture_header += '\n';
    const std::string expected(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    if (dump == expected)
        return;

    if (fixture_header != header)
    {
        GTEST_SKIP()
            << "output differs from a fixture written on another platform\n"
            << "  fixture: " << fixture_header
            << "  this:    " << header
            << "exact equality is only claimed on the fixture's platform";
    }

    // Name the pixels that differ.
    std::istringstream a(dump), e(expected);
    std::string la, le, label;
    std::size_t differing = 0;
    std::string last_reported;
    while (std::getline(e, le))
    {
        if (!std::getline(a, la))
        {
            ADD_FAILURE() << "output ends before the fixture";
            break;
        }
        if (le.rfind("P ", 0) == 0)
            label = le;
        if (la != le && label != last_reported)
        {
            ++differing;
            last_reported = label;
            ADD_FAILURE() << "differs: " << label << "\n  expected: " << le << "\n  actual:   " << la;
        }
    }
    FAIL() << differing << " pixels differ from the fixture";
}

TEST(Transitions, PathsAsDocumented)
{
    const auto all = golden::characterisation_pixels();
    using CQ = ccd::CurveQA;
    const int START = static_cast<int>(CQ::Start), END = static_cast<int>(CQ::End);
    const int CLEAR = static_cast<int>(CQ::InsufficientClear), SNOW = static_cast<int>(CQ::PersistentSnow);

    // Select at and around the two thresholds.
    {
        const auto r = run_detect(pixel(all, "select_clear_at_threshold"));
        expect_shapes("select_clear_at_threshold", r, {{6, 0.0, 19}});   // Standard: Extend's curve_qa
        EXPECT_EQ(r.mask.count(), 25u);
    }
    expect_shapes("select_clear_below_threshold",
        run_detect(pixel(all, "select_clear_below_threshold")), {{CLEAR, 0.0, 24}});
    expect_shapes("select_snow_at_threshold",
        run_detect(pixel(all, "select_snow_at_threshold")), {{SNOW, 0.0, 81}});
    expect_shapes("select_snow_below_threshold",
        run_detect(pixel(all, "select_snow_below_threshold")), {{CLEAR, 0.0, 20}});

    // Early exits: no segments, and the processing mask (rule 5): the clear
    // and snow observations of snow_too_few, the clear ones of the others.
    const std::pair<const char*, std::size_t> early[] = {
        {"snow_too_few", 11}, {"standard_too_few", 12}, {"variogram_nan", 61}};
    for (const auto& [name, kept] : early)
    {
        const auto& p = pixel(all, name);
        const auto r = run_detect(p);
        EXPECT_TRUE(r.models.empty()) << name;
        EXPECT_EQ(r.mask.size(), p.dates.size()) << name;
        EXPECT_EQ(r.mask.count(), kept) << name;
    }

    // No room for Initialize: straight to EndCatch, one End segment over all 20.
    expect_shapes("standard_no_room", run_detect(pixel(all, "standard_no_room")), {{END, 0.0, 20}});

    // Never stable: Initialize exhausted with no segment, EndCatch over everything.
    expect_shapes("constant_detection_band",
        run_detect(pixel(all, "constant_detection_band")), {{END, 0.0, 228}});

    // StartCatch: a Start segment up to the window Lookback stopped at, which
    // is where the next segment starts and the Start segment's break day.
    {
        const auto r = run_detect(pixel(all, "start_catch"));
        expect_shapes("start_catch", r, {{START, 0.0, 10}, {8, 0.0, 212}});
        EXPECT_EQ(r.models[0].break_day, r.models[1].start_day);
    }

    // Lookback near prev_break at a dense cadence (rule 2), after a change:
    // the observations between the first break and the second segment
    // belong to no segment.
    {
        const auto& p = pixel(all, "lookback_near_break");
        const auto r = run_detect(p);
        expect_shapes("lookback_near_break", r, {{4, 1.0, 13}, {8, 0.0, 476}});
        EXPECT_LT(r.models[0].break_day, r.models[1].start_day);
        // Series end after the last full peek (rule 6): the curve ends
        // before the observation the peek starts at, its break day.
        EXPECT_LT(r.models[1].end_day, r.models[1].break_day);
    }

    // The pixels whose output a rule decides (see the rules above): their
    // shapes, so a path change reads as one even off the fixture platform,
    // and where a fix keeps the shapes, what it does move.
    {
        // Rule 2 decides only the mask: Lookback near the break reads no
        // observation of the previous segment, so it drops none that
        // reading below the break would (493 kept; 483 when it reads
        // below).
        const auto r = run_detect(pixel(all, "lookback_near_break_decides"));
        expect_shapes("lookback_near_break_decides", r,
            {{6, 1.0, 22}, {8, 1.0, 188}, {8, 0.0, 241}});
        EXPECT_EQ(r.mask.count(), 493u);
    }
    {
        // Rule 1 decides the shapes: the second segment's Lookback, peeking
        // the full adjusted peek, reaches back to 127 observations. Rule 7
        // decides only the first segment's fit, refitted at 24 observations.
        const auto r = run_detect(pixel(all, "lookback_far_peek_decides"));
        expect_shapes("lookback_far_peek_decides", r,
            {{8, 1.0, 99}, {8, 1.0, 127}, {8, 0.0, 60}});
        expect_rmse("lookback_far_peek_decides", r, 0,
            {35.81290728, 48.70786646, 47.4006856, 124.7866627, 71.73959318,
             66.24394152, 231.2815984});
    }
    {
        // Rule 4 decides the fits of the first two segments: Initialize
        // fits COEFFICIENT_MIN = 6 coefficients.
        const auto r = run_detect(pixel(all, "coefficient_min_6"));
        expect_shapes("coefficient_min_6", r,
            {{8, 1.0, 107}, {8, 1.0, 33}, {8, 0.0, 307}});
        expect_rmse("coefficient_min_6", r, 0,
            {33.84482982, 41.67944514, 50.41066257, 122.4179905, 92.36000081,
             65.21865399, 223.7726019});
        expect_rmse("coefficient_min_6", r, 1,
            {61.42278107, 58.15066275, 75.88209759, 179.3754244, 114.9040179,
             75.02686201, 172.548894});
    }

    // The peek guard sends the second Initialize to EndCatch.
    expect_shapes("peek_guard_tail",
        run_detect(pixel(all, "peek_guard_tail")), {{8, 1.0, 310}, {END, 0.0, 90}});

    // Series end after an outlier drop: the break day is the observation
    // after the dropped one, so end day < break day, and the drop is in the mask.
    {
        const auto& p = pixel(all, "series_end_after_outlier");
        const auto r = run_detect(p);
        expect_shapes("series_end_after_outlier", r, {{8, 0.0, 222}});
        EXPECT_LT(r.models[0].end_day, r.models[0].break_day);
        EXPECT_EQ(r.mask.count(), p.dates.size() - 1);
    }

    // A series end at peek 32 (rule 6): the curve that reaches the end is
    // the last one, so nothing runs after it, and the spikes among the last
    // observations, which no segment reads, stay in the mask.
    {
        const auto& p = pixel(all, "dense_series_end");
        const auto r = run_detect(p);
        expect_shapes("dense_series_end", r, {{8, 0.0, 256}});
        EXPECT_LT(r.models[0].end_day, r.models[0].break_day);

        std::size_t dropped_after = 0;
        for (std::size_t t = 0; t < p.dates.size(); ++t)
            if (!r.mask[t] && p.dates[t] > r.models[0].break_day)
                ++dropped_after;
        EXPECT_EQ(dropped_after, 0u);
        EXPECT_EQ(r.mask.count(), p.dates.size());
    }
}
