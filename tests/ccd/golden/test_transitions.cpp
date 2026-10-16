//==============================================================================
//
// Characterization of the Standard procedure's transitions.
//
// The golden corpus (test_golden.cpp) is broad but random: it takes most of
// the Standard procedure's transitions many times and a few never. The pixels
// of characterisation.hpp are built for the transitions, guards and pyccd
// quirks that the corpus misses, and for the boundaries of the procedure
// choice. Their detect() output is pinned, byte for byte, in
// transitions_fixture.txt, which ccdcpp v0.6.1 wrote. The refactor of the
// Standard procedure into an explicit state machine must reproduce it
// exactly, quirks included (the quirks change only in 0.8.0).
//
// Everything here goes through ccd::detect(), the public entry point, and
// reads only its result, so the tests hold whatever the procedure's insides
// become, and stay valid beside a second change-detection procedure.
//
// Three tests:
//
//   * Transitions.InputsAreTheSameOnEveryPlatform: a fingerprint of each
//     pixel's input (dates, QA, every spectral value's bits), pinned. The
//     pixels are whole DN, so the fingerprints hold on any platform; a
//     generator whose output depends on the compiler fails here first.
//
//   * Transitions.BitIdenticalToFixture: the dump against the fixture, as
//     test_golden.cpp does it (a different platform is a skip, not a failure).
//     To regenerate, from a build of the reference code only, never from the
//     change under test:  CCD_TRANSITIONS_WRITE=1 ./test_transitions
//
//   * Transitions.PathsAsDocumented: the structure of each result (segment
//     count, curve_qa, change probability, observation counts, the mask) and
//     the observable signature of the quirk a pixel is for, and on the
//     pixels whose output a quirk decides without moving that structure, the
//     values it decides, to a tolerance. Platform free.
//
// The pyccd quirks the refactor keeps exactly, numbered as in the design
// notes (each is decided for 0.8.0):
//
//   1  far from the previous break, Lookback peeks peek - 1 observations (the
//      adjusted peek, Cadence::peek, minus one)
//   2  near the series start, Lookback's peek reads below the previous break
//   3  Lookback compares against the initialization models' RMSE
//   4  Initialize fits 4 coefficients and the peek basis has 8, whatever
//      COEFFICIENT_MIN / COEFFICIENT_MAX say
//   5  the early exits return an empty processing mask
//   6  the series-end check in Extend is dead (a), so a series-end segment's
//      end day is its break day (b), and after it the outer guard passes
//      again at peek >= 25: at 25 Initialize exits at once, at peek >= 26 it
//      runs on the last observations and can mark exclusions no segment
//      uses (c)
//   7  Extend refits while the window has < RETRAIN_MIN_OBS observations but
//      compares against the model RMSE while it has <= RETRAIN_MIN_OBS
//
// Which pixel takes which transition. The states are the machine's
// (Select, the whole-series fit, Prepare, Initialize, Lookback, StartCatch,
// the peek guard, Extend, Restart, EndCatch); "corpus" names a pixel
// of the golden corpus that also takes it, as test_golden.cpp labels it.
// Every reachable transition is taken by at least one pixel of the two
// fixtures; this table was measured with a trace-instrumented build of
// v0.6.1.
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
//     too few -> Done, empty mask        snow_too_few (11 < MEOW_SIZE)
//     fit                                the three whole-series select_* pixels
//   Prepare
//     too few -> Done, empty mask        standard_too_few (12 <= MEOW_SIZE)
//     peek not adjusted                  start_catch, series_end_after_outlier
//     peek adjusted 7-12                 corpus c8:dn:0,0
//     peek adjusted 13-25                lookback_below_break (19),
//                                        peek_guard_tail (19)
//     peek adjusted >= 26                reentry_after_series_end (32)
//     variogram lag raised               start_catch and most Standard pixels
//                                        (not select_clear_at_threshold: its
//                                        gaps exceed VARIOGRAM_MIN_LAG_DAYS)
//     variogram NaN -> Done, empty mask  variogram_nan
//     no room for Initialize -> EndCatch  standard_no_room
//   Initialize
//     grow: span < DAY_DELTA             start_catch, peek_guard_tail, ...
//     grow: Tmask flags all              corpus c8:dn:2,3
//     grow: candidate too short in time  lookback_below_break, reentry_*
//     grow: candidate too few samples    corpus pyccd:dn
//     Tmask commit                       lookback_below_break, reentry_*
//     shift: unstable                    start_catch, constant_detection_band
//                                        (red constant: 0/0 in stable(), so
//                                        never stable)
//     stable                             every pixel with a segment from
//                                        Extend
//     exhausted, no segment -> EndCatch  constant_detection_band
//     exhausted after a segment -> EndCatch  lookback_below_break_decides (after
//                                        a re-entry); corpus c2:c2
//   Lookback
//     skipped (start == prev_break)      series_end_after_outlier, ...
//     peek far: peek - 1 obs (quirk 1)   start_catch,
//                                        lookback_peek_minus_one_decides
//     peek to index 0, prev_break == 0   peek_guard_tail
//     peek to index 0 BELOW prev_break (quirk 2)  lookback_below_break,
//                                        lookback_below_break_decides
//     peek to prev_break                 reentry_after_series_end
//     change, first segment              start_catch
//     change, later segment: the
//       observations between the break
//       and the window are in no segment lookback_below_break
//     outlier drop                       reentry_after_series_end; corpus
//                                        bounds:dn
//     include one                        lookback_below_break, peek_guard_tail
//     reached prev_break                 peek_guard_tail, reentry_*
//     compares against the
//       initialization RMSE (quirk 3)    every Lookback step above
//   StartCatch
//     emits a Start segment              start_catch; corpus bounds:dn
//     off: gap <= peek                   every first segment but start_catch
//     off: later segment                 peek_guard_tail, lookback_below_break
//   Peek guard
//     pass                               every pixel with an Extend segment
//     fail -> EndCatch                   peek_guard_tail
//   Extend
//     refit: untrained / < RETRAIN_MIN_OBS / span factor
//                                        lookback_below_break (all three)
//     no refit; no refit at exactly
//       RETRAIN_MIN_OBS (quirk 7)        lookback_peek_minus_one_decides;
//                                        corpus pyccd:dn
//     compare: model RMSE (<= 24 obs)    series_end_after_outlier
//     compare: seasonal RMSE             series_end_after_outlier
//     coefficients 4 / 6 / 8             select_clear_at_threshold (4, 6),
//                                        lookback_below_break (4, 8)
//     4-coefficient Initialize, 8-term
//       peek basis (quirk 4)             every Extend step above; with
//                                        COEFFICIENT_MIN = 6,
//                                        coefficient_min_6
//     change                             lookback_below_break, peek_guard_tail
//     outlier drop                       series_end_after_outlier
//     grow                               every Extend
//     series end after a grow: end day
//       == break day (quirk 6b)          lookback_below_break,
//                                        reentry_after_series_end
//     series end after an outlier drop:
//       break day after the dropped one  series_end_after_outlier
//     curve_qa is the last step's count,
//       not the fit's                    corpus c2:c2
//     the dead series-end check (quirk 6a)   never taken (an assert in
//                                        standard::extend's grow)
//   Restart
//     room -> Initialize                 lookback_below_break, peek_guard_tail
//     no room -> EndCatch                start_catch, ...
//     room after a series-end segment
//       (quirk 6c: the guard passes at
//       peek >= 25, Initialize runs at
//       peek >= 26), and in it a Tmask
//       commit and a Lookback drop that
//       no segment uses                  reentry_after_series_end
//   EndCatch
//     End segment                        standard_no_room,
//                                        constant_detection_band,
//                                        peek_guard_tail
//     none (prev_break + peek >= n)      every series-end pixel
//   Early exits return an empty mask (quirk 5)
//                                        snow_too_few, standard_too_few,
//                                        variogram_nan
//
// What each decided 0.8.0 fix changes in these fixtures. Measured
// on v0.6.1 by applying each fix alone to a scratch copy: the pixels whose
// dump changes, so the 0.8.0 work knows which pins it is expected to move and
// a refactor that moves one of them has changed behaviour.
//
//   1  Lookback reads the full peek   lookback_peek_minus_one_decides
//                                      (golden corpus: none)
//   2  Lookback stops at prev_break    lookback_below_break_decides
//                                      (golden corpus: none)
//   4  Initialize reads COEFFICIENT_MIN  coefficient_min_6 (at the defaults
//                                      the fix changes nothing)
//   5  early exits keep the mask       variogram_nan; standard_too_few and
//                                      128 corpus pixels; snow_too_few and 18
//                                      corpus pixels
//   6a series-end check `>=`           7 of these pixels; 328 corpus pixels
//   6c series end goes to EndCatch     reentry_after_series_end
//                                      (golden corpus: none)
//   7  refit while <= RETRAIN_MIN_OBS  lookback_peek_minus_one_decides;
//                                      177 corpus pixels
//   (3 is intent and keeps the output.)
//
// Off the fixture platform PathsAsDocumented still fails on each fix alone:
// 1, 5 and 6 through the shapes and masks, 2 through
// lookback_below_break_decides's mask count, 4 and 7 through the RMSE values
// of coefficient_min_6 and lookback_peek_minus_one_decides (checked by
// applying each to v0.6.1).
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

// The values fixes 4 and 7 decide on pixels whose shapes they leave alone:
// each band's RMSE of one segment. Each fix, applied alone to v0.6.1, moves
// every one of these by at least 8.8e-4 relative (7: 8.8e-4 to 4.8e-2; 4:
// 4.2e-3 to 1.0e-1); building v0.6.1 with -ffp-contract=off, which stands in
// for another compiler's contraction, moves them by under 1e-13, and
// -ffp-contract=fast not at all. A relative 1e-6 sits between the two.
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
        EXPECT_NEAR(bands[b].score.rmse, expected[b], DECIDED_RTOL * expected[b])
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
        {"lookback_below_break", 0xaa29a7010494c9b1ull},
        {"lookback_below_break_decides", 0x2de711abc976bdfdull},
        {"lookback_peek_minus_one_decides", 0xcbd41435e2ee30deull},
        {"peek_guard_tail", 0xd019800e2ee6f191ull},
        {"series_end_after_outlier", 0x73e2bfe02ecda114ull},
        {"reentry_after_series_end", 0x2b119249d8626b0full},
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
        expect_shapes("select_clear_at_threshold", r, {{6, 0.0, 20}});   // Standard: Extend's curve_qa
        EXPECT_EQ(r.mask.count(), 25u);
    }
    expect_shapes("select_clear_below_threshold",
        run_detect(pixel(all, "select_clear_below_threshold")), {{CLEAR, 0.0, 24}});
    expect_shapes("select_snow_at_threshold",
        run_detect(pixel(all, "select_snow_at_threshold")), {{SNOW, 0.0, 81}});
    expect_shapes("select_snow_below_threshold",
        run_detect(pixel(all, "select_snow_below_threshold")), {{CLEAR, 0.0, 20}});

    // Early exits: no segments and an empty (size 0) processing mask (quirk 5).
    for (const char* name : {"snow_too_few", "standard_too_few", "variogram_nan"})
    {
        const auto r = run_detect(pixel(all, name));
        EXPECT_TRUE(r.models.empty()) << name;
        EXPECT_EQ(r.mask.size(), 0u) << name;
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
        expect_shapes("start_catch", r, {{START, 0.0, 10}, {8, 0.0, 213}});
        EXPECT_EQ(r.models[0].break_day, r.models[1].start_day);
    }

    // Lookback below prev_break (quirk 2), after a change: the observations
    // between the first break and the second segment belong to no segment.
    {
        const auto& p = pixel(all, "lookback_below_break");
        const auto r = run_detect(p);
        expect_shapes("lookback_below_break", r, {{4, 1.0, 13}, {8, 0.0, 477}});
        EXPECT_LT(r.models[0].break_day, r.models[1].start_day);
        EXPECT_EQ(r.models[1].end_day, r.models[1].break_day);   // series end after a grow (quirk 6b)
    }

    // The pixels whose output a quirk decides (see the table above): their
    // shapes, so a path change reads as one even off the fixture platform,
    // and where a fix keeps the shapes, what it does move.
    {
        // Fix 2 keeps every segment and moves only the mask: it keeps 10
        // more observations (493).
        const auto r = run_detect(pixel(all, "lookback_below_break_decides"));
        expect_shapes("lookback_below_break_decides", r,
            {{6, 1.0, 22}, {8, 1.0, 188}, {8, 0.0, 242}});
        EXPECT_EQ(r.mask.count(), 483u);
    }
    {
        // Fix 1 moves the shapes; fix 7 only the first segment's fit.
        const auto r = run_detect(pixel(all, "lookback_peek_minus_one_decides"));
        expect_shapes("lookback_peek_minus_one_decides", r,
            {{8, 1.0, 99}, {8, 1.0, 109}, {8, 0.0, 61}});
        expect_rmse("lookback_peek_minus_one_decides", r, 0,
            {36.24151163, 49.6614777, 47.72221929, 123.7609953, 68.2605607,
             66.3021312, 236.0055942});
    }
    {
        // Fix 4 moves the fits of the first two segments.
        const auto r = run_detect(pixel(all, "coefficient_min_6"));
        expect_shapes("coefficient_min_6", r,
            {{8, 1.0, 107}, {8, 1.0, 33}, {8, 0.0, 308}});
        expect_rmse("coefficient_min_6", r, 0,
            {33.98750948, 41.98759122, 50.12372659, 117.7534882, 93.13544933,
             64.747995, 211.9343041});
        expect_rmse("coefficient_min_6", r, 1,
            {65.4233493, 63.69229807, 84.5716549, 187.0748164, 118.5062873,
             78.32135257, 167.267371});
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

    // Re-entry after a series-end segment at peek 32 (quirk 6c): no second
    // segment, but the mask carries exclusions dated after the only segment's
    // break day, which no segment used.
    {
        const auto& p = pixel(all, "reentry_after_series_end");
        const auto r = run_detect(p);
        expect_shapes("reentry_after_series_end", r, {{8, 0.0, 257}});
        EXPECT_EQ(r.models[0].end_day, r.models[0].break_day);   // quirk 6b

        std::size_t unused = 0;
        for (std::size_t t = 0; t < p.dates.size(); ++t)
            if (!r.mask[t] && p.dates[t] > r.models[0].break_day)
                ++unused;
        EXPECT_EQ(unused, 4u);
    }
}
