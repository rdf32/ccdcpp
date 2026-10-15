//==============================================================================
//
// Characterisation of the Standard procedure's transitions.
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
// Two tests:
//
//   * Transitions.BitIdenticalToFixture: the dump against the fixture, as
//     test_golden.cpp does it (a different platform is a skip, not a failure).
//     To regenerate, from a build of the reference code only, never from the
//     change under test:  CCD_TRANSITIONS_WRITE=1 ./test_transitions
//
//   * Transitions.PathsAsDocumented: the structure of each result (segment
//     count, curve_qa, change probability, observation counts, the mask) and
//     the observable signature of the quirk a pixel is for. Platform free.
//
// The pyccd quirks the refactor keeps exactly, numbered as in the design
// notes (each is decided for 0.8.0):
//
//   1  far from the previous break, Lookback peeks PEEK_SIZE - 1 observations
//   2  near the series start, Lookback's peek reads below the previous break
//   3  Lookback compares against the initialisation models' RMSE
//   4  Initialise fits 4 coefficients and the peek basis has 8, whatever
//      COEFFICIENT_MIN / COEFFICIENT_MAX say
//   5  the early exits return an empty processing mask
//   6  the series-end check in Extend is dead (a), so a series-end segment's
//      end day is its break day (b), and at peek >= 25 the machine re-enters
//      Initialise after it and can mark exclusions no segment uses (c)
//   7  Extend refits while the window has < RETRAIN_MIN_OBS observations but
//      compares against the model RMSE while it has <= RETRAIN_MIN_OBS
//
// Which pixel takes which transition. The states are those of the planned
// machine (Select, the whole-series fit, Prepare, Initialise, Lookback,
// StartCatch, the peek guard, Extend, Restart, Tail); "corpus" names a pixel
// of the golden corpus that also takes it, as test_golden.cpp labels it.
// Every reachable transition is taken by at least one pixel of the two
// fixtures; this table was measured with a trace-instrumented build of
// v0.6.1.
//
//   Select (procedure choice, ccd.cpp detect_pixel)
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
//     no room for Initialise -> Tail     standard_no_room
//   Initialise
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
//     exhausted, no segment -> Tail      constant_detection_band
//     exhausted after a segment -> Tail  lookback_below_break_decides (after
//                                        a re-entry); corpus c2:c2
//   Lookback
//     skipped (start == prev_break)      series_end_after_outlier, ...
//     peek far: PEEK_SIZE-1 obs (quirk 1)  start_catch,
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
//       initialisation RMSE (quirk 3)    every Lookback step above
//   StartCatch
//     emits a Start segment              start_catch; corpus bounds:dn
//     off: gap <= peek                   every first segment but start_catch
//     off: later segment                 peek_guard_tail, lookback_below_break
//   Peek guard
//     pass                               every pixel with an Extend segment
//     fail -> Tail                       peek_guard_tail
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
//     4-coefficient Initialise, 8-term
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
//     the dead series-end check (quirk 6a)   unreachable
//   Restart
//     room -> Initialise                 lookback_below_break, peek_guard_tail
//     no room -> Tail                    start_catch, ...
//     room after a series-end segment
//       (quirk 6c, peek >= 25), and in
//       it a Tmask commit and a Lookback
//       drop that no segment uses        reentry_after_series_end
//   Tail
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
//   1  Lookback reads PEEK_SIZE        lookback_peek_minus_one_decides
//                                      (golden corpus: none)
//   2  Lookback stops at prev_break    lookback_below_break_decides
//                                      (golden corpus: none)
//   4  Initialise reads COEFFICIENT_MIN  coefficient_min_6 (at the defaults
//                                      the fix changes nothing)
//   5  early exits keep the mask       variogram_nan; standard_too_few and
//                                      128 corpus pixels; snow_too_few and 18
//                                      corpus pixels
//   6a series-end check `>=`           7 of these pixels; 328 corpus pixels
//   6c series end goes to Tail         reentry_after_series_end
//                                      (golden corpus: none)
//   7  refit while <= RETRAIN_MIN_OBS  lookback_peek_minus_one_decides;
//                                      177 corpus pixels
//   (3 is intent and keeps the output.)
//
//==============================================================================

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
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
    throw std::runtime_error("no characterisation pixel " + name);
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

} // namespace

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

    // No room for Initialise: straight to Tail, one End segment over all 20.
    expect_shapes("standard_no_room", run_detect(pixel(all, "standard_no_room")), {{END, 0.0, 20}});

    // Never stable: Initialise exhausted with no segment, Tail over everything.
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
    // shapes, so a path change reads as one even off the fixture platform.
    expect_shapes("lookback_below_break_decides",
        run_detect(pixel(all, "lookback_below_break_decides")),
        {{6, 1.0, 22}, {8, 1.0, 188}, {8, 0.0, 242}});
    expect_shapes("lookback_peek_minus_one_decides",
        run_detect(pixel(all, "lookback_peek_minus_one_decides")),
        {{8, 1.0, 99}, {8, 1.0, 109}, {8, 0.0, 61}});
    expect_shapes("coefficient_min_6",
        run_detect(pixel(all, "coefficient_min_6")),
        {{8, 1.0, 107}, {8, 1.0, 33}, {8, 0.0, 308}});

    // The peek guard sends the second Initialise to Tail.
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
