//==============================================================================
//
// The per-pixel records below detect_pixel: MaskedSeries (the processing
// mask and the observations it keeps), Variogram, and the Standard
// procedure's Cadence. The procedures' outputs are held by the golden and transitions
// fixtures; these tests hold the records' own contracts.
//
//   * MaskedSeries: after reset() and after any sequence of drops, dates() and
//     spectral() equal apply_mask(workspace, mask()) and size() equals
//     mask().count(); each drop moves the caller's window by the one rule
//     (a bound above the dropped position moves down by one); a rejected
//     Tmask removal changes nothing.
//   * Cadence: measure_cadence() is adjust_peek() and
//     adjust_change_threshold() of the caller's PEEK_SIZE and
//     CHANGE_THRESHOLD.
//   * HarmonicWorkspace: refers to the caller's options and gives only const
//     access to them.
//
//==============================================================================

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ccd/harmonic/harmonic.hpp"
#include "ccd/procedure/fit.hpp"
#include "ccd/procedure/masked_series.hpp"
#include "ccd/procedure/standard.hpp"

namespace
{

using ccd::index_t;

// A 3-band pixel of T observations, every value distinct: band b at
// observation t is 1000 b + t, dated 16 days apart with one repeated date.
struct Pixel
{
    static constexpr index_t B = 3;
    index_t T;
    std::vector<std::int64_t> dates;
    std::vector<double> spectral;
    std::vector<std::uint8_t> qas;
    ccd::HarmonicOptions options;

    explicit Pixel(index_t T_) : T(T_), dates(T_), spectral(B * T_), qas(T_, 0)
    {
        for (index_t t = 0; t < T; ++t)
            dates[t] = 730000 + 16 * static_cast<std::int64_t>(t) - (t > 5 ? 16 : 0);
        for (index_t b = 0; b < B; ++b)
            for (index_t t = 0; t < T; ++t)
                spectral[b * T + t] = 1000.0 * b + t;
    }

    ccd::HarmonicWorkspace workspace() const
    {
        return ccd::HarmonicWorkspace(
            ccd::ArrayView<const std::int64_t, 1>::contiguous(dates.data(), {T}),
            ccd::ArrayView<const double, 2>::contiguous(spectral.data(), {B, T}),
            ccd::ArrayView<const std::uint8_t, 1>::contiguous(qas.data(), {T}),
            options
        );
    }
};

// Every third observation masked out.
ccd::ProcessingMask thinned(index_t T)
{
    ccd::ProcessingMask mask(T);
    for (index_t t = 0; t < T; ++t)
        mask.set(t, t % 3 != 1);
    return mask;
}

// The MaskedSeries invariant: the same observations apply_mask() gives for its
// mask, in the same order, and size() == mask().count().
void expect_in_step(const ccd::HarmonicWorkspace& ws, const ccd::MaskedSeries& s)
{
    const ccd::MaskedData expected = ccd::apply_mask(ws, s.mask());

    ASSERT_EQ(s.size(), s.mask().count());
    ASSERT_EQ(s.size(), expected.observations);
    ASSERT_EQ(s.spectral().extent(0), expected.bands);

    for (index_t i = 0; i < s.size(); ++i)
    {
        EXPECT_EQ(s.dates()(i), expected.dates[i]) << "position " << i;
        for (index_t b = 0; b < expected.bands; ++b)
            EXPECT_EQ(s.spectral()(b, i), expected.spectral[b * expected.observations + i])
                << "band " << b << " position " << i;
    }
}

std::vector<std::int64_t> to_vector(ccd::ArrayView<const std::int64_t, 1> v)
{
    std::vector<std::int64_t> out;
    for (index_t i = 0; i < v.size(); ++i)
        out.push_back(v(i));
    return out;
}

} // namespace

TEST(MaskedSeries, ResetIsApplyMask)
{
    const Pixel px(40);
    const auto ws = px.workspace();

    ccd::MaskedSeries s;
    s.reset(ws, thinned(px.T));
    expect_in_step(ws, s);

    // A reused MaskedSeries holds only the new pixel's observations.
    ccd::ProcessingMask all(px.T);
    for (index_t t = 0; t < px.T; ++t)
        all.set(t, true);
    s.reset(ws, all);
    expect_in_step(ws, s);
    EXPECT_EQ(s.size(), px.T);
}

TEST(MaskedSeries, DropsKeepMaskAndObservationsInStep)
{
    const Pixel px(60);
    const auto ws = px.workspace();

    ccd::MaskedSeries s;
    s.reset(ws, thinned(px.T));

    // First, last, middle, and the same position repeatedly.
    ccd::Window w(0, 1);
    for (const index_t position : {index_t{0}, index_t{37}, index_t{17}, index_t{5}, index_t{5}, index_t{5}})
    {
        ASSERT_LT(position, s.size());
        const std::int64_t gone = s.dates()(position);
        const std::int64_t next = s.dates()(position + 1);

        s.drop(position, w);
        expect_in_step(ws, s);

        // The observation after it takes its position.
        EXPECT_EQ(s.dates()(position), next);
        EXPECT_NE(s.dates()(position), gone);
    }
    EXPECT_EQ(s.size(), 40u - 6u);
}

// The coordinate rule: a window bound above the dropped position moves down
// by one, one at or below it stays.
TEST(MaskedSeries, DropMovesTheWindowWithTheObservations)
{
    const Pixel px(30);
    const auto ws = px.workspace();
    ccd::MaskedSeries s;

    const auto dropped = [&](index_t position, ccd::Window w)
    {
        s.reset(ws, thinned(px.T));
        s.drop(position, w);
        return std::make_pair(w.start, w.stop);
    };

    // Lookback: the observation just before the window.
    EXPECT_EQ(dropped(4, {5, 12}), std::make_pair(index_t{4}, index_t{11}));
    // Extend: the first peek observation, at window.stop.
    EXPECT_EQ(dropped(12, {5, 12}), std::make_pair(index_t{5}, index_t{12}));
    // Inside the window, and its first observation.
    EXPECT_EQ(dropped(8, {5, 12}), std::make_pair(index_t{5}, index_t{11}));
    EXPECT_EQ(dropped(5, {5, 12}), std::make_pair(index_t{5}, index_t{11}));
    // After the window.
    EXPECT_EQ(dropped(15, {5, 12}), std::make_pair(index_t{5}, index_t{12}));
}

TEST(MaskedSeries, RejectedOutlierDropChangesNothing)
{
    const Pixel px(40);
    const auto ws = px.workspace();

    ccd::MaskedSeries s;
    s.reset(ws, thinned(px.T));

    const ccd::ProcessingMask mask_before = s.mask();
    const auto dates_before = to_vector(s.dates());

    ccd::Window w(4, 14);
    ccd::ProcessingMask outliers(w.size());
    outliers.set(0, true);
    outliers.set(3, true);
    outliers.set(9, true);

    std::vector<std::int64_t> asked;
    const bool dropped = s.drop_outliers_if(outliers, w,
        [&](ccd::ArrayView<const std::int64_t, 1> shortened)
        {
            asked = to_vector(shortened);
            return false;
        });

    EXPECT_FALSE(dropped);
    EXPECT_EQ(w.start, 4u);
    EXPECT_EQ(w.stop, 14u);
    EXPECT_EQ(s.mask().data(), mask_before.data());
    EXPECT_EQ(to_vector(s.dates()), dates_before);
    expect_in_step(ws, s);

    // `accept` was asked with the shortened window: the window's dates
    // without the three flagged.
    std::vector<std::int64_t> expected;
    for (index_t j = 0; j < 10; ++j)
        if (j != 0 && j != 3 && j != 9)
            expected.push_back(dates_before[4 + j]);
    EXPECT_EQ(asked, expected);
}

TEST(MaskedSeries, AcceptedOutlierDropCommitsMaskObservationsAndWindow)
{
    const Pixel px(40);
    const auto ws = px.workspace();

    ccd::MaskedSeries s;
    s.reset(ws, thinned(px.T));
    const auto dates_before = to_vector(s.dates());
    const index_t size_before = s.size();

    ccd::Window w(4, 14);
    ccd::ProcessingMask outliers(w.size());
    outliers.set(0, true);
    outliers.set(3, true);
    outliers.set(9, true);

    const bool dropped = s.drop_outliers_if(outliers, w,
        [](ccd::ArrayView<const std::int64_t, 1>) { return true; });

    EXPECT_TRUE(dropped);
    EXPECT_EQ(w.start, 4u);
    EXPECT_EQ(w.stop, 11u);
    EXPECT_EQ(s.size(), size_before - 3);
    expect_in_step(ws, s);

    // The flagged dates are gone, the rest kept in order.
    std::vector<std::int64_t> expected;
    for (index_t i = 0; i < size_before; ++i)
        if (i != 4 && i != 7 && i != 13)
            expected.push_back(dates_before[i]);
    EXPECT_EQ(to_vector(s.dates()), expected);
}

TEST(MaskedSeries, CountUntilAndTakeMask)
{
    const Pixel px(20);
    const auto ws = px.workspace();

    ccd::ProcessingMask all(px.T);
    for (index_t t = 0; t < px.T; ++t)
        all.set(t, true);

    ccd::MaskedSeries s;
    s.reset(ws, all);

    // dates[5] == dates[6]: both are on or before that day.
    EXPECT_EQ(s.count_until(px.dates[5]), 7u);
    EXPECT_EQ(s.count_until(px.dates[5] - 1), 5u);
    EXPECT_EQ(s.count_until(px.dates[0] - 1), 0u);
    EXPECT_EQ(s.count_until(px.dates[px.T - 1]), px.T);

    const ccd::ProcessingMask taken = s.take_mask();
    EXPECT_EQ(taken.size(), px.T);
    EXPECT_EQ(taken.count(), px.T);
    EXPECT_EQ(s.size(), 0u);
}

TEST(Variogram, ReadsAndUsable)
{
    const ccd::Variogram v{{1.5, 2.5, 3.5}};
    EXPECT_EQ(v[1], 2.5);
    EXPECT_TRUE(v.usable());

    const ccd::Variogram nan{{1.5, std::numeric_limits<double>::quiet_NaN(), 3.5}};
    EXPECT_FALSE(nan.usable());
}

// measure_cadence() is the two adjust_* kernels of the caller's options,
// at the reference cadence (no adjustment) and at a denser one.
TEST(StandardCadence, MeasureCadenceIsTheAdjustedPeekAndThreshold)
{
    for (const std::int64_t gap : {16, 8, 5})
    {
        std::vector<std::int64_t> dates(100);
        for (std::size_t i = 0; i < dates.size(); ++i)
            dates[i] = 730000 + gap * static_cast<std::int64_t>(i);
        const auto view = ccd::ArrayView<const std::int64_t, 1>::contiguous(
            dates.data(), {dates.size()});

        const ccd::HarmonicOptions h;
        const ccd::standard::Cadence c = ccd::standard::measure_cadence(view, h);

        const index_t peek = ccd::adjust_peek(view, h.PEEK_SIZE, h.PEEK_REFERENCE_DAYS);
        EXPECT_EQ(c.peek, peek) << gap;
        EXPECT_EQ(c.change_threshold,
            ccd::adjust_change_threshold(peek, h.PEEK_SIZE, h.CHANGE_THRESHOLD,
                h.DETECTION_BANDS.size(), h.CHANGE_PROBABILITY)) << gap;

        if (gap == 16)
        {
            EXPECT_EQ(c.peek, h.PEEK_SIZE);
            EXPECT_EQ(c.change_threshold, h.CHANGE_THRESHOLD);
        }
        else
        {
            EXPECT_GT(c.peek, h.PEEK_SIZE) << gap;
            EXPECT_LT(c.change_threshold, h.CHANGE_THRESHOLD) << gap;
        }
    }

    // Fewer than two dates: the caller's values.
    const std::int64_t one = 730000;
    const ccd::HarmonicOptions h;
    const auto c = ccd::standard::measure_cadence(
        ccd::ArrayView<const std::int64_t, 1>::contiguous(&one, {1}), h);
    EXPECT_EQ(c.peek, h.PEEK_SIZE);
    EXPECT_EQ(c.change_threshold, h.CHANGE_THRESHOLD);
}

// The workspace refers to the caller's options, cannot be built from a
// temporary, and gives only const access: nothing below detect_pixel can
// write them.
TEST(HarmonicWorkspace, RefersToTheCallerOptionsConstOnly)
{
    const Pixel px(10);
    const auto ws = px.workspace();
    EXPECT_EQ(&ws.options(), &px.options);

    using Dates = ccd::ArrayView<const std::int64_t, 1>;
    using Spectral = ccd::ArrayView<const double, 2>;
    using Qas = ccd::ArrayView<const std::uint8_t, 1>;
    static_assert(!std::is_constructible_v<ccd::HarmonicWorkspace,
        Dates, Spectral, Qas, ccd::HarmonicOptions&&>);
    static_assert(std::is_same_v<
        decltype(std::declval<ccd::HarmonicWorkspace&>().options()),
        const ccd::HarmonicOptions&>);
}
