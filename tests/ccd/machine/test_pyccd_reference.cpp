//==============================================================================
//
// The three procedures on pyccd's reference pixel (test_3657_3610, 443
// observations) against pyccd 2021.7.19's output: every segment's days,
// observation count, change probability (1e-8) and curve_qa, each band's
// rmse, intercept, magnitude and coefficients (1e-6), and the processing
// mask, bit for bit. The references are in tests/ccd/data
// (<procedure>_reference.csv and <procedure>_mask_reference.csv), written by
// make_procedure_reference.py. InsufficientClear and PermanentSnow run as
// fit_whole_series(), Standard as the machine, each on its "ccdc" chain's
// mask.
//
// And the Atlanta pixel whose half-day median gap pyccd's np.median keeps,
// through detect(), against pyccd's values pinned here.
//
//==============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.hpp"
#include "reference.hpp"
#include "series.hpp"

#include "ccd/ccd.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/procedure.hpp"
#include "ccd/machine/detector.hpp"   // internal (src/)
#include "ccd/results/segment.hpp"
#include "ccd/strategies/edge_segments/lasso.hpp"

namespace
{

struct ExpectedBand
{
    double rmse;
    double intercept;
    double magnitude;
    std::vector<double> coefficients;
};

struct ExpectedModel
{
    int start_day;
    int end_day;
    int break_day;
    int observation_count;
    double change_probability;
    int curve_qa;
    std::array<ExpectedBand, 7> bands;
};

// pyccd's segments, one per row: the days, the count, the change probability
// and curve_qa, then per band its rmse, intercept, magnitude and seven
// coefficients.
std::vector<ExpectedModel> read_models_reference(const std::string& filename)
{
    std::vector<ExpectedModel> models;
    for (const auto& row : golden::read_reference_rows(filename))
    {
        std::size_t at = 0;
        auto next = [&]()
        {
            if (at == row.size())
                throw std::runtime_error(filename + ": a row of " + std::to_string(row.size())
                                         + " values; a segment has 76");
            return row[at++];
        };

        ExpectedModel m{};
        m.start_day = static_cast<int>(next());
        m.end_day = static_cast<int>(next());
        m.break_day = static_cast<int>(next());
        m.observation_count = static_cast<int>(next());
        m.change_probability = next();
        m.curve_qa = static_cast<int>(next());

        for (auto& band : m.bands)
        {
            band.rmse = next();
            band.intercept = next();
            band.magnitude = next();
            for (int i = 0; i < 7; ++i)
                band.coefficients.push_back(next());
        }

        models.push_back(std::move(m));
    }
    return models;
}

// pyccd's processing mask, one 0 or 1 per row.
std::vector<int> read_mask_reference(const std::string& filename)
{
    std::vector<int> mask;
    for (const auto& row : golden::read_reference_rows(filename))
        mask.push_back(static_cast<int>(row.at(0)));
    return mask;
}

// The segment's days, count, change probability and curve_qa are pyccd's.
void expect_segment_matches(const ccd::ChangeModel& actual, const ExpectedModel& expected)
{
    EXPECT_EQ(actual.start_day, expected.start_day);
    EXPECT_EQ(actual.end_day, expected.end_day);
    EXPECT_EQ(actual.break_day, expected.break_day);
    EXPECT_EQ(actual.observation_count, expected.observation_count);
    EXPECT_NEAR(actual.change_probability, expected.change_probability, 1e-8);
    EXPECT_EQ(static_cast<int>(actual.curve_qa), expected.curve_qa);
}

// Each band's fit is pyccd's.
void expect_fits_match(const ccd::ChangeModel& actual, const ExpectedModel& expected)
{
    for (std::size_t b = 0; b < 7; ++b)
    {
        SCOPED_TRACE("band " + std::to_string(b));
        const auto& actual_band = actual.bands[b];
        const auto& expected_band = expected.bands[b];

        EXPECT_NEAR(actual_band.rmse, expected_band.rmse, 1e-6);
        EXPECT_NEAR(actual_band.intercept, expected_band.intercept, 1e-6);
        EXPECT_NEAR(actual_band.magnitude, expected_band.magnitude, 1e-6);

        const auto& coef = actual_band.coefficients;
        ASSERT_EQ(coef.size(), expected_band.coefficients.size());
        for (std::size_t i = 0; i < coef.size(); ++i)
            EXPECT_NEAR(coef[i], expected_band.coefficients[i], 1e-6) << "coefficient " << i;
    }
}

void expect_mask_matches(const ccd::ProcessingMask& mask, const std::vector<int>& expected)
{
    ASSERT_EQ(mask.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(static_cast<int>(mask.test(i)), expected[i]);
}

golden::Series reference_pixel()
{
    return golden::read_csv(CCD_TEST_DATA_DIR "/test_3657_3610_observations.csv", false);
}

// fit_whole_series() of `procedure` on the reference pixel, its chain `chain`
// of "ccdc", against `name`_reference.csv and `name`_mask_reference.csv.
void expect_whole_series_matches(ccd::PixelProcedure procedure, const std::string& chain,
                                 const std::string& name)
{
    const golden::Series data = reference_pixel();
    const golden::PixelView px = data.view();

    const ccd::Input input;
    const ccd::PixelSeries series(px.dates, px.spectral, px.qas);

    ccd::LassoOptions solver_options;
    ccd::LassoSolver solver(solver_options);
    ccd::LassoWorkspace lworkspace(data.size());

    const ccd::FitResult result = ccd::fit_whole_series(
        procedure,
        series,
        golden::chain_mask(series, input, chain, golden::ccdc()),
        static_cast<ccd::index_t>(
            golden::parameter<std::int64_t>(golden::resolved().stages, "edge_segments", "min_observations")),
        static_cast<ccd::index_t>(
            golden::parameter<std::int64_t>(golden::resolved().stages, "edge_segments", "coefficients.min")),
        lworkspace,
        solver
    );

    const auto expected = read_models_reference(name + "_reference.csv");
    const auto expected_mask = read_mask_reference(name + "_mask_reference.csv");

    ASSERT_EQ(result.models.size(), 1);
    expect_segment_matches(result.models[0], expected.at(0));
    expect_fits_match(result.models[0], expected.at(0));
    expect_mask_matches(result.mask, expected_mask);
}

} // namespace

TEST(PyccdReference, InsufficientClearMatchesPyccd)
{
    expect_whole_series_matches(ccd::PixelProcedure::InsufficientClear, "insufficient_clear", "clear");
}

TEST(PyccdReference, PermanentSnowMatchesPyccd)
{
    expect_whole_series_matches(ccd::PixelProcedure::PermanentSnow, "snow", "snow");
}

//------------------------------------------------------------------------------
// Segments 0 and 1 grow through a window of exactly refit's min_obs (24)
// observations, which ccdcpp refits and pyccd does not (it refits while the
// window holds fewer), so their fits are ccdcpp's own and differ from
// pyccd's; the other three are fitted as pyccd fits them.
//------------------------------------------------------------------------------
TEST(PyccdReference, StandardMatchesPyccd)
{
    golden::Series data = reference_pixel();
    const std::size_t T = data.size();

    const ccd::Input input;
    const ccd::config::Shared hshared;

    // The Standard chain's thermal transform, applied here: the machine reads
    // the block as it is.
    const ccd::index_t thermal_idx = input.index("thermal");
    for (std::size_t t = 0; t < T; ++t)
        data.spectral[thermal_idx * T + t] = data.spectral[thermal_idx * T + t] * 10.0 - 27315.0;

    const golden::PixelView px = data.view();
    const ccd::PixelSeries series(px.dates, px.spectral, px.qas);

    ccd::machine::Detector detector;
    const ccd::FitResult result = detector.fit(
        ccd::PixelProcedure::Standard,
        series,
        golden::chain_mask(series, input, "standard", golden::ccdc()),
        hshared,
        golden::strategies_of()
    );

    const auto expected = read_models_reference("standard_reference.csv");
    const auto expected_mask = read_mask_reference("standard_mask_reference.csv");

    ASSERT_EQ(result.models.size(), expected.size());
    ASSERT_EQ(expected.size(), 5);

    for (std::size_t m = 0; m < result.models.size(); ++m)
    {
        SCOPED_TRACE("segment " + std::to_string(m));
        expect_segment_matches(result.models[m], expected[m]);

        if (m <= 1)
        {
            double furthest = 0.0;
            for (std::size_t b = 0; b < 7; b++)
                furthest = std::max(furthest,
                    std::abs(result.models[m].bands[b].rmse - expected[m].bands[b].rmse));
            EXPECT_GT(furthest, 1e-3) << "segment " << m << " should be refitted at 24";
            continue;
        }

        expect_fits_match(result.models[m], expected[m]);
    }

    expect_mask_matches(result.mask, expected_mask);
}

//------------------------------------------------------------------------------
// Atlanta tile 024013, chunk 00030008, pixel (481, 393): a Standard pixel
// whose stat window has an even number of gaps with middle values 8 and 9.
// pyccd's np.median gives 8.5 and a peek window of round(96 / 8.501) = 11;
// a median that truncates to 8 gives round(96 / 8.001) = 12, the lower change
// threshold that goes with it, and different outliers in Extend
// (727551 kept, 732104 dropped), so the segment ended one observation early.
//
// The input is Collection 1 DN, fill rows dropped, the reflectance bound
// applied in the QA (code 4). The reference is pyccd 2021.7.19 on it, with
// the RLM convergence test as |x0 - x| > tol, a stable date sort and a stable
// argsort in find_closest_doy. The segment reaches the series end: ccdcpp
// ends it before the observation its last full peek starts at, its break
// day, so its end day and observation count are one observation short of
// pyccd's (738751, 960; pyccd grows the window once more before its
// series-end check). Its fits, and the mask, are pyccd's.
//------------------------------------------------------------------------------
TEST(PyccdReference, HalfDayMedianGapMatchesPyccd)
{
    const golden::Series data =
        golden::read_csv(CCD_TEST_DATA_DIR "/test_atlanta_024013_00030008_481_393.csv", false);
    const std::size_t T = data.size();
    ASSERT_EQ(T, 1760u);

    const auto result = golden::detect(golden::spec_of(ccd::Input{}), data.view());

    ASSERT_EQ(result.models.size(), 1u);
    const auto& m = result.models[0];
    EXPECT_EQ(m.start_day, 724440);
    EXPECT_EQ(m.end_day, 738727);
    EXPECT_EQ(m.break_day, 738751);
    EXPECT_EQ(m.observation_count, 959);
    EXPECT_EQ(m.change_probability, 0.0);
    EXPECT_EQ(static_cast<int>(m.curve_qa), 8);

    // pyccd's intercept and RMSE per band.
    const double expected[7][2] = {
        {13856.652256036561, 104.84273922901038},
        {17861.914274833238, 108.69803572362223},
        {21401.59535008651, 122.95137094618717},
        {1122.2419983962932, 297.2795540223367},
        {36779.271735154354, 258.6001913744581},
        {27717.356902630087, 180.5723178104279},
        {1652.8467279281745, 358.25635528737234},
    };
    ASSERT_EQ(m.bands.size(), 7u);
    for (std::size_t b = 0; b < 7; ++b)
    {
        EXPECT_NEAR(m.bands[b].intercept, expected[b][0], 1e-6) << "band " << b;
        EXPECT_NEAR(m.bands[b].rmse, expected[b][1], 1e-6) << "band " << b;
    }

    std::size_t kept = 0;
    bool kept_727551 = false, kept_732104 = false;
    for (std::size_t t = 0; t < T; ++t)
    {
        if (!result.mask[t])
            continue;
        ++kept;
        kept_727551 |= data.dates[t] == 727551;
        kept_732104 |= data.dates[t] == 732104;
    }
    EXPECT_EQ(kept, 973u);
    EXPECT_FALSE(kept_727551);
    EXPECT_TRUE(kept_732104);
}
