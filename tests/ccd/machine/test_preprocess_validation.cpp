//==============================================================================
//
// The preprocessing plan's validation (input/mask/chain.hpp): every plan
// validate() rejects, and that a configuration whose plan it rejects is
// refused before any pixel runs.
//
//==============================================================================

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.hpp"

#include "ccd/ccd.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/chain.hpp"

#include "dump.hpp"
#include "preprocess_reference.hpp"
#include "series.hpp"
#include "units.hpp"
#include "synthetic.hpp"
#include "synthetic_bands.hpp"
#include "band_profiles.hpp"

namespace
{

namespace pp = ccd::preprocess;

using ccd::index_t;

using golden::PixelView;

using namespace golden::chains;

// validate() throws invalid_argument with `fragment` in its message.
void expect_rejected(const pp::PreprocessPlan& plan, index_t bands, const std::string& fragment)
{
    try
    {
        pp::validate(plan, bands);
        ADD_FAILURE() << "accepted; expected: " << fragment;
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
}

} // namespace

TEST(PlanValidation, CcdcPlanIsValid)
{
    EXPECT_NO_THROW(pp::validate(plan_of(Options{}), 7));
    EXPECT_NO_THROW(pp::validate(plan_of(physical_options()), 7));
    for (const auto& profile : golden::bands::profiles())
        EXPECT_NO_THROW(pp::validate(golden::plan_of(profile.config, profile.h), profile.h.bands()))
            << profile.name;
}

TEST(PlanValidation, BandIndexNotBelowBandCount)
{
    const Options h;
    const pp::Range r{{0.0, false}, {1.0, false}};

    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.snow.chain[6] = pp::Step::thermal_range(7, r);
        expect_rejected(plan, 7, "snow step 6 (ThermalRange): band 7");
    }
    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.standard.chain[2] = pp::Step::band_range({0, 1, 9}, r);
        expect_rejected(plan, 7, "standard step 2 (BandRange): band 9");
    }
    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.insufficient_clear.chain[4].band = 8;
        expect_rejected(plan, 7, "insufficient_clear step 4 (GreenMedian): band 8");
    }
    {
        pp::PreprocessPlan plan = plan_of(h);
        plan.standard.transform->band = 7;
        expect_rejected(plan, 7, "standard transform: band 7");
    }
    // The plan is valid for 7 bands and not for 6: thermal is band 6.
    expect_rejected(plan_of(h), 6, "band 6 is not below the band count 6");
}

TEST(PlanValidation, GreenMedianWithoutBand)
{
    pp::PreprocessPlan plan = plan_of(Options{});
    plan.insufficient_clear.chain[4].band = ccd::NO_BAND;
    expect_rejected(plan, 7, "insufficient_clear step 4 (GreenMedian): needs a band");
}

TEST(PlanValidation, TransformBandInBandRange)
{
    pp::PreprocessPlan plan = plan_of(Options{});
    plan.snow.transform = pp::Affine{3, 2.0, 1.0};
    expect_rejected(plan, 7, "snow transform: band 3 is also tested by step 2 (BandRange)");

    // The same band outside every BandRange is accepted.
    plan.snow.transform = pp::Affine{6, 2.0, 1.0};
    EXPECT_NO_THROW(pp::validate(plan, 7));
}

TEST(PlanValidation, AddQaNotFollowedByOnePerDate)
{
    const Options options;
    const ccd::DataSpec& h = options.spec;
    {
        // AddQa last.
        pp::PreprocessPlan plan = plan_of(options);
        plan.standard.chain.push_back(pp::Step::add_qa(pp::qa_set({h.qa_snow})));
        expect_rejected(plan, 7, "standard step 4 (AddQa): must be followed by a OnePerDate");
    }
    {
        // AddQa followed by another step.
        pp::PreprocessPlan plan = plan_of(options);
        plan.snow.chain[5] = pp::Step::keep_qa(pp::qa_set({h.qa_clear}));
        expect_rejected(plan, 7, "snow step 4 (AddQa): must be followed by a OnePerDate");
    }
}

TEST(PlanValidation, ChainOverCapacity)
{
    pp::Chain chain;
    for (index_t i = 0; i < pp::CHAIN_CAPACITY; ++i)
        chain.push_back(pp::Step::one_per_date());
    EXPECT_THROW(chain.push_back(pp::Step::one_per_date()), std::invalid_argument);

    pp::PreprocessPlan plan = plan_of(Options{});
    plan.standard.chain.count = pp::CHAIN_CAPACITY + 1;
    expect_rejected(plan, 7, "standard chain: holds 9 steps; at most 8");
}

// A configuration whose plan validate() refuses (an add_qa without its
// one_per_date) is refused when it is resolved, before any pixel: detect() and
// detect_cube() throw rather than landing in CubeOutput::failures.
TEST(PlanValidation, EntryPointsReject)
{
    const golden::Cube c = golden::make_cube(11, golden::Regime::Cadence16, 2, 2);
    const std::size_t T = c.T;
    const ccd::DataSpec h;

    ccd::config::Config bad = golden::pyccd();
    auto& snow_steps = bad.preprocess.at("snow").steps;
    snow_steps.erase(snow_steps.begin() + 5);   // the one_per_date after add_qa

    const PixelView px = golden::view_of(c.dates.data(), c.spectral.data(), 7, c.qas.data(), T);
    EXPECT_THROW(golden::detect(golden::spec_of(h, bad), px), std::invalid_argument);

    EXPECT_THROW(
        ccd::detect_cube(
            golden::spec_of(h, bad),
            ccd::ArrayView<const std::int64_t, 1>::contiguous(c.dates.data(), {T}),
            ccd::ArrayView<const double, 4>::contiguous(c.spectral.data(), {c.H, c.W, 7, T}),
            ccd::ArrayView<const std::uint8_t, 3>::contiguous(c.qas.data(), {c.H, c.W, T}),
            1),
        std::invalid_argument);
}
