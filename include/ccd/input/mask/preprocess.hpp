#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>

#include "ccd/core/types.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/input/pmask.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/input/harmonic.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/mask/filter.hpp"

namespace ccd
{

namespace preprocess
{

//==============================================================================
//
// Preprocessing plans
//
// What each procedure does to a pixel's observations before it fits them:
// an optional affine transform of one band, then a chain of filter steps
// that builds the processing mask.
//
//   PreprocessPlan     one ProcedurePlan per procedure (standard, snow,
//                      insufficient_clear); Run::pixel picks the one of
//                      the procedure it selected.
//   ProcedurePlan      the transform and the chain of one procedure.
//   Chain              up to CHAIN_CAPACITY steps, stored inline, run in
//                      order by apply().
//   Step               one filter: a StepKind and the parameters it reads.
//
// default_plan(input, shared) is the plan of an input and its resolved
// shared values, and the one a Run runs when it is given none. A plan is checked against the band
// count once per call by validate(), never per pixel.
//
// apply() starts from a mask that keeps every observation; each step then
// narrows it (KeepQa, ThermalRange, BandRange, GreenMedian, Dedupe) or
// widens it (AddQa). Each step is one loop over the time series, chosen by a
// switch on its kind.
//
//==============================================================================

// The bound and range types of the thermal and reflectance filters; see
// filter::in_range() for the comparison each inclusive flag selects.
using Bound = filter::Bound;
using Range = filter::Range;

// A set of QA codes, indexed by code.
using QaSet = std::bitset<256>;

inline QaSet qa_set(std::initializer_list<std::uint8_t> codes)
{
    QaSet set;
    for (const std::uint8_t code : codes)
        set.set(code);
    return set;
}

enum class StepKind : std::uint8_t
{
    // Keep only the observations whose QA code is in `codes`.
    KeepQa,

    // Also keep every observation whose QA code is in `codes`, whatever an
    // earlier step decided. Must be followed by Dedupe.
    AddQa,

    // Keep only the observations whose `band` value is in_range() of
    // `range` (filter::apply_thermal_filter).
    ThermalRange,

    // Keep only the observations on which every one of `bands` is
    // in_range() of `range`, the bands tested in list order
    // (filter::apply_saturation_filter).
    BandRange,

    // Drop the kept observations whose `band` value is at or above the
    // median of that band over the kept observations dated <= `max_date`,
    // plus `green_offset`, at the dates `scope` names
    // (filter::apply_green_median_filter).
    GreenMedian,

    // Keep the first kept observation of each date
    // (filter::remove_duplicate_dates).
    Dedupe
};

const char* name(StepKind kind);

//------------------------------------------------------------------------------
//
// One step. `kind` says which of the other fields it reads:
//
//   KeepQa, AddQa   codes
//   ThermalRange    band, range
//   BandRange       bands, range
//   GreenMedian     band, green_offset, max_date, scope
//   Dedupe          none
//
// The named constructors below fill exactly those.
//
//------------------------------------------------------------------------------

struct Step
{
    StepKind kind = StepKind::Dedupe;

    QaSet codes;
    BandList bands;
    index_t band = NO_BAND;
    Range range{Bound{0.0, false}, Bound{0.0, false}};

    scalar_t green_offset = 0.0;
    std::int64_t max_date = 0;
    GreenFilterScope scope = GreenFilterScope::StatWindow;

    static Step keep_qa(const QaSet& codes)
    {
        Step s;
        s.kind = StepKind::KeepQa;
        s.codes = codes;
        return s;
    }

    static Step add_qa(const QaSet& codes)
    {
        Step s;
        s.kind = StepKind::AddQa;
        s.codes = codes;
        return s;
    }

    static Step thermal_range(index_t band, const Range& range)
    {
        Step s;
        s.kind = StepKind::ThermalRange;
        s.band = band;
        s.range = range;
        return s;
    }

    static Step band_range(const BandList& bands, const Range& range)
    {
        Step s;
        s.kind = StepKind::BandRange;
        s.bands = bands;
        s.range = range;
        return s;
    }

    static Step green_median(
        index_t band,
        scalar_t green_offset,
        std::int64_t max_date,
        GreenFilterScope scope
    )
    {
        Step s;
        s.kind = StepKind::GreenMedian;
        s.band = band;
        s.green_offset = green_offset;
        s.max_date = max_date;
        s.scope = scope;
        return s;
    }

    static Step dedupe()
    {
        Step s;
        s.kind = StepKind::Dedupe;
        return s;
    }
};

//------------------------------------------------------------------------------
//
// Chain
//
// An ordered list of at most CHAIN_CAPACITY steps, stored inline so that a
// plan allocates nothing. push_back() onto a full chain throws
// std::invalid_argument; `count` is public, and validate() rejects a chain
// whose count exceeds the capacity.
//
//------------------------------------------------------------------------------

constexpr index_t CHAIN_CAPACITY = 8;

struct Chain
{
    std::array<Step, CHAIN_CAPACITY> steps{};
    index_t count = 0;

    Chain() = default;

    Chain(std::initializer_list<Step> list)
    {
        for (const Step& step : list)
            push_back(step);
    }

    void push_back(const Step& step);

    index_t size() const
    {
        return count;
    }

    bool empty() const
    {
        return count == 0;
    }

    const Step& operator[](index_t i) const
    {
        return steps[i];
    }

    Step& operator[](index_t i)
    {
        return steps[i];
    }

    const Step* begin() const
    {
        return steps.data();
    }

    const Step* end() const
    {
        return steps.data() + count;
    }
};

//------------------------------------------------------------------------------
//
// The transform of one band of the pixel's per-pixel copy, before the chain
// runs:
//
//     value <- value * scale + offset
//
// Run::pixel applies it to its private scalar_t copy; the caller's buffer is
// never written.
//
//------------------------------------------------------------------------------

struct Affine
{
    index_t band = NO_BAND;
    scalar_t scale = 1.0;
    scalar_t offset = 0.0;
};

struct ProcedurePlan
{
    std::optional<Affine> transform;
    Chain chain;
};

struct PreprocessPlan
{
    ProcedurePlan standard;
    ProcedurePlan snow;
    ProcedurePlan insufficient_clear;
};

// run_pixel() and run_cube() take a plan as they take a QaTable: shared,
// read-only, by every pixel and thread of the call.
using PreprocessPlanPtr = std::shared_ptr<const PreprocessPlan>;

//------------------------------------------------------------------------------
//
// The plan of `input` and the shared values a run resolved for it
// (config/shared.hpp: the green filter in the input's units).
//
// Every procedure's chain begins with the same four steps:
//
//   KeepQa {qa_clear, qa_water}
//   ThermalRange (the "thermal" row, therm_min/therm_max and their flags)
//                                    -- omitted for an input without one
//   BandRange (saturation_bands, refl_min/refl_max and their flags)
//   Dedupe
//
// after which
//
//   standard             ends;
//   snow                 adds AddQa {qa_snow}, Dedupe;
//   insufficient_clear   adds GreenMedian (the "green" row, shared
//                        green_filter, stat_ord, green_filter_scope), Dedupe.
//
// standard's transform is thermal_transform(input); the other two have
// none, so their ThermalRange tests the caller's own unit (see the "Units"
// block in input.hpp).
//
//------------------------------------------------------------------------------

PreprocessPlan default_plan(const Input& input, const config::Shared& shared);

//------------------------------------------------------------------------------
//
// The Standard procedure's transform
//
//     thermal <- thermal * thermal_scale + thermal_offset
//
// of the input's "thermal" row, which puts it into the unit its ThermalRange
// step tests; none when that is the identity (1.0, 0.0) or the input has no
// thermal row. The defaults are pyccd's collection-1 transform, Kelvin x 10 ->
// Celsius x 100 (the collection-2 form `* 0.00341802 + 149.0 - 273.15`
// yields plain Celsius).
//
// default_plan() gives it to Standard only, matching pyccd, where
// kelvin_to_celsius() is called inside standard_procedure(). See input.hpp's
// "Units" block for the consequences for the other two procedures.
//
//------------------------------------------------------------------------------
std::optional<Affine> thermal_transform(const Input& input);

//------------------------------------------------------------------------------
//
// The plan check a Run makes once per call, before any work.
//
// Throws std::invalid_argument, naming the procedure and the step, when a
// procedure's
//
//   * chain holds more than CHAIN_CAPACITY steps;
//   * transform band, ThermalRange band, GreenMedian band or a BandRange
//     band is not below `bands`;
//   * GreenMedian step has no band (NO_BAND);
//   * transform band is in one of its chain's BandRange steps;
//   * AddQa step is not immediately followed by a Dedupe step.
//
//------------------------------------------------------------------------------

void validate(const PreprocessPlan& plan, index_t bands);

//------------------------------------------------------------------------------
//
// Run `chain` over one pixel's (B, T) `spectral`, its dates and its QA
// codes. `mask` is resized to T, set to keep every observation, and then
// narrowed or widened by each step in order.
//
// The chain is assumed valid for `spectral`'s band count (validate()).
//
//------------------------------------------------------------------------------

void apply(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask
);

// apply()'s steps on `mask` as it is (T entries): a chain after another.
void apply_steps(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask
);

} // namespace preprocess

using PreprocessPlanPtr = preprocess::PreprocessPlanPtr;

} // namespace ccd
