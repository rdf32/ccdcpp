#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/input/processing_mask.hpp"
#include "ccd/core/array_view.hpp"
#include "ccd/input/data_spec.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/reason.hpp"
#include "ccd/input/procedure.hpp"

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
//   PreprocessPlan     one ProcedurePlan per procedure (standard,
//                      insufficient_clear, snow); Run::prepare picks the one of
//                      the procedure it selected (of()).
//   ProcedurePlan      the transform and the chain of one procedure.
//   Chain              up to CHAIN_CAPACITY steps, stored inline, run in
//                      order by apply().
//   Step               one filter: a StepKind and the parameters it reads.
//
// A plan is the configuration's preprocess section resolved against an
// data spec (config::resolve(), plan_of() in run/algorithm.hpp): the
// steps by name with their parameters in physical units, as data units,
// rows and codes. It is checked against the band count by validate() when it
// is resolved, never per pixel.
//
// apply() starts from a mask that keeps every observation; each step then
// narrows it (KeepQa, ThermalRange, BandRange, GreenMedian, OnePerDate) or
// widens it (AddQa). Each step is one loop over the time series, chosen by a
// switch on its kind; a Custom step runs its own rule (CustomStep).
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
    // earlier step decided. Must be followed by OnePerDate.
    AddQa,

    // Keep only the observations whose `band` value is in_range() of
    // `range` (filter::apply_thermal_range).
    ThermalRange,

    // Keep only the observations on which every one of `bands` is
    // in_range() of `range`, the bands tested in list order
    // (filter::apply_band_range).
    BandRange,

    // Drop the kept observations whose `band` value is at or above the
    // median of that band over the kept observations dated <= `max_date`,
    // plus `green_offset`, at the dates `scope` names
    // (filter::apply_green_median).
    GreenMedian,

    // Keep the first kept observation of each date
    // (filter::apply_one_per_date).
    OnePerDate,

    // A step a registry adds beyond these (config::Registry::add_step()):
    // `custom` decides.
    Custom
};

const char* name(StepKind kind);

//------------------------------------------------------------------------------
//
// A step beyond the built-in kinds: what a registered step type's make()
// returns as its Step's `custom` (config::Registry::add_step()). Immutable;
// shared by every pixel and thread of a run.
//
//------------------------------------------------------------------------------

class CustomStep
{
public:
    virtual ~CustomStep() = default;

    // Narrows or widens `mask` (T entries) over one pixel, as apply() runs a
    // built-in step.
    virtual void apply(
        ArrayView<const std::int64_t, 1> dates,
        ArrayView<const scalar_t, 2> spectral,
        ArrayView<const std::uint8_t, 1> qas,
        ProcessingMask& mask
    ) const = 0;

    // Throws std::invalid_argument when the step reads a row not below
    // `bands` (validate()).
    virtual void validate(index_t bands) const = 0;

    // Its name and fields, as a plan describes a step.
    virtual std::string describe() const = 0;
};

//------------------------------------------------------------------------------
//
// One step. `kind` says which of the other fields it reads:
//
//   KeepQa, AddQa   codes
//   ThermalRange    band, range
//   BandRange       bands, range
//   GreenMedian     band, green_offset, max_date, scope
//   OnePerDate      none
//   Custom          custom
//
// The named constructors below fill exactly those.
//
//------------------------------------------------------------------------------

struct Step
{
    StepKind kind = StepKind::OnePerDate;

    QaSet codes;
    BandList bands;
    index_t band = NO_BAND;
    Range range{Bound{0.0, false}, Bound{0.0, false}};

    scalar_t green_offset = 0.0;
    std::int64_t max_date = 0;
    GreenFilterScope scope = GreenFilterScope::StatWindow;

    std::shared_ptr<const CustomStep> custom;

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

    static Step one_per_date()
    {
        Step s;
        s.kind = StepKind::OnePerDate;
        return s;
    }

    static Step of(std::shared_ptr<const CustomStep> custom)
    {
        Step s;
        s.kind = StepKind::Custom;
        s.custom = std::move(custom);
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
// Run::prepare applies it (apply()) to its private scalar_t copy; the caller's
// buffer is never written.
//
//------------------------------------------------------------------------------

struct Affine
{
    index_t band = NO_BAND;
    scalar_t scale = 1.0;
    scalar_t offset = 0.0;
};

// One value through `affine`: the one expression the transform pass and
// stored_range() evaluate.
inline scalar_t apply(const Affine& affine, scalar_t value)
{
    return value * affine.scale + affine.offset;
}

// The range of stored values whose transform lies in `range`: for every
// double v, in_range(v, stored_range(range, affine)) is
// in_range(apply(affine, v), range), NaN included (a NaN stays NaN through
// the transform). For a positive finite scale and a finite offset apply() is
// non-decreasing in v, so each end is the first or last double on its side
// of the bound, found by bisection over the doubles in order; another scale
// or offset is refused (std::invalid_argument). A thermal_range step of a
// procedure that does not transform its thermal row tests the stored row
// with it, so its bounds are degrees Celsius on every data spec (docs/units.md).
// On Collection-1 DN (Kelvin x 10, the default transform) -93.2 and 70.7
// degrees are 1799.5 and 3438.5 exactly: 179.95 and 343.85 kelvin.
Range stored_range(const Range& range, const Affine& affine);

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

    // The plan of `procedure`: plan.of(PixelProcedure::PermanentSnow) is
    // plan.snow.
    const ProcedurePlan& of(PixelProcedure procedure) const
    {
        switch (procedure)
        {
            case PixelProcedure::InsufficientClear:
                return insufficient_clear;

            case PixelProcedure::PermanentSnow:
                return snow;

            case PixelProcedure::Standard:
                break;
        }

        return standard;
    }

    ProcedurePlan& of(PixelProcedure procedure)
    {
        return const_cast<ProcedurePlan&>(std::as_const(*this).of(procedure));
    }
};


//------------------------------------------------------------------------------
//
// The Standard procedure's transform
//
//     thermal <- thermal * thermal_scale + thermal_offset
//
// of the spec's "thermal" row, which puts it into the unit its ThermalRange
// step tests; none when that is the identity (1.0, 0.0) or the spec has no
// thermal row. The defaults are pyccd's collection-1 transform, Kelvin x 10 ->
// Celsius x 100 (the collection-2 form `* 0.00341802 + 149.0 - 273.15`
// yields plain Celsius).
//
// A procedure whose configuration reads thermal = "celsius" (config::Thermal)
// reads its thermal row through it: "ccdc" does for Standard only, matching pyccd, where
// kelvin_to_celsius() is called inside standard_procedure(). The thermal
// range of the procedures that do not read it transformed tests the stored
// row against the bounds taken back through it (stored_range()), so every
// procedure's range is in degrees Celsius (docs/units.md).
//
//------------------------------------------------------------------------------
std::optional<Affine> thermal_transform(const DataSpec& spec);

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
//   * AddQa step is not immediately followed by a OnePerDate step;
//   * Custom step's validate() refuses `bands`.
//
//------------------------------------------------------------------------------

void validate(const PreprocessPlan& plan, index_t bands);

//------------------------------------------------------------------------------
//
// Run `chain` over one pixel's (B, T) `spectral`, its dates and its QA
// codes. `mask` is resized to T, set to keep every observation, and then
// narrowed or widened by each step in order.
//
// With `reasons`, it is resized to T too, Reason::InSegment everywhere, and
// each step writes the code of what it removes (reason_of() its kind) and
// InSegment where it takes an observation back: reasons[t] is InSegment
// exactly where mask[t] keeps.
//
// The chain is assumed valid for `spectral`'s band count (validate()).
//
//------------------------------------------------------------------------------

void apply(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask,
    std::vector<std::uint8_t>* reasons = nullptr
);

// apply()'s steps on `mask` (and `reasons`) as they are (T entries): a chain
// after another.
void apply_steps(
    const Chain& chain,
    ArrayView<const std::int64_t, 1> dates,
    ArrayView<const scalar_t, 2> spectral,
    ArrayView<const std::uint8_t, 1> qas,
    ProcessingMask& mask,
    std::vector<std::uint8_t>* reasons = nullptr
);

// The reason code of an observation a step of `kind` removes: KeepQa's is
// Reason::Qa, OnePerDate's Reason::DuplicateDate, ... (AddQa removes none).
Reason reason_of(StepKind kind);

} // namespace preprocess


} // namespace ccd
