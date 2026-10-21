#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/config/preprocess.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/mask/preprocess.hpp"

namespace ccd
{
namespace preprocess
{

//==============================================================================
//
// The library's preprocessing step types
//
// What the configuration's preprocess section names (config/preprocess.hpp),
// registered by name in register_core_strategies(). A step type S provides
//
//     struct S {
//         struct Params { ... };                     // its parameters
//         static const std::vector<config::Field<Params>>& schema();
//         static std::optional<Step> make(const Params&, const Input&,
//                                         const config::StepContext&);
//     };
//
// and optionally `static void validate(const Params&)`, which throws
// std::invalid_argument for parameters the schema's bounds cannot say. A
// Params member's default is its value on the default Input (Collection-1
// DN, the Landsat rows), a schema's default_value the physical one, as for
// the strategies (config/params.hpp). make() builds the step from parameters
// resolved against `input` (its units, rows and codes), or none where the
// step does not apply to the input; `context` is where the step is (its
// procedure, whether the procedure applies the thermal transform, its
// position; config/preprocess.hpp). A std::invalid_argument it throws is
// refused naming the step ("preprocess standard step 2 'band_range': ..."). A registry's other step types are the
// same, and return a Step::of() a preprocess::CustomStep.
//
//==============================================================================

// keep_qa: keep only the observations whose QA code is one of `codes`.
struct KeepQa
{
    struct Params
    {
        std::vector<std::int64_t> codes = {0, 1};
    };
    static const std::vector<config::Field<Params>>& schema();
    static std::optional<Step> make(const Params& params, const Input& input,
                                   const config::StepContext& context);
};

// add_qa: also keep every observation whose QA code is one of `codes`,
// whatever an earlier step decided; a dedupe must follow it.
struct AddQa
{
    struct Params
    {
        std::vector<std::int64_t> codes = {3};
    };
    static const std::vector<config::Field<Params>>& schema();
    static std::optional<Step> make(const Params& params, const Input& input,
                                   const config::StepContext& context);
};

// thermal_range: keep only the observations whose thermal row lies within
// (min, max), in degrees Celsius, on every input: resolved to the row's unit
// after the input's transform, and in a procedure that does not transform
// its thermal row taken back through the transform (stored_range()), so the
// stored row is tested as its transform would be (docs/units.md). An input
// without a "thermal" row skips it.
struct ThermalRange
{
    struct Params
    {
        double min = MIN_CELSIUS;
        double max = MAX_CELSIUS;
        bool min_inclusive = false;
        bool max_inclusive = false;
    };
    static const std::vector<config::Field<Params>>& schema();
    static std::optional<Step> make(const Params& params, const Input& input,
                                   const config::StepContext& context);
};

// band_range: keep only the observations on which every one of `bands`
// lies within (min, max), in reflectance, the bands tested in order: roles,
// rows, or "reflective" (the default), every row whose role is not thermal
// or other, in row order. An empty list is refused.
struct BandRange
{
    struct Params
    {
        BandList bands = {0, 1, 2, 3, 4, 5};
        double min = 0.0;
        double max = 10000.0;
        bool min_inclusive = false;
        bool max_inclusive = false;
    };
    static const std::vector<config::Field<Params>>& schema();
    static void validate(const Params& params);
    static std::optional<Step> make(const Params& params, const Input& input,
                                   const config::StepContext& context);
};

// green_median: InsufficientClear's cloud screen. Drop the kept
// observations whose green row is at or above the median green of the kept
// observations dated up to the shared statistics_end, plus `offset` (reflectance),
// at the dates `scope` names (GreenFilterScope: "all", "stat_window",
// "off").
struct GreenMedian
{
    struct Params
    {
        double offset = 400.0;
        std::string scope = "stat_window";
        std::int64_t max_date = std::numeric_limits<std::int64_t>::max();
    };
    static const std::vector<config::Field<Params>>& schema();
    static void validate(const Params& params);
    static std::optional<Step> make(const Params& params, const Input& input,
                                   const config::StepContext& context);
};

// dedupe: keep the first kept observation of each date.
struct Dedupe
{
    struct Params
    {
    };
    static const std::vector<config::Field<Params>>& schema();
    static std::optional<Step> make(const Params& params, const Input& input,
                                   const config::StepContext& context);
};

} // namespace preprocess
} // namespace ccd
