#pragma once

#include <concepts>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd/config/config.hpp"
#include "ccd/config/params.hpp"
#include "ccd/config/preprocess.hpp"
#include "ccd/input/data_spec.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The registry of strategies, steps and presets
//
// The catalogue a configuration (config/config.hpp) names its strategies and
// steps in: for each stage, its strategy types by name; the preprocessing
// step types by name; and the named configurations, the presets. Each type
// comes with its parameters' schema, which config::resolve()
// (config/resolve.hpp) checks and converts a configuration's values by.
//
// A strategy type S for a stage interface I provides:
//
//     struct S final : I {
//         struct Params { ... };                     // its parameters, with defaults
//         static const std::vector<Field<Params>>& schema();
//         explicit S(const Params&);
//     };
//
// and is registered once, by name, in its module's registration function
// (register_core_strategies(), strategies/library.cpp):
//
//     registry.add<I, S>("span-factor");
//
// A preprocessing step type (input/mask/steps.hpp) is registered the same
// way, by the name the configuration's chains give it, and its make() is
// handed where each step of it is (StepContext, config/preprocess.hpp):
//
//     registry.add_step<preprocess::KeepQa>("keep_qa");
//
// Strategy objects and plans are immutable: built once per resolution and
// shared by every thread.
//
//==============================================================================

// What add<Interface, S>() requires of a strategy type S: an Interface with
// default-constructible Params, their schema, and a constructor from them.
template<class S, class Interface>
concept StrategyFor =
    std::derived_from<S, Interface> && std::default_initializable<typename S::Params> &&
    std::constructible_from<S, const typename S::Params&> && requires {
        { S::schema() } -> std::same_as<const std::vector<Field<typename S::Params>>&>;
    };

// What add_step<S>() requires of a preprocessing step type S: default-
// constructible Params, their schema, and make() (input/mask/steps.hpp).
template<class S>
concept PreprocessingStep =
    std::default_initializable<typename S::Params> &&
    requires(const typename S::Params& params, const DataSpec& spec, const StepContext& context) {
        { S::schema() } -> std::same_as<const std::vector<Field<typename S::Params>>&>;
        { S::make(params, spec, context) } -> std::convertible_to<std::optional<preprocess::Step>>;
    };

// The parameters `values` give a registered type, back by name as its Params
// holds them: std::invalid_argument, naming `where`, for what apply() or the
// type's validate() refuses, and against `spec` (null: none) for a band list
// naming a row it does not have, or repeating one. `values` are converted
// (in the spec's units and rows, or as written).
using ParameterCheck =
    std::function<Values(const Values& values, const DataSpec* spec, const std::string& where)>;

// A stage's strategy type: its parameters, their check, and its strategy
// (as the stage's interface) from resolved values.
struct StrategyType
{
    std::vector<ParameterInfo> info;
    ParameterCheck check;
    std::function<std::shared_ptr<const void>(const Values&)> make;
};

// A preprocessing step type: its parameters, their check, and its step from
// resolved values on a data spec, at a place in a chain (none where the step
// does not apply to the spec), refusals naming `where`.
struct StepType
{
    std::vector<ParameterInfo> info;
    ParameterCheck check;
    std::function<std::optional<preprocess::Step>(const Values&, const DataSpec&,
                                                  const StepContext&, const std::string& where)>
        make;
};

class Registry
{
public:

    // Registers the strategy type Strategy of Interface's stage as `name`;
    // std::logic_error for a name the stage already has.
    template<class Interface, StrategyFor<Interface> Strategy>
    void add(const std::string& name)
    {
        auto& types = strategies_[Interface::stage];
        if (types.contains(name))
            throw std::logic_error(std::string(Interface::stage) + " strategy '" + name +
                                   "' registered twice");
        const std::string where = std::string(Interface::stage) + " '" + name + "'";
        types[name] = StrategyType{
            info_of(Strategy::schema()),
            [](const Values& values, const DataSpec* spec, const std::string& at)
            {
                typename Strategy::Params params;
                apply(Strategy::schema(), values, params, at);
                if (spec)
                    check_band_lists(Strategy::schema(), params, spec->bands(), at);
                return to_values(Strategy::schema(), params);
            },
            [where](const Values& values) -> std::shared_ptr<const void>
            {
                typename Strategy::Params params;
                apply(Strategy::schema(), values, params, where);
                return std::static_pointer_cast<const void>(
                    std::shared_ptr<const Interface>(std::make_shared<const Strategy>(params)));
            }};
    }

    // Registers the preprocessing step type S (input/mask/steps.hpp) as
    // `name`, the name a configuration's chains give it; std::logic_error
    // for a name already registered.
    template<PreprocessingStep S>
    void add_step(const std::string& name)
    {
        if (steps_.contains(name))
            throw std::logic_error("preprocessing step '" + name + "' registered twice");
        steps_[name] = StepType{
            info_of(S::schema()),
            [](const Values& values, const DataSpec* spec, const std::string& where)
            {
                typename S::Params params;
                apply(S::schema(), values, params, where);
                // A step type may have `static void validate(const Params&)`.
                if constexpr (requires(const typename S::Params& p) { S::validate(p); })
                {
                    try
                    {
                        S::validate(params);
                    }
                    catch (const std::invalid_argument& e)
                    {
                        throw std::invalid_argument(where + " " + e.what());
                    }
                }
                if (spec)
                    check_band_lists(S::schema(), params, spec->bands(), where);
                return to_values(S::schema(), params);
            },
            [](const Values& values, const DataSpec& spec, const StepContext& context,
               const std::string& where)
            {
                typename S::Params params;
                apply(S::schema(), values, params, where);
                try
                {
                    return S::make(params, spec, context);
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::invalid_argument(where + ": " + e.what());
                }
            }};
    }

    // The strategy `config` names for Interface's stage. Throws
    // std::invalid_argument on a strategy the stage does not have, naming the
    // ones it does, and on parameters the strategy refuses. `config` is
    // resolved: every parameter, the shared ones included, in the spec's
    // units and rows.
    template<class Interface>
    std::shared_ptr<const Interface> make(const StageConfig& config) const
    {
        // The type stores the strategy as its Interface, so the cast back is
        // to the type it was made as.
        return std::static_pointer_cast<const Interface>(
            strategy(Interface::stage, config.strategy).make(config.params));
    }

    // The strategy type `name` of `stage`, or std::invalid_argument naming
    // the strategies the stage has.
    const StrategyType& strategy(const std::string& stage, const std::string& name) const;

    // The step type `name`, or std::invalid_argument naming `where` and the
    // step types.
    const StepType& step(const std::string& name, const std::string& where) const;

    // The strategy names of each stage, for listings and tests.
    std::map<std::string, std::vector<std::string>> names() const;

    // Each strategy's parameters with their sources and defaults, by stage
    // and name, as a copy: hold it to iterate one stage's.
    std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>> schemas() const;

    // The parameters a stage of `strategy` in `stage` sets, at their
    // defaults (physical, bands by role), refusing an unknown strategy as
    // strategy() does. The shared and derived ones are not among them.
    Values defaults(const std::string& stage, const std::string& strategy) const;

    // Each step type's parameters with their sources and defaults, by name.
    std::map<std::string, std::vector<ParameterInfo>> step_schemas() const;

    // The parameters a step of type `step` sets, at their defaults
    // (physical, bands by role, codes by name); std::invalid_argument naming
    // the step types for a name not registered.
    Values step_defaults(const std::string& step) const;

    // Named configurations ("ccdc", and an extension's): add_preset()
    // registers one (std::logic_error for a name already registered),
    // preset() returns it (std::invalid_argument naming the presets for
    // another name), presets() lists them.
    void add_preset(const std::string& name, Config config);
    const Config& preset(const std::string& name) const;
    std::vector<std::string> presets() const;

private:

    std::map<std::string, std::map<std::string, StrategyType>> strategies_;
    std::map<std::string, StepType> steps_;
    std::map<std::string, Config> presets_;
};

} // namespace config
} // namespace ccd
