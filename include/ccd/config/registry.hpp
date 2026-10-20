#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/input/input.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The algorithm's configuration and the registry of strategies
//
// A configuration (Config) holds the shared values (config/shared.hpp) and,
// for each stage of the algorithm, the strategy that decides it and that
// strategy's parameters (StageConfig). It is written in physical terms --
// reflectance, days, observations, probabilities, bands by role -- so it
// means the same on every input. The registry resolves it against an Input
// (resolve()): reflectance values times Input::reflectance_unit, roles to
// rows, shared values into the stages that read them, chi-squared thresholds
// from probabilities for the input's detection-band count; then it builds
// the strategy objects, by name, validating every parameter against its
// strategy's schema.
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
// (register_core_strategies(), strategies/register_core.cpp):
//
//     registry.add<I, S>("span-factor");
//
// Strategy objects are immutable: built once per resolution and shared by
// every thread.
//
//==============================================================================

// One parameter of a strategy, as its schema declares it: for listings, the
// Python binding and the docs. A parameter its stage cannot set (a Shared or
// Chi2 source) has no default.
struct ParameterInfo
{
    std::string name;
    Unit unit;
    std::string bounds;   // describe() of its bounds
    std::string doc;
    Source source;
    std::string from;     // the shared value it reads; empty for Own
    std::optional<Value> default_value;
};

struct StageConfig
{
    std::string strategy;
    Values params;
};

inline bool operator==(const StageConfig& a, const StageConfig& b)
{
    return a.strategy == b.strategy && a.params == b.params;
}

inline bool operator!=(const StageConfig& a, const StageConfig& b)
{
    return !(a == b);
}

// The stages, by name.
using ConfigTree = std::map<std::string, StageConfig>;

// A configuration: the shared values and the stages. A shared value it
// leaves out is shared_schema()'s default; a stage's parameter, its
// strategy's default.
struct Config
{
    Values shared;
    ConfigTree stages;
};

inline bool operator==(const Config& a, const Config& b)
{
    return a.shared == b.shared && a.stages == b.stages;
}

inline bool operator!=(const Config& a, const Config& b)
{
    return !(a == b);
}

// A configuration resolved against an input: what a run runs.
struct Resolved
{
    // Every stage's every parameter spelled out, in the input's units and
    // rows, shared values and derived thresholds included.
    ConfigTree stages;
    // The shared values, in the input's units and rows.
    Shared shared;
    // One strategy per stage, built from `stages`.
    Strategies strategies;
    // Over the input and the resolved configuration (stages and shared):
    // equal for equal runs, on any platform.
    std::uint64_t digest = 0;
};

// Throws std::invalid_argument, naming `where`, when `list` names a band an
// input of `bands` bands does not have, or repeats one.
void check_band_list(const BandList& list, index_t bands, const std::string& where);

// A digest of a set of values, of the stages, and of a configuration: equal
// for equal ones (every value by name, every stage, strategy name and
// parameter), on any platform.
std::uint64_t digest(const Values& values);
std::uint64_t digest(const ConfigTree& stages);
std::uint64_t digest(const Config& config);

class Registry
{
public:

    template<class Interface, class Strategy>
    void add(const std::string& name)
    {
        auto& stage = factories_[Interface::stage];
        if (stage.count(name))
            throw std::logic_error(std::string(Interface::stage) + " strategy '" + name +
                                   "' registered twice");
        std::vector<ParameterInfo>& info = schemas_[Interface::stage][name];
        const Values members = to_values(Strategy::schema(), typename Strategy::Params{});
        for (const auto& field : Strategy::schema())
        {
            std::optional<Value> fallback;
            if (field.source == Source::Own)
                fallback = field.default_value ? *field.default_value : members.at(field.name);
            info.push_back(ParameterInfo{field.name, field.unit, describe(field.bounds),
                                         field.doc, field.source,
                                         field.from ? field.from : "", fallback});
        }
        const std::string where = std::string(Interface::stage) + " '" + name + "'";
        resolvers_[Interface::stage][name] =
            [where, info](const Values& given, const Resolving& with)
        {
            const Values values = with.stage(info, given, where);
            typename Strategy::Params params;
            apply(Strategy::schema(), values, params, where);
            if (with.input)
                for (const auto& field : Strategy::schema())
                    if (const auto* m = std::get_if<BandList Strategy::Params::*>(&field.member))
                        check_band_list(params.**m, with.input->bands(),
                                        where + " parameter '" + field.name + "'");
            return to_values(Strategy::schema(), params);
        };
        stage[name] = [where](const Values& values) -> std::shared_ptr<const void>
        {
            typename Strategy::Params params;
            apply(Strategy::schema(), values, params, where);
            return std::static_pointer_cast<const void>(
                std::shared_ptr<const Interface>(std::make_shared<const Strategy>(params)));
        };
    }

    // The strategy `config` names for Interface's stage. Throws
    // std::invalid_argument on a strategy the stage does not have, naming the
    // ones it does, and on parameters the strategy refuses. `config` is
    // resolved: every parameter, the shared ones included, in the input's
    // units and rows.
    template<class Interface>
    std::shared_ptr<const Interface> make(const StageConfig& config) const
    {
        const Factory* factory = find(Interface::stage, config.strategy);
        // The factory stores the strategy as its Interface, so the cast back
        // is to the type it was made as.
        return std::static_pointer_cast<const Interface>((*factory)(config.params));
    }

    // The strategy names of each stage, for listings and tests.
    std::map<std::string, std::vector<std::string>> names() const;

    // Each strategy's parameters with their sources and defaults, by stage
    // and name.
    const std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>>&
    schemas() const
    {
        return schemas_;
    }

    // The parameters a stage of `strategy` in `stage` sets, at their
    // defaults (physical, bands by role), refusing an unknown strategy as
    // make() does. The shared and derived ones are not among them.
    Values defaults(const std::string& stage, const std::string& strategy) const;

    // Throws std::invalid_argument for what resolve() refuses on any input:
    // a stage the algorithm does not have or a missing one, an unknown
    // strategy, shared value or parameter, a stage setting a shared or
    // derived parameter, a value of the wrong type or outside its bounds as
    // written, an unknown role, and what a strategy needs that the stages
    // before it do not provide.
    void check(const Config& config) const;

    // `config` against `input`: every stage's parameters resolved (its
    // defaults filling what `config` leaves out, reflectance values times
    // input.reflectance_unit, roles as input's rows, the shared values and
    // the derived thresholds filled in), the strategies built, and the
    // digest. Refuses what check() refuses, and a role the input lacks or a
    // band list naming a row it does not have, or repeating one. `input`
    // must pass validate() (ccd.hpp).
    Resolved resolve(const Config& config, const Input& input) const;

    // The strategies of a resolved tree. Throws std::invalid_argument on a
    // stage the tree lacks, a strategy its stage does not have, parameters
    // the strategy refuses, or a strategy needing what the stages before it
    // do not provide.
    Strategies build(const ConfigTree& resolved) const;

    // Named configurations ("ccdc", and an extension's): add_preset()
    // registers one (std::logic_error for a name already registered),
    // preset() returns it (std::invalid_argument naming the presets for
    // another name), presets() lists them.
    void add_preset(const std::string& name, Config config);
    const Config& preset(const std::string& name) const;
    std::vector<std::string> presets() const;

private:

    // How one resolution fills a stage's values (registry.cpp): against an
    // input, or (check()) as written.
    class Resolving
    {
    public:
        // The stage's values for its schema `info`: given, or the
        // defaults, converted to the input's units and rows (Own); the
        // shared values (Shared); the derived thresholds (Chi2).
        Values stage(const std::vector<ParameterInfo>& info, const Values& given,
                     const std::string& where) const;

        const Input* input = nullptr;   // null: as written
        Values shared;                  // resolved shared values
        index_t detection_bands = 0;
    };

    using Factory = std::function<std::shared_ptr<const void>(const Values&)>;
    using Resolver = std::function<Values(const Values&, const Resolving&)>;

    // The factory of `strategy` in `stage`, or std::invalid_argument naming
    // the strategies the stage has.
    const Factory* find(const std::string& stage, const std::string& strategy) const;

    // The stages of `config` resolved `with` (the shared values already in
    // it), every stage of the algorithm required.
    ConfigTree resolve_stages(const Config& config, const Resolving& with) const;

    std::map<std::string, std::map<std::string, Factory>> factories_;
    std::map<std::string, std::map<std::string, Resolver>> resolvers_;
    std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>> schemas_;
    std::map<std::string, Config> presets_;
};

} // namespace config
} // namespace ccd
