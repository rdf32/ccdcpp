#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd/config/params.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The algorithm's configuration and the registry of strategies
//
// A configuration names, for each stage of the algorithm, the strategy that
// decides it and that strategy's parameters (StageConfig). The registry turns
// a StageConfig into the strategy object, by name, validating the parameters
// against the strategy's schema.
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
// Strategy objects are immutable: built once per call and shared by every
// thread.
//
//==============================================================================

// One parameter of a strategy, as its schema declares it: for listings, the
// Python binding and the docs.
struct ParameterInfo
{
    std::string name;
    Unit unit;
    std::string bounds;   // describe() of its bounds
    std::string doc;
    Value default_value;
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

// The whole configuration, by stage name.
using ConfigTree = std::map<std::string, StageConfig>;

// Throws std::invalid_argument, naming `where`, when `list` names a band an
// input of `bands` bands does not have, or repeats one.
void check_band_list(const BandList& list, index_t bands, const std::string& where);

// A digest of a configuration: equal for equal trees (every stage, strategy
// name and parameter, by name), on any platform.
std::uint64_t digest(const ConfigTree& config);

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
        {
            std::vector<ParameterInfo>& info = schemas_[Interface::stage][name];
            const Values defaults = to_values(Strategy::schema(), typename Strategy::Params{});
            for (const auto& field : Strategy::schema())
                info.push_back(ParameterInfo{field.name, field.unit, describe(field.bounds),
                                             field.doc, defaults.at(field.name)});
        }
        band_checks_[Interface::stage][name] =
            [name](const Values& values, index_t bands, const std::string& where)
        {
            typename Strategy::Params params;
            apply(Strategy::schema(), values, params, where);
            for (const auto& field : Strategy::schema())
                if (const auto* member = std::get_if<BandList Strategy::Params::*>(&field.member))
                    check_band_list(params.**member, bands,
                                    where + " parameter '" + field.name + "'");
        };
        stage[name] = [name](const Values& values) -> std::shared_ptr<const void>
        {
            typename Strategy::Params params;
            apply(Strategy::schema(), values, params,
                  std::string(Interface::stage) + " '" + name + "'");
            return std::static_pointer_cast<const void>(
                std::shared_ptr<const Interface>(std::make_shared<const Strategy>(params)));
        };
    }

    // The strategy `config` names for Interface's stage. Throws
    // std::invalid_argument on a strategy the stage does not have, naming the
    // ones it does, and on parameters the strategy refuses.
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

    // Each strategy's parameters with their defaults, by stage and name.
    const std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>>&
    schemas() const
    {
        return schemas_;
    }

    // Throws std::invalid_argument when a band-list parameter of a strategy
    // `config` names, as set there or by default, names a band an input of
    // `bands` bands does not have, or repeats one. Strategies the registry
    // does not hold are not checked (build_strategies() refuses them).
    void check_bands(const ConfigTree& config, index_t bands) const;

    // The parameters of `strategy` in `stage` at their defaults (refusing
    // an unknown one as make() does).
    Values defaults(const std::string& stage, const std::string& strategy) const;

    // Named configurations ("ccdc", and an extension's): add_preset()
    // registers one, preset() returns it (std::invalid_argument naming the
    // presets for another name), presets() lists them.
    void add_preset(const std::string& name, ConfigTree config);
    const ConfigTree& preset(const std::string& name) const;
    std::vector<std::string> presets() const;

private:

    using Factory = std::function<std::shared_ptr<const void>(const Values&)>;

    // The factory of `strategy` in `stage`, or std::invalid_argument naming
    // the strategies the stage has.
    const Factory* find(const std::string& stage, const std::string& strategy) const;

    std::map<std::string, std::map<std::string, Factory>> factories_;
    using BandCheck = std::function<void(const Values&, index_t, const std::string&)>;
    std::map<std::string, std::map<std::string, BandCheck>> band_checks_;
    std::map<std::string, std::map<std::string, std::vector<ParameterInfo>>> schemas_;
    std::map<std::string, ConfigTree> presets_;
};

} // namespace config
} // namespace ccd
