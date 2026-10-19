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

struct StageConfig
{
    std::string strategy;
    Values params;
};

// The whole configuration, by stage name.
using ConfigTree = std::map<std::string, StageConfig>;

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

private:

    using Factory = std::function<std::shared_ptr<const void>(const Values&)>;

    // The factory of `strategy` in `stage`, or std::invalid_argument naming
    // the strategies the stage has.
    const Factory* find(const std::string& stage, const std::string& strategy) const;

    std::map<std::string, std::map<std::string, Factory>> factories_;
};

} // namespace config
} // namespace ccd
