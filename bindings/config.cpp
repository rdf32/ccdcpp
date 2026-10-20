#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ccd/strategies/register_core.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/strategies/register_core.hpp"

#include "config.hpp"

namespace py = pybind11;

//==============================================================================
//
// The algorithm's configuration, in Python
//
// A configuration crosses into Python as a dict of stages, each
// {"strategy": name, "params": {name: value}}: a value is a bool, an int, a
// float, a str or a list of ints (a band list). python/ccdcpp/config.py wraps
// it (ccdcpp.Config); these functions are its C++ half.
//
//==============================================================================

namespace
{

ccd::config::Value value_from_python(const py::handle& v, const std::string& where)
{
    // bool before int: a Python bool is an int.
    if (py::isinstance<py::bool_>(v))
        return ccd::config::flag(v.cast<bool>());
    if (py::isinstance<py::int_>(v) || (!py::isinstance<py::float_>(v) && py::hasattr(v, "__index__")))
    {
        // numpy integers too (their __index__).
        try
        {
            return ccd::config::integer(py::reinterpret_steal<py::int_>(PyNumber_Index(v.ptr())).cast<std::int64_t>());
        }
        catch (const py::cast_error&)
        {
            throw py::value_error(where + " does not fit a 64-bit integer");
        }
    }
    if (py::isinstance<py::float_>(v))
        return ccd::config::real(v.cast<double>());
    if (py::isinstance<py::str>(v))
        return ccd::config::Value(v.cast<std::string>());
    if (py::isinstance<py::list>(v) || py::isinstance<py::tuple>(v))
    {
        std::vector<std::int64_t> bands;
        for (const py::handle& b : v)
        {
            if (py::isinstance<py::bool_>(b) || py::isinstance<py::float_>(b) ||
                !(py::isinstance<py::int_>(b) || py::hasattr(b, "__index__")))
                throw py::value_error(where + " is a list of band indices (ints)");
            try
            {
                bands.push_back(py::reinterpret_steal<py::int_>(PyNumber_Index(b.ptr())).cast<std::int64_t>());
            }
            catch (const py::cast_error&)
            {
                throw py::value_error(where + " holds a band index that does not fit 64 bits");
            }
        }
        return ccd::config::Value(std::move(bands));
    }
    throw py::value_error(where + " is a bool, an int, a float, a str or a list of ints, not " +
                          std::string(py::str(py::type::of(v).attr("__name__"))));
}

py::object value_to_python(const ccd::config::Value& value)
{
    return std::visit([](const auto& v) -> py::object { return py::cast(v); }, value);
}

const char* unit_name(ccd::config::Unit unit)
{
    switch (unit)
    {
        case ccd::config::Unit::None:         return "none";
        case ccd::config::Unit::Observations: return "observations";
        case ccd::config::Unit::Days:         return "days";
        case ccd::config::Unit::Reflectance:  return "reflectance";
        case ccd::config::Unit::Bands:        return "bands";
    }
    return "?";
}

// The configuration's strategies: refuses (ValueError) what build_strategies
// refuses, and band lists not below `bands` when given.
void check(const ccd::config::ConfigTree& tree, std::int64_t bands)
{
    try
    {
        ccd::strategies::build_strategies(ccd::strategies::library_registry(), tree);
        if (bands >= 0)
            ccd::strategies::library_registry().check_bands(
                tree, static_cast<ccd::index_t>(bands));
    }
    catch (const std::invalid_argument& e)
    {
        throw py::value_error(e.what());
    }
}

} // namespace

ConfigPtr config_from_python(const py::object& config);

namespace
{

// A configuration that must be one: TypeError for None.
ConfigPtr required(const py::object& config)
{
    if (config.is_none())
        throw py::type_error("config is a ccdcpp.Config or a dict of stages, not None");
    return config_from_python(config);
}

} // namespace

ConfigPtr config_from_python(const py::object& config)
{
    if (config.is_none())
        return std::make_shared<const ccd::config::ConfigTree>(
            ccd::strategies::library_registry().preset("ccdc"));
    // A ccdcpp.Config holds its tree in `_tree`.
    const py::object tree_obj = py::hasattr(config, "_tree") ? config.attr("_tree") : config;
    if (!py::isinstance<py::dict>(tree_obj))
        throw py::type_error("config is a ccdcpp.Config or a dict of stages");

    ccd::config::ConfigTree tree;
    for (const auto& [stage_obj, chosen] : tree_obj.cast<py::dict>())
    {
        const std::string stage = py::str(stage_obj);
        if (!py::isinstance<py::dict>(chosen))
            throw py::value_error("stage '" + stage + "' is a dict {\"strategy\": ..., \"params\": ...}");
        const py::dict entry = chosen.cast<py::dict>();
        if (!entry.contains("strategy"))
            throw py::value_error("stage '" + stage + "' names no strategy");
        ccd::config::StageConfig out;
        out.strategy = py::str(entry["strategy"]);
        if (entry.contains("params") && !py::isinstance<py::dict>(entry["params"]))
            throw py::value_error("stage '" + stage + "' params is a dict of parameters");
        if (entry.contains("params"))
            for (const auto& [name, value] : entry["params"].cast<py::dict>())
            {
                const std::string n = py::str(name);
                out.params[n] = value_from_python(
                    value, stage + " '" + out.strategy + "' parameter '" + n + "'");
            }
        tree[stage] = std::move(out);
    }
    return std::make_shared<const ccd::config::ConfigTree>(std::move(tree));
}

py::dict config_to_python(const ccd::config::ConfigTree& tree)
{
    py::dict out;
    for (const auto& [stage, chosen] : tree)
    {
        py::dict params;
        for (const auto& [name, value] : chosen.params)
            params[py::str(name)] = value_to_python(value);
        py::dict entry;
        entry["strategy"] = chosen.strategy;
        entry["params"] = params;
        out[py::str(stage)] = entry;
    }
    return out;
}

void bind_config(py::module_& m)
{
    m.def(
        "_config_preset",
        [](const std::string& name)
        { return config_to_python(ccd::strategies::library_registry().preset(name)); },
        py::arg("name"),
        "A preset of the library's registry, as a dict of stages (input units: C1 DN for \"ccdc\").");
    m.def(
        "_config_digest",
        [](const py::object& config) { return ccd::config::digest(*required(config)); },
        py::arg("config"),
        "The configuration's digest: equal for equal configurations, on any platform.");
    m.def(
        "_config_check",
        [](const py::object& config, std::int64_t bands)
        { check(*required(config), bands); },
        py::arg("config"), py::arg("bands") = -1,
        "Raises ValueError for what a run would refuse in the configuration.");
    m.def(
        "_strategies",
        []()
        {
            py::dict out;
            for (const auto& [stage, strategies] : ccd::strategies::library_registry().schemas())
            {
                py::dict by_name;
                for (const auto& [name, params] : strategies)
                {
                    py::list list;
                    for (const auto& p : params)
                    {
                        py::dict d;
                        d["name"] = p.name;
                        d["unit"] = unit_name(p.unit);
                        d["bounds"] = p.bounds;
                        d["doc"] = p.doc;
                        d["default"] = value_to_python(p.default_value);
                        list.append(d);
                    }
                    by_name[py::str(name)] = list;
                }
                out[py::str(stage)] = by_name;
            }
            return out;
        },
        "Every stage's strategies and their parameters, from the library's registry.");
}
