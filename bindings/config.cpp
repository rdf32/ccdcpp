#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ccd/config/registry.hpp"
#include "ccd/config/shared.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/run/algorithm.hpp"
#include "ccd/strategies/register_core.hpp"

#include "config.hpp"

namespace py = pybind11;

//==============================================================================
//
// The algorithm's configuration, in Python
//
// A configuration crosses into Python as a dict {"shared": {name: value},
// "stages": {stage: {"strategy": name, "params": {name: value}}}}: a value
// is a bool, an int, a float, a str, a list of ints (band rows) or a list of
// strs (band roles). python/ccdcpp/config.py wraps it (ccdcpp.Config) and
// holds its compiled Algorithm; these functions are their C++ half.
//
//==============================================================================

namespace
{

std::int64_t index_of(const py::handle& v, const std::string& where)
{
    try
    {
        return py::reinterpret_steal<py::int_>(PyNumber_Index(v.ptr())).cast<std::int64_t>();
    }
    catch (const py::cast_error&)
    {
        throw py::value_error(where + " does not fit a 64-bit integer");
    }
}

bool is_index(const py::handle& v)
{
    return !py::isinstance<py::bool_>(v) && !py::isinstance<py::float_>(v) &&
           (py::isinstance<py::int_>(v) || py::hasattr(v, "__index__"));
}

ccd::config::Value value_from_python(const py::handle& v, const std::string& where)
{
    // bool before int: a Python bool is an int.
    if (py::isinstance<py::bool_>(v))
        return ccd::config::flag(v.cast<bool>());
    if (is_index(v))   // numpy integers too (their __index__)
        return ccd::config::integer(index_of(v, where));
    if (py::isinstance<py::float_>(v))
        return ccd::config::real(v.cast<double>());
    if (py::isinstance<py::str>(v))
        return ccd::config::Value(v.cast<std::string>());
    if (py::isinstance<py::list>(v) || py::isinstance<py::tuple>(v))
    {
        const py::sequence items = py::reinterpret_borrow<py::sequence>(v);
        if (py::len(items) > 0 && py::isinstance<py::str>(items[0]))
        {
            std::vector<std::string> roles;
            for (const py::handle& b : items)
            {
                if (!py::isinstance<py::str>(b))
                    throw py::value_error(where + " is a list of band roles (strs) or rows (ints)");
                roles.push_back(b.cast<std::string>());
            }
            return ccd::config::roles(std::move(roles));
        }
        std::vector<std::int64_t> bands;
        for (const py::handle& b : items)
        {
            if (!is_index(b))
                throw py::value_error(where + " is a list of band roles (strs) or rows (ints)");
            bands.push_back(index_of(b, where));
        }
        return ccd::config::Value(std::move(bands));
    }
    throw py::value_error(where + " is a bool, an int, a float, a str or a list of band roles "
                          "or rows, not " +
                          std::string(py::str(py::type::of(v).attr("__name__"))));
}

py::object value_to_python(const ccd::config::Value& value)
{
    return std::visit([](const auto& v) -> py::object { return py::cast(v); }, value);
}

ccd::config::Values values_from_python(const py::handle& values, const std::string& where)
{
    if (!py::isinstance<py::dict>(values))
        throw py::value_error(where + " is a dict of values");
    ccd::config::Values out;
    for (const auto& [name, value] : values.cast<py::dict>())
    {
        const std::string n = py::str(name);
        out[n] = value_from_python(value, where + " '" + n + "'");
    }
    return out;
}

py::dict values_to_python(const ccd::config::Values& values)
{
    py::dict out;
    for (const auto& [name, value] : values)
        out[py::str(name)] = value_to_python(value);
    return out;
}

ccd::config::ConfigTree stages_from_python(const py::handle& stages)
{
    if (!py::isinstance<py::dict>(stages))
        throw py::value_error("a configuration's stages are a dict of stages");
    ccd::config::ConfigTree tree;
    for (const auto& [stage_obj, chosen] : stages.cast<py::dict>())
    {
        const std::string stage = py::str(stage_obj);
        if (!py::isinstance<py::dict>(chosen))
            throw py::value_error("stage '" + stage + "' is a dict {\"strategy\": ..., \"params\": ...}");
        const py::dict entry = chosen.cast<py::dict>();
        if (!entry.contains("strategy"))
            throw py::value_error("stage '" + stage + "' names no strategy");
        ccd::config::StageConfig out;
        out.strategy = py::str(entry["strategy"]);
        if (entry.contains("params"))
        {
            if (!py::isinstance<py::dict>(entry["params"]))
                throw py::value_error("stage '" + stage + "' params is a dict of parameters");
            for (const auto& [name, value] : entry["params"].cast<py::dict>())
            {
                const std::string n = py::str(name);
                out.params[n] = value_from_python(
                    value, stage + " '" + out.strategy + "' parameter '" + n + "'");
            }
        }
        tree[stage] = std::move(out);
    }
    return tree;
}

py::dict stages_to_python(const ccd::config::ConfigTree& tree)
{
    py::dict out;
    for (const auto& [stage, chosen] : tree)
    {
        py::dict entry;
        entry["strategy"] = chosen.strategy;
        entry["params"] = values_to_python(chosen.params);
        out[py::str(stage)] = entry;
    }
    return out;
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

const char* source_name(ccd::config::Source source)
{
    switch (source)
    {
        case ccd::config::Source::Own:    return "own";
        case ccd::config::Source::Shared: return "shared";
        case ccd::config::Source::Chi2:   return "chi2";
    }
    return "?";
}

const ccd::config::Registry& registry()
{
    return ccd::strategies::library_registry();
}

// A resolution as Python data: the stages, the shared values (in the input's
// units and rows) and the digest.
py::dict resolved_to_python(const ccd::config::Resolved& r)
{
    py::dict out;
    out["stages"] = stages_to_python(r.stages);
    out["shared"] = values_to_python(ccd::config::to_values(ccd::config::shared_schema(), r.shared));
    out["digest"] = r.digest;
    return out;
}

} // namespace

ccd::config::Config config_from_python(const py::handle& config)
{
    if (!py::isinstance<py::dict>(config))
        throw py::type_error("a configuration is a dict {\"shared\": ..., \"stages\": ...}");
    const py::dict d = config.cast<py::dict>();
    for (const auto& [key, _] : d)
    {
        const std::string k = py::str(key);
        if (k != "shared" && k != "stages")
            throw py::value_error("a configuration holds \"shared\" and \"stages\", not '" + k +
                                  "'");
    }
    ccd::config::Config out;
    if (d.contains("shared"))
        out.shared = values_from_python(d["shared"], "shared value");
    if (d.contains("stages"))
        out.stages = stages_from_python(d["stages"]);
    return out;
}

py::dict config_to_python(const ccd::config::Config& config)
{
    py::dict out;
    out["shared"] = values_to_python(config.shared);
    out["stages"] = stages_to_python(config.stages);
    return out;
}

ccd::AlgorithmPtr algorithm_from_python(const py::object& config)
{
    if (config.is_none())
        return ccd::Algorithm::preset("ccdc");
    if (py::isinstance<ccd::Algorithm>(config))
        return config.cast<std::shared_ptr<ccd::Algorithm>>();
    // A ccdcpp.Config holds its compiled Algorithm (a property of its class:
    // compiling it may raise, which an instance's hasattr would hide).
    if (py::hasattr(py::type::of(config), "_algorithm"))
        return config.attr("_algorithm").cast<std::shared_ptr<ccd::Algorithm>>();
    if (!py::isinstance<py::dict>(config))
        throw py::type_error("config is a ccdcpp.Config, a dict {\"shared\": ..., \"stages\": "
                             "...} or None");
    return std::make_shared<const ccd::Algorithm>(config_from_python(config));
}

void bind_config(py::module_& m)
{
    m.attr("NO_STAT_CUTOFF") = ccd::config::NO_STAT_CUTOFF;

    py::class_<ccd::Algorithm, std::shared_ptr<ccd::Algorithm>>(
        m,
        "Algorithm",
        "A compiled configuration: checked once, as it is built, and resolved "
        "against each input it runs on once. Immutable; ccdcpp.Config holds "
        "one.")
    .def(py::init([](const py::dict& config)
                  { return std::make_shared<ccd::Algorithm>(config_from_python(config)); }),
         py::arg("config"),
         "The Algorithm of a configuration's dict; ValueError for what it refuses.")
    .def("config", [](const ccd::Algorithm& a) { return config_to_python(a.config()); },
         "The configuration, as a dict.")
    .def("resolve",
         [](const ccd::Algorithm& a, const ccd::Input& input)
         { return resolved_to_python(*a.for_input(input)); },
         py::arg("input"),
         "The configuration resolved against `input` (cached): {\"stages\", "
         "\"shared\", \"digest\"}, every value in the input's units and rows.");

    m.def(
        "_config_preset",
        [](const std::string& name) { return config_to_python(registry().preset(name)); },
        py::arg("name"),
        "A preset of the library's registry, as a configuration's dict (in physical terms).");
    m.def(
        "_presets",
        []() { return registry().presets(); },
        "The names of the library registry's presets.");
    m.def(
        "_config_digest",
        [](const py::dict& config) { return ccd::config::digest(config_from_python(config)); },
        py::arg("config"),
        "The configuration's digest: equal for equal configurations, on any platform.");
    m.def(
        "_config_check",
        [](const py::dict& config, const std::optional<ccd::Input>& input)
        {
            const ccd::config::Config c = config_from_python(config);
            registry().check(c);
            if (input)
                registry().resolve(c, *input);
        },
        py::arg("config"), py::arg("input") = py::none(),
        "Raises ValueError for what a run would refuse in the configuration; with `input`, "
        "what it refuses against that input too.");
    m.def(
        "_config_resolve",
        [](const py::dict& config, const ccd::Input& input)
        { return resolved_to_python(registry().resolve(config_from_python(config), input)); },
        py::arg("config"), py::arg("input"),
        "The configuration resolved against `input`: {\"stages\", \"shared\", \"digest\"}.");
    m.def(
        "_strategies",
        []()
        {
            py::dict out;
            for (const auto& [stage, strategies] : registry().schemas())
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
                        d["source"] = source_name(p.source);
                        d["from"] = p.from;
                        d["default"] = p.default_value ? value_to_python(*p.default_value)
                                                       : py::none();
                        list.append(d);
                    }
                    by_name[py::str(name)] = list;
                }
                out[py::str(stage)] = by_name;
            }
            return out;
        },
        "Every stage's strategies and their parameters, from the library's registry: a "
        "parameter whose source is \"shared\" or \"chi2\" reads the shared value `from` and "
        "has no default of its own.");
    m.def(
        "_shared",
        []()
        {
            const ccd::config::Values defaults = ccd::config::shared_defaults();
            py::list out;
            for (const auto& f : ccd::config::shared_schema())
            {
                py::dict d;
                d["name"] = f.name;
                d["unit"] = unit_name(f.unit);
                d["bounds"] = ccd::config::describe(f.bounds);
                d["doc"] = f.doc;
                d["default"] = value_to_python(defaults.at(f.name));
                out.append(d);
            }
            return out;
        },
        "The shared values of a configuration, with their physical defaults.");

    m.def(
        "chi2_quantile",
        &ccd::chi2_quantile,
        py::arg("p"),
        py::arg("df"),
        "The chi-squared quantile at probability p with df degrees of "
        "freedom, from the same boost call the change test's "
        "threshold adjustment makes. chi2_quantile(0.99, 5) is the "
        "change_threshold the shared change_probability gives five "
        "detection bands."
    );
}
