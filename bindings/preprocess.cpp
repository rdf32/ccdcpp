#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ccd/core/constants.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/mask/preprocess.hpp"

#include "config.hpp"

namespace py = pybind11;
namespace pp = ccd::preprocess;

//------------------------------------------------------------------------------
// Preprocessing plans: ccd::preprocess (input/mask/preprocess.hpp) as Python
// classes, for inspection.
//
// A plan is a configuration's preprocess section resolved against an input:
// plan_of(config, input) returns it. The classes are read-only copies; plans
// compare with == field by field; describe() and repr() print them one step
// a line.
//------------------------------------------------------------------------------

namespace
{

//------------------------------------------------------------------------------
// Text
//------------------------------------------------------------------------------

// `v` as an integer when it is one, else the shortest %g form that reads back
// as `v`.
std::string number(double v)
{
    if (std::isnan(v))
        return "nan";

    char buf[32];
    if (v == std::trunc(v) && std::fabs(v) < 1e16)
    {
        std::snprintf(buf, sizeof buf, "%.0f", v);
        return buf;
    }
    for (int precision = 1; precision <= 17; ++precision)
    {
        std::snprintf(buf, sizeof buf, "%.*g", precision, v);
        if (std::strtod(buf, nullptr) == v)
            break;
    }
    return buf;
}

std::string band(ccd::index_t b)
{
    return b == ccd::NO_BAND ? std::string("NO_BAND") : std::to_string(b);
}

std::string range_text(const pp::Range& r)
{
    return number(r.lo.value) + (r.lo.inclusive ? " <= " : " < ") + "x" +
           (r.hi.inclusive ? " <= " : " < ") + number(r.hi.value);
}

std::string codes_text(const pp::QaSet& codes)
{
    std::string out = "{";
    bool first = true;
    for (std::size_t c = 0; c < codes.size(); ++c)
    {
        if (!codes[c])
            continue;
        if (!first)
            out += ", ";
        out += std::to_string(c);
        first = false;
    }
    return out + "}";
}

std::string bands_text(const ccd::BandList& bands)
{
    std::string out = "[";
    for (ccd::index_t i = 0; i < bands.size(); ++i)
        out += (i ? ", " : "") + std::to_string(bands[i]);
    return out + "]";
}

const char* scope_name(ccd::GreenFilterScope scope)
{
    switch (scope)
    {
        case ccd::GreenFilterScope::All:        return "All";
        case ccd::GreenFilterScope::StatWindow: return "StatWindow";
        case ccd::GreenFilterScope::Off:        return "Off";
    }
    return "unknown";
}

// One step, without its kind: the fields the kind reads.
std::string step_fields(const pp::Step& s)
{
    switch (s.kind)
    {
        case pp::StepKind::KeepQa:
        case pp::StepKind::AddQa:
            return "codes " + codes_text(s.codes);

        case pp::StepKind::ThermalRange:
            return "band " + band(s.band) + ", " + range_text(s.range);

        case pp::StepKind::BandRange:
            return "bands " + bands_text(s.bands) + ", " + range_text(s.range);

        case pp::StepKind::GreenMedian:
            return "band " + band(s.band) + ", drop x >= median(" +
                   (s.max_date == ccd::config::NO_STATISTICS_END
                        ? std::string("every date")
                        : "dates <= " + std::to_string(s.max_date)) +
                   ") + " + number(s.green_offset) + ", scope " + scope_name(s.scope);

        case pp::StepKind::Dedupe:
            return "";

        case pp::StepKind::Custom:
            return s.custom ? s.custom->describe() : std::string("none");
    }
    return "";
}

std::string step_text(const pp::Step& s)
{
    const std::string fields = step_fields(s);
    return std::string(pp::name(s.kind)) + (fields.empty() ? "" : " " + fields);
}

std::string affine_text(const pp::Affine& a)
{
    return "band " + band(a.band) + " <- x * " + number(a.scale) +
           (std::signbit(a.offset) ? " - " : " + ") + number(std::fabs(a.offset));
}

std::string procedure_text(const char* name, const pp::ProcedurePlan& p)
{
    std::ostringstream out;
    out << name << "\n";
    out << "  transform  "
        << (p.transform ? affine_text(*p.transform) : std::string("none")) << "\n";
    for (ccd::index_t i = 0; i < p.chain.size(); ++i)
        out << "  " << i << "  " << step_text(p.chain[i]) << "\n";
    return out.str();
}

std::string plan_text(const pp::PreprocessPlan& p)
{
    std::string out =
        procedure_text("standard", p.standard) +
        procedure_text("snow", p.snow) +
        procedure_text("insufficient_clear", p.insufficient_clear);
    out.pop_back();
    return out;
}

//------------------------------------------------------------------------------
// Equality, field by field.
//------------------------------------------------------------------------------

bool same(const pp::Range& a, const pp::Range& b)
{
    return a.lo.value == b.lo.value && a.lo.inclusive == b.lo.inclusive &&
           a.hi.value == b.hi.value && a.hi.inclusive == b.hi.inclusive;
}

bool same(const pp::Step& a, const pp::Step& b)
{
    return a.kind == b.kind && a.codes == b.codes &&
           a.bands.to_vector() == b.bands.to_vector() && a.band == b.band &&
           same(a.range, b.range) && a.green_offset == b.green_offset &&
           a.max_date == b.max_date && a.scope == b.scope && a.custom == b.custom;
}

bool same(const pp::Chain& a, const pp::Chain& b)
{
    if (a.size() != b.size())
        return false;
    for (ccd::index_t i = 0; i < a.size(); ++i)
    {
        if (!same(a[i], b[i]))
            return false;
    }
    return true;
}

bool same(const std::optional<pp::Affine>& a, const std::optional<pp::Affine>& b)
{
    if (a.has_value() != b.has_value())
        return false;
    return !a || (a->band == b->band && a->scale == b->scale && a->offset == b->offset);
}

bool same(const pp::ProcedurePlan& a, const pp::ProcedurePlan& b)
{
    return same(a.transform, b.transform) && same(a.chain, b.chain);
}

bool same(const pp::PreprocessPlan& a, const pp::PreprocessPlan& b)
{
    return same(a.standard, b.standard) && same(a.snow, b.snow) &&
           same(a.insufficient_clear, b.insufficient_clear);
}

std::vector<int> from_codes(const pp::QaSet& set)
{
    std::vector<int> out;
    for (std::size_t c = 0; c < set.size(); ++c)
    {
        if (set[c])
            out.push_back(static_cast<int>(c));
    }
    return out;
}

ccd::index_t wrap_index(const pp::Chain& c, std::ptrdiff_t i)
{
    const auto n = static_cast<std::ptrdiff_t>(c.size());
    if (i < 0)
        i += n;
    if (i < 0 || i >= n)
        throw py::index_error("chain index out of range");
    return static_cast<ccd::index_t>(i);
}

} // namespace

void bind_preprocess(py::module_& m)
{
    m.attr("CHAIN_CAPACITY") = pp::CHAIN_CAPACITY;

    py::enum_<pp::StepKind>(
        m,
        "StepKind",
        "What a preprocessing Step does to the processing mask: KeepQa, "
        "AddQa, ThermalRange, BandRange, GreenMedian, Dedupe, or Custom (a "
        "step type a registry adds)."
    )
    .value("KeepQa", pp::StepKind::KeepQa)
    .value("AddQa", pp::StepKind::AddQa)
    .value("ThermalRange", pp::StepKind::ThermalRange)
    .value("BandRange", pp::StepKind::BandRange)
    .value("GreenMedian", pp::StepKind::GreenMedian)
    .value("Dedupe", pp::StepKind::Dedupe)
    .value("Custom", pp::StepKind::Custom);

    py::class_<pp::Step>(
        m,
        "Step",
        "One filter of a resolved preprocessing chain, in the input's units, "
        "rows and codes; `kind` says which fields it reads: codes (KeepQa, "
        "AddQa), band, min, max, inclusive (ThermalRange), bands, min, max, "
        "inclusive (BandRange), band, green_offset, max_date, scope "
        "(GreenMedian). A range keeps min < x < max; an inclusive end also "
        "keeps x equal to it. NaN passes only a range inclusive at both ends."
    )
    .def_readonly("kind", &pp::Step::kind)
    .def_property_readonly("codes", [](const pp::Step& s) { return from_codes(s.codes); },
        "QA codes, ascending.")
    .def_property_readonly("bands", [](const pp::Step& s) { return s.bands.to_vector(); },
        "Band rows tested, in order.")
    .def_readonly("band", &pp::Step::band, "The band row read.")
    .def_property_readonly("min", [](const pp::Step& s) { return s.range.lo.value; },
        "The lower bound.")
    .def_property_readonly("max", [](const pp::Step& s) { return s.range.hi.value; },
        "The upper bound.")
    .def_property_readonly("inclusive",
        [](const pp::Step& s) { return std::make_pair(s.range.lo.inclusive, s.range.hi.inclusive); },
        "(min, max): whether a value equal to each bound is kept.")
    .def_readonly("green_offset", &pp::Step::green_offset, "Added to the median.")
    .def_readonly("max_date", &pp::Step::max_date,
        "The last date of the median's window, an ordinal.")
    .def_readonly("scope", &pp::Step::scope, "Which dates are dropped, a GreenFilterScope.")
    .def("__eq__", [](const pp::Step& a, const pp::Step& b) { return same(a, b); },
        py::is_operator())
    .def("__repr__", [](const pp::Step& s) { return "<Step " + step_text(s) + ">"; });

    py::class_<pp::Chain>(m, "Chain", "A resolved chain: its Steps, run in order.")
    .def("__len__", &pp::Chain::size)
    .def("__getitem__",
        [](const pp::Chain& c, std::ptrdiff_t i) { return c[wrap_index(c, i)]; },
        py::arg("index"))
    .def("__iter__",
        [](const pp::Chain& c) { return py::make_iterator(c.begin(), c.end()); },
        py::keep_alive<0, 1>())
    .def_property_readonly("steps",
        [](const pp::Chain& c) { return std::vector<pp::Step>(c.begin(), c.end()); },
        "The steps, as a list.")
    .def("__eq__", [](const pp::Chain& a, const pp::Chain& b) { return same(a, b); },
        py::is_operator())
    .def("__repr__",
        [](const pp::Chain& c)
        {
            std::string out = "<Chain [";
            for (ccd::index_t i = 0; i < c.size(); ++i)
                out += (i ? "; " : "") + step_text(c[i]);
            return out + "]>";
        });

    py::class_<pp::Affine>(
        m,
        "Affine",
        "The transform of one band of the per-pixel copy, before the chain "
        "runs: value <- value * scale + offset (the input's thermal "
        "transform). The caller's array is never written."
    )
    .def_readonly("band", &pp::Affine::band)
    .def_readonly("scale", &pp::Affine::scale)
    .def_readonly("offset", &pp::Affine::offset)
    .def("__eq__",
        [](const pp::Affine& a, const pp::Affine& b)
        {
            return same(std::optional<pp::Affine>(a), std::optional<pp::Affine>(b));
        },
        py::is_operator())
    .def("__repr__", [](const pp::Affine& a) { return "<Affine " + affine_text(a) + ">"; });

    py::class_<pp::ProcedurePlan>(
        m,
        "ProcedurePlan",
        "One procedure's resolved preprocessing: an optional Affine transform, "
        "then a Chain that builds the processing mask."
    )
    .def_readonly("transform", &pp::ProcedurePlan::transform)
    .def_readonly("chain", &pp::ProcedurePlan::chain)
    .def("__eq__",
        [](const pp::ProcedurePlan& a, const pp::ProcedurePlan& b) { return same(a, b); },
        py::is_operator())
    .def("__repr__",
        [](const pp::ProcedurePlan& p)
        {
            std::string text = procedure_text("ProcedurePlan", p);
            text.pop_back();
            return text;
        });

    py::class_<pp::PreprocessPlan>(
        m,
        "PreprocessPlan",
        "A configuration's preprocessing resolved against an input "
        "(plan_of()): a ProcedurePlan per procedure, standard, snow "
        "(PermanentSnow) and insufficient_clear. A run runs the one of the "
        "procedure each pixel selects."
    )
    .def_readonly("standard", &pp::PreprocessPlan::standard)
    .def_readonly("snow", &pp::PreprocessPlan::snow)
    .def_readonly("insufficient_clear", &pp::PreprocessPlan::insufficient_clear)
    .def("describe", &plan_text,
        "The plan as text: per procedure, its transform and one numbered "
        "line per step.")
    .def("__eq__",
        [](const pp::PreprocessPlan& a, const pp::PreprocessPlan& b) { return same(a, b); },
        py::is_operator())
    .def("__repr__", &plan_text);

    m.def(
        "plan_of",
        [](const py::object& config, const ccd::Input& input)
        { return algorithm_from_python(config)->for_input(input)->plan; },
        py::arg("config"),
        py::arg("input"),
        "The PreprocessPlan `config` runs on `input` (`config` as detect "
        "takes it): its preprocess section resolved against the input, the "
        "QA code names as its codes, the bands by role as its rows, the "
        "bounds in its units; a thermal_range step is left out of an input "
        "without a thermal row."
    );
}
