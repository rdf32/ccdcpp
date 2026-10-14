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

#include "ccd/constants.hpp"
#include "ccd/harmonic/harmonic.hpp"
#include "ccd/preprocess/preprocess.hpp"

namespace py = pybind11;
namespace pp = ccd::preprocess;

//------------------------------------------------------------------------------
// Preprocessing plans: ccd::preprocess (preprocess/preprocess.hpp) as Python
// classes.
//
// The classes are mutable value types, like HarmonicOptions. The entry points
// take a plan as preprocess= and copy it before the GIL is released, so a
// call never reads an object Python can change underneath it. Plans compare
// with == field by field; describe() and repr() print them one step a line.
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
            return "band " + band(s.band) + ", drop x >= median(dates <= " +
                   std::to_string(s.max_date) + ") + " + number(s.green_offset) +
                   ", scope " + scope_name(s.scope);

        case pp::StepKind::Dedupe:
            return "";
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
           a.max_date == b.max_date && a.scope == b.scope;
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

//------------------------------------------------------------------------------
// Conversions
//------------------------------------------------------------------------------

pp::QaSet to_codes(const std::vector<int>& codes)
{
    pp::QaSet set;
    for (const int code : codes)
    {
        if (code < 0 || code > 255)
            throw py::value_error(
                "QA code " + std::to_string(code) + " is not in 0..255");
        set.set(static_cast<std::size_t>(code));
    }
    return set;
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

pp::Range to_range(double lo, double hi, const std::pair<bool, bool>& inclusive)
{
    return pp::Range{pp::Bound{lo, inclusive.first}, pp::Bound{hi, inclusive.second}};
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

pp::Chain to_chain(const std::vector<pp::Step>& steps)
{
    pp::Chain chain;
    for (const pp::Step& s : steps)
        chain.push_back(s);
    return chain;
}

} // namespace

void bind_preprocess(py::module_& m)
{
    m.attr("CHAIN_CAPACITY") = pp::CHAIN_CAPACITY;

    py::enum_<pp::StepKind>(
        m,
        "StepKind",
        "What a preprocessing Step does to the processing mask: KeepQa, "
        "AddQa, ThermalRange, BandRange, GreenMedian, Dedupe."
    )
    .value("KeepQa", pp::StepKind::KeepQa)
    .value("AddQa", pp::StepKind::AddQa)
    .value("ThermalRange", pp::StepKind::ThermalRange)
    .value("BandRange", pp::StepKind::BandRange)
    .value("GreenMedian", pp::StepKind::GreenMedian)
    .value("Dedupe", pp::StepKind::Dedupe);

    //--------------------------------------------------------------------------
    // Step
    //--------------------------------------------------------------------------

    py::class_<pp::Step>(
        m,
        "Step",
        "One filter of a preprocessing chain. Build one with a named "
        "constructor; `kind` says which fields it reads:\n\n"
        "  keep_qa(codes)        keep only observations with these QA codes\n"
        "  add_qa(codes)         also keep these codes; Dedupe must follow\n"
        "  thermal_range(band, min, max, inclusive=(False, False))\n"
        "                        keep observations whose band is in range\n"
        "  band_range(bands, min, max, inclusive=(False, False))\n"
        "                        keep observations with every band in range\n"
        "  green_median(band, green_offset, max_date, scope)\n"
        "                        drop observations at or above the median of\n"
        "                        band (dates <= max_date) plus green_offset\n"
        "  dedupe()              keep the first observation of each date\n\n"
        "A range keeps min < x < max; an inclusive end also keeps x equal to "
        "it. NaN passes only a range inclusive at both ends."
    )
    .def_static("keep_qa",
        [](const std::vector<int>& codes) { return pp::Step::keep_qa(to_codes(codes)); },
        py::arg("codes"))
    .def_static("add_qa",
        [](const std::vector<int>& codes) { return pp::Step::add_qa(to_codes(codes)); },
        py::arg("codes"))
    .def_static("thermal_range",
        [](ccd::index_t band, double min, double max, std::pair<bool, bool> inclusive)
        {
            return pp::Step::thermal_range(band, to_range(min, max, inclusive));
        },
        py::arg("band"), py::arg("min"), py::arg("max"),
        py::arg("inclusive") = std::make_pair(false, false))
    .def_static("band_range",
        [](const std::vector<ccd::index_t>& bands, double min, double max,
           std::pair<bool, bool> inclusive)
        {
            return pp::Step::band_range(ccd::BandList(bands), to_range(min, max, inclusive));
        },
        py::arg("bands"), py::arg("min"), py::arg("max"),
        py::arg("inclusive") = std::make_pair(false, false))
    .def_static("green_median", &pp::Step::green_median,
        py::arg("band"), py::arg("green_offset"), py::arg("max_date"), py::arg("scope"))
    .def_static("dedupe", &pp::Step::dedupe)

    .def_readwrite("kind", &pp::Step::kind)
    .def_property("codes",
        [](const pp::Step& s) { return from_codes(s.codes); },
        [](pp::Step& s, const std::vector<int>& codes) { s.codes = to_codes(codes); },
        "QA codes (KeepQa, AddQa), ascending.")
    .def_property("bands",
        [](const pp::Step& s) { return s.bands.to_vector(); },
        [](pp::Step& s, const std::vector<ccd::index_t>& b) { s.bands = ccd::BandList(b); },
        "Band rows tested, in order (BandRange).")
    .def_readwrite("band", &pp::Step::band,
        "Band row read (ThermalRange, GreenMedian).")
    .def_property("min",
        [](const pp::Step& s) { return s.range.lo.value; },
        [](pp::Step& s, double v) { s.range.lo.value = v; },
        "Lower bound (ThermalRange, BandRange).")
    .def_property("max",
        [](const pp::Step& s) { return s.range.hi.value; },
        [](pp::Step& s, double v) { s.range.hi.value = v; },
        "Upper bound (ThermalRange, BandRange).")
    .def_property("inclusive",
        [](const pp::Step& s)
        {
            return std::make_pair(s.range.lo.inclusive, s.range.hi.inclusive);
        },
        [](pp::Step& s, std::pair<bool, bool> v)
        {
            s.range.lo.inclusive = v.first;
            s.range.hi.inclusive = v.second;
        },
        "(min, max): whether a value equal to each bound is kept.")
    .def_readwrite("green_offset", &pp::Step::green_offset,
        "Added to the median (GreenMedian).")
    .def_readwrite("max_date", &pp::Step::max_date,
        "Last date of the median's window, an ordinal (GreenMedian).")
    .def_readwrite("scope", &pp::Step::scope,
        "Which dates are dropped, a GreenFilterScope (GreenMedian).")
    .def("__eq__", [](const pp::Step& a, const pp::Step& b) { return same(a, b); },
        py::is_operator())
    .def("__repr__", [](const pp::Step& s) { return "<Step " + step_text(s) + ">"; });

    //--------------------------------------------------------------------------
    // Chain
    //--------------------------------------------------------------------------

    py::class_<pp::Chain>(
        m,
        "Chain",
        "An ordered list of at most CHAIN_CAPACITY Steps, run in order. "
        "Chain(steps) and append() raise ValueError past the capacity."
    )
    .def(py::init<>())
    .def(py::init(&to_chain), py::arg("steps"))
    .def("append", &pp::Chain::push_back, py::arg("step"))
    .def("__len__", &pp::Chain::size)
    .def("__getitem__",
        [](pp::Chain& c, std::ptrdiff_t i) -> pp::Step& { return c[wrap_index(c, i)]; },
        py::return_value_policy::reference_internal,
        py::arg("index"))
    .def("__setitem__",
        [](pp::Chain& c, std::ptrdiff_t i, const pp::Step& s) { c[wrap_index(c, i)] = s; },
        py::arg("index"), py::arg("step"))
    .def("__iter__",
        [](const pp::Chain& c) { return py::make_iterator(c.begin(), c.end()); },
        py::keep_alive<0, 1>())
    .def_property_readonly("steps",
        [](const pp::Chain& c) { return std::vector<pp::Step>(c.begin(), c.end()); },
        "The steps, as a list of copies.")
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

    //--------------------------------------------------------------------------
    // Affine, ProcedurePlan, PreprocessPlan
    //--------------------------------------------------------------------------

    py::class_<pp::Affine>(
        m,
        "Affine",
        "A transform of one band of the per-pixel copy, before the chain "
        "runs: value <- value * scale + offset. The caller's array is never "
        "written."
    )
    .def(py::init(
        [](ccd::index_t band, double scale, double offset)
        {
            return pp::Affine{band, scale, offset};
        }),
        py::arg("band"), py::arg("scale") = 1.0, py::arg("offset") = 0.0)
    .def_readwrite("band", &pp::Affine::band)
    .def_readwrite("scale", &pp::Affine::scale)
    .def_readwrite("offset", &pp::Affine::offset)
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
        "One procedure's preprocessing: an optional Affine transform, then a "
        "Chain that builds the processing mask."
    )
    .def(py::init(
        [](std::optional<pp::Affine> transform, const pp::Chain& chain)
        {
            return pp::ProcedurePlan{transform, chain};
        }),
        py::arg("transform") = py::none(), py::arg("chain") = pp::Chain())
    .def_readwrite("transform", &pp::ProcedurePlan::transform)
    .def_readwrite("chain", &pp::ProcedurePlan::chain)
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
        "A ProcedurePlan per procedure: standard, snow (PermanentSnow) and "
        "insufficient_clear. detect, detect_cube and detect_cube_columns run "
        "the one of the procedure each pixel selects; pass it as preprocess=. "
        "default_plan(hoptions) is the plan they run without one."
    )
    .def(py::init(
        [](const pp::ProcedurePlan& standard,
           const pp::ProcedurePlan& snow,
           const pp::ProcedurePlan& insufficient_clear)
        {
            return pp::PreprocessPlan{standard, snow, insufficient_clear};
        }),
        py::arg("standard") = pp::ProcedurePlan(),
        py::arg("snow") = pp::ProcedurePlan(),
        py::arg("insufficient_clear") = pp::ProcedurePlan())
    .def_readwrite("standard", &pp::PreprocessPlan::standard)
    .def_readwrite("snow", &pp::PreprocessPlan::snow)
    .def_readwrite("insufficient_clear", &pp::PreprocessPlan::insufficient_clear)
    .def("describe", &plan_text,
        "The plan as text: per procedure, its transform and one numbered "
        "line per step.")
    .def("__eq__",
        [](const pp::PreprocessPlan& a, const pp::PreprocessPlan& b) { return same(a, b); },
        py::is_operator())
    .def("__repr__", &plan_text);

    m.def(
        "default_plan",
        &pp::default_plan,
        py::arg("hoptions"),
        "The PreprocessPlan `hoptions` describe, the one detect runs without "
        "preprocess=. Every procedure keeps QA_CLEAR and QA_WATER, tests "
        "THERMAL_IDX against THERM_MIN/THERM_MAX (omitted without a thermal "
        "band), tests SATURATION_BANDS against REFL_MIN/REFL_MAX and dedupes; "
        "snow then adds QA_SNOW and dedupes, insufficient_clear runs the "
        "green median filter and dedupes. Only standard has a transform: "
        "THERMAL_SCALE/THERMAL_OFFSET, none when that is the identity."
    );

    m.def(
        "validate_plan",
        &pp::validate,
        py::arg("plan"),
        py::arg("bands"),
        "Raise ValueError, naming the procedure and the step, unless `plan` "
        "fits an input of `bands` bands: every band below `bands`, a "
        "GreenMedian band given, no transformed band in a BandRange of its "
        "procedure, and every AddQa followed by a Dedupe. detect* run this "
        "check on preprocess= before any work."
    );
}
