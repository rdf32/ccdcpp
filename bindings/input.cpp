#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ccd/core/constants.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/qa_table.hpp"

namespace py = pybind11;


namespace
{

// Every field of ccd::Input, by its Python keyword, and how a value sets it.
using Setter = void (*)(ccd::Input&, const py::handle&);

template<class T, T ccd::Input::*Member>
void set_member(ccd::Input& input, const py::handle& value)
{
    input.*Member = value.cast<T>();
}

void set_qa_table(ccd::Input& input, const py::handle& value)
{
    input.qa_table = value.is_none() ? nullptr : value.cast<std::shared_ptr<ccd::QaTable>>();
}

const std::vector<std::pair<const char*, Setter>>& input_fields()
{
    using ccd::Input;
    using ccd::scalar_t;
    static const std::vector<std::pair<const char*, Setter>> fields = {
        {"qa_fill", set_member<std::uint8_t, &Input::qa_fill>},
        {"qa_clear", set_member<std::uint8_t, &Input::qa_clear>},
        {"qa_water", set_member<std::uint8_t, &Input::qa_water>},
        {"qa_shadow", set_member<std::uint8_t, &Input::qa_shadow>},
        {"qa_snow", set_member<std::uint8_t, &Input::qa_snow>},
        {"qa_cloud", set_member<std::uint8_t, &Input::qa_cloud>},
        {"qa_cirrus1", set_member<std::uint8_t, &Input::qa_cirrus1>},
        {"qa_cirrus2", set_member<std::uint8_t, &Input::qa_cirrus2>},
        {"qa_occlusion", set_member<std::uint8_t, &Input::qa_occlusion>},
        {"qa_table", set_qa_table},
        {"roles", set_member<std::vector<std::string>, &Input::roles>},
        {"reflectance_unit", set_member<scalar_t, &Input::reflectance_unit>},
        {"reflectance_offset", set_member<scalar_t, &Input::reflectance_offset>},
        {"thermal_scale", set_member<scalar_t, &Input::thermal_scale>},
        {"thermal_offset", set_member<scalar_t, &Input::thermal_offset>},
        {"thermal_unit", set_member<scalar_t, &Input::thermal_unit>},
        {"saturated", set_member<std::optional<scalar_t>, &Input::saturated>},
    };
    return fields;
}

// `input` with the keyword `fields` set; TypeError, naming `where` and the
// fields, for another keyword or a value of the wrong type.
ccd::Input with_fields(ccd::Input input, const py::kwargs& fields, const char* where)
{
    for (const auto& [key, value] : fields)
    {
        const std::string name = py::str(key);
        const auto& known = input_fields();
        const auto it = std::find_if(known.begin(), known.end(),
                                     [&](const auto& f) { return name == f.first; });
        if (it == known.end())
        {
            std::string names;
            for (const auto& f : known)
                names += std::string(names.empty() ? "" : ", ") + f.first;
            throw py::type_error(std::string(where) + " has no field '" + name +
                                 "'; the fields are " + names);
        }
        try
        {
            it->second(input, value);
        }
        catch (const py::cast_error&)
        {
            throw py::type_error(std::string(where) + ": " + name + " is " +
                                 py::repr(value).cast<std::string>() +
                                 ", which it cannot hold");
        }
    }
    return input;
}

} // namespace


void bind_input(py::module_& m)
{
    // A band index that names no band.
    m.attr("NO_BAND") = ccd::NO_BAND;

    // The largest band count, and the capacity of each band list.
    m.attr("CCD_MAX_BANDS") = ccd::CCD_MAX_BANDS;

    // The roles a row can play.
    m.attr("ROLES") = py::tuple(py::cast(ccd::ROLES));

    py::enum_<ccd::GreenFilterScope>(
        m,
        "GreenFilterScope",
        "Which observations the green median filter drops. The threshold is "
        "the median green of the masked observations dated <= stat_ord plus "
        "the step's offset; a masked observation whose green is >= it is "
        "dropped at any date (All), only when dated <= stat_ord (StatWindow, "
        "the scope pyccd computes the mask for), or never (Off). With no "
        "masked observation dated <= stat_ord nothing is dropped. The "
        "green_median step's scope names them \"all\", \"stat_window\" and "
        "\"off\"."
    )
    .value("All", ccd::GreenFilterScope::All)
    .value("StatWindow", ccd::GreenFilterScope::StatWindow)
    .value("Off", ccd::GreenFilterScope::Off);

    py::class_<ccd::Input>(
        m,
        "Input",
        "What the caller's data is: its QA codes and the table that decodes "
        "its raw QA, a role per row of the spectral array, and its units. It "
        "holds no algorithm parameter: those are the configuration's, "
        "resolved against the Input when a run is built. The defaults are "
        "Landsat Collection-1 DN as pyccd consumes it, QA as codes. "
        "ccdcpp.SensorProfile.input() builds one for a sensor.\n\n"
        "The input says only how to read the data. The configuration says "
        "what to do with it, in physical units. The run translates once: 0–1 "
        "reflectance × 10000 = the 0–10000 DN check, exactly as today.\n\n"
        "Immutable: Input(**fields) takes every field by keyword, each "
        "defaulting as below, and replace(**fields) returns a copy with those "
        "fields changed. Equal and hashed by value.\n\n"
        "Example: ccdcpp.Input(reflectance_unit=1.0, thermal_scale=1.0, "
        "thermal_offset=0.0, thermal_unit=1.0)  # reflectance, Celsius / 100"
    )
    .def(py::init([](const py::kwargs& fields) { return with_fields(ccd::Input{}, fields, "Input()"); }))
    .def("replace",
         [](const ccd::Input& self, const py::kwargs& fields)
         { return with_fields(self, fields, "Input.replace()"); },
         "A copy of this input with the fields given by keyword changed.")

    //--------------------------------------------------------------------------
    // Band layout
    //--------------------------------------------------------------------------
    .def_property_readonly(
        "roles",
        [](const ccd::Input& self) { return self.roles; },
        "A role (ccdcpp.ROLES) per row of the spectral array, in row order "
        "(default blue, green, red, nir, swir1, swir2, thermal). Its length is "
        "the band count every call must pass. \"green\" is required; without "
        "\"thermal\" the thermal transform and filter are skipped."
    )
    .def("bands", &ccd::Input::bands, "The band count: one row per role.")
    .def("index", &ccd::Input::index, py::arg("role"),
         "The row of `role`, or ccdcpp.NO_BAND when the input has none.")

    //--------------------------------------------------------------------------
    // Units
    //
    // The physical-unit input is one divisor, 10000, on the data, with
    // reflectance its unit:
    //
    //     spectral[:6] = dn[:6] / 1e4             # reflectance 0..1
    //     spectral[6]  = (k10 * 10 - 27315) / 1e4 # degrees Celsius / 100
    //
    //     i = ccdcpp.Input(reflectance_unit=1.0, thermal_scale=1.0,
    //                      thermal_offset=0.0, thermal_unit=1.0)
    //
    // See docs/units.md for the measured equivalence and why plain Celsius
    // is not the physical unit.
    //--------------------------------------------------------------------------
    .def_readonly(
        "reflectance_unit", &ccd::Input::reflectance_unit,
        "The value one unit of reflectance has in the data (default 10000, "
        "Collection-1 DN; 1 for reflectance): reflectance r is held as r * "
        "reflectance_unit + reflectance_offset. The configuration's "
        "reflectance levels (band_range's bounds) resolve that way, its "
        "differences (alpha, coef_floor, green_median's offset) times "
        "reflectance_unit only."
    )
    .def_readonly(
        "reflectance_offset", &ccd::Input::reflectance_offset,
        "The data's value of reflectance 0 (default 0). Collection-2 DN, r = "
        "DN * 2.75e-5 - 0.2, is reflectance_unit 1 / 2.75e-5 and "
        "reflectance_offset 0.2 / 2.75e-5."
    )
    .def_readonly("thermal_scale", &ccd::Input::thermal_scale,
        "A procedure that transforms thermal reads thermal * thermal_scale + "
        "thermal_offset (default 10, Kelvin x 10 to Celsius x 100), in the "
        "unit thermal_unit names.")
    .def_readonly("thermal_offset", &ccd::Input::thermal_offset,
        "See thermal_scale (default -27315).")
    .def_readonly("thermal_unit", &ccd::Input::thermal_unit,
        "The value 100 degrees Celsius has in the thermal row after the "
        "transform (default 10000, Celsius x 100; 1 for Celsius / 100). The "
        "configuration's degrees Celsius c resolve to c * 100 * thermal_unit "
        "/ 10000. It goes with the transform: Kelvin x 10 through (10, "
        "-27315) is 10000, Celsius / 100 through the identity is 1.")
    .def_readonly(
        "saturated", &ccd::Input::saturated,
        "The value a saturated observation holds, as the sensor writes it "
        "(20000 in Collection-1 DN), or None (the default) when the input "
        "does not declare one. A step or strategy that reads it refuses an "
        "input without one."
    )

    //--------------------------------------------------------------------------
    // QA codes
    //
    // The codes the procedures compare `qas` against: the caller's uint8
    // codes, or a QaTable's output, which ccdcpp.qa_tables builds from these
    // fields. See input.hpp for what reads each one.
    //--------------------------------------------------------------------------
    .def_readonly("qa_fill", &ccd::Input::qa_fill,
        "Fill: not a valid observation (default 255).")
    .def_readonly("qa_clear", &ccd::Input::qa_clear,
        "Clear: kept for the fit (default 0).")
    .def_readonly("qa_water", &ccd::Input::qa_water,
        "Water: kept for the fit (default 1).")
    .def_readonly("qa_shadow", &ccd::Input::qa_shadow,
        "Cloud shadow (default 2). Not read by the procedures, so excluded "
        "from the fit; the QA tables write it.")
    .def_readonly("qa_snow", &ccd::Input::qa_snow,
        "Snow: the snow percentage and PermanentSnow (default 3).")
    .def_readonly("qa_cloud", &ccd::Input::qa_cloud,
        "Cloud: excluded from the fit (default 4).")
    .def_readonly("qa_cirrus1", &ccd::Input::qa_cirrus1,
        "Informational; not read by ccdcpp (default 8).")
    .def_readonly("qa_cirrus2", &ccd::Input::qa_cirrus2,
        "Informational; not read by ccdcpp (default 9).")
    .def_readonly("qa_occlusion", &ccd::Input::qa_occlusion,
        "Informational; not read by ccdcpp (default 10).")
    .def_property_readonly(
        "qa_table",
        [](const ccd::Input& self)
        { return std::const_pointer_cast<ccd::QaTable>(self.qa_table); },
        "The ccdcpp.QaTable that decodes raw uint8 or uint16 qas into the "
        "qa_* codes (ccdcpp.qa_tables builds one with this input's codes), "
        "or None (the default): qas are the codes, uint8."
    )

    .def("__copy__", [](const ccd::Input& self) { return self; })
    .def("__deepcopy__", [](const ccd::Input& self, const py::dict&) { return self; })
    .def("__eq__", [](const ccd::Input& a, const ccd::Input& b) { return a == b; },
         py::is_operator())
    .def("__hash__", [](const ccd::Input& self) { return ccd::digest(self); })
    .def("digest", [](const ccd::Input& self) { return ccd::digest(self); },
         "A digest of every field: equal for equal inputs, on any platform.")
;
}
