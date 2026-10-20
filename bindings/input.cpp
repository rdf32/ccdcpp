#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "ccd/config/shared.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/input/input.hpp"
#include "ccd/input/qa_table.hpp"

namespace py = pybind11;


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
        "Which observations the InsufficientClear green median filter drops. "
        "The threshold is the median green of the masked observations dated "
        "<= stat_ord plus the green filter; a masked observation whose green "
        "is >= it is dropped at any date (All), only when dated <= stat_ord "
        "(StatWindow, the scope pyccd computes the mask for), or never (Off). "
        "With no masked observation dated <= stat_ord nothing is dropped. The "
        "configuration's shared green_filter_scope names them \"all\", "
        "\"stat_window\" and \"off\"."
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
        "ccdcpp.SensorProfile.input() builds one for a sensor."
    )
    .def(py::init<>())

    //--------------------------------------------------------------------------
    // Band layout
    //--------------------------------------------------------------------------
    .def_property(
        "roles",
        [](const ccd::Input& self) { return self.roles; },
        [](ccd::Input& self, const std::vector<std::string>& roles) { self.roles = roles; },
        "A role (ccdcpp.ROLES) per row of the spectral array, in row order "
        "(default blue, green, red, nir, swir1, swir2, thermal). Its length is "
        "the band count every call must pass. \"green\" is required; without "
        "\"thermal\" the thermal transform and filter are skipped."
    )
    .def_property(
        "saturation_bands",
        [](const ccd::Input& self) { return self.saturation_bands.to_vector(); },
        [](ccd::Input& self, const std::vector<ccd::index_t>& bands)
        { self.saturation_bands = ccd::BandList(bands); },
        "Rows tested against refl_min and refl_max, in order (default [0, 1, "
        "2, 3, 4, 5]). May be empty; must not hold the thermal row."
    )
    .def("bands", &ccd::Input::bands, "The band count: one row per role.")
    .def("index", &ccd::Input::index, py::arg("role"),
         "The row of `role`, or ccdcpp.NO_BAND when the input has none.")

    //--------------------------------------------------------------------------
    // Units
    //
    // The physical-unit input is one divisor, 10000, on the data and on
    // every bound, with reflectance its unit:
    //
    //     spectral[:6] = dn[:6] / 1e4             # reflectance 0..1
    //     spectral[6]  = (k10 * 10 - 27315) / 1e4 # degrees Celsius / 100
    //
    //     i = ccdcpp.Input()
    //     i.reflectance_unit = 1.0
    //     i.thermal_scale, i.thermal_offset = 1.0, 0.0
    //     i.refl_min, i.refl_max = 0.0, 1.0
    //     i.therm_min, i.therm_max = -0.9320, 0.7070
    //
    // See input.hpp for the measured equivalence and why plain Celsius is not
    // the physical unit.
    //--------------------------------------------------------------------------
    .def_readwrite(
        "reflectance_unit", &ccd::Input::reflectance_unit,
        "The value one unit of reflectance has in the data (default 10000, "
        "Collection-1 DN; 1 for reflectance). The configuration's reflectance "
        "values (alpha, coef_floor, the green filter) are multiplied by it."
    )
    .def_readwrite("thermal_scale", &ccd::Input::thermal_scale,
        "The Standard procedure reads thermal * thermal_scale + thermal_offset "
        "(default 10, Kelvin x 10 to Celsius x 100).")
    .def_readwrite("thermal_offset", &ccd::Input::thermal_offset,
        "See thermal_scale (default -27315).")
    .def_readwrite("refl_min", &ccd::Input::refl_min,
        "Lower saturation bound of the saturation_bands (default 0).")
    .def_readwrite("refl_max", &ccd::Input::refl_max,
        "Upper saturation bound of the saturation_bands (default 10000).")
    .def_readwrite("therm_min", &ccd::Input::therm_min,
        "Lower bound of the thermal row, after the transform (default -9320).")
    .def_readwrite("therm_max", &ccd::Input::therm_max,
        "Upper bound of the thermal row, after the transform (default 7070).")
    .def_readwrite(
        "refl_min_inclusive", &ccd::Input::refl_min_inclusive,
        "Keep a saturation-band value equal to refl_min (default False, "
        "pyccd's exclusive 0 < x < 10000, which rejects NaN). NaN passes the "
        "saturation filter only with both refl_* flags True."
    )
    .def_readwrite("refl_max_inclusive", &ccd::Input::refl_max_inclusive,
        "Keep a saturation-band value equal to refl_max (default False). See "
        "refl_min_inclusive.")
    .def_readwrite(
        "therm_min_inclusive", &ccd::Input::therm_min_inclusive,
        "Keep a thermal value equal to therm_min (default False, pyccd's "
        "exclusive bound, which rejects NaN). NaN passes the thermal filter "
        "only with both therm_* flags True."
    )
    .def_readwrite("therm_max_inclusive", &ccd::Input::therm_max_inclusive,
        "Keep a thermal value equal to therm_max (default False). See "
        "therm_min_inclusive.")

    //--------------------------------------------------------------------------
    // QA codes
    //
    // The codes the procedures compare `qas` against: the caller's uint8
    // codes, or a QaTable's output, which ccdcpp.qa_tables builds from these
    // fields. See input.hpp for what reads each one.
    //--------------------------------------------------------------------------
    .def_readwrite("qa_fill", &ccd::Input::qa_fill,
        "Fill: not a valid observation (default 255).")
    .def_readwrite("qa_clear", &ccd::Input::qa_clear,
        "Clear: kept for the fit (default 0).")
    .def_readwrite("qa_water", &ccd::Input::qa_water,
        "Water: kept for the fit (default 1).")
    .def_readwrite("qa_shadow", &ccd::Input::qa_shadow,
        "Cloud shadow (default 2). Not read by the procedures, so excluded "
        "from the fit; the QA tables write it.")
    .def_readwrite("qa_snow", &ccd::Input::qa_snow,
        "Snow: the snow percentage and PermanentSnow (default 3).")
    .def_readwrite("qa_cloud", &ccd::Input::qa_cloud,
        "Cloud: excluded from the fit (default 4).")
    .def_readwrite("qa_cirrus1", &ccd::Input::qa_cirrus1,
        "Informational; not read by ccdcpp (default 8).")
    .def_readwrite("qa_cirrus2", &ccd::Input::qa_cirrus2,
        "Informational; not read by ccdcpp (default 9).")
    .def_readwrite("qa_occlusion", &ccd::Input::qa_occlusion,
        "Informational; not read by ccdcpp (default 10).")
    .def_property(
        "qa_table",
        [](const ccd::Input& self)
        { return std::const_pointer_cast<ccd::QaTable>(self.qa_table); },
        [](ccd::Input& self, std::shared_ptr<ccd::QaTable> table)
        { self.qa_table = std::move(table); },
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
