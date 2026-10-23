#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <memory>
#include <string>

#include "ccd/input/data_spec.hpp"
#include "ccd/input/qa_tables.hpp"

namespace py = pybind11;

//------------------------------------------------------------------------------
// QaTable and the ccdcpp.qa_tables builders.
//
// Python holds a table as std::shared_ptr<ccd::QaTable>, the holder pybind11
// needs; the class exposes no way to write to it, so it is as read-only as
// the std::shared_ptr<const QaTable> the core takes, and the conversion
// between the two is a pointer copy. An Input holds one as its qa_table.
//------------------------------------------------------------------------------

namespace
{

std::shared_ptr<ccd::QaTable> held(ccd::QaTablePtr table)
{
    return std::const_pointer_cast<ccd::QaTable>(std::move(table));
}

std::shared_ptr<ccd::QaTable> from_codes(const py::object& obj)
{
    auto codes = py::array_t<std::uint8_t, py::array::c_style>::ensure(obj);

    if (!codes)
        throw py::type_error(
            "codes cannot be converted to uint8 without an unsafe cast"
        );

    if (codes.ndim() != 1 ||
        static_cast<ccd::index_t>(codes.shape(0)) != ccd::QA_TABLE_SIZE)
    {
        throw py::value_error(
            "codes must have shape (" + std::to_string(ccd::QA_TABLE_SIZE) +
            ",), one code per uint16 value"
        );
    }

    ccd::QaTable::Codes copy;
    std::copy(codes.data(), codes.data() + ccd::QA_TABLE_SIZE, copy.begin());
    return std::make_shared<ccd::QaTable>(copy);
}

} // namespace

void bind_qa_tables(py::module_& m)
{
    m.attr("QA_TABLE_SIZE") = ccd::QA_TABLE_SIZE;

    py::class_<ccd::QaTable, std::shared_ptr<ccd::QaTable>>(
        m,
        "QaTable",
        "A lookup from raw QA values to the QA codes detect reads: entry v "
        "is the code of raw value v, for every uint16 v. An Input holds one "
        "as its qa_table to decode a raw uint8 or uint16 qas array per pixel. "
        "Build one with ccdcpp.qa_tables, or from 65,536 uint8 codes. "
        "Immutable."
    )
    .def(
        py::init(&from_codes),
        py::arg("codes"),
        "A table from a (65536,) array of codes, entry v for raw value v."
    )
    .def_property_readonly(
        "codes",
        [](const ccd::QaTable& t)
        {
            return py::array_t<std::uint8_t>(
                static_cast<py::ssize_t>(ccd::QA_TABLE_SIZE),
                t.code.data()
            );
        },
        "A copy of the (65536,) uint8 codes."
    )
    .def_property_readonly(
        "digest",
        [](const ccd::QaTable& t) { return t.digest; },
        "A digest of the codes: equal for equal tables, on any platform."
    )
    .def("__eq__", [](const ccd::QaTable& a, const ccd::QaTable& b)
         { return a.digest == b.digest && a.code == b.code; }, py::is_operator())
    .def("__hash__", [](const ccd::QaTable& t) { return t.digest; });

    py::module_ sub = m.def_submodule(
        "qa_tables",
        "QaTable builders, one per QA product. Each writes the qa_* codes of "
        "the input it is given (an Input, default Input()), so build with "
        "the input the run uses. Precedence is most-excluding "
        "first: fill, cloud, shadow, snow, water, clear. The Landsat tables "
        "are validated for production use. The HLS and Sentinel-2 tables "
        "follow the products' definitions, are tested, and decode as that "
        "documentation says on every value present in the real sample the "
        "HLS and Sentinel-2 presets were validated on (MGRS tile T16SGC, "
        "2020-2023); no production run has used them yet."
    );

    sub.def(
        "landsat_c2_qa_pixel",
        [](const ccd::DataSpec& o)
        {
            return held(ccd::qa_tables::landsat_c2_qa_pixel(o));
        },
        py::arg("input") = ccd::DataSpec{},
        "Landsat Collection 2 QA_PIXEL (uint16). Bit 0 fill; bits 3, 1, 2 "
        "(cloud, dilated cloud, cirrus) cloud; bit 4 shadow; bit 5 snow; "
        "bit 7 water; otherwise clear. Bits 6 and 8-15 are not read."
    );

    sub.def(
        "landsat_c1_pixel_qa",
        [](const ccd::DataSpec& o)
        {
            return held(ccd::qa_tables::landsat_c1_pixel_qa(o));
        },
        py::arg("input") = ccd::DataSpec{},
        "Landsat Collection 1 pixel_qa (uint16), in pyccd's order. Bit 0 "
        "fill; bit 5 cloud; bit 3 shadow; bit 4 snow; bit 2 water; bit 1 "
        "clear; bits 8 and 9 both set, or bit 10, clear; otherwise fill "
        "(pyccd raises there)."
    );

    sub.def(
        "hls_fmask",
        [](const ccd::DataSpec& o, bool exclude_high_aerosol)
        {
            return held(ccd::qa_tables::hls_fmask(o, exclude_high_aerosol));
        },
        py::arg("input") = ccd::DataSpec{},
        py::arg("exclude_high_aerosol") = false,
        "HLS v2.0 Fmask (uint8). 255 fill; bits 1, 2 (cloud, adjacent to "
        "cloud or shadow) cloud; bit 0, reserved and unused in v2.0 (Fmask "
        "folds cirrus into cloud), cloud should it ever be set; bit 3 "
        "shadow; bit 4 snow; bit 5 water; otherwise clear. Bits 6-7 "
        "(aerosol) are ignored unless exclude_high_aerosol, which makes high "
        "aerosol (11) cloud. Values above 255 are fill. Decodes as the HLS "
        "v2.0 User Guide says on all 43 values present in the T16SGC "
        "2020-2023 sample (38 in L30, 43 in S30); no production run has "
        "used it yet."
    );

    sub.def(
        "sentinel2_scl",
        [](const ccd::DataSpec& o)
        {
            return held(ccd::qa_tables::sentinel2_scl(o));
        },
        py::arg("input") = ccd::DataSpec{},
        "Sentinel-2 L2A SCL (uint8). 0, 1 fill; 2, 3 (cast shadows, cloud "
        "shadows) shadow; 4, 5 clear; 6 water; 7-10 (unclassified, cloud "
        "medium and high probability, thin cirrus) cloud; 11 snow; anything "
        "else fill. Decodes as SentiWiki says on every class present in the "
        "T16SGC 2020-2023 sample (0, 2-11); no production run has used it "
        "yet."
    );
}
