#pragma once

// The resume record as Python data (bindings/resume.cpp): the header as a
// dict, one pixel's record as a dict, a cube's records as named columns in
// raster order, and the stored segments a resume takes back, in the shapes
// detect() and detect_cube() return them.

#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

#include "ccd/ccd.hpp"

namespace py = pybind11;

py::dict header_to_python(const ccd::ResumeHeader& header);
ccd::ResumeHeader header_from_python(const py::dict& header);

py::dict record_to_python(const ccd::PixelRecord& record);
ccd::PixelRecord record_from_python(const py::dict& record);

// detect()'s "models": a list of dicts with a "bands" list each.
std::vector<ccd::ChangeModel> segments_from_python(const py::list& models);

// The records of a cube's H x W pixels, raster order; a pixel that failed has
// a default record. With `status`, also each pixel's ResumeStatus.
py::dict record_columns(
    const ccd::CubeResult& cube,
    ccd::index_t H,
    ccd::index_t W,
    ccd::index_t B,
    bool status
);

// What run_cube()'s resume reads: the header, record_columns()'s dict and
// detect_cube()'s segment columns.
ccd::CubeResumeFrom cube_from_python(
    const py::dict& header,
    const py::dict& records,
    const py::dict& segments,
    ccd::index_t H,
    ccd::index_t W,
    ccd::index_t B
);

// Each record column's name and numpy dtype, as record_columns() writes them.
py::dict resume_schema();
