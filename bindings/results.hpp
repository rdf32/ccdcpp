#pragma once

// The results as Python data (bindings/results.cpp), each shape next to its
// inverse: what detect() and detect_cube() return, and what resume= takes
// back of it.

#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

#include "ccd/ccd.hpp"

namespace py = pybind11;

py::dict header_to_python(const ccd::ResumeHeader& header);
ccd::ResumeHeader header_from_python(const py::dict& header);

// detect()'s "models": a list of dicts with a "bands" list each.
py::list models_to_python(const std::vector<ccd::ChangeModel>& models);
std::vector<ccd::ChangeModel> models_from_python(const py::list& models);

// detect()'s "resume": one pixel's record as a dict.
py::dict record_to_python(const ccd::PixelRecord& record);
ccd::PixelRecord record_from_python(const py::dict& record);

// detect_cube()'s output: one row per (pixel, segment), its counts and, with
// `masks`, the cube's processing masks (moved out of `cube`); and back, the
// stored segments of each of the H x W pixels, raster order.
py::dict segment_columns(
    ccd::CubeOutput& cube,
    ccd::index_t H,
    ccd::index_t W,
    ccd::index_t B,
    ccd::index_t T,
    bool masks
);
std::vector<std::vector<ccd::ChangeModel>> segments_from_columns(
    const py::dict& segments,
    ccd::index_t H,
    ccd::index_t W,
    ccd::index_t B
);

// detect_cube()'s "resume": the records of a cube's H x W pixels, raster
// order; a pixel that failed has a default record. With `status`, also each
// pixel's ResumeStatus. And back, each pixel's record.
py::dict record_columns(
    const ccd::CubeOutput& cube,
    ccd::index_t H,
    ccd::index_t W,
    ccd::index_t B,
    bool status
);
std::vector<ccd::PixelRecord> records_from_columns(
    const py::dict& records,
    ccd::index_t H,
    ccd::index_t W,
    ccd::index_t B
);

// detect_cube()'s "failures": (row, col, message) tuples.
py::list failures_to_python(const std::vector<ccd::CubeFailure>& failures);

// Each record column's name and numpy dtype, as record_columns() writes them.
py::dict resume_schema();
