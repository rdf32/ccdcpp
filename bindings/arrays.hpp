#pragma once

// The arrays of detect() and detect_cube(), as the run reads them
// (bindings/arrays.cpp): one intake for both calls, taken while the GIL is
// held, whose views the run reads with the GIL released.
//
// `spectral` is taken as it is when it is float64, float32 or uint16 in native
// byte order with non-negative strides -- any layout, C-contiguous or not --
// and read through a strided view; the run converts each pixel to float64 in
// its own buffer, so a uint16 cube is never widened as a whole. Anything else
// is converted to a C-contiguous float64 array first, under numpy safe
// casting. Without the spec's QA table `qas` is QA codes, uint8 under the
// same two rules; a uint16 `qas` holds raw QA values, which have no meaning
// without a table, and is a TypeError. With a table `qas` is raw values, uint8
// or uint16, anything else converted to uint16 under numpy safe casting.
// `dates` is C-contiguous int64, converted under numpy safe casting.
//
// Nothing is written to the arrays: a plan's transform happens in the
// run's per-pixel copy. Only the ranks are checked here, as the views need
// them; the run checks the lengths, the data spec and the dates.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ccd/ccd.hpp"

namespace py = pybind11;

// A call's views, and the buffers they read, kept alive.
template<typename Spectral, typename Qa>
struct Arrays
{
    py::array dates_array;
    py::array spectral_array;
    py::array qas_array;

    ccd::ArrayView<const std::int64_t, 1> dates;
    Spectral spectral;
    Qa qas;
};

// One pixel's: dates (T), spectral (B, T), qas (T).
struct PixelArrays : Arrays<ccd::PixelSpectral, ccd::PixelQa>
{
    ccd::index_t B = 0;
};

PixelArrays pixel_arrays(
    const py::object& dates,
    const py::object& spectral,
    const py::object& qas,
    bool with_table
);

// A cube's: dates (T), spectral (H, W, B, T), qas (H, W, T).
struct CubeArrays : Arrays<ccd::CubeSpectral, ccd::CubeQa>
{
    ccd::index_t H = 0;
    ccd::index_t W = 0;
    ccd::index_t B = 0;
    ccd::index_t T = 0;
};

CubeArrays cube_arrays(
    const py::object& dates,
    const py::object& spectral,
    const py::object& qas,
    bool with_table
);
