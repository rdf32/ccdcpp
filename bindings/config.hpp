#pragma once

#include <pybind11/pybind11.h>

#include "ccd/config/registry.hpp"
#include "ccd/run/algorithm.hpp"

// A Python configuration as the Algorithm a run runs: None for the "ccdc"
// preset's; a ccdcpp.Config (its compiled `_algorithm`); an
// _extension.Algorithm; or a configuration's dict {"shared": {...}, <stage>:
// {"strategy": ..., <parameter>: ...}, "preprocess": {...}}, compiled here. TypeError on a value of
// no configuration type, ValueError for what the Algorithm refuses.
ccd::AlgorithmPtr algorithm_from_python(const pybind11::object& config);

// A configuration's dict as Python holds it, and back.
ccd::config::Config config_from_python(const pybind11::handle& config);
pybind11::dict config_to_python(const ccd::config::Config& config);
