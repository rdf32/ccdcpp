#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "ccd/config/registry.hpp"
#include "ccd/run/algorithm.hpp"

// The Algorithm a call runs: its config=, an Algorithm, or the "ccdc"
// preset's for None.
ccd::AlgorithmPtr algorithm_or_ccdc(std::shared_ptr<ccd::Algorithm> config);

// A configuration's dict as Python holds it, and back.
ccd::config::Config config_from_python(const pybind11::handle& config);
pybind11::dict config_to_python(const ccd::config::Config& config);
