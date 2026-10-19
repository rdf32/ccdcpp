#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "ccd/config/registry.hpp"

using ConfigPtr = std::shared_ptr<const ccd::config::ConfigTree>;

// A Python configuration (a ccdcpp.Config, a dict of stages, or None) as a
// run's RunSpec::config: null for None. ValueError on a value of no
// configuration type; the strategies and parameters are checked by the run.
ConfigPtr config_from_python(const pybind11::object& config);

// A configuration as the dict of stages Python holds.
pybind11::dict config_to_python(const ccd::config::ConfigTree& tree);
