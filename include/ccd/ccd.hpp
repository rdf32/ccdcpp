#pragma once

// ccdcpp's C++ API: the entry points detect and detect_cube (run/run.hpp)
// and the headers a caller and a strategy extension use.

#include "ccd/core/types.hpp"
#include "ccd/core/array_view.hpp"

#include "ccd/config/registry.hpp"
#include "ccd/input/input.hpp"
#include "ccd/machine/fit.hpp"
#include "ccd/input/qa_tables.hpp"
#include "ccd/results/record.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/run/run.hpp"
