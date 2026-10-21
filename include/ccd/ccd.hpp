#pragma once

// ccdcpp's C++ API: the entry points detect and detect_cube (run/run.hpp),
// and every header a caller or a strategy extension uses. The built-in
// strategies' own headers (strategies/<stage>/<name>.hpp) are included where
// they are used.

#include "ccd/core/array_view.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/maths.hpp"
#include "ccd/core/types.hpp"

#include "ccd/config/params.hpp"
#include "ccd/config/preprocess.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/config/shared.hpp"

#include "ccd/input/input.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/preprocess.hpp"
#include "ccd/input/mask/steps.hpp"
#include "ccd/input/masked_series.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/pmask.hpp"
#include "ccd/input/qa_statistics.hpp"
#include "ccd/input/qa_table.hpp"
#include "ccd/input/qa_tables.hpp"

#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/regression/robust_solver.hpp"

#include "ccd/machine/fit.hpp"
#include "ccd/machine/kernels.hpp"
#include "ccd/machine/pixel_procedure.hpp"
#include "ccd/machine/suspended.hpp"

#include "ccd/strategies/capabilities.hpp"
#include "ccd/strategies/coefficients.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/per_thread.hpp"
#include "ccd/strategies/register_core.hpp"
#include "ccd/strategies/stages.hpp"

#include "ccd/results/diagnostics.hpp"
#include "ccd/results/record.hpp"

#include "ccd/run/algorithm.hpp"
#include "ccd/run/run.hpp"
