#pragma once

// ccdcpp's C++ API: the entry points detect and detect_cube (run/run.hpp),
// and every header a caller or a strategy extension uses. The built-in
// strategies' own headers (strategies/<stage>/<name>.hpp) are included where
// they are used.

#include "ccd/core/array_view.hpp"
#include "ccd/core/band_list.hpp"
#include "ccd/core/build_info.hpp"
#include "ccd/core/constants.hpp"
#include "ccd/core/maths.hpp"
#include "ccd/core/types.hpp"
#include "ccd/core/window.hpp"

#include "ccd/config/config.hpp"
#include "ccd/config/params.hpp"
#include "ccd/config/preprocess.hpp"
#include "ccd/config/presets.hpp"
#include "ccd/config/registry.hpp"
#include "ccd/config/resolve.hpp"
#include "ccd/config/shared.hpp"

#include "ccd/input/data_spec.hpp"
#include "ccd/input/mask/filter.hpp"
#include "ccd/input/mask/chain.hpp"
#include "ccd/input/mask/reason.hpp"
#include "ccd/input/mask/steps.hpp"
#include "ccd/input/masked_series.hpp"
#include "ccd/input/pixel_series.hpp"
#include "ccd/input/procedure.hpp"
#include "ccd/input/processing_mask.hpp"
#include "ccd/input/qa_statistics.hpp"
#include "ccd/input/qa_table.hpp"
#include "ccd/input/qa_tables.hpp"

#include "ccd/regression/coefficients.hpp"
#include "ccd/regression/harmonic_basis.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/regression/robust_solver.hpp"

#include "ccd/machine/suspended.hpp"

#include "ccd/strategies/capabilities.hpp"
#include "ccd/strategies/library.hpp"
#include "ccd/strategies/model.hpp"
#include "ccd/strategies/per_thread.hpp"
#include "ccd/strategies/pyccd/peek_statistics.hpp"
#include "ccd/strategies/stages.hpp"

#include "ccd/results/diagnostics.hpp"
#include "ccd/results/record.hpp"
#include "ccd/results/segment.hpp"

#include "ccd/run/algorithm.hpp"
#include "ccd/run/run.hpp"
