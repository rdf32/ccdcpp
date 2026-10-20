#pragma once

// A run's RunSpec from the input's options and the configuration the tests
// vary (config.hpp).

#include "ccd/ccd.hpp"
#include "ccd/strategies/register_core.hpp"

#include "config.hpp"

namespace golden
{

inline ccd::RunSpec spec_of(const ccd::HarmonicOptions& h,
                            ccd::config::ConfigTree config = golden::ccdc(),
                            ccd::QaTablePtr table = nullptr,
                            ccd::PreprocessPlanPtr plan = nullptr,
                            bool record = false)
{
    ccd::RunSpec spec;
    spec.hoptions = h;
    spec.config = std::move(config);
    spec.qa_table = std::move(table);
    spec.plan = std::move(plan);
    spec.record = record;
    return spec;
}

// The strategies of a configuration, from the library's registry.
inline ccd::Strategies strategies_of(const ccd::config::ConfigTree& config = golden::ccdc())
{
    return ccd::strategies::build_strategies(ccd::strategies::library_registry(), config);
}

} // namespace golden
