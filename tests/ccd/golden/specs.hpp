#pragma once

// A run's RunSpec from the input's options and the algorithm's values the
// tests vary (golden::Algorithm, algorithm.hpp).

#include "ccd/ccd.hpp"
#include "ccd/strategies/register_core.hpp"

#include "algorithm.hpp"

namespace golden
{

inline ccd::RunSpec spec_of(const ccd::HarmonicOptions& h,
                            const golden::Algorithm& a = {},
                            ccd::QaTablePtr table = nullptr,
                            ccd::PreprocessPlanPtr plan = nullptr,
                            bool record = false)
{
    ccd::RunSpec spec;
    spec.hoptions = h;
    spec.config = golden::config_of(a);
    spec.qa_table = std::move(table);
    spec.plan = std::move(plan);
    spec.record = record;
    return spec;
}

// The same with 0.10's defaults but for the Lasso's parameters.
inline ccd::RunSpec spec_of(const ccd::HarmonicOptions& h,
                            const ccd::LassoOptions& lasso,
                            ccd::QaTablePtr table = nullptr,
                            ccd::PreprocessPlanPtr plan = nullptr,
                            bool record = false)
{
    golden::Algorithm a;
    a.lasso = lasso;
    return spec_of(h, a, std::move(table), std::move(plan), record);
}

// The strategies of that configuration, from the library's registry.
inline ccd::Strategies strategies_of(const golden::Algorithm& a = {})
{
    return ccd::strategies::build_strategies(ccd::strategies::library_registry(),
                                             golden::config_of(a));
}

} // namespace golden
