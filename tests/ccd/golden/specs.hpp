#pragma once

// A run's RunSpec from the input and the algorithm the tests vary
// (config.hpp).

#include <utility>

#include "ccd/ccd.hpp"
#include "ccd/strategies/register_core.hpp"

#include "config.hpp"

namespace golden
{

inline ccd::RunSpec spec_of(const ccd::Input& input,
                            ccd::AlgorithmPtr algorithm = ccd::Algorithm::preset("ccdc"),
                            ccd::QaTablePtr table = nullptr,
                            ccd::PreprocessPlanPtr plan = nullptr,
                            bool record = false)
{
    ccd::RunSpec spec;
    spec.input = input;
    spec.algorithm = std::move(algorithm);
    spec.qa_table = std::move(table);
    spec.plan = std::move(plan);
    spec.record = record;
    return spec;
}

// The same, of a configuration (compiled here, once per call).
inline ccd::RunSpec spec_of(const ccd::Input& input,
                            ccd::config::Config config,
                            ccd::QaTablePtr table = nullptr,
                            ccd::PreprocessPlanPtr plan = nullptr,
                            bool record = false)
{
    return spec_of(input, algorithm(std::move(config)), std::move(table), std::move(plan),
                   record);
}

} // namespace golden
