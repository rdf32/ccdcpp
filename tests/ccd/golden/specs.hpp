#pragma once

// A run's RunSpec from the input and the algorithm the tests vary
// (config.hpp), with the processing masks the dumps hold, and an input with a
// QA table.

#include <utility>

#include "ccd/ccd.hpp"
#include "ccd/strategies/register_core.hpp"

#include "config.hpp"

namespace golden
{

inline ccd::RunSpec spec_of(const ccd::Input& input,
                            ccd::AlgorithmPtr algorithm = pyccd_algorithm(),
                            bool record = false)
{
    ccd::RunSpec spec;
    spec.encoding = input;
    spec.algorithm = std::move(algorithm);
    spec.record = record;
    spec.masks = true;
    return spec;
}

// The same, of a configuration (compiled here, once per call).
inline ccd::RunSpec spec_of(const ccd::Input& input, ccd::config::Config config,
                            bool record = false)
{
    return spec_of(input, algorithm(std::move(config)), record);
}

// `input` decoding its raw QA through `table`.
inline ccd::Input with_table(ccd::Input input, ccd::QaTablePtr table)
{
    input.qa_table = std::move(table);
    return input;
}

} // namespace golden
