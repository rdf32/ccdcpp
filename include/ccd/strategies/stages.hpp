#pragma once

#include <memory>

#include "ccd/types.hpp"

namespace ccd
{

//==============================================================================
//
// The stages of the algorithm
//
// The machine (procedure/standard.hpp) is the CCDC family's fixed skeleton:
// its phases and routes. What each phase decides is a stage, decided by the
// strategy the run's configuration names for it (config/registry.hpp). A
// strategy returns its decision; only the machine routes.
//
// Each stage is an interface with its name in `stage`; its strategies live
// one per file under strategies/<stage>/. Strategy objects are immutable and
// shared by every thread of a call.
//
// The stages so far (the others are still decided inside the machine):
//
//   refit   when Extend refits the monitoring model
//
//==============================================================================

//------------------------------------------------------------------------------
// refit: whether Extend refits the model before this peek test. The first
// test after a start always fits; this decides the tests after it.
//
//   window_size   observations in the model window now
//   window_span   days the model window spans now
//   fit_span      days it spanned when the model was last fitted
//------------------------------------------------------------------------------
struct RefitSchedule
{
    static constexpr const char* stage = "refit";

    virtual ~RefitSchedule() = default;
    virtual bool due(index_t window_size, scalar_t window_span, scalar_t fit_span) const = 0;
};

// One call's strategies, one per stage, built from its configuration.
struct Strategies
{
    std::shared_ptr<const RefitSchedule> refit;
};

} // namespace ccd
