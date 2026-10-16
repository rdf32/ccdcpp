#pragma once

#include <string>

#include "ccd/types.hpp"
#include "ccd/logger.hpp"

#include "ccd/procedure/fit.hpp"

namespace ccd
{

//==============================================================================
//
// The change-detection machines' shared parts
//
// A change-detection procedure is a per-pixel state machine: a record of
// everything that survives from one step to the next, a set of transitions
// (each a function from a read-only context and the record to the next
// phase), and one loop that steps it. Each machine owns its phases, its exit
// reasons, its record and its transitions; what they share is composed in,
// never inherited:
//
//   * MaskedSeries (masked_series.hpp): the pixel's processing mask and
//     kept observations;
//   * Cursor (below): where the model window is;
//   * the segments written so far (ChangeModel, fit.hpp);
//   * the kernels of kernels.hpp, Initialise among them;
//   * drive() (below), the loop, with its trace.
//
//==============================================================================

//------------------------------------------------------------------------------
// Cursor
//
// Where a machine is in the pixel's series, in masked coordinates (positions
// in MaskedSeries::dates()):
//
//   window       the model window, the observations a model is fitted to
//   prev_break   the position where the previous segment stopped, the
//                break observation of the last change; 0 before the first
//
// A MaskedSeries::drop() moves `window` with the observations; `prev_break`
// stays where it is.
//------------------------------------------------------------------------------
struct Cursor
{
    Window window;
    index_t prev_break = 0;
};

//------------------------------------------------------------------------------
// drive
//
// Steps `state` through its machine until its phase is Done: the only loop a
// machine has. `Machine` names the machine's types and its step; for the
// trace, the machine's namespace also has to_string(Phase) and
// summary(const State&) -> std::string, found by argument-dependent lookup:
//
//     struct Machine
//     {
//         using Context = ...;   // what the transitions read, never write
//         using State   = ...;   // the per-pixel record; has `Phase phase`
//         using Phase   = ...;   // an enum class with a Done
//         static constexpr const char* name = "...";
//         static Phase step(const Context&, State&);
//     };
//
// The trace is one DEBUG line per transition through the logger, so the log
// level is its only switch, and below DEBUG it compiles to nothing:
//
//     [DEBUG] <name>: <phase> -> <next phase> | <state summary>
//
//------------------------------------------------------------------------------
template<typename Machine>
void drive(
    const typename Machine::Context& context,
    typename Machine::State& state
)
{
    using Phase = typename Machine::Phase;

    while (state.phase != Phase::Done)
    {
        const Phase next = Machine::step(context, state);

        LOG_DEBUG(
            Machine::name << ": "
            << to_string(state.phase) << " -> " << to_string(next)
            << " | " << summary(state)
        );

        state.phase = next;
    }
}

} // namespace ccd
