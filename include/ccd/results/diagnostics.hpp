#pragma once

#include <cstdint>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/input/procedure.hpp"
#include "ccd/machine/suspended.hpp"

// CMake sets CCD_OBSERVE for every target of the library; a translation unit
// built without it gets the default, the hooks compiled in.
#ifndef CCD_OBSERVE
#define CCD_OBSERVE 1
#endif

//==============================================================================
//
// Diagnostics: what a run did, and what it does as it does it
//
// With RunSpec::diagnostics a pixel's result holds its PixelDiagnostics: its
// procedure, why its machine stopped, its observation counts and its peak,
// and per segment the machine's work on it (SegmentDiagnostics). The
// counters run on every pixel, a few additions per step; they are copied
// into the result only when asked for.
//
// An Observer is told each step of the machine on the pixels a run observes
// (RunSpec::observer, RunSpec::observe): every phase transition with where
// the cursor is, every Extend change test with its outcome and statistics,
// every observation a stage drops, every segment written, and how the pixel
// ends. TextObserver prints them; the Python binding records them as rows
// (detect(trace=True)).
//
// The hooks are compiled in with CCD_OBSERVE (CMake's option, ON by default):
// a pixel without an observer then costs one pointer test per hook. Built
// with CCD_OBSERVE=OFF they are gone and an observer is never called.
//
//==============================================================================

namespace ccd
{

// The machine's work on one segment, from the segment before it to its own
// writing: its start search's Initialize steps, Lookback's steps, Extend's
// change tests and refits, and the observations dropped on the way (the
// window screen's, Lookback's and Extend's, and those the change before it
// dropped after writing the segment it ended). The Start segment, written
// between the first stable window and its Extend, holds zeros (that work is
// the first curve's), as do the segments of the whole-series procedures and
// a resume's carried ones.
struct SegmentDiagnostics
{
    index_t initialize_steps = 0;
    index_t lookback_steps = 0;
    index_t tests = 0;
    index_t refits = 0;
    index_t dropped = 0;
};

// One pixel's run:
//
//   procedure   the mode's choice
//   exit        why the machine stopped (machine::Exit); None for the
//               whole-series procedures, which take no machine
//   clear       the observations the preprocessing kept
//   kept        the observations the processing mask keeps after the fit
//   peek        the change test's peek for the pixel (PeekPlan::peek); 0
//               without a machine or before Prepare chose it
//   segments    one per segment of the result, in order
struct PixelDiagnostics
{
    PixelProcedure procedure = PixelProcedure::Standard;
    machine::Exit exit = machine::Exit::None;
    index_t clear = 0;
    index_t kept = 0;
    index_t peek = 0;
    std::vector<SegmentDiagnostics> segments;
};

struct ObserverEvent
{
    enum class Kind : std::uint8_t
    {
        Transition,   // phase -> next; outcome (what an Initialize, Lookback
                      // or Extend step decided), window, prev_break,
                      // observations
        Test,         // an Extend change test: day (the peek's first), outcome, values
                      // (the change test's first statistic, chi2's change_magnitude)
        Drop,         // an observation dropped: day, outcome (its reason code's
                      // name, input/processing_mask.hpp)
        Segment,      // a segment written: day and day_to (its first and last
                      // day), outcome (its curve_qa, break day and change
                      // probability), observations (its count)
        Exit          // the pixel's end: outcome (the exit)
    };

    Kind kind;
    const char* machine = "";
    const char* phase = "";
    const char* next = "";
    index_t window_start = 0;
    index_t window_stop = 0;
    index_t prev_break = 0;
    index_t observations = 0;
    std::int64_t day = 0;
    std::int64_t day_to = 0;
    std::string outcome;
    std::vector<scalar_t> values;
};

const char* to_string(ObserverEvent::Kind kind);

class Observer
{
public:
    virtual ~Observer() = default;

    // An event of pixel `pixel` (detect: 0; detect_cube: row * W + col).
    // Called from the thread running that pixel: an observer of a cube run
    // is called from several threads at once.
    virtual void event(index_t pixel, const ObserverEvent& e) = 0;
};

// Where a machine reports: the observer and the pixel, or no observer.
struct Observe
{
    Observer* observer = nullptr;
    index_t pixel = 0;
};

// Prints one line per event, serialised across threads, for debugging in C++
// (it compiles away with CCD_OBSERVE=OFF):
//
//     pixel 0: transition Initialize -> Lookback (stable) window (4, 16) prev_break 0 n 230
//     pixel 0: test 731203 no change [3.1, 2.7, 4.0, 1.2, 0.9, 2.2]
class TextObserver final : public Observer
{
public:
    explicit TextObserver(std::ostream& out) : out_(out) {}
    void event(index_t pixel, const ObserverEvent& e) override;

private:
    std::ostream& out_;
    std::mutex mutex_;
};

} // namespace ccd

#if CCD_OBSERVE
// Reports `event` (an ObserverEvent built only when there is an observer).
#define CCD_OBSERVE_EVENT(observe, ...)                                   \
    do {                                                                  \
        if ((observe).observer)                                           \
            (observe).observer->event((observe).pixel, __VA_ARGS__);      \
    } while (0)
#else
// The event is not built; `observe` is read only so that a caller's
// parameter is not unused.
#define CCD_OBSERVE_EVENT(observe, ...) do { (void)(observe); } while (0)
#endif
