#pragma once

#include <cstdint>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include "ccd/core/types.hpp"

// CMake sets CCD_OBSERVE for every target of the library; a translation unit
// built without it gets the default, the hooks compiled in.
#ifndef CCD_OBSERVE
#define CCD_OBSERVE 1
#endif

//==============================================================================
//
// Diagnostics: what a run does, as it does it
//
// An Observer is told each step of the machine on the pixels a run observes
// (RunSpec::observer, RunSpec::observe): every phase transition with where
// the cursor is, every Extend change test with its outcome and statistics,
// every observation a test drops, every segment written, and how the pixel
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

struct ObserverEvent
{
    enum class Kind : std::uint8_t
    {
        Transition,   // phase -> next; window, prev_break, observations
        Test,         // an Extend change test: day (the peek's first), outcome, values
        Drop,         // an observation dropped: day, outcome (why)
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

    // An event of pixel `pixel` (run_pixel: 0; run_cube: row * W + col).
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

// Prints one line per event, serialised across threads:
//
//     pixel 0: transition Initialize -> Lookback window (4, 16) prev_break 0 n 230
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
#define CCD_OBSERVE_EVENT(observe, ...) do {} while (0)
#endif
