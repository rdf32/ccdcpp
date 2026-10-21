#pragma once

#include <cstdint>
#include <vector>

#include "ccd/core/types.hpp"
#include "ccd/core/window.hpp"

//==============================================================================
//
// Where the Standard machine stopped: the part of its state a resume record
// holds (PixelRecord::standard, results/record.hpp). The machine itself is
// internal to the library.
//
//==============================================================================

namespace ccd
{

namespace standard
{

// Why the machine stopped where it did: the early exits, and what ended the
// segment loop (the routes into EndCatch). The resume record holds it; the
// result does not carry it.
enum class Exit : std::uint8_t
{
    None,                 // still running
    TooFewObservations,   // Prepare: too few observations to start (the start
                          // search's too_few(); pyccd: at most meow_size)
    VariogramNaN,         // Prepare: a NaN in the variogram
    InitializeExhausted,  // Initialize: no stable window before the end
    PeekBeyondEnd,        // no full peek after the stable window
    NoRoomToInitialize,   // no room for the start search's first candidate
                          // after the cursor
    SeriesEnd,            // Extend: a curve reached the series end
    NoFit                 // Extend: the series ended before the model could
                          // be fitted on the window, or it could not be
};

const char* to_string(Exit exit);

// Where the machine stopped, for a later resume (record.hpp): the record at
// the first size guard that failed, before anything after it wrote. Every
// segment written by then is final; a longer series makes the same
// decisions up to here, then passes the guard this one failed.
//
//   exit            the guard (Exit::None while running); after an early
//                   exit, and for the whole-series fit, a longer series is
//                   run from the start
//   window,         the cursor at the guard, masked coordinates
//   prev_break
//   fit_stop        Extend's fit window's stop (SeriesEnd only; the fit
//                   window starts at window.start)
//   grow            SeriesEnd after a test whose peek reached the last
//                   observation: that step's grow is pending (false after a
//                   drop that left no full peek). A resume that ignored it
//                   would test that peek again, with the same outcome, and
//                   grow then: one step more, the same result
//   num_coef,       SeriesEnd: the last test's coefficient count and each
//   magnitude       band's magnitude, which write the provisional segment
//                   again when a longer series still ends there
//   final_segments  the segments written before the guard, those of the run
//                   it resumed included
struct Suspended
{
    Exit exit = Exit::None;
    Window window;
    index_t prev_break = 0;
    index_t fit_stop = 0;
    bool grow = false;
    index_t num_coef = 0;
    std::vector<scalar_t> magnitude;
    index_t final_segments = 0;
};

} // namespace standard

} // namespace ccd
