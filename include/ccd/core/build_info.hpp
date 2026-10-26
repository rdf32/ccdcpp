#pragma once

#include <string>

namespace ccd
{

// What this build of the library is: what to report next to a benchmark or a
// bug, and how to tell a build without OpenMP (detect_cube() on one thread)
// from one with it. Set when the library is configured and compiled.
struct BuildInfo
{
    std::string version;        // the project version, "1.1.0"
    std::string commit;         // the source tree's commit at configure time:
                                // the 40-character hash, "-dirty" appended when
                                // a tracked file differs from it; "unknown"
                                // outside git
    std::string describe;       // `git describe --always --dirty --tags` then,
                                // "v1.1.0-3-gabc1234"; "unknown" outside git
    bool openmp = false;        // compiled with OpenMP
    long openmp_version = 0;    // _OPENMP (yyyymm of the OpenMP spec), 0 without
    std::string compiler;       // the compiler's id and version, "AppleClang 17.0.0"
    long cxx_standard = 0;      // __cplusplus, 202002 for C++20
    bool observe = false;       // CCD_OBSERVE: the diagnostics hooks compiled in
    std::string build_type;     // the CMake build type ("Release"), empty for none
};

// This build's BuildInfo.
BuildInfo build_info();

} // namespace ccd
