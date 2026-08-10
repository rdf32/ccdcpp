#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ccd/array_view.hpp"

// -----------------------------------------------------------------------------
// Log levels
// -----------------------------------------------------------------------------

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO  1
#define LOG_LEVEL_ERROR 2

#ifndef MAX_LOG_LEVEL
// #define MAX_LOG_LEVEL LOG_LEVEL_DEBUG
#define MAX_LOG_LEVEL LOG_LEVEL_ERROR
#endif


// -----------------------------------------------------------------------------
// Internal implementation
// -----------------------------------------------------------------------------

#if MAX_LOG_LEVEL <= LOG_LEVEL_DEBUG

#define LOG_DEBUG(...)                                      \
    do {                                                    \
        std::ostringstream _log_stream;                    \
        _log_stream << __VA_ARGS__;                         \
        std::cout << "[DEBUG] " << _log_stream.str()       \
                  << '\n';                                  \
    } while (0)

#else

#define LOG_DEBUG(...) do {} while (0)

#endif


#if MAX_LOG_LEVEL <= LOG_LEVEL_INFO

#define LOG_INFO(...)                                       \
    do {                                                    \
        std::ostringstream _log_stream;                    \
        _log_stream << __VA_ARGS__;                         \
        std::cout << "[INFO] " << _log_stream.str()        \
                  << '\n';                                  \
    } while (0)

#else

#define LOG_INFO(...) do {} while (0)

#endif


#define LOG_ERROR(...)                                      \
    do {                                                    \
        std::ostringstream _log_stream;                    \
        _log_stream << __VA_ARGS__;                         \
        std::cerr << "[ERROR] " << _log_stream.str()       \
                  << '\n';                                  \
    } while (0)



template <typename T>
std::ostream& operator<<(
    std::ostream& os,
    const std::vector<T>& values
)
{
    os << "[";

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            os << ", ";

        os << values[i];
    }

    return os << "]";
}

template <typename T>
std::ostream& operator<<(
    std::ostream& os,
    const ccd::ArrayView<T, 1>& values
)
{
    os << "[";

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            os << ", ";

        os << values(i);
    }

    return os << "]";
}