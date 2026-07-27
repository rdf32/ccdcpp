cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------
# CMake policies
# --------------------------------------------------

# Use modern BoostConfig package behavior
cmake_policy(SET CMP0167 NEW)

# Use extracted archive timestamps for FetchContent
cmake_policy(SET CMP0135 NEW)


# --------------------------------------------------
# Project
# --------------------------------------------------

project(
    ccdcpp
    VERSION 0.1.0
    LANGUAGES CXX
)


# --------------------------------------------------
# Build options
# --------------------------------------------------

option(
    CCD_BUILD_TESTS
    "Build CCD unit tests"
    ON
)


# --------------------------------------------------
# C++ configuration
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# --------------------------------------------------
# Compiler warnings
# --------------------------------------------------

if(MSVC)

    add_compile_options(
        /W4
        /permissive-
    )

else()

    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
    )

endif()


# --------------------------------------------------
# Dependencies
# --------------------------------------------------

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)


# --------------------------------------------------
# Core library
# --------------------------------------------------

add_library(
    ccd_core

    src/ccd/regression/lasso_solver.cpp
    src/ccd/regression/robust_solver.cpp

    src/ccd/harmonic/filter.cpp

    # Future additions:
    # src/ccd/harmonic/tmask.cpp
    # src/ccd/harmonic/observation_filter.cpp
)


target_include_directories(
    ccd_core

    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)


target_compile_features(
    ccd_core

    PUBLIC
        cxx_std_17
)


target_link_libraries(
    ccd_core

    PUBLIC
        Eigen3::Eigen
        Boost::boost
)


# --------------------------------------------------
# Tests
# --------------------------------------------------

if(CCD_BUILD_TESTS)

    include(CTest)
    enable_testing()


    include(FetchContent)


    FetchContent_Declare(
        googletest

        URL
        https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip

        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )


    # Required for MSVC
    set(
        gtest_force_shared_crt
        ON
        CACHE BOOL ""
        FORCE
    )


    FetchContent_MakeAvailable(googletest)


    # -----------------------------
    # Lasso test executable
    # -----------------------------

    add_executable(
        test_lasso

        tests/ccd/regression/test_lasso.cpp
    )


    target_link_libraries(
        test_lasso

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_lasso

        PRIVATE
            cxx_std_17
    )

    # -----------------------------
    # Robust test executable
    # -----------------------------

    add_executable(
        test_robust

        tests/ccd/regression/test_robust.cpp
    )


    target_link_libraries(
        test_robust

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_robust

        PRIVATE
            cxx_std_17
    )

    # Copy Python reference data into the build directory so the
    # test can load it using a relative path.
    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/regression/lasso_reference.csv
        ${PROJECT_BINARY_DIR}/lasso_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/regression/robust_reference.csv
        ${PROJECT_BINARY_DIR}/robust_reference.csv
        COPYONLY
    )

    include(GoogleTest)

    gtest_discover_tests(test_lasso)
    gtest_discover_tests(test_robust)

endif()


# --------------------------------------------------
# Install (optional future support)
# --------------------------------------------------

include(GNUInstallDirs)

install(
    TARGETS ccd_core
    EXPORT ccdTargets
)


install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# cmake -S . -B build -G "Visual Studio 17 2022" `
#     -DCMAKE_PREFIX_PATH=C:\Users\rylie\miniconda3\envs\geo\Library

# cmake --build build --config Debug

# ctest --test-dir build -C Debug --output-on-failure