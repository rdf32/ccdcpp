cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------
# CMake policies
# --------------------------------------------------

# Use modern BoostConfig package behavior
cmake_policy(SET CMP0167 NEW)

# Use extracted archive timestamps for FetchContent
cmake_policy(SET CMP0135 NEW)


# -----------------------------
# Build options
# -----------------------------

# for profiling
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
# set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
# set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG")
# set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /DEBUG")
# for profiling

option(
    CCD_ENABLE_OPENMP
    "Enable OpenMP support"
    ON
)


option(
    CCD_BUILD_PYTHON
    "Build Python bindings"
    ON
)

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
        $<$<CONFIG:Debug>:/W4>
        $<$<CONFIG:Debug>:/permissive->
    )

else()

    add_compile_options(
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
        $<$<CONFIG:Debug>:-Wpedantic>
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
    src/ccd/harmonic/harmonic.cpp

    src/ccd/procedure/fit.cpp
    # src/ccd/procedure/standard.cpp

    # src/ccd/ccd.cpp
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

# -----------------------------
# Optional OpenMP support
# -----------------------------

if(CCD_ENABLE_OPENMP)

    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)

        message(STATUS "CCD: OpenMP enabled")

        target_link_libraries(
            ccd_core
            PUBLIC
                OpenMP::OpenMP_CXX
        )

    else()

        message(STATUS "CCD: OpenMP requested but not found")

    endif()

else()

    message(STATUS "CCD: OpenMP disabled")

endif()

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

    # -----------------------------
    # Clear test executable
    # -----------------------------

    add_executable(
        test_clear

        tests/ccd/procedure/test_clear.cpp
    )


    target_link_libraries(
        test_clear

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_clear

        PRIVATE
            cxx_std_17
    )

    # -----------------------------
    # Snow test executable
    # -----------------------------

    add_executable(
        test_snow

        tests/ccd/procedure/test_snow.cpp
    )


    target_link_libraries(
        test_snow

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_snow

        PRIVATE
            cxx_std_17
    )


    # add_executable(
    #     test_standard

    #     tests/ccd/procedure/test_standard.cpp
    # )


    # target_link_libraries(
    #     test_standard

    #     PRIVATE
    #         ccd_core
    #         GTest::gtest_main
    # )


    # target_compile_features(
    #     test_standard

    #     PRIVATE
    #         cxx_std_17
    # )

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

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/test_3657_3610_observations.csv
        ${PROJECT_BINARY_DIR}/test_3657_3610_observations.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/clear_reference.csv
        ${PROJECT_BINARY_DIR}/clear_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/clear_mask_reference.csv
        ${PROJECT_BINARY_DIR}/clear_mask_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/snow_reference.csv
        ${PROJECT_BINARY_DIR}/snow_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/snow_mask_reference.csv
        ${PROJECT_BINARY_DIR}/snow_mask_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/standard_reference.csv
        ${PROJECT_BINARY_DIR}/standard_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/standard_mask_reference.csv
        ${PROJECT_BINARY_DIR}/standard_mask_reference.csv
        COPYONLY
    )
    include(GoogleTest)

    gtest_discover_tests(test_lasso)
    gtest_discover_tests(test_robust)
    gtest_discover_tests(test_clear)
    gtest_discover_tests(test_snow)
    # gtest_discover_tests(test_standard)

endif()

# add_executable(
#     test_scratch

#     tests/scratch.cpp
# )


# target_link_libraries(
#     test_scratch

#     PRIVATE
#         ccd_core
# )

# add_executable(
#     test_ccd

#     tests/ccd/procedure/test_ccd.cpp
# )


# target_link_libraries(
#     test_ccd

#     PRIVATE
#         ccd_core
# )


# target_compile_features(
#     test_ccd

#     PRIVATE
#         cxx_std_17
# )

# --------------------------------------------------
# Python bindings
# --------------------------------------------------

if(CCD_BUILD_PYTHON)

    # Use modern Python discovery
    set(PYBIND11_FINDPYTHON ON)

    find_package(pybind11 CONFIG REQUIRED)


    pybind11_add_module(
        ccd_python

        bindings/module.cpp
        bindings/options.cpp
        bindings/regression.cpp
        bindings/fit.cpp
        bindings/pmask.cpp
        bindings/quality.cpp
        bindings/detect.cpp
    )

    # Python import name:
    # import ccdcpp
    # produces:
    # ccdcpp.cp312-win_amd64.pyd

    set_target_properties(
        ccd_python
        PROPERTIES
            OUTPUT_NAME "ccdcpp"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )

    target_link_libraries(
        ccd_python

        PRIVATE
            ccd_core
    )

    target_include_directories(
        ccd_python

        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_compile_features(
        ccd_python

        PRIVATE
            cxx_std_17
    )

    # -----------------------------
    # Install into Python package
    #
    # wheel layout:
    # site-packages/
    #   ccdcpp/
    #       ccdcpp.cp312-win_amd64.pyd
    #
    # -----------------------------
    install(
        TARGETS
            ccd_python
        LIBRARY DESTINATION
            ccdcpp
        RUNTIME DESTINATION
            ccdcpp
    )

endif()


# --------------------------------------------------
# Install (optional future support)
# --------------------------------------------------

option(
    CCD_INSTALL_CPP
    "Install C++ library headers"
    OFF
)


if(CCD_INSTALL_CPP)

    include(GNUInstallDirs)

    install(
        TARGETS ccd_core
        EXPORT ccdTargets
    )

    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

endif()

# cmake -S . -B build -G "Visual Studio 17 2022" `
#     -DCMAKE_PREFIX_PATH=C:\Users\rylie\miniconda3\envs\geo\Library

# cmake --build build --config Debug

# ctest --test-dir build -C Debug --output-on-failure