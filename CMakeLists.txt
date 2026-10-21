cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------
# CMake policies
# --------------------------------------------------

# Each is set only where this CMake knows the policy: both are newer than
# the minimum above, and a CMake without them has their old behavior only.

# find_package(Boost) reads Boost's own BoostConfig.cmake (Boost 1.70 and
# later ship it) instead of CMake's FindBoost module, removed in 3.30.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# A FetchContent URL download without DOWNLOAD_EXTRACT_TIMESTAMP is stamped
# with the time of extraction. googletest below sets that option itself
# (the archive's timestamps), so this only governs a later download.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()


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
    VERSION 0.10.0
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

# Strategy extensions built into the library: directories, relative to the
# source directory, each with an extension.cmake that appends to
#   CCD_EXTENSION_SOURCES        its library sources
#   CCD_EXTENSION_INCLUDE_DIRS   its include directories
#   CCD_EXTENSION_REGISTRATIONS  its registration functions' names, each a
#                                void ccd::strategies::NAME(config::Registry&)
#                                the library calls (library_registry())
set(CCD_EXTENSIONS "" CACHE STRING "Strategy extension directories to build into the library")

option(
    CCD_OBSERVE
    "Compile in the diagnostics hooks (results/diagnostics.hpp); OFF removes them"
    ON
)

option(
    CCD_BUILD_GOLDEN_DRIVER
    "Build golden_driver, the golden check's dump driver (tools/linux-compare)"
    OFF
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
# Strategy extensions
# --------------------------------------------------

set(CCD_EXTENSION_SOURCES "")
set(CCD_EXTENSION_INCLUDE_DIRS "")
set(CCD_EXTENSION_REGISTRATIONS "")
set(extensions ${CCD_EXTENSIONS})
list(REMOVE_DUPLICATES extensions)
foreach(extension IN LISTS extensions)
    set(CCD_EXTENSION_DIR ${PROJECT_SOURCE_DIR}/${extension})
    if(NOT EXISTS ${CCD_EXTENSION_DIR}/extension.cmake)
        message(FATAL_ERROR "CCD_EXTENSIONS: ${extension} has no extension.cmake")
    endif()
    include(${CCD_EXTENSION_DIR}/extension.cmake)
endforeach()
list(REMOVE_DUPLICATES CCD_EXTENSION_REGISTRATIONS)
set(CCD_EXTENSION_DECLARATIONS "")
set(CCD_EXTENSION_CALLS "")
foreach(registration IN LISTS CCD_EXTENSION_REGISTRATIONS)
    if(NOT registration MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        message(FATAL_ERROR "CCD_EXTENSIONS: ${registration} is not a function name")
    endif()
    string(APPEND CCD_EXTENSION_DECLARATIONS "void ${registration}(config::Registry& registry);\n")
    string(APPEND CCD_EXTENSION_CALLS "    ${registration}(registry);\n")
endforeach()
configure_file(
    ${PROJECT_SOURCE_DIR}/src/ccd/strategies/register_extensions.cpp.in
    ${PROJECT_BINARY_DIR}/generated/register_extensions.cpp
)


# --------------------------------------------------
# Core library
# --------------------------------------------------

add_library(
    ccd_core

    src/ccd/core/maths.cpp
    src/ccd/core/window.cpp

    src/ccd/regression/lasso_solver.cpp
    src/ccd/regression/robust_solver.cpp
    src/ccd/regression/harmonic_basis.cpp

    src/ccd/input/mask/filter.cpp
    src/ccd/input/procedure.cpp
    src/ccd/input/input.cpp
    src/ccd/input/qa_statistics.cpp

    src/ccd/input/mask/chain.cpp
    src/ccd/input/mask/steps.cpp

    src/ccd/input/masked_series.cpp
    src/ccd/machine/standard.cpp

    src/ccd/run/algorithm.cpp
    src/ccd/run/cube.cpp
    src/ccd/run/resume.cpp
    src/ccd/run/run.cpp
    src/ccd/config/convert.cpp
    src/ccd/config/digest.cpp
    src/ccd/config/params.cpp
    src/ccd/config/presets.cpp
    src/ccd/config/registry.cpp
    src/ccd/config/resolve.cpp
    src/ccd/config/preprocess.cpp
    src/ccd/config/shared.cpp
    src/ccd/strategies/change_test/chi2.cpp
    src/ccd/strategies/capabilities.cpp
    src/ccd/strategies/edge_segments/lasso.cpp
    src/ccd/strategies/lookback/pyccd.cpp
    src/ccd/strategies/mode/pyccd.cpp
    src/ccd/strategies/model/lasso.cpp
    src/ccd/strategies/noise/variogram.cpp
    src/ccd/strategies/outlier/floored.cpp
    src/ccd/strategies/pyccd/peek_statistics.cpp
    src/ccd/strategies/library.cpp
    src/ccd/strategies/refit/span_factor.cpp
    src/ccd/strategies/segment_report/model_fit.cpp
    src/ccd/strategies/stability_rule/pyccd.cpp
    src/ccd/strategies/start_search/pyccd.cpp
    src/ccd/strategies/window_screen/tmask.cpp
    src/ccd/input/qa_table.cpp
    src/ccd/input/qa_tables.cpp
    src/ccd/results/diagnostics.cpp
    src/ccd/results/record.cpp
    ${PROJECT_BINARY_DIR}/generated/register_extensions.cpp
    ${CCD_EXTENSION_SOURCES}
)


target_compile_definitions(
    ccd_core

    PUBLIC
        CCD_OBSERVE=$<BOOL:${CCD_OBSERVE}>
)

target_include_directories(
    ccd_core

    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CCD_EXTENSION_INCLUDE_DIRS}
)


target_compile_features(
    ccd_core

    PUBLIC
        cxx_std_17
)


# The static core is linked into the Python extension, a shared object, so it is built
# position-independent; Linux linkers refuse non-PIC objects in a shared library.
set_target_properties(
    ccd_core

    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
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


    include(GoogleTest)

    # A test executable: `name` built from SOURCES, linked with the library and
    # gtest_main, with the shared test headers (tests/ccd/support) and DEFINES
    # as its compile definitions; INTERNAL gives it
    # the library's internal headers (src/), which the library does not
    # export. Its tests are discovered unless NO_DISCOVER, for the tests that
    # discover theirs in shards (below). Every test opens its CSVs in
    # tests/ccd/data by absolute path (CCD_TEST_DATA_DIR), so it passes from
    # any working directory, not only under ctest.
    function(ccd_add_gtest name)
        cmake_parse_arguments(PARSE_ARGV 1 test "INTERNAL;NO_DISCOVER" "" "SOURCES;DEFINES")
        add_executable(${name} ${test_SOURCES})
        target_link_libraries(${name} PRIVATE ccd_core GTest::gtest_main)
        target_compile_features(${name} PRIVATE cxx_std_17)
        target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/ccd/support)
        target_compile_definitions(
            ${name} PRIVATE CCD_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/ccd/data" ${test_DEFINES})
        if(test_INTERNAL)
            target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
        endif()
        if(NOT test_NO_DISCOVER)
            gtest_discover_tests(${name})
        endif()
    endfunction()

    # The regression solvers against scikit-learn.
    ccd_add_gtest(test_lasso SOURCES tests/ccd/regression/test_lasso.cpp)
    ccd_add_gtest(test_robust SOURCES tests/ccd/regression/test_robust.cpp)

    # The three procedures on the pyccd reference pixel, and the Atlanta
    # pixel, against pyccd. See tests/ccd/machine/test_pyccd_reference.cpp.
    ccd_add_gtest(test_pyccd_reference INTERNAL SOURCES tests/ccd/machine/test_pyccd_reference.cpp)

    # The unit-convention guard, on the pyccd reference pixel.
    ccd_add_gtest(test_units SOURCES tests/ccd/machine/test_units.cpp)

    # The observation filters on hand-built series: the range bounds and
    # their flags, and the green median filter's scope.
    ccd_add_gtest(test_filters SOURCES tests/ccd/machine/test_filters.cpp)

    # Entry-point input checks: band count and timestep agreement.
    ccd_add_gtest(test_validation SOURCES tests/ccd/machine/test_validation.cpp)

    # The per-pixel records below Run::pixel: MaskedSeries, BandNoise.
    ccd_add_gtest(test_masked_series INTERNAL SOURCES tests/ccd/machine/test_masked_series.cpp)

    # The shared kernels against the per-site loops they stand for, bit for
    # bit. See tests/ccd/machine/test_kernels.cpp.
    ccd_add_gtest(test_kernels INTERNAL SOURCES tests/ccd/machine/test_kernels.cpp)

    # The strategy registry and parameter schemas, each strategy against what
    # it stands for, and the configuration's preprocessing section. See
    # tests/ccd/strategies/test_registry.cpp, test_strategies.cpp and
    # test_registry_preprocess.cpp.
    ccd_add_gtest(
        test_registry
        SOURCES
            tests/ccd/strategies/test_registry.cpp
            tests/ccd/strategies/test_strategies.cpp
            tests/ccd/strategies/test_registry_preprocess.cpp
    )

    # The Standard machine: its routing and steps from hand-built records,
    # its routes under test-only strategies, and its exit reasons on whole
    # pixels. See tests/ccd/machine/test_standard_*.cpp.
    ccd_add_gtest(
        test_machine
        INTERNAL
        SOURCES
            tests/ccd/machine/test_standard_steps.cpp
            tests/ccd/machine/test_standard_hooks.cpp
            tests/ccd/machine/test_standard_whole_pixel.cpp
    )

    # The preprocessing plan: the "ccdc" plan (plan_of()) against the
    # hand-written chains, custom plans through detect(), and plan
    # validation. See tests/ccd/machine/test_preprocess.cpp.
    ccd_add_gtest(test_preprocess SOURCES tests/ccd/machine/test_preprocess.cpp)

    # apply() timed against the hand-written chains
    # (tools/bench/bench_preprocess.cpp): built with the tests, so it keeps
    # building, and not a test.
    if(UNIX)
        add_executable(bench_preprocess tools/bench/bench_preprocess.cpp)
        target_link_libraries(bench_preprocess PRIVATE ccd_core)
        target_compile_features(bench_preprocess PRIVATE cxx_std_17)
        target_include_directories(bench_preprocess PRIVATE ${PROJECT_SOURCE_DIR}/tests/ccd/support)
        target_compile_definitions(
            bench_preprocess PRIVATE CCD_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/ccd/data")
    endif()

    # The bit-exact gate: detect() and detect_cube() against a committed
    # hexfloat fixture. See tests/ccd/golden/test_golden.cpp.
    ccd_add_gtest(
        test_golden
        SOURCES tests/ccd/golden/test_golden.cpp
        DEFINES CCD_GOLDEN_FIXTURE="${PROJECT_SOURCE_DIR}/tests/ccd/golden/golden_fixture.txt"
    )

    # The Standard procedure's transitions, guards and pyccd quirks that the
    # golden corpus misses, on pixels built for them, against a fixture
    # written by v0.6.1. See tests/ccd/golden/test_transitions.cpp.
    ccd_add_gtest(
        test_transitions
        SOURCES tests/ccd/golden/test_transitions.cpp
        DEFINES
            CCD_TRANSITIONS_FIXTURE="${PROJECT_SOURCE_DIR}/tests/ccd/golden/transitions_fixture.txt"
    )

    # The resume record: what detect() with RunSpec::record writes. See
    # tests/ccd/machine/test_record.cpp.
    ccd_add_gtest(test_record SOURCES tests/ccd/machine/test_record.cpp)

    # Resume equals the full run. See tests/ccd/golden/test_resume.cpp.
    ccd_add_gtest(test_resume NO_DISCOVER SOURCES tests/ccd/golden/test_resume.cpp)

    # Prefix stability: a run on the observations up to a date keeps every
    # final segment of the run on the whole series. See
    # tests/ccd/golden/test_prefix.cpp.
    ccd_add_gtest(test_prefix NO_DISCOVER SOURCES tests/ccd/golden/test_prefix.cpp)

    # The sweeps over cuts (tests/ccd/support/series_corpus.hpp) run as shards:
    # "sample" in every run, "restN" labelled corpus, so `ctest -LE corpus` is
    # the quick tier. CMake before 3.22 cannot filter discovery: no label.
    if(CMAKE_VERSION VERSION_LESS 3.22)
        gtest_discover_tests(test_prefix)
        gtest_discover_tests(test_resume PROPERTIES TIMEOUT 600)
    else()
        gtest_discover_tests(test_prefix TEST_FILTER "-*/rest*" TEST_LIST test_prefix_quick)
        gtest_discover_tests(
            test_prefix
            TEST_FILTER "*/rest*"
            TEST_LIST test_prefix_corpus
            PROPERTIES LABELS corpus
        )
        gtest_discover_tests(
            test_resume
            TEST_FILTER "-*/rest*"
            TEST_LIST test_resume_quick
            PROPERTIES TIMEOUT 600
        )
        gtest_discover_tests(
            test_resume
            TEST_FILTER "*/rest*"
            TEST_LIST test_resume_corpus
            PROPERTIES TIMEOUT 600 LABELS corpus
        )
    endif()

    # Band layouts other than Landsat's: a second fixture of synthetic
    # profiles, and the layout equivalences. See tests/ccd/golden/test_bands.cpp.
    ccd_add_gtest(
        test_bands
        SOURCES tests/ccd/golden/test_bands.cpp
        DEFINES CCD_BANDS_FIXTURE="${PROJECT_SOURCE_DIR}/tests/ccd/golden/bands_fixture.txt"
    )

    # The cadence and change-probability options: defaults spelled out equal
    # the defaults, and each option moves the result its way. See
    # tests/ccd/golden/test_cadence.cpp.
    ccd_add_gtest(test_cadence SOURCES tests/ccd/golden/test_cadence.cpp)

    # QA lookup tables: raw QA decoded through a table against the codes
    # passed directly. See tests/ccd/golden/test_qa_tables.cpp.
    ccd_add_gtest(test_qa_tables INTERNAL SOURCES tests/ccd/golden/test_qa_tables.cpp)

endif()

# The golden check's driver: dumps every output of detect() and detect_cube()
# on the synthetic corpus and the CSV pixels (tools/linux-compare/README.md).
# Its one source is tools/linux-compare/golden_driver.cpp, which
# linux-compare also compiles against older trees by hand.
if(CCD_BUILD_GOLDEN_DRIVER)

    add_executable(
        golden_driver

        tools/linux-compare/golden_driver.cpp
    )

    target_include_directories(
        golden_driver

        PRIVATE
            ${PROJECT_SOURCE_DIR}/tests/ccd/support
    )

    target_link_libraries(
        golden_driver

        PRIVATE
            ccd_core
    )

    target_compile_features(
        golden_driver

        PRIVATE
            cxx_std_17
    )

endif()

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
        bindings/config.cpp
        bindings/input.cpp
        bindings/qa_tables.cpp
        bindings/preprocess.cpp
        bindings/arrays.cpp
        bindings/detect.cpp
        bindings/results.cpp
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

    # ccdcpp.__version__. pyproject.toml carries the same number; the
    # python_bindings test checks that they agree.
    target_compile_definitions(
        ccd_python

        PRIVATE
            CCD_VERSION="${PROJECT_VERSION}"
    )

    target_compile_features(
        ccd_python

        PRIVATE
            cxx_std_17
    )

    # The Python entry points, against the module just built: tests/python/
    # test_<name>.py as python_<name>. real_fixtures runs the HLS and
    # Sentinel-2 presets on real pixels against pinned outputs
    # (tests/python/fixtures/t16sgc_2020_2023).
    if(CCD_BUILD_TESTS)

        foreach(
            python_test

            resume
            config
            bindings
            stub
            package
            qa_tables
            profiles_landsat
            profiles_api
            profiles_qa_options
            profiles_band_order
            profiles_sensors
            docs
            preprocess
            real_fixtures
            qa_references
        )
            add_test(
                NAME python_${python_test}
                COMMAND
                    ${Python_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/python/test_${python_test}.py
                    $<TARGET_FILE_DIR:ccd_python>
                    ${PROJECT_SOURCE_DIR}
            )
        endforeach()

        # The Landsat QA tables against the millworks nlcd plugin's table and
        # pyccd's qabitval. Both are found through PYTHONPATH, set for this
        # test alone from CCD_REFERENCE_PYTHONPATH (os.pathsep-separated, e.g.
        # <millworks>/plugins/nlcd/src:<pyccd>). Skipped when neither imports.
        set(
            CCD_REFERENCE_PYTHONPATH "" CACHE STRING
            "PYTHONPATH for python_qa_references: the nlcd plugin's src and pyccd"
        )

        set_tests_properties(
            python_qa_references
            PROPERTIES
                SKIP_RETURN_CODE 77
                ENVIRONMENT "PYTHONPATH=${CCD_REFERENCE_PYTHONPATH}"
        )

    endif()

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
# pip install . ^
#   --config-settings=cmake.args="-G;Visual Studio 17 2022;-A;x64;-DCMAKE_PREFIX_PATH=C:\Users\rylie\miniconda3\envs\geo\Library"
# pip install . --config-settings=cmake.args="-G Visual Studio 17 2022 -A x64" --config-settings=cmake.build-type=Release