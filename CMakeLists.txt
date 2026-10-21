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
    src/ccd/run/run.cpp
    src/ccd/config/params.cpp
    src/ccd/config/registry.cpp
    src/ccd/config/preprocess.cpp
    src/ccd/config/shared.cpp
    src/ccd/strategies/change_test/chi2.cpp
    src/ccd/strategies/capabilities.cpp
    src/ccd/strategies/edge_fit/lasso.cpp
    src/ccd/strategies/lookback/pyccd.cpp
    src/ccd/strategies/mode/pyccd.cpp
    src/ccd/strategies/model/lasso.cpp
    src/ccd/strategies/noise_floor/variogram.cpp
    src/ccd/strategies/outlier/floored.cpp
    src/ccd/strategies/pyccd/peek_statistics.cpp
    src/ccd/strategies/register_core.cpp
    src/ccd/strategies/refit/span_factor.cpp
    src/ccd/strategies/segment_fit/model_fit.cpp
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

        tests/ccd/machine/test_clear.cpp
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

        tests/ccd/machine/test_snow.cpp
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


    add_executable(
        test_standard

        tests/ccd/machine/test_standard.cpp
    )


    target_link_libraries(
        test_standard

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_standard

        PRIVATE
            cxx_std_17
    )

    # The unit-convention guard. Shares test_standard's input file, which the
    # configure_file below already copies into the build directory.
    add_executable(
        test_units

        tests/ccd/machine/test_units.cpp
    )


    target_link_libraries(
        test_units

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_units

        PRIVATE
            cxx_std_17
    )

    # The observation filters on hand-built series: the range bounds and
    # their flags, and the green median filter's scope.
    add_executable(
        test_filters

        tests/ccd/machine/test_filters.cpp
    )


    target_link_libraries(
        test_filters

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_filters

        PRIVATE
            cxx_std_17
    )

    # The preprocessing plan: the "ccdc" plan (plan_of()) against the
    # hand-written chains, custom plans through detect(), plan
    # validation, and a microbenchmark of apply(). See
    # tests/ccd/machine/test_preprocess.cpp.
    add_executable(
        test_preprocess

        tests/ccd/machine/test_preprocess.cpp
    )


    target_link_libraries(
        test_preprocess

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_preprocess

        PRIVATE
            cxx_std_17
    )

    # Entry-point input checks: band count and timestep agreement.
    add_executable(
        test_validation

        tests/ccd/machine/test_validation.cpp
    )


    target_link_libraries(
        test_validation

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_validation

        PRIVATE
            cxx_std_17
    )

    # The per-pixel records below Run::pixel: MaskedSeries, Variogram.
    add_executable(
        test_masked_series

        tests/ccd/machine/test_masked_series.cpp
    )


    target_link_libraries(
        test_masked_series

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_masked_series

        PRIVATE
            cxx_std_17
    )

    # The strategy registry and parameter schemas. See
    # tests/ccd/strategies/test_registry.cpp.
    add_executable(
        test_registry

        tests/ccd/strategies/test_registry.cpp
    )

    target_link_libraries(
        test_registry

        PRIVATE
            ccd_core
            GTest::gtest_main
    )

    target_compile_features(
        test_registry

        PRIVATE
            cxx_std_17
    )

    # The shared kernels against the per-site loops they stand for, bit for
    # bit. See tests/ccd/machine/test_kernels.cpp.
    add_executable(
        test_kernels

        tests/ccd/machine/test_kernels.cpp
    )


    target_link_libraries(
        test_kernels

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_kernels

        PRIVATE
            cxx_std_17
    )

    target_compile_definitions(
        test_kernels

        PRIVATE
            CCD_PYCCD_PIXEL_CSV="${PROJECT_SOURCE_DIR}/tests/ccd/machine/test_3657_3610_observations.csv"
            CCD_C2_PIXEL_CSV="${PROJECT_SOURCE_DIR}/tests/ccd/machine/test002_c2.csv"
    )

    # The Standard machine's routing from hand-built records, and its exit
    # reasons on whole pixels. See tests/ccd/machine/test_machine.cpp.
    add_executable(
        test_machine

        tests/ccd/machine/test_machine.cpp
    )


    target_link_libraries(
        test_machine

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_machine

        PRIVATE
            cxx_std_17
    )

    # The bit-exact gate: detect() and detect_cube() against a committed
    # hexfloat fixture. See tests/ccd/golden/test_golden.cpp.
    add_executable(
        test_golden

        tests/ccd/golden/test_golden.cpp
    )


    target_link_libraries(
        test_golden

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_golden

        PRIVATE
            cxx_std_17
    )

    target_compile_definitions(
        test_golden

        PRIVATE
            CCD_GOLDEN_FIXTURE="${PROJECT_SOURCE_DIR}/tests/ccd/golden/golden_fixture.txt"
    )

    # The Standard procedure's transitions, guards and pyccd quirks that the
    # golden corpus misses, on pixels built for them, against a fixture
    # written by v0.6.1. See tests/ccd/golden/test_transitions.cpp.
    add_executable(
        test_transitions

        tests/ccd/golden/test_transitions.cpp
    )


    target_link_libraries(
        test_transitions

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_transitions

        PRIVATE
            cxx_std_17
    )

    target_compile_definitions(
        test_transitions

        PRIVATE
            CCD_TRANSITIONS_FIXTURE="${PROJECT_SOURCE_DIR}/tests/ccd/golden/transitions_fixture.txt"
    )

    # Resume equals the full run. See tests/ccd/golden/test_resume.cpp.
    add_executable(
        test_resume

        tests/ccd/golden/test_resume.cpp
    )


    target_link_libraries(
        test_resume

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_resume

        PRIVATE
            cxx_std_17
    )

    target_compile_definitions(
        test_resume

        PRIVATE
            CCD_PREFIX_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/ccd/machine"
    )

    # The resume record: what detect() with RunSpec::record writes. See
    # tests/ccd/machine/test_record.cpp.
    add_executable(
        test_record

        tests/ccd/machine/test_record.cpp
    )


    target_include_directories(
        test_record

        PRIVATE
            ${PROJECT_SOURCE_DIR}/tests/ccd/golden
    )


    target_link_libraries(
        test_record

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_record

        PRIVATE
            cxx_std_17
    )

    # Prefix stability: a run on the observations up to a date keeps every
    # final segment of the run on the whole series. See
    # tests/ccd/golden/test_prefix.cpp.
    add_executable(
        test_prefix

        tests/ccd/golden/test_prefix.cpp
    )


    target_link_libraries(
        test_prefix

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_prefix

        PRIVATE
            cxx_std_17
    )

    target_compile_definitions(
        test_prefix

        PRIVATE
            CCD_PREFIX_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/ccd/machine"
    )

    # Band layouts other than Landsat's: a second fixture of synthetic
    # profiles, and the layout equivalences. See tests/ccd/golden/test_bands.cpp.
    add_executable(
        test_bands

        tests/ccd/golden/test_bands.cpp
    )


    target_link_libraries(
        test_bands

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_bands

        PRIVATE
            cxx_std_17
    )

    target_compile_definitions(
        test_bands

        PRIVATE
            CCD_BANDS_FIXTURE="${PROJECT_SOURCE_DIR}/tests/ccd/golden/bands_fixture.txt"
    )

    # The cadence and change-probability options: defaults spelled out equal
    # the defaults, and each option moves the result its way. See
    # tests/ccd/golden/test_cadence.cpp.
    add_executable(
        test_cadence

        tests/ccd/golden/test_cadence.cpp
    )


    target_link_libraries(
        test_cadence

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_cadence

        PRIVATE
            cxx_std_17
    )

    # QA lookup tables: raw QA decoded through a table against the codes
    # passed directly. See tests/ccd/golden/test_qa_tables.cpp.
    add_executable(
        test_qa_tables

        tests/ccd/golden/test_qa_tables.cpp
    )


    target_link_libraries(
        test_qa_tables

        PRIVATE
            ccd_core
            GTest::gtest_main
    )


    target_compile_features(
        test_qa_tables

        PRIVATE
            cxx_std_17
    )

    # The tests that reach the library's internal headers (src/), which the
    # library does not export.
    foreach(internal_test IN ITEMS
        test_kernels
        test_machine
        test_masked_series
        test_qa_tables
        test_standard
    )
        target_include_directories(
            ${internal_test}

            PRIVATE
                ${PROJECT_SOURCE_DIR}/src
        )
    endforeach()

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/test002_c2.csv
        ${PROJECT_BINARY_DIR}/test002_c2.csv
        COPYONLY
    )

    # Copy Python reference data into the build directory, where the
    # tests open it (CCD_TEST_DATA_DIR below).
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
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/test_3657_3610_observations.csv
        ${PROJECT_BINARY_DIR}/test_3657_3610_observations.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/test_atlanta_024013_00030008_481_393.csv
        ${PROJECT_BINARY_DIR}/test_atlanta_024013_00030008_481_393.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/clear_reference.csv
        ${PROJECT_BINARY_DIR}/clear_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/clear_mask_reference.csv
        ${PROJECT_BINARY_DIR}/clear_mask_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/snow_reference.csv
        ${PROJECT_BINARY_DIR}/snow_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/snow_mask_reference.csv
        ${PROJECT_BINARY_DIR}/snow_mask_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/standard_reference.csv
        ${PROJECT_BINARY_DIR}/standard_reference.csv
        COPYONLY
    )

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/machine/standard_mask_reference.csv
        ${PROJECT_BINARY_DIR}/standard_mask_reference.csv
        COPYONLY
    )
    # The tests that read the CSVs copied above open them under the build
    # directory by absolute path, so they pass from any working directory,
    # not only under ctest.
    foreach(
        csv_test

        test_lasso
        test_robust
        test_clear
        test_snow
        test_standard
        test_units
        test_preprocess
        test_golden
    )
        target_compile_definitions(
            ${csv_test}

            PRIVATE
                CCD_TEST_DATA_DIR="${PROJECT_BINARY_DIR}"
        )
    endforeach()

    include(GoogleTest)

    gtest_discover_tests(test_lasso)
    gtest_discover_tests(test_robust)
    gtest_discover_tests(test_clear)
    gtest_discover_tests(test_snow)
    gtest_discover_tests(test_standard)
    gtest_discover_tests(test_units)
    gtest_discover_tests(test_filters)
    gtest_discover_tests(test_validation)
    gtest_discover_tests(test_masked_series)
    gtest_discover_tests(test_kernels)
    gtest_discover_tests(test_registry)
    gtest_discover_tests(test_machine)
    gtest_discover_tests(test_preprocess)
    gtest_discover_tests(test_golden)
    gtest_discover_tests(test_transitions)
    gtest_discover_tests(test_record)

    # The sweeps over cuts (tests/ccd/golden/series_corpus.hpp) run as shards:
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
    gtest_discover_tests(test_bands)
    gtest_discover_tests(test_qa_tables)
    gtest_discover_tests(test_cadence)

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
            ${PROJECT_SOURCE_DIR}/tests/ccd/golden
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
        bindings/detect.cpp
        bindings/resume.cpp
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

    # The Python entry points, against the module just built.
    if(CCD_BUILD_TESTS)

        add_test(
            NAME python_resume
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_resume.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        add_test(
            NAME python_config
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_config.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )
        add_test(
            NAME python_bindings
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_bindings.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        add_test(
            NAME python_stub
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_stub.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        add_test(
            NAME python_qa_tables
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_qa_tables.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        add_test(
            NAME python_profiles
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_profiles.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        add_test(
            NAME python_preprocess
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_preprocess.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        # The HLS and Sentinel-2 presets on real pixels, against pinned
        # outputs (tests/python/fixtures/t16sgc_2020_2023).
        add_test(
            NAME python_real_fixtures
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_real_fixtures.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
        )

        # The Landsat QA tables against the millworks nlcd plugin's table and
        # pyccd's qabitval. Both are found through PYTHONPATH, set for this
        # test alone from CCD_REFERENCE_PYTHONPATH (os.pathsep-separated, e.g.
        # <millworks>/plugins/nlcd/src:<pyccd>). Skipped when neither imports.
        set(
            CCD_REFERENCE_PYTHONPATH "" CACHE STRING
            "PYTHONPATH for python_qa_references: the nlcd plugin's src and pyccd"
        )

        add_test(
            NAME python_qa_references
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_qa_references.py
                $<TARGET_FILE_DIR:ccd_python>
                ${PROJECT_SOURCE_DIR}
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