cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------
# CMake policies
# --------------------------------------------------

# Use modern BoostConfig package behavior
#cmake_policy(SET CMP0167 NEW)

# Use extracted archive timestamps for FetchContent
#cmake_policy(SET CMP0135 NEW)


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
    VERSION 0.4.0
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
    src/ccd/procedure/standard.cpp

    src/ccd/ccd.cpp
    src/ccd/qa_table.cpp
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


    add_executable(
        test_standard

        tests/ccd/procedure/test_standard.cpp
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

        tests/ccd/procedure/test_units.cpp
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

        tests/ccd/procedure/test_filters.cpp
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

    # Entry-point input checks: band count and timestep agreement.
    add_executable(
        test_validation

        tests/ccd/procedure/test_validation.cpp
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

    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/ccd/procedure/test002_c2.csv
        ${PROJECT_BINARY_DIR}/test002_c2.csv
        COPYONLY
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
    gtest_discover_tests(test_standard)
    gtest_discover_tests(test_units)
    gtest_discover_tests(test_filters)
    gtest_discover_tests(test_validation)
    gtest_discover_tests(test_golden)
    gtest_discover_tests(test_bands)
    gtest_discover_tests(test_qa_tables)
    gtest_discover_tests(test_cadence)

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

add_executable(
    test_ccd

    tests/ccd/procedure/test_ccd.cpp
)


target_link_libraries(
    test_ccd

    PRIVATE
        ccd_core
)


target_compile_features(
    test_ccd

    PRIVATE
        cxx_std_17
)

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
        bindings/qa_tables.cpp
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
            NAME python_bindings
            COMMAND
                ${Python_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/python/test_bindings.py
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