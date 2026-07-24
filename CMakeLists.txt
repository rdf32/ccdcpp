cmake_minimum_required(VERSION 3.20)

project(
    ccdcpp
    VERSION 0.1.0
    LANGUAGES CXX
)

# -----------------------------
# C++ standard
# -----------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Third part dependencies
# -----------------------------

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)