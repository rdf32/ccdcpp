# A strategy extension: built into the library by
#   cmake -DCCD_EXTENSIONS=<this directory, relative to the source directory>
# CMake sets CCD_EXTENSION_DIR to this directory while it reads this file.
list(APPEND CCD_EXTENSION_INCLUDE_DIRS ${CCD_EXTENSION_DIR})
list(APPEND CCD_EXTENSION_SOURCES ${CCD_EXTENSION_DIR}/outlier/residual_cap.cpp)
list(APPEND CCD_EXTENSION_REGISTRATIONS register_outlier_residual_cap)
