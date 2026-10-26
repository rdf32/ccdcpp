#include "ccd/core/build_info.hpp"

// CCD_VERSION, CCD_COMMIT, CCD_COMPILER and CCD_BUILD_TYPE are this file's
// compile definitions (CMakeLists.txt); CCD_OBSERVE is the library's.

namespace ccd
{

BuildInfo build_info()
{
    BuildInfo info;
    info.version = CCD_VERSION;
    info.commit = CCD_COMMIT;
#ifdef _OPENMP
    info.openmp = true;
    info.openmp_version = _OPENMP;
#endif
    info.compiler = CCD_COMPILER;
    info.cxx_standard = __cplusplus;
    info.observe = CCD_OBSERVE;
    info.build_type = CCD_BUILD_TYPE;
    return info;
}

} // namespace ccd
