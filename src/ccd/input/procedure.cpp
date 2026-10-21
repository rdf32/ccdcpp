#include "ccd/input/procedure.hpp"

namespace ccd
{

const char* name(PixelProcedure procedure)
{
    switch (procedure)
    {
        case PixelProcedure::InsufficientClear:
            return "insufficient_clear";

        case PixelProcedure::PermanentSnow:
            return "snow";

        case PixelProcedure::Standard:
            break;
    }

    return "standard";
}

} // namespace ccd
