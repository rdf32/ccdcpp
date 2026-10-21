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

std::optional<PixelProcedure> procedure_named(const std::string& name)
{
    for (const PixelProcedure procedure : PROCEDURES)
        if (name == ccd::name(procedure))
            return procedure;
    return std::nullopt;
}

} // namespace ccd
