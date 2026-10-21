#include "ccd/config/preprocess.hpp"

#include "ccd/input/procedure.hpp"

namespace ccd
{
namespace config
{

const std::vector<std::string>& procedures()
{
    static const std::vector<std::string> names(
        [] {
            std::vector<std::string> out;
            for (PixelProcedure procedure : PROCEDURES)
                out.push_back(name(procedure));
            return out;
        }());
    return names;
}

const char* name(Thermal thermal)
{
    return thermal == Thermal::Celsius ? "celsius" : "stored";
}

std::optional<Thermal> thermal_named(const std::string& name)
{
    for (const Thermal thermal : {Thermal::Stored, Thermal::Celsius})
        if (name == config::name(thermal))
            return thermal;
    return std::nullopt;
}

} // namespace config
} // namespace ccd
