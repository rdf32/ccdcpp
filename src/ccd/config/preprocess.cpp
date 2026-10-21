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

} // namespace config
} // namespace ccd
