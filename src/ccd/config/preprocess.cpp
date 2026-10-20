#include "ccd/config/preprocess.hpp"

namespace ccd
{
namespace config
{

const std::vector<std::string>& procedures()
{
    static const std::vector<std::string> names = {"standard", "snow", "insufficient_clear"};
    return names;
}

} // namespace config
} // namespace ccd
