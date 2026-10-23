#include "ccd/config/config.hpp"

#include <type_traits>

#include "../core/fnv.hpp"
#include "digest.hpp"

namespace ccd
{
namespace config
{

//==============================================================================
//
// The digests of configurations and resolutions
//
// Each value is hashed by name, with its type (Value's index) and its bits
// (core/fnv.hpp), so equal configurations hash equal on any platform and an
// integer differs from the equal real.
//
//==============================================================================

namespace
{

void hash(detail::Fnv& h, const Values& values)
{
    h.integer(static_cast<std::int64_t>(values.size()));
    for (const auto& [name, value] : values)
    {
        h.text(name);
        h.byte(static_cast<std::uint8_t>(value.index()));
        std::visit(
            [&](const auto& v)
            {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bool>)
                    h.flag(v);
                else if constexpr (std::is_same_v<T, std::int64_t>)
                    h.integer(v);
                else if constexpr (std::is_same_v<T, double>)
                    h.real(v);
                else if constexpr (std::is_same_v<T, std::string>)
                    h.text(v);
                else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
                {
                    h.integer(static_cast<std::int64_t>(v.size()));
                    for (const std::int64_t b : v)
                        h.integer(b);
                }
                else if constexpr (std::is_same_v<T, std::vector<std::string>>)
                {
                    h.integer(static_cast<std::int64_t>(v.size()));
                    for (const std::string& r : v)
                        h.text(r);
                }
                else if constexpr (std::is_same_v<T, std::vector<double>>)
                {
                    h.integer(static_cast<std::int64_t>(v.size()));
                    for (const double r : v)
                        h.real(r);
                }
                else
                {
                    h.integer(static_cast<std::int64_t>(v.size()));
                    for (const bool f : v)
                        h.flag(f);
                }
            },
            value);
    }
}

void hash(detail::Fnv& h, const ConfigTree& stages)
{
    h.integer(static_cast<std::int64_t>(stages.size()));
    for (const auto& [stage, chosen] : stages)
    {
        h.text(stage);
        h.text(chosen.strategy);
        hash(h, chosen.params);
    }
}

void hash(detail::Fnv& h, const PreprocessConfig& preprocess)
{
    h.integer(static_cast<std::int64_t>(preprocess.size()));
    for (const auto& [procedure, chain] : preprocess)
    {
        h.text(procedure);
        h.text(name(chain.thermal));
        h.integer(static_cast<std::int64_t>(chain.steps.size()));
        for (const StepConfig& step : chain.steps)
        {
            h.text(step.step);
            hash(h, step.params);
        }
    }
}

} // namespace

std::uint64_t digest(const Config& config)
{
    detail::Fnv h;
    hash(h, config.shared);
    hash(h, config.stages);
    hash(h, config.preprocess);
    return h.value();
}

std::uint64_t resolution_digest(const DataSpec& spec, const Values& shared,
                                const ConfigTree& stages, const PreprocessConfig& preprocess)
{
    detail::Fnv h;
    h.integer(static_cast<std::int64_t>(digest(spec)));
    hash(h, shared);
    hash(h, stages);
    hash(h, preprocess);
    return h.value();
}

} // namespace config
} // namespace ccd
