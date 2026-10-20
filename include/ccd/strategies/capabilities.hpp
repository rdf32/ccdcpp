#pragma once

#include <cstdint>
#include <string>

namespace ccd
{

//==============================================================================
//
// Capabilities
//
// What one stage's strategy gives the stages after it at an Extend peek, and
// what a strategy needs from the ones before it. The model provides its
// fits' residuals and RMSE; the change test provides the peek's comparison
// RMSE (PeekResult). Registry::build() refuses a configuration in which a
// strategy needs what no earlier stage's strategy provides, naming both. The
// per-observation statistics a change test provides and an outlier rule
// reads go by name (ChangeTest::statistics(), OutlierRule::statistics()).
//
//==============================================================================

enum class Capability : std::uint8_t
{
    PeekResiduals = 1 << 0,       // Model::peek_residuals()
    ModelRmse = 1 << 1,           // Model::rmse()
    InWindowResiduals = 1 << 2,   // Model::residuals()
    ComparisonRmse = 1 << 3,      // PeekResult::comparison_rmse
    PredictionCovariance = 1 << 4 // the model is a least-squares fit whose
                                  // factor gives its predictions' covariance
                                  // (a test reading it downcasts to it)
};

// A set of capabilities.
class Capabilities
{
public:

    constexpr Capabilities() = default;
    constexpr Capabilities(Capability c) : bits_(static_cast<std::uint8_t>(c)) {}

    constexpr Capabilities operator|(Capabilities other) const
    {
        return Capabilities(static_cast<std::uint8_t>(bits_ | other.bits_));
    }

    // The capabilities of this set that `other` lacks.
    constexpr Capabilities missing_from(Capabilities other) const
    {
        return Capabilities(static_cast<std::uint8_t>(bits_ & ~other.bits_));
    }

    constexpr bool empty() const { return bits_ == 0; }

    // "peek_residuals, model_rmse", in declaration order.
    std::string names() const;

private:

    constexpr explicit Capabilities(std::uint8_t bits) : bits_(bits) {}

    std::uint8_t bits_ = 0;
};

constexpr Capabilities operator|(Capability a, Capability b)
{
    return Capabilities(a) | Capabilities(b);
}

} // namespace ccd
