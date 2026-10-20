#include "ccd/strategies/capabilities.hpp"

namespace ccd
{

std::string Capabilities::names() const
{
    // By bit, in Capability's order.
    static const char* const name[] = {"peek_residuals", "model_rmse", "in_window_residuals",
                                       "comparison_rmse", "peek_statistic",
                                       "prediction_covariance"};
    static_assert(1u << (sizeof(name) / sizeof(name[0]) - 1) ==
                      static_cast<unsigned>(Capability::PredictionCovariance),
                  "a name for every capability");
    std::string out;
    for (unsigned i = 0; i < sizeof(name) / sizeof(name[0]); ++i)
        if (bits_ & (1u << i))
            out += std::string(out.empty() ? "" : ", ") + name[i];
    return out;
}

} // namespace ccd
