#include "ccd/config/shared.hpp"

namespace ccd
{

namespace config
{

const std::vector<Field<Shared>>& shared_schema()
{
    static const std::vector<Field<Shared>> s = {
        {"detection_bands", &Shared::detection_bands, Unit::Bands, any(),
         "the bands of the change magnitude, in the order it sums them",
         Source::Own, nullptr, roles({"green", "red", "nir", "swir1", "swir2"})},
        {"change_probability", &Shared::change_probability, Unit::None, open_interval(0, 1),
         "the probability of the change tests (change_test, stability_rule, lookback)"},
        {"outlier_probability", &Shared::outlier_probability, Unit::None, open_interval(0, 1),
         "the probability of the outlier tests (outlier, lookback)"},
        {"alpha", &Shared::alpha, Unit::ReflectanceDelta, at_least(0),
         "the Lasso's L1 penalty", Source::Own, nullptr, real(1e-4)},
        {"coef_floor", &Shared::coef_floor, Unit::ReflectanceDelta, at_least(0),
         "the least coefficient size the Lasso's tolerance scales", Source::Own, nullptr,
         real(1e-4)},
        {"max_iter", &Shared::max_iter, Unit::None, at_least(0),
         "the Lasso's coordinate descent sweeps at most"},
        {"tolerance", &Shared::tolerance, Unit::None, at_least(0),
         "a Lasso sweep whose largest update is at most this times max(largest "
         "coefficient, coef_floor) checks the duality gap"},
        {"coefficient_min", &Shared::coefficient_min, Unit::None, at_least(1),
         "coefficients of a short window's fit, of a start window's and of an edge fit"},
        {"coefficient_mid", &Shared::coefficient_mid, Unit::None, at_least(1),
         "coefficients from coefficient_mid * num_obs_factor observations"},
        {"coefficient_max", &Shared::coefficient_max, Unit::None, at_least(1),
         "coefficients from coefficient_max * num_obs_factor observations; the basis width"},
        {"num_obs_factor", &Shared::num_obs_factor, Unit::None, at_least(1),
         "observations per coefficient"},
        {"min_observations", &Shared::min_observations, Unit::Observations, at_least(1),
         "the fewest observations a curve is fitted on: a start window's size and what a start "
         "needs after it, and a whole-series curve's least"},
        {"stat_ord", &Shared::stat_ord, Unit::None, any(),
         "the statistics cutoff, an ordinal day (9223372036854775807: none)"},
    };
    return s;
}

Values shared_defaults()
{
    Values out = to_values(shared_schema(), Shared{});
    for (const auto& field : shared_schema())
        if (field.default_value)
            out[field.name] = *field.default_value;
    return out;
}

bool operator==(const Shared& a, const Shared& b)
{
    return to_values(shared_schema(), a) == to_values(shared_schema(), b);
}

} // namespace config
} // namespace ccd
