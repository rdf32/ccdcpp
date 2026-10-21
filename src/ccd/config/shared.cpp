#include "ccd/config/shared.hpp"

namespace ccd
{

namespace config
{

namespace
{

constexpr const char* COEFFICIENTS = "coefficients";
constexpr const char* LASSO = "lasso";

// `values` with "<table>." before each name.
Values prefixed(const std::string& table, const Values& values)
{
    Values out;
    for (const auto& [name, value] : values)
        out[table + "." + name] = value;
    return out;
}

// The values of `values` in `table`, by their names there; or (an empty
// table) the ones in none.
Values in_table(const std::string& table, const Values& values)
{
    Values out;
    for (const auto& [name, value] : values)
    {
        const auto dot = name.find('.');
        if (table.empty() ? dot == std::string::npos
                          : dot == table.size() && name.compare(0, dot, table) == 0)
            out[table.empty() ? name : name.substr(dot + 1)] = value;
    }
    return out;
}

void add(std::vector<ParameterInfo>& out, const std::string& table,
         std::vector<ParameterInfo> info)
{
    for (ParameterInfo& p : info)
    {
        if (!table.empty())
            p.name = table + "." + p.name;
        out.push_back(std::move(p));
    }
}

} // namespace

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
        {"min_observations", &Shared::min_observations, Unit::Observations, at_least(1),
         "the fewest observations a curve is fitted on: a start window's size and what a start "
         "needs after it, and a whole-series curve's least"},
        {"statistics_end", &Shared::statistics_end, Unit::None, any(),
         "the statistics cutoff, an ordinal day (9223372036854775807: none)"},
    };
    return s;
}

const std::vector<Field<CoefficientRule>>& coefficients_schema()
{
    static const std::vector<Field<CoefficientRule>> s = {
        {"min", &CoefficientRule::min, Unit::None, at_least(1),
         "coefficients below mid * per observations; a start window's and an edge segment's"},
        {"mid", &CoefficientRule::mid, Unit::None, at_least(1),
         "coefficients below max * per observations"},
        {"max", &CoefficientRule::max, Unit::None, at_least(1),
         "coefficients otherwise; the basis width"},
        {"per", &CoefficientRule::per, Unit::None, at_least(1), "observations per coefficient"},
    };
    return s;
}

const std::vector<Field<LassoOptions>>& lasso_schema()
{
    static const std::vector<Field<LassoOptions>> s = {
        {"alpha", &LassoOptions::alpha, Unit::ReflectanceDelta, at_least(0),
         "the L1 penalty", Source::Own, nullptr, real(1e-4)},
        {"coef_floor", &LassoOptions::coef_floor, Unit::ReflectanceDelta, at_least(0),
         "the least coefficient size the tolerance scales", Source::Own, nullptr, real(1e-4)},
        {"max_iter", &LassoOptions::max_iter, Unit::None, at_least(0),
         "coordinate descent sweeps at most"},
        {"tolerance", &LassoOptions::tolerance, Unit::None, at_least(0),
         "a sweep whose largest update is at most this times max(largest coefficient, "
         "coef_floor) checks the duality gap"},
    };
    return s;
}

const std::vector<SharedTable>& shared_tables()
{
    static const std::vector<SharedTable> tables = {
        {COEFFICIENTS, "how many harmonic coefficients a curve gets"},
        {LASSO, "the Lasso every fitting stage uses"},
    };
    return tables;
}

const std::vector<ParameterInfo>& shared_parameters()
{
    static const std::vector<ParameterInfo> info = []
    {
        std::vector<ParameterInfo> out;
        add(out, "", info_of(shared_schema()));
        add(out, COEFFICIENTS, info_of(coefficients_schema()));
        add(out, LASSO, info_of(lasso_schema()));
        return out;
    }();
    return info;
}

Values shared_defaults()
{
    Values out;
    for (const ParameterInfo& p : shared_parameters())
        out[p.name] = *p.default_value;
    return out;
}

Values shared_values(const Shared& shared)
{
    Values out = to_values(shared_schema(), shared);
    for (auto& [name, value] :
         prefixed(COEFFICIENTS, to_values(coefficients_schema(), shared.coefficients)))
        out[name] = std::move(value);
    for (auto& [name, value] : prefixed(LASSO, to_values(lasso_schema(), shared.lasso)))
        out[name] = std::move(value);
    return out;
}

void apply_shared(const Values& values, Shared& shared)
{
    apply(shared_schema(), in_table("", values), shared, "shared");
    apply(coefficients_schema(), in_table(COEFFICIENTS, values), shared.coefficients,
          std::string("shared ") + COEFFICIENTS);
    apply(lasso_schema(), in_table(LASSO, values), shared.lasso, std::string("shared ") + LASSO);
}

bool operator==(const Shared& a, const Shared& b)
{
    return shared_values(a) == shared_values(b);
}

} // namespace config
} // namespace ccd
