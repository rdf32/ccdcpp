#pragma once

#include <memory>
#include <vector>

#include "ccd/config/params.hpp"
#include "ccd/regression/lasso_solver.hpp"
#include "ccd/strategies/stages.hpp"

namespace ccd
{
namespace strategies
{

//------------------------------------------------------------------------------
// model "lasso": pyccd's monitoring model. Each band is a Lasso regression on
// the harmonic basis of the window's dates (regression/harmonic_basis.hpp),
// solved by coordinate descent (regression/lasso_solver.hpp) with these
// parameters; its RMSE has the model's coefficient count as its degrees of
// freedom.
//------------------------------------------------------------------------------
struct Lasso final : MonitoringModel
{
    struct Params
    {
        index_t max_iter = 1000;
        scalar_t alpha = 1.0;
        scalar_t tolerance = 1e-4;
        scalar_t coef_floor = 1.0;
    };

    static const std::vector<config::Field<Params>>& schema();

    explicit Lasso(const Params& p) : p_(p) {}

    Capabilities provides() const override;
    std::unique_ptr<Model> make_model() const override;

    // The solver's options these parameters are.
    LassoOptions solver_options() const;

private:
    Params p_;
};

} // namespace strategies
} // namespace ccd
