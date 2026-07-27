#pragma once

#include <vector>
#include <algorithm>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"

namespace ccd
{


struct LassoOptions {
    
    //--------------------------------------------------------------------------
    // Optimization Parameters
    //--------------------------------------------------------------------------
    // Maximum number of coordinate descent iterations.
    index_t max_iter = 1000;

    // L1 regularization strength.
    // Larger values produce more sparse coefficients.
    scalar_t alpha = static_cast<scalar_t>(1.0);

    // Convergence tolerance.
    // Used for coefficient updates and dual gap stopping criteria.
    scalar_t tolerance = static_cast<scalar_t>(1e-4);

    bool fit_intercept = true;

    bool warm_start = false;

    //--------------------------------------------------------------------------
    // Evaluation Parameters
    //--------------------------------------------------------------------------
    // Apply degrees-of-freedom correction when calculating RMSE.
    // Matches the unbiased RMSE calculation:
    //     sqrt(RSS / (n - p))
    // instead of:
    //     sqrt(RSS / n)
    bool unbiased_rmse = true;
};

struct LassoScore {
    scalar_t rmse;
    scalar_t magn;
    std::vector<scalar_t> residuals;
};

class LassoModel {
public:

    LassoModel() = default;

    LassoModel(
        index_t iter,
        scalar_t bias,
        const std::vector<scalar_t>& weights
    )
        : iter_(iter),
        bias_(bias),
        weights_(weights)
    {}
    //--------------------------------------------------------------------------
    // Prediction
    //--------------------------------------------------------------------------
    // Computes:
    //      y_hat = X * weights + bias
    // Stores output into &preds
    std::vector<scalar_t> predict(
        ArrayView<const scalar_t, 2> X
    ) const;

    index_t iterations() const noexcept
    {
        return iter_;
    }

    scalar_t intercept() const noexcept
    {
        return bias_;
    }

    const std::vector<scalar_t>& coefficients() const noexcept
    {
        return weights_;
    }

private:
    index_t  iter_  = 0;
    scalar_t bias_  = 0.0;

    std::vector<scalar_t> weights_;
};

//==============================================================================
//
// LassoSolver
//
// Coordinate descent Lasso regression solver.
//
// Solves:
//
//      minimize:
//          (1 / 2n) ||y - Xw||^2 + lambda ||w||_1
//
// using coordinate descent.
//
// The solver does not own memory.
//
// Workspace owns:
//      - coefficients
//      - predictions
//      - residuals
//
// Solver owns:
//      - references to workspace/options
//
// Typical usage:
//
//      LassoWorkspace workspace(features, samples);
//
//      LassoOptions options;
//
//      LassoSolver solver(workspace, options);
//
//      solver.fit(X, y);
//
//      solver.predict(X);
//
//==============================================================================

class LassoSolver {
public:
    //--------------------------------------------------------------------------
    // Construction
    //--------------------------------------------------------------------------
    // LassoWorkspace& workspace,
    explicit LassoSolver(
        const LassoOptions& options
    ) noexcept;

    //--------------------------------------------------------------------------
    // Model fitting
    //--------------------------------------------------------------------------
    // Fits coefficients:
    //      y = X * weights + bias
    // X:
    //      shape = (samples, features)
    // y:
    //      shape = (samples)
    LassoModel fit(
        ArrayView<const scalar_t, 2> X,
        ArrayView<const scalar_t, 1> y
    );

private:
    static scalar_t soft_threshold(
        scalar_t rho,
        scalar_t lambda
    ) noexcept;

private:
    LassoOptions options_;
};

LassoScore score(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& preds,
    index_t num_coefficients,
    bool unbiased_rmse = true
);

std::vector<scalar_t> lasso_basis(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
);

struct LassoResult {
    LassoModel model;
    LassoScore score;
};

} // namespace ccd