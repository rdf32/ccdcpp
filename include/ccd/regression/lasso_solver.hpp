#pragma once

#include <array>
#include <vector>
#include <algorithm>

#include "ccd/types.hpp"
#include "ccd/constants.hpp"
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

//==============================================================================
// CCD Lasso Workspace
//
// Reused for every fit.
// No coefficient allocation.
//==============================================================================
struct LassoWorkspace
{
    std::array<scalar_t, CCD_MAX_COEFS> weights{}; // fit
    std::vector<scalar_t> y_cent; // fit
    std::vector<scalar_t> y_resi; // fit

    std::vector<scalar_t> predictions; // prediction & score
    std::vector<scalar_t> residuals; // score

    explicit LassoWorkspace(index_t max_samples) // build outside loop
    {
        y_cent.resize(max_samples);
        y_resi.resize(max_samples);
        predictions.resize(max_samples);
        residuals.resize(max_samples);
    }

    void resize(index_t samples) // call inside loop
    {
        y_cent.resize(samples);
        y_resi.resize(samples);
        predictions.resize(samples);
        residuals.resize(samples);
    }

    void reset() // call inside loop
    {
        weights.fill(scalar_t(0));
    }
};

struct LassoProblem
{
    std::vector<scalar_t> X_store;   // original harmonic basis
    std::vector<scalar_t> X_center;  // centered copy for fitting

    std::array<scalar_t, CCD_MAX_COEFS> X_mean{};
    std::array<scalar_t, CCD_MAX_COEFS> column_norm2{};

    index_t n_samples = 0;

    ArrayView<const scalar_t, 2> X() const noexcept
    {
        return ArrayView<const scalar_t, 2>::contiguous(
            X_store.data(),
            {n_samples, CCD_MAX_COEFS}
        );
    }
};

class LassoModel {
public:

    LassoModel() = default;

    LassoModel(
        index_t iter,
        scalar_t bias,
        const std::array<scalar_t, CCD_MAX_COEFS>& weights
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
    void predict(
        ArrayView<const scalar_t, 2> X,
        std::vector<scalar_t>& preds
    ) const;

    index_t iterations() const noexcept
    {
        return iter_;
    }

    scalar_t intercept() const noexcept
    {
        return bias_;
    }

    const std::array<scalar_t, CCD_MAX_COEFS>& coefficients() const noexcept
    {
        return weights_;
    }

private:
    index_t  iter_  = 0;
    scalar_t bias_  = 0.0;

    std::array<scalar_t, CCD_MAX_COEFS> weights_{};
};

//==============================================================================
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
        LassoWorkspace& workspace,
        const LassoProblem& problem,
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

LassoProblem lasso_basis(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
);

struct LassoScore {
    scalar_t rmse;
    scalar_t magn;
    std::vector<scalar_t> residuals;
};

struct LassoResult {
    LassoModel model;
    LassoScore score;
};

LassoScore score(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& predictions,
    std::vector<scalar_t>& residuals,
    index_t num_coefficients,
    bool unbiased_rmse = true
);


} // namespace ccd