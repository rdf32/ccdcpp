#include "ccd/regression/lasso_solver.hpp"

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

#include "ccd/maths.hpp"
#include "ccd/constants.hpp"



namespace ccd
{

// per band solver
// constructor
LassoSolver::LassoSolver(
    const LassoOptions& options
) noexcept
    :
    options_(options)
{}

scalar_t LassoSolver::soft_threshold(
    scalar_t rho, 
    scalar_t lambda
) noexcept 
{
    if (rho < -lambda) {
        return rho + lambda;
    }
    if (rho > lambda) {
        return rho - lambda;
    }
    return 0.0;
}

LassoModel LassoSolver::fit(
    LassoWorkspace& workspace,
    const LassoProblem& problem,
    ArrayView<const scalar_t, 1> y
)
{   
    // workspace -> y_cent, y_resi, weights
    // problem -> X, X_store, X_center, X_mean, column_norm2
    auto X = problem.X();
    const index_t n_samples  = X.extent(0);
    const index_t n_features = X.extent(1);

    // workspace_.resize(n_samples); // resize scratch space maybe do this outside fit 
    // workspace_.reset(); // reset scratch space (weights filled to 0) maybe do this outside fit

    const auto& X_mean = problem.X_mean;
    const auto& X_center = problem.X_center;
    const auto& column_norm2 = problem.column_norm2;
    //----------------------------------------------------------
    // Center y
    //----------------------------------------------------------
    scalar_t y_mean = 0.0;
    for(index_t i = 0; i < n_samples; ++i) {
        y_mean += y(i);
    }

    y_mean /= n_samples;
    scalar_t y_norm2 = 0.0;
    for(index_t i = 0; i < n_samples; ++i) {
        workspace.y_cent[i] = y(i) - y_mean;
        workspace.y_resi[i] = workspace.y_cent[i];
        y_norm2 += workspace.y_cent[i] * workspace.y_cent[i];
    }

    //----------------------------------------------------------
    // Coordinate Descent
    //----------------------------------------------------------
    index_t iter;
    for(iter = 0; iter < options_.max_iter; ++iter) {

        scalar_t max_update = 0.0;
        scalar_t max_coef = 0.0;
        for(index_t j = 0; j < n_features; ++j) {
            if(column_norm2[j] == 0.0) {continue;}
            //--------------------------------------------------
            // Add old contribution back into residual
            //--------------------------------------------------
            scalar_t old_w = workspace.weights[j];
            if (old_w != 0.0) {
                for(index_t i = 0; i < n_samples; ++i) {
                    workspace.y_resi[i] += X_center[i * n_features + j] * old_w;
                }
            }
            //--------------------------------------------------
            // rho = X_j^T r
            //--------------------------------------------------
            scalar_t rho=0.0;
            for(index_t i = 0; i < n_samples; ++i) {
                rho += X_center[i * n_features + j] * workspace.y_resi[i];
            }
            //--------------------------------------------------
            // Soft threshold
            //--------------------------------------------------
            scalar_t new_w = soft_threshold(rho, options_.alpha * n_samples) / column_norm2[j];
            workspace.weights[j] = new_w;
            //--------------------------------------------------
            // Remove new contribution
            //--------------------------------------------------
            if(new_w != 0.0) {
                for(index_t i = 0; i < n_samples; ++i) {
                    workspace.y_resi[i] -= X_center[i * n_features + j] * new_w;
                }
            }
            max_update = std::max(max_update, std::abs(new_w - old_w));
            max_coef = std::max(max_coef, std::abs(new_w));
        }

        //------------------------------------------------------
        // First stopping criterion
        //------------------------------------------------------
        if(max_update <= options_.tolerance * std::max(max_coef, scalar_t(1.0))) {
            //--------------------------------------------------
            // Compute dual gap
            //--------------------------------------------------
            scalar_t residual_norm2 = 0.0;
            for(index_t i = 0; i < n_samples; ++i) {
                residual_norm2 += workspace.y_resi[i] * workspace.y_resi[i];
            }
            scalar_t max_abs_XTr = 0.0;
            for(index_t j = 0; j < n_features; ++j) {

                scalar_t v = 0.0;
                for(index_t i = 0; i < n_samples; ++i) {
                    v += X_center[i * n_features + j] * workspace.y_resi[i];
                }
                max_abs_XTr  = std::max(max_abs_XTr,std::abs(v));
            }

            scalar_t dual_scale=1.0;
            if(max_abs_XTr > options_.alpha * n_samples) {
                dual_scale = (options_.alpha * n_samples) / max_abs_XTr;
            }

            scalar_t theta_norm2 = 0.0;
            scalar_t y_theta = 0.0;
            for(index_t i = 0; i < n_samples; ++i) {
                scalar_t theta = workspace.y_resi[i] * dual_scale;

                theta_norm2 += theta * theta;
                y_theta += workspace.y_cent[i] * theta;
            }

            scalar_t l1 = 0.0;
            for(auto wi : workspace.weights){
                l1 += std::abs(wi);
            }

            scalar_t primal = residual_norm2 / (2.0 * n_samples) + options_.alpha * l1;

            scalar_t dual = y_theta / n_samples - theta_norm2 / (2.0 * n_samples);

            scalar_t gap = primal - dual;

            if(gap <= options_.tolerance * y_norm2 / n_samples) {   
                break;
            }
        }
    }
    //----------------------------------------------------------
    // Recover intercept
    //----------------------------------------------------------
    scalar_t intercept = y_mean;
    for(index_t j = 0; j < n_features; ++j){
        intercept -= X_mean[j] * workspace.weights[j];
    }

    return LassoModel(
        iter + 1,
        intercept,
        workspace.weights // copy of weights
    );
}

void LassoModel::predict(
    ccd::ArrayView<const scalar_t, 2> X,
    std::vector<scalar_t>& predictions
) const 
{   
    const index_t num_samp = X.extent(0);
    const index_t num_feat = X.extent(1);
    
    assert(weights_.size() == num_feat);

    for (index_t i = 0; i < num_samp; ++i) {

        scalar_t prediction = bias_;
        for (index_t j = 0; j < num_feat; ++j) {
            prediction += X(i, j) * weights_[j];
        }
        predictions[i] = prediction;
    }
    return;
}

LassoScore score(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& predictions,
    std::vector<scalar_t>& residuals,
    index_t num_coefficients,
    bool unbiased_rmse
) {
    assert(y.size() == predictions.size());

    scalar_t rss = 0.0;
    for (index_t i = 0; i < y.size(); ++i) {
        const scalar_t residual = y(i) - predictions[i];
        residuals[i] = residual;
        rss += residual * residual;
    }

    scalar_t denominator = static_cast<scalar_t>(y.size());
    if (unbiased_rmse) {
        // Degrees of freedom correction (matches Python calc_rmse)
        assert(y.size() > num_coefficients);
        denominator -= static_cast<scalar_t>(num_coefficients);
    }
    return LassoScore{
        std::sqrt(rss / denominator), // rmse
        median_absolute(residuals),   // magnitude
        residuals                     // residuals
    };
}

LassoProblem lasso_basis(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
) {
    const index_t n_samples = dates.size();
    // Current CCD implementation:
    //
    // column layout:
    //
    // 0: time
    // 1: cos(annual)
    // 2: sin(annual)
    // 3: cos(2 annual)
    // 4: sin(2 annual)
    // 5: cos(3 annual)
    // 6: sin(3 annual)
    LassoProblem problem;
    problem.n_samples = n_samples;
    problem.X_store.resize(n_samples * CCD_MAX_COEFS);
    problem.X_center.resize(n_samples * CCD_MAX_COEFS);

    const scalar_t omega = 2.0 * PI / AVG_DAYS_YR;

    for(index_t i = 0; i < n_samples; ++i) {

        const scalar_t t =
            static_cast<scalar_t>(
                dates(i)
            );
        const scalar_t wt = omega * t;

        problem.X_store[i * CCD_MAX_COEFS + 0] = t;
        problem.X_store[i * CCD_MAX_COEFS + 1] = std::cos(wt);
        problem.X_store[i * CCD_MAX_COEFS + 2] = std::sin(wt);

        // initialize 0 //
        problem.X_store[i * CCD_MAX_COEFS + 3] = scalar_t{0};
        problem.X_store[i * CCD_MAX_COEFS + 4] = scalar_t{0};
        problem.X_store[i * CCD_MAX_COEFS + 5] = scalar_t{0};
        problem.X_store[i * CCD_MAX_COEFS + 6] = scalar_t{0};

        if(num_coefficients >= 6) {
            const scalar_t w2t =
                static_cast<scalar_t>(2.0) * wt;

            problem.X_store[i * CCD_MAX_COEFS + 3] = std::cos(w2t);
            problem.X_store[i * CCD_MAX_COEFS + 4] = std::sin(w2t);
        }

        if(num_coefficients >= 8) {
            const scalar_t w3t =
                static_cast<scalar_t>(3.0) * wt;

            problem.X_store[i * CCD_MAX_COEFS + 5] = std::cos(w3t);
            problem.X_store[i * CCD_MAX_COEFS + 6] = std::sin(w3t);
        }
    }
    //----------------------------------------------------------
    // Center X & Column Norms
    //----------------------------------------------------------
    for(index_t j = 0; j < CCD_MAX_COEFS; ++j) {
        scalar_t s = 0.0;
        scalar_t mean = 0.0;
        for(index_t i = 0; i < n_samples; ++i) {
            mean += problem.X_store[i * CCD_MAX_COEFS + j];
        }
        mean /= n_samples;
        problem.X_mean[j] = mean;

        for(index_t i = 0; i < n_samples; ++i) {
            scalar_t val = problem.X_store[i * CCD_MAX_COEFS + j] - mean;
            problem.X_center[i * CCD_MAX_COEFS + j] = val;
            scalar_t v = val;
            s += v * v;
        }
        problem.column_norm2[j] = s;
    }
    return problem;
};

} // namespace ccd
