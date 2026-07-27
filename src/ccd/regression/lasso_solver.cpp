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

LassoModel LassoSolver::fit(
    ArrayView<const scalar_t, 2> X,
    ArrayView<const scalar_t, 1> y
)
{
    const index_t n_samples  = X.extent(0);
    const index_t n_features = X.extent(1);

    std::vector<scalar_t> weights(n_features, scalar_t{0});

    //----------------------------------------------------------
    // Allocate scratch
    //----------------------------------------------------------
    std::vector<scalar_t> X_mean(n_features, 0.0);
    std::vector<scalar_t> Xc(n_samples * n_features);

    std::vector<scalar_t> yc(n_samples);
    std::vector<scalar_t> residual(n_samples);

    std::vector<scalar_t> column_norm2(n_features);

    //----------------------------------------------------------
    // Center X
    //----------------------------------------------------------

    for(index_t j = 0; j < n_features; ++j) {
        scalar_t mean = 0.0;
        for(index_t i = 0; i < n_samples; ++i) {
            mean += X(i, j);
        }
        mean /= n_samples;
        X_mean[j] = mean;

        for(index_t i = 0; i < n_samples; ++i)
            Xc[i * n_features + j] = X(i, j) - mean;
    }

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
        yc[i] = y(i) - y_mean;
        residual[i] = yc[i];
        y_norm2 += yc[i] * yc[i];
    }

    //----------------------------------------------------------
    // Column norms
    //----------------------------------------------------------
    for(index_t j = 0; j < n_features; ++j) {

        scalar_t s = 0.0;
        for(index_t i = 0; i < n_samples; ++i) {
            scalar_t v = Xc[i * n_features + j];
            s+= v * v;
        }
        column_norm2[j] = s;
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
            scalar_t old_w = weights[j];
            if (old_w != 0.0) {
                for(index_t i = 0; i < n_samples; ++i) {
                    residual[i] += Xc[i * n_features + j] * old_w;
                }
            }
            //--------------------------------------------------
            // rho = X_j^T r
            //--------------------------------------------------
            scalar_t rho=0.0;
            for(index_t i = 0; i < n_samples; ++i) {
                rho += Xc[i * n_features + j] * residual[i];
            }
            //--------------------------------------------------
            // Soft threshold
            //--------------------------------------------------
            scalar_t new_w = soft_threshold(rho, options_.alpha * n_samples) / column_norm2[j];
            weights[j] = new_w;
            //--------------------------------------------------
            // Remove new contribution
            //--------------------------------------------------
            if(new_w != 0.0) {
                for(index_t i = 0; i < n_samples; ++i) {
                    residual[i] -= Xc[i * n_features + j] * new_w;
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
                residual_norm2 += residual[i] * residual[i];
            }
            scalar_t max_abs_XTr = 0.0;
            for(index_t j = 0; j < n_features; ++j) {

                scalar_t v = 0.0;
                for(index_t i = 0; i < n_samples; ++i) {
                    v += Xc[i * n_features + j] * residual[i];
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
                scalar_t theta = residual[i] * dual_scale;

                theta_norm2 += theta * theta;
                y_theta += yc[i] * theta;
            }

            scalar_t l1 = 0.0;
            for(auto wi : weights){
                l1 += std::abs(wi);
            }

            scalar_t primal = residual_norm2 / (2.0 * n_samples) + options_.alpha*l1;

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
        intercept -= X_mean[j] * weights[j];
    }

    return LassoModel(
        iter + 1,
        intercept,
        weights // copy of weights
    );
}

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

std::vector<scalar_t> LassoModel::predict(
    ccd::ArrayView<const scalar_t, 2> X
) const 
{   
    const index_t num_samp = X.extent(0);
    const index_t num_feat = X.extent(1);
    std::vector<scalar_t> preds(num_samp);
    
    assert(weights_.size() == num_feat);

    for (index_t i = 0; i < num_samp; ++i) {

        scalar_t prediction = bias_;
        for (index_t j = 0; j < num_feat; ++j) {
            prediction += X(i, j) * weights_[j];
        }
        preds[i] = prediction;
    }
    return preds;
}

LassoScore score(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& preds,
    index_t num_coefficients,
    bool unbiased_rmse
) {
    assert(y.size() == preds.size());
    std::vector<scalar_t> residuals(preds.size());

    scalar_t rss = 0.0;
    for (index_t i = 0; i < y.size(); ++i) {
        const scalar_t residual = y(i) - preds[i];
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

std::vector<scalar_t> lasso_basis(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
)
{
    const index_t num_observations = dates.size();

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
    //
    constexpr index_t NUM_FEATURES = 7;

    std::vector<scalar_t> matrix(
        num_observations * NUM_FEATURES,
        scalar_t{0}
    );

    const scalar_t omega = 2.0 * PI / AVG_DAYS_YR;

    for(index_t i = 0; i < num_observations; ++i) {

        const scalar_t t =
            static_cast<scalar_t>(
                dates(i)
            );
        const scalar_t wt = omega * t;

        matrix[i * NUM_FEATURES + 0] = t;
        matrix[i * NUM_FEATURES + 1] = std::cos(wt);
        matrix[i * NUM_FEATURES + 2] = std::sin(wt);

        if(num_coefficients >= 6) {
            const scalar_t w2t =
                static_cast<scalar_t>(2.0) * wt;

            matrix[i * NUM_FEATURES + 3] = std::cos(w2t);
            matrix[i * NUM_FEATURES + 4] = std::sin(w2t);
        }

        if(num_coefficients >= 8) {
            const scalar_t w3t =
                static_cast<scalar_t>(3.0) * wt;

            matrix[i * NUM_FEATURES + 5] = std::cos(w3t);
            matrix[i * NUM_FEATURES + 6] = std::sin(w3t);
        }
    }

    return matrix;
}

} // namespace ccd
