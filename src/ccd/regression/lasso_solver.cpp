#include "ccd/regression/lasso_solver.hpp"

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

#include "ccd/core/maths.hpp"
#include "ccd/core/constants.hpp"



namespace ccd
{

namespace
{

//------------------------------------------------------------------------------
// sum_i a[i] * b[i], in four interleaved partial sums.
//
// Why four sums and not one. A floating-point sum written as one loop,
//
//     s += a[i] * b[i];
//
// is one chain of additions: each waits for the one before it, because the
// compiler may not reorder them. C++ fixes the order of floating-point
// operations as written, and (x + y) + z rounds differently from
// x + (y + z), so a reordered sum is a different result, not the same one
// faster. The loop then runs at the latency of an add (about 3-4 cycles
// each), however many the CPU could start per cycle.
//
// Four sums over every fourth term are four independent chains the CPU runs
// at once, and the coordinate descent's two dot products (rho = X_j^T r and
// the duality gap's X^T r) are most of the fit's time. It is the same sum in
// another order: exact arithmetic gives the same value, floating point a
// value rounded at other places, accurate to the same order (each order's
// error is bounded by about n * 1.1e-16 times the sum of |a[i] * b[i]|; four
// shorter chains, if anything, less). The order is part of the output:
// changing it, or the number of sums, changes results in their last bits
// and is a results version (record.hpp).
//------------------------------------------------------------------------------
inline scalar_t dot(const scalar_t* a, const scalar_t* b, index_t n)
{
    scalar_t s0 = 0.0;
    scalar_t s1 = 0.0;
    scalar_t s2 = 0.0;
    scalar_t s3 = 0.0;
    index_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
}

} // namespace

void LassoWorkspace::build_basis(
    ArrayView<const std::int64_t, 1> dates,
    index_t num_coefficients
) {
    basis.build(dates, num_coefficients);
    n_samples = basis.samples();
    X_center.resize(CCD_MAX_COEFS * n_samples);

    const auto X = basis.X();
    //----------------------------------------------------------
    // Center X & Column Norms
    //----------------------------------------------------------
    for(index_t j = 0; j < CCD_MAX_COEFS; ++j) {
        scalar_t s = 0.0;
        scalar_t mean = 0.0;
        for(index_t i = 0; i < n_samples; ++i) {
            mean += X(j, i);
        }
        mean /= n_samples;
        X_mean[j] = mean;

        for(index_t i = 0; i < n_samples; ++i) {
            scalar_t val = X(j, i) - mean;
            X_center[j * n_samples + i] = val;
            scalar_t v = val;
            s += v * v;
        }
        column_norm2[j] = s;
    }
}

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
    ArrayView<const scalar_t, 1> y
)
{   
    // workspace -> y_cent, y_resi, weights
    // problem -> X, X_center, X_mean, column_norm2
    auto X = workspace.X();
    const index_t num_feat = X.extent(0);
    const index_t num_samp = X.extent(1);

    // workspace_.resize(n_samples); // resize scratch space maybe do this outside fit 
    workspace.reset(); // reset scratch space (weights filled to 0)

    const auto& X_mean   = workspace.X_mean;
    const auto& X_center = workspace.X_center;
    const auto& c_norm2  = workspace.column_norm2;
    //----------------------------------------------------------
    // Center y
    //----------------------------------------------------------
    scalar_t y_mean = 0.0;
    for(index_t i = 0; i < num_samp; ++i) {
        y_mean += y(i);
    }

    y_mean /= num_samp;
    scalar_t y_norm2 = 0.0;
    for(index_t i = 0; i < num_samp; ++i) {
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
        for(index_t j = 0; j < num_feat; ++j) {
            if(c_norm2[j] == 0.0) {continue;}
            //--------------------------------------------------
            // Add old contribution back into residual
            //--------------------------------------------------
            scalar_t old_w = workspace.weights[j];
            if (old_w != 0.0) {
                for(index_t i = 0; i < num_samp; ++i) {
                    workspace.y_resi[i] += X_center[j * num_samp + i] * old_w;
                }
            }
            //--------------------------------------------------
            // rho = X_j^T r
            //--------------------------------------------------
            const scalar_t rho = dot(X_center.data() + j * num_samp, workspace.y_resi.data(), num_samp);
            //--------------------------------------------------
            // Soft threshold
            //--------------------------------------------------
            scalar_t new_w = soft_threshold(rho, options_.alpha * num_samp) / c_norm2[j];
            workspace.weights[j] = new_w;
            //--------------------------------------------------
            // Remove new contribution
            //--------------------------------------------------
            if(new_w != 0.0) {
                for(index_t i = 0; i < num_samp; ++i) {
                    workspace.y_resi[i] -= X_center[j * num_samp + i] * new_w;
                }
            }
            max_update = std::max(max_update, std::abs(new_w - old_w));
            max_coef = std::max(max_coef, std::abs(new_w));
        }

        //------------------------------------------------------
        // First stopping criterion
        //------------------------------------------------------
        // `coef_floor` defaults to 1.0, the value that used to be a literal
        // here. It is an absolute magnitude, so it is a statement about the
        // unit of y -- see LassoOptions::coef_floor.
        if(max_update <= options_.tolerance * std::max(max_coef, options_.coef_floor)) {
            //--------------------------------------------------
            // Compute dual gap
            //--------------------------------------------------
            scalar_t residual_norm2 = 0.0;
            for(index_t i = 0; i < num_samp; ++i) {
                residual_norm2 += workspace.y_resi[i] * workspace.y_resi[i];
            }
            scalar_t max_abs_XTr = 0.0;
            for(index_t j = 0; j < num_feat; ++j) {

                const scalar_t v = dot(X_center.data() + j * num_samp, workspace.y_resi.data(), num_samp);
                max_abs_XTr  = std::max(max_abs_XTr,std::abs(v));
            }

            scalar_t dual_scale=1.0;
            if(max_abs_XTr > options_.alpha * num_samp) {
                dual_scale = (options_.alpha * num_samp) / max_abs_XTr;
            }

            scalar_t theta_norm2 = 0.0;
            scalar_t y_theta = 0.0;
            for(index_t i = 0; i < num_samp; ++i) {
                scalar_t theta = workspace.y_resi[i] * dual_scale;

                theta_norm2 += theta * theta;
                y_theta += workspace.y_cent[i] * theta;
            }

            scalar_t l1 = 0.0;
            for(auto wi : workspace.weights){
                l1 += std::abs(wi);
            }

            scalar_t primal = residual_norm2 / (2.0 * num_samp) + options_.alpha * l1;

            scalar_t dual = y_theta / num_samp - theta_norm2 / (2.0 * num_samp);

            scalar_t gap = primal - dual;

            if(gap <= options_.tolerance * y_norm2 / num_samp) {   
                break;
            }
        }
    }
    //----------------------------------------------------------
    // Recover intercept
    //----------------------------------------------------------
    scalar_t intercept = y_mean;
    for(index_t j = 0; j < num_feat; ++j){
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
    const index_t num_feat = X.extent(0);
    const index_t num_samp = X.extent(1);
    
    assert(weights_.size() == num_feat);

    for (index_t i = 0; i < num_samp; ++i) {

        scalar_t prediction = bias_;
        for (index_t j = 0; j < num_feat; ++j) {
            prediction += X(j, i) * weights_[j];
        }
        predictions[i] = prediction;
    }
    return;
}

scalar_t rmse(
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
    return std::sqrt(rss / denominator);
}

LassoScore score(
    ArrayView<const scalar_t, 1> y,
    const std::vector<scalar_t>& predictions,
    std::vector<scalar_t>& residuals,
    index_t num_coefficients,
    bool unbiased_rmse
) {
    const scalar_t r = rmse(
        y,
        predictions,
        residuals,
        num_coefficients,
        unbiased_rmse
    );

    return LassoScore{
        r,                            // rmse
        median_absolute(residuals),   // magnitude
        residuals                     // residuals
    };
}

} // namespace ccd
