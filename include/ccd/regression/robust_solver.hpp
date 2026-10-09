#pragma once

#include <limits>
#include <vector>
#include <Eigen/Dense>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"

namespace ccd
{

constexpr scalar_t EPS =
    std::numeric_limits<scalar_t>::epsilon();

struct RobustOptions {
    index_t max_iter = 50;
    scalar_t tol = 1e-8;
    scalar_t tune = 4.685;
    scalar_t scale_constant = 0.6745;
};

struct RobustScore {
    scalar_t rmse;
    scalar_t magn;
    std::vector<scalar_t> residuals;
};

class RobustModel {
public:
    RobustModel(
        index_t iter,
        const Eigen::VectorXd& coefficients
    )
        : iter_(iter),
        coefficients_(coefficients)
    {}
 
    Eigen::VectorXd predict(
        Eigen::Ref<const Eigen::MatrixXd> X
    ) const;

    index_t iterations() const noexcept
    {
        return iter_;
    }

    const Eigen::VectorXd& coefficients() const noexcept
    {
        return coefficients_;
    }


private:
    index_t  iter_;
    Eigen::VectorXd coefficients_;
};

//==============================================================================
//
// RobustDesign
//
// Everything a robust fit needs that depends on X alone, computed once for
// every y fitted against the same X -- tmask() fits each of its bands against
// one basis:
//
//  * the SVD of X, which is the decomposition the initial unweighted solve
//    uses (the unit weights leave X unchanged, so the weighted form of that
//    solve is this same decomposition),
//  * the leverage adjustment 1/sqrt(1 - h_i), built on first use because a
//    fit whose initial residual scale is below EPS never reads it,
//
// plus the buffers and the SVD object the IRLS solves reuse.
//
// It refers to X; X must outlive it.
//
//==============================================================================

class RobustDesign
{
public:

    explicit RobustDesign(
        Eigen::Ref<const Eigen::MatrixXd> X
    );

    RobustDesign(const RobustDesign&) = delete;
    RobustDesign& operator=(const RobustDesign&) = delete;

    Eigen::Ref<const Eigen::MatrixXd> X() const noexcept
    {
        return X_;
    }

    Eigen::Index rows() const noexcept
    {
        return X_.rows();
    }

    const Eigen::VectorXd& leverage_adjustment();

private:

    friend class RobustSolver;

    using SVD = Eigen::BDCSVD<
        Eigen::MatrixXd,
        Eigen::ComputeThinU | Eigen::ComputeThinV
    >;

    Eigen::Ref<const Eigen::MatrixXd> X_;

    SVD unweighted_svd_;

    bool have_adjustment_ = false;
    Eigen::VectorXd adjustment_;

    // weighted-solve scratch
    Eigen::VectorXd sqrt_w_;
    Eigen::MatrixXd Xw_;
    Eigen::VectorXd yw_;
    SVD weighted_svd_;
};

class RobustSolver
{
public:

    explicit RobustSolver(
        const RobustOptions& options
    ) noexcept;

    RobustModel fit(
        Eigen::Ref<const Eigen::MatrixXd> X,
        Eigen::Ref<const Eigen::VectorXd> y
    );

    // Same fit, with the X-only work taken from (and cached in) `design`.
    RobustModel fit(
        RobustDesign& design,
        Eigen::Ref<const Eigen::VectorXd> y
    );


private:

    Eigen::VectorXd weighted_fit(
        RobustDesign& design,
        Eigen::Ref<const Eigen::VectorXd> y,
        const Eigen::VectorXd& weights,
        Eigen::VectorXd& residuals
    );

    scalar_t mad(
        const Eigen::VectorXd& x
    ) const;

    // scratch for mad()
    mutable std::vector<scalar_t> mad_values_;

    Eigen::VectorXd bisquare(
        const Eigen::VectorXd& residuals,
        scalar_t c
    ) const;

    bool converged(
        const Eigen::VectorXd& old_coef,
        const Eigen::VectorXd& new_coef
    ) const;


private:
    RobustOptions options_;
};

RobustScore score(
    Eigen::Ref<const Eigen::VectorXd> y,
    Eigen::Ref<const Eigen::VectorXd> preds,
    index_t num_coefficients,
    bool unbiased_rmse = true
);

Eigen::MatrixXd tmask_basis(
    ArrayView<const std::int64_t, 1> dates
);

// score

}