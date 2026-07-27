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
    Eigen::VectorXd residuals;
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


private:

    Eigen::VectorXd weighted_fit(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& weights,
        Eigen::VectorXd& residuals
    );

    scalar_t mad(
        const Eigen::VectorXd& x
    ) const;

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

// score

}