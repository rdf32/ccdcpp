#pragma once

#include <vector>

#include <Eigen/Dense>

#include "ccd/types.hpp"

namespace ccd
{

class RobustLinearModel
{
public:

    explicit RobustLinearModel(
        index_t max_iter = 50,
        scalar_t tol = 1e-8
    );

    void fit(
        Eigen::Ref<const Eigen::MatrixXd> X,
        Eigen::Ref<const Eigen::VectorXd> y
    );

    Eigen::VectorXd predict(
        Eigen::Ref<const Eigen::MatrixXd> X
    ) const;


    const Eigen::VectorXd& coefficients() const
    {
        return coefficients_;
    }

    const Eigen::VectorXd& weights() const
    {
        return weights_;
    }

    const Eigen::VectorXd& residuals() const
    {
        return residuals_;
    }

    scalar_t rmse() const
    {
        return rmse_;
    }


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

    index_t max_iter_;
    scalar_t tol_;

    scalar_t tune_ = 4.685;
    scalar_t scale_constant_ = 0.6745;

    Eigen::VectorXd coefficients_;
    Eigen::VectorXd weights_;
    Eigen::VectorXd residuals_;

    scalar_t scale_ = 0.0;
    scalar_t rmse_ = 0.0;
};

}