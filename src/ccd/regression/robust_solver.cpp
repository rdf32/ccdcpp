#include <vector>
#include <Eigen/Dense>

#include "ccd/types.hpp"
#include "ccd/array_view.hpp"
#include "ccd/regression/robust_solver.hpp"

namespace ccd
{

RobustSolver::RobustSolver(
    const RobustOptions& options
) noexcept
    : options_(options)
{}

RobustModel RobustSolver::fit(
    Eigen::Ref<const Eigen::MatrixXd> X,
    Eigen::Ref<const Eigen::VectorXd> y
) {
    //--------------------------------------------------------
    // Initial OLS
    //--------------------------------------------------------
    Eigen::VectorXd ones =
        Eigen::VectorXd::Ones(
            y.size()
        );

    Eigen::VectorXd resid;
    Eigen::VectorXd coefficients =
        weighted_fit(
            X,
            y,
            ones,
            resid
        );

    scalar_t scale = 
        mad(resid);

    if(scale < EPS)
        return RobustModel(
            0,
            coefficients
        );

    //--------------------------------------------------------
    // leverage adjustment
    //--------------------------------------------------------
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);

    Eigen::MatrixXd R =
        qr.matrixQR()
        .topLeftCorner(
            X.cols(),
            X.cols()
        )
        .template triangularView<Eigen::Upper>();

    Eigen::MatrixXd Rinv =
        R.inverse();

    Eigen::MatrixXd E =
        X * Rinv;

    Eigen::VectorXd leverage =
        Eigen::VectorXd::Zero(X.rows());

    for(index_t i=0;i<X.rows();i++)
    {
        leverage[i] =
            E.row(i).squaredNorm();

        leverage[i] =
            std::min(0.9999, leverage[i]);
    }

    Eigen::VectorXd adj =
        (1.0 / (1.0 - leverage.array()).sqrt()).matrix();
    //--------------------------------------------------------
    // IRLS
    //--------------------------------------------------------
    Eigen::VectorXd weights =
        Eigen::VectorXd::Ones(
            y.size()
        );
    
    index_t iter = 0;
    for(; iter < options_.max_iter; iter++)
    {
        Eigen::VectorXd old =
            coefficients;

        resid =
            y - X * coefficients;

        resid =
            resid.array() * adj.array();

        scalar_t y_mean = y.mean();
        scalar_t y_std =
            std::sqrt(
                (y.array() - y_mean)
                .square()
                .mean()
            );
        
        scale =
            std::max(
                EPS * y_std,
                mad(resid)
            );
        
        weights =
            bisquare(
                resid / scale,
                options_.tune
            );

        coefficients =
            weighted_fit(
                X,
                y,
                weights,
                resid
            );


        if(converged(old, coefficients))
        {
            break;
        }

    }

    return RobustModel(
            iter + 1,
            coefficients
        );
 
}

Eigen::VectorXd RobustModel::predict(
    Eigen::Ref<const Eigen::MatrixXd> X
) const {
    return X * coefficients_;
}

} // namespace ccd

//--------------------------------------------------------
// final statistics
//--------------------------------------------------------
// residuals_ =
//     y - X*coefficients_;


// rmse_ =
//     std::sqrt(
//         residuals_
//         .array()
//         .square()
//         .mean()
//     );