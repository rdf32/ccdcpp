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
            std::max(EPS * y_std, mad(resid));
        
        weights =
            bisquare(resid / scale, options_.tune);

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

//------------------------------------------------------------
// Weighted least squares
//------------------------------------------------------------
Eigen::VectorXd RobustSolver::weighted_fit(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& w,
    Eigen::VectorXd& residuals
)
{
    Eigen::VectorXd sqrt_w =
        w.array().sqrt();

    Eigen::MatrixXd Xw =
        X.array().colwise()
        * sqrt_w.array();

    Eigen::VectorXd yw =
        y.array() *
        sqrt_w.array();

    // Eigen::VectorXd beta =
    //     Xw.colPivHouseholderQr()
    //     .solve(yw);

    Eigen::BDCSVD<Eigen::MatrixXd> svd(
        Xw,
        Eigen::ComputeThinU |
        Eigen::ComputeThinV
    );

    Eigen::VectorXd beta =
        svd.solve(yw);

    residuals =
        y - X * beta;

    return beta;
}

//------------------------------------------------------------
// Median absolute deviation
//------------------------------------------------------------
scalar_t RobustSolver::mad(
    const Eigen::VectorXd& x
) const
{
    std::vector<scalar_t> values;
    values.reserve(x.size());

    for(auto v : x)
        values.push_back(std::abs(v));

    std::sort(
        values.begin(),
        values.end()
    );

    // Python version uses median(rs[4:])
    // because this is Landsat time-series specific
    // If you want exact numpy median:
    // remove this offset.
    index_t start = 
        std::min<index_t>(4, values.size());

    std::vector<scalar_t> trimmed(
        values.begin() + start,
        values.end()
    );

    scalar_t med;
    index_t n = trimmed.size();

    if(n % 2)
    {
        med = trimmed[n/2];
    }
    else
    {
        med =
            0.5 * (trimmed[n/2-1] + trimmed[n/2]);
    }

    return med / options_.scale_constant;
}

//------------------------------------------------------------
// Tukey bisquare weights
//------------------------------------------------------------
Eigen::VectorXd RobustSolver::bisquare(
    const Eigen::VectorXd& r,
    scalar_t c
) const
{
    Eigen::VectorXd w(r.size());

    for(index_t i = 0; i < r.size(); i++)
    {
        scalar_t x = std::abs(r[i]);

        if(x < c)
        {
            scalar_t t = r[i] / c;
            w[i] = std::pow(1.0 - t*t, 2.0);
        }
        else
        {
            w[i] = 0.0;
        }
    }
    return w;
}

bool RobustSolver::converged(
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b
) const
{
    return ((a - b).array().abs().maxCoeff() < options_.tol);
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