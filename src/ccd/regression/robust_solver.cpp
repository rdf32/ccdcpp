#include "ccd/regression/robust_solver.hpp"

#include <algorithm>
#include <vector>
#include <Eigen/Dense>

#include "ccd/maths.hpp"
#include "ccd/constants.hpp"


namespace ccd
{

RobustSolver::RobustSolver(
    const RobustOptions& options
) noexcept
    : options_(options)
{}

RobustDesign::RobustDesign(
    Eigen::Ref<const Eigen::MatrixXd> X
)
    : X_(X),
      unweighted_svd_(X_)
{}

//------------------------------------------------------------
// Leverage adjustment, 1 / sqrt(1 - h_i)
//------------------------------------------------------------
const Eigen::VectorXd& RobustDesign::leverage_adjustment()
{
    if (have_adjustment_)
        return adjustment_;

    const auto& X = X_;

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

    for(index_t i = 0; i < static_cast<index_t>(X.rows()); ++i)
    {
        leverage[i] =
            E.row(i).squaredNorm();

        leverage[i] =
            std::min(0.9999, leverage[i]);
    }

    adjustment_ =
        (1.0 / (1.0 - leverage.array()).sqrt()).matrix();

    have_adjustment_ = true;
    return adjustment_;
}

RobustModel RobustSolver::fit(
    Eigen::Ref<const Eigen::MatrixXd> X,
    Eigen::Ref<const Eigen::VectorXd> y
) {
    RobustDesign design(X);
    return fit(design, y);
}

RobustModel RobustSolver::fit(
    RobustDesign& design,
    Eigen::Ref<const Eigen::VectorXd> y
) {
    const auto X = design.X();

    //--------------------------------------------------------
    // Initial OLS
    //
    // Unit weights: sqrt(1) * X is X and sqrt(1) * y is y, so
    // the weighted solve is the shared SVD of X applied to y.
    //--------------------------------------------------------
    Eigen::VectorXd coefficients =
        design.unweighted_svd_.solve(y);

    Eigen::VectorXd resid =
        y - X * coefficients;

    scalar_t scale = 
        mad(resid);

    if(scale < EPS)
        return RobustModel(
            1,
            coefficients
        );

    //--------------------------------------------------------
    // leverage adjustment
    //--------------------------------------------------------
    const Eigen::VectorXd& adj =
        design.leverage_adjustment();
    //--------------------------------------------------------
    // IRLS
    //--------------------------------------------------------
    Eigen::VectorXd weights =
        Eigen::VectorXd::Ones(
            y.size()
        );

    // y is fixed across iterations, so are its mean and spread.
    const scalar_t y_mean = y.mean();
    const scalar_t y_std =
        std::sqrt(
            (y.array() - y_mean)
            .square()
            .mean()
        );
    
    index_t iter = 1;
    for(; iter < options_.max_iter; iter++)
    {   

        Eigen::VectorXd old =
            coefficients;

        resid =
            y - X * coefficients;

        resid =
            resid.array() * adj.array();
        
        scale =
            std::max(EPS * y_std, mad(resid));
        
        weights =
            bisquare(resid / scale, options_.tune);

        coefficients =
            weighted_fit(
                design,
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
    RobustDesign& design,
    Eigen::Ref<const Eigen::VectorXd> y,
    const Eigen::VectorXd& w,
    Eigen::VectorXd& residuals
)
{
    const auto X = design.X();

    auto& sqrt_w = design.sqrt_w_;
    auto& Xw = design.Xw_;
    auto& yw = design.yw_;
    auto& svd = design.weighted_svd_;

    sqrt_w =
        w.array().sqrt();

    Xw =
        sqrt_w.asDiagonal() * X;

    yw =
        y.array() *
        sqrt_w.array();

    svd.compute(Xw);

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
    auto& values = mad_values_;
    values.clear();
    values.reserve(x.size());

    for(auto v : x)
        values.push_back(std::abs(v));

    // Python version uses median(rs[4:])
    // because this is Landsat time-series specific
    // If you want exact numpy median:
    // remove this offset.
    index_t start = 
        std::min<index_t>(4, values.size());

    index_t n = values.size() - start;

    if(n == 0)
        return 0.0;

    //--------------------------------------------------------
    // The median of the sorted values past `start` is one or
    // two order statistics of the whole set: select them
    // rather than sort. For an even count the lower middle is
    // the largest value left of the upper one.
    //--------------------------------------------------------
    const auto upper =
        values.begin() + start + n/2;

    std::nth_element(
        values.begin(),
        upper,
        values.end()
    );

    scalar_t med;

    if(n % 2)
    {
        med = *upper;
    }
    else
    {
        const scalar_t lower =
            *std::max_element(
                values.begin(),
                upper
            );

        med =
            0.5 * (lower + *upper);
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

    for(index_t i = 0; i < static_cast<index_t>(r.size()); ++i) 
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

RobustScore score(
    Eigen::Ref<const Eigen::VectorXd> y,
    Eigen::Ref<const Eigen::VectorXd> preds,
    index_t num_coefficients,
    bool unbiased_rmse
)
{
    assert(y.size() == preds.size());

    std::vector<scalar_t> residuals(y.size());

    scalar_t rss = 0.0;

    for (index_t i = 0; i < static_cast<index_t>(y.size()); ++i)
    {
        const scalar_t residual = y(i) - preds(i);
        residuals[i] = residual;
        rss += residual * residual;
    }

    scalar_t denominator = static_cast<scalar_t>(y.size());

    if (unbiased_rmse)
    {
        assert(static_cast<index_t>(y.size()) > num_coefficients);
        denominator -= static_cast<scalar_t>(num_coefficients);
    }

    return RobustScore{
        std::sqrt(rss / denominator),
        median_absolute(residuals),
        std::move(residuals)
    };
}


Eigen::MatrixXd tmask_basis(
    ArrayView<const std::int64_t, 1> dates
)
{      
    const index_t n = dates.size();
    constexpr index_t num_features = 5;

    Eigen::MatrixXd matrix(n, num_features);
    
    const scalar_t annual_cycle =
        2.0 * PI / AVG_DAYS_YR;

    const scalar_t years =
        std::ceil(
            (static_cast<scalar_t>(dates(n - 1)) -
            static_cast<scalar_t>(dates(0))) /
            AVG_DAYS_YR
        );

    const scalar_t observation_cycle = 
        annual_cycle / years;

    for(index_t i = 0; i < n; ++i)
    {
        const scalar_t d =
            static_cast<scalar_t>(dates(i));

        matrix(i, 0) =
            std::cos(annual_cycle * d);

        matrix(i, 1) =
            std::sin(annual_cycle * d);

        matrix(i, 2) =
            std::cos(observation_cycle * d);

        matrix(i, 3) =
            std::sin(observation_cycle * d);


        matrix(i, 4) = 1.0;
    }

    return matrix;
}

} // namespace ccd
