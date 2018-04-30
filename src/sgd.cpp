#include <RcppArmadillo.h>

class optimisation_objective {
private :
    arma::mat X;
    arma::colvec y;

public :
    arma::colvec theta;

    double alpha;
    double gamma;

    std::size_t N;
    std::size_t M;

    optimisation_objective(const arma::mat & X_, const arma::colvec & y_, const arma::colvec & theta_initial, const double & alpha_, const double & gamma_) :
        X(X_), y(y_), theta(theta_initial), alpha(alpha_), gamma(gamma_), N(X_.n_rows), M(X_.n_cols) {  };

    double squared_error()
    {
        double residual = 0.0;
        for (std::size_t n = 0; n < N; n++)
        {
            const arma::rowvec &X_n = X.row(n);
            const double residual_n = arma::as_scalar(y.row(n) - X_n * theta);
            residual += std::pow(residual_n, 2.0);
        }

        return residual;
    }

    arma::colvec squared_error_gradient_traditional()
    {
        arma::colvec gradient(M);
        gradient.fill(0.0);

        for (std::size_t n = 0; n < N; n++)
        {
            const arma::rowvec &X_n = X.row(n);
            const double residual_n = arma::as_scalar(y[n] - X_n * theta);
            for (std::size_t m = 0; m < M; m++)
            {
                gradient.row(m) += -2.0 * residual_n * X_n[m];
            }
        }

        return gradient;
    }

    arma::colvec squared_error_gradient_index(const std::size_t &n)
    {
        arma::colvec gradient(M);
        gradient.fill(0.0);

        const arma::rowvec &X_n = X.row(n);
        const double residual_n = arma::as_scalar(y[n] - X_n * theta);
        for (std::size_t m = 0; m < M; m++)
        {
            gradient.row(m) += -2.0 * residual_n * X_n[m];
        }

        return gradient;
    }
};


//' @title Optimisation procedure
//'
//' @description Example of gradient and stochastic gradient descent method for minimising the sum of squares.
//'
//' @param X Design matrix.
//' @param y Response.
//' @param theta_initial The initial choice of theta.
//' @param alpha The step size.
//' @param tolerance The tolerance imposed on the absolute difference.
//' @param max_iterations The maximum number of iterations.
//' @param trace_limit When should the trace be displayed.
//' @param trace TRUE/FALSE: Should the trace be shown?
//'
//' @return The estimated paramters.
//' @example inst/examples/sgd.R
//[[Rcpp::export()]]
Rcpp::List gd(const arma::mat & X, const arma::colvec & y, const arma::colvec & theta_initial, const double & alpha,
              const double & tolerance, const std::size_t & max_iterations, const std::size_t & trace_limit, const bool & trace)
{
    optimisation_objective oo(X, y, theta_initial, alpha, 0.0);

    double new_error = oo.squared_error();
    std::size_t i = 1;
    bool converged = false;
    while (!converged)
    {
        arma::colvec gradient_current = oo.squared_error_gradient_traditional();
        arma::colvec theta_new = oo.theta - oo.alpha * gradient_current;

        oo.theta = theta_new;

        double old_error = new_error;
        new_error = oo.squared_error();

        double difference = std::abs(old_error - new_error);
        converged = (difference < tolerance) | (i > max_iterations);

        if (trace & ((i == 1) | ((i % trace_limit) == 0) | converged))
        {
            Rcpp::Rcout << "Iteration: " << i << "\n"
                        << "\tDifference: " << difference << "\n"
                        << "\tTheta: " << oo.theta.t() << "\n";
        }

        i++;
    }

    return Rcpp::List::create(Rcpp::Named("theta") = oo.theta);
}

//' @title Stochastic gradient descent
//'
//' @description Example of stochastic gradient descent method for minimising the sum of squares.
//'
//' @param X Design matrix.
//' @param y Response.
//' @param theta_initial The initial choice of theta.
//' @param alpha The step size.
//' @param gamma The belief in the current momentum.
//' @param sgd_iterations The number of iterations used, when utilising stochastic gradiend descent.
//' @param trace_limit When should the trace be displayed.
//' @param trace TRUE/FALSE: Should the trace be shown?
//'
//' @return The estimated paramters.
//' @example inst/examples/sgd.R
//[[Rcpp::export()]]
Rcpp::List sgd(const arma::mat & X, const arma::colvec & y, const arma::colvec & theta_initial, const double & alpha,
               const double & gamma, const std::size_t & sgd_iterations, const std::size_t & trace_limit, const bool & trace)
{
    optimisation_objective oo(X, y, theta_initial, alpha, gamma);
    for (std::size_t i = 0; i < sgd_iterations; i++)
    {
        arma::colvec theta_outer = oo.theta;
        arma::colvec momentum_current = oo.squared_error_gradient_index(0);

        arma::colvec theta_current = oo.theta;
        for (std::size_t n = 0; n < oo.N; n++)
        {
            arma::colvec momentum_old = momentum_current;
            momentum_current = oo.gamma * momentum_old + (1.0 - oo.gamma) * oo.squared_error_gradient_index(n);

            theta_current = oo.theta;
            oo.theta = theta_current - oo.alpha * momentum_current;
        }

        if (trace & ((i == 1) | ((i % trace_limit) == 0) | (i == (sgd_iterations - 1))))
        {
            Rcpp::Rcout << "Iteration: " << i << "\n"
                        << "\tDifference: " << oo.theta.t() - theta_outer.t()
                        << "\tTheta: " << oo.theta.t() << "\n";
        }
    }

    return Rcpp::List::create(Rcpp::Named("theta") = oo.theta);
}

