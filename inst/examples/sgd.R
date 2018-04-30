##
library("tidyverse")

trace_limit = 25
number_of_simulations = 1000
number_of_covariates = 6
number_of_observations = 10000
theta = matrix(rnorm(number_of_covariates, 0, 4), ncol = 1)
parameters_list = vector("list", number_of_simulations)
for (i in 1:number_of_simulations) {
    if ((i == 1) | (i == number_of_simulations) | ((i %% trace_limit) == 0))
        cat("Iteration:", i, "\n")

    X = matrix(rnorm(number_of_observations * number_of_covariates), ncol = number_of_covariates, nrow = number_of_observations)
    y = X %*% theta + rnorm(number_of_observations, 0, number_of_covariates)

    initial_theta = matrix(rnorm(number_of_covariates, 0, 4), ncol = 1)
    gd_optimised = gd(X = X, y = y, theta_initial = initial_theta, alpha = 0.000001,
                      tolerance = 1e-6, max_iterations = 1000, trace_limit = 1, trace = FALSE)

    sgd_optimised = sgd(X = X, y = y, theta_initial = initial_theta, alpha = 0.0001, gamma = 0.9, sgd_iterations = 10,
                        trace_limit = 1, trace = FALSE)

    parameters_list[[i]] = data.frame(Iteration = i, ParameterNames = paste0("beta[", 1:number_of_covariates, "]"), Parameters = theta[, 1],
                                      ParametersGD = gd_optimised$theta[, 1], ParametersSGD = sgd_optimised$theta[, 1], stringsAsFactors = FALSE)
}

(parameters_tibble <- parameters_list %>% bind_rows() %>%
        group_by(ParameterNames) %>%
        summarise(Parameters = mean(Parameters), GD = mean(ParametersGD), SGD = mean(ParametersSGD)))

parameters_tibble %>%
    mutate(RelativeDifferenceGD = (Parameters - GD) / Parameters, RelativeDifferenceSGD = (Parameters - SGD) / Parameters) %>%
    select(-Parameters, -GD, -SGD)

#
parameters_tibble = bind_rows(parameters_list) %>%
    mutate(GD = ParametersGD, SGD = ParametersSGD) %>%
    select(-Parameters, -ParametersGD, -ParametersSGD) %>%
    gather(Method, Parameters, -(Iteration:ParameterNames))

true_parameters_tibble = bind_rows(parameters_list) %>% group_by(ParameterNames) %>% distinct(Parameters)
estimated_parameters_tibble <- parameters_tibble %>% group_by(ParameterNames, Method) %>%
    summarise(Parameters = mean(Parameters))

ggplot(parameters_tibble, aes(x = Iteration, y = Parameters, colour = Method)) +
    geom_line(aes(colour = NULL)) +
    geom_hline(data = true_parameters_tibble, aes(yintercept = Parameters, colour = NULL), size = 0.75, colour = "grey") +
    geom_hline(data = estimated_parameters_tibble, aes(yintercept = Parameters, colour = Method), size = 0.75) +
    facet_grid(Method ~ ParameterNames, labeller = label_parsed) +
    theme_bw() + theme(legend.position = "top")

##
library("microbenchmark")

number_of_covariates = 6
number_of_observations = 10000
theta = matrix(rnorm(number_of_covariates, 0, 4), ncol = 1)

X = matrix(rnorm(number_of_observations * number_of_covariates), ncol = number_of_covariates, nrow = number_of_observations)
y = X %*% theta + rnorm(number_of_observations, 0, number_of_covariates)

microbenchmark(gd_optimised = gd(X = X, y = y, theta_initial = rep(0, number_of_covariates), alpha = 0.000001,
                                 tolerance = 1e-6, max_iterations = 1000, trace_limit = 1, trace = FALSE),
               sgd_optimised = sgd(X = X, y = y, theta_initial = rep(0, number_of_covariates), alpha = 0.0001, gamma = 0.9, sgd_iterations = 10,
                                   trace_limit = 1, trace = FALSE),
               times = 100)
