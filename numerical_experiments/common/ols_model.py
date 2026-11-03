"""
Linear regression module.
"""

import numpy as np


def _get_linear_data_params(config):
    """
    Extract the linear Gaussian data parameters from the config.
    """
    data_config = config["data"]
    x_dim = int(data_config["x_dim"])
    beta = np.asarray(data_config["beta"], dtype=float)
    intercept = float(data_config["intercept"])
    noise_std = float(data_config["noise_std"])
    n_train = int(data_config["n_train"])
    n_test = int(data_config["n_test"])
    return x_dim, beta, intercept, noise_std, n_train, n_test


def _add_intercept(x):
    """
    Add an intercept column of ones to the feature matrix.
    """
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([ones, x], axis=1)


def _simulate_linear_gaussian_data(n_samples, config, rng):
    """
    Simulate data from a linear Gaussian model.
    """
    x_dim, beta, intercept, noise_std, _, _ = _get_linear_data_params(config)

    x = rng.randn(n_samples, x_dim)
    noise = noise_std * rng.randn(n_samples)
    y = intercept + x @ beta + noise

    return x, y


def _weighted_linear_regression_fit(x, y, sample_weight=None, ridge=1e-10):
    """
    Fit weighted linear regression with an intercept term.

    The solution is:
        beta_hat = (X^T W X + ridge I)^{-1} X^T W y
    """
    x_with_intercept = _add_intercept(x)

    if sample_weight is None:
        sample_weight = np.ones(x.shape[0], dtype=float)

    sample_weight = np.asarray(sample_weight, dtype=float)
    sample_weight = np.maximum(sample_weight, 0.0)

    weighted_x = x_with_intercept * sample_weight.reshape(-1, 1)
    xtwx = x_with_intercept.T @ weighted_x
    xtwy = x_with_intercept.T @ (sample_weight * y)

    dim = x_with_intercept.shape[1]
    beta_full = np.linalg.solve(xtwx + ridge * np.eye(dim), xtwy)

    return {
        "beta_full": beta_full,
    }


def _predict_linear_regression(model_state, x):
    """
    Predict y values using the fitted linear regression model.
    """
    x_with_intercept = _add_intercept(x)
    beta_full = model_state["beta_full"]
    return x_with_intercept @ beta_full


def load_real_data(config):
    """
    Load or simulate real train/test data.
    """
    seed = int(config["seed"])
    rng = np.random.RandomState(seed)

    _, _, _, _, n_train, n_test = _get_linear_data_params(config)

    train_x, train_y = _simulate_linear_gaussian_data(n_train, config, rng)
    test_x, test_y = _simulate_linear_gaussian_data(n_test, config, rng)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }


def fit_reference_model(train_x, train_y, config):
    """
    Fit the reference model on real training data.
    """
    ridge = float(config["model"]["ridge"])

    return _weighted_linear_regression_fit(
        x=train_x,
        y=train_y,
        sample_weight=None,
        ridge=ridge,
    )


def compute_psi(x, y, reference_model, config):
    """
    Compute per-sample psi vectors for squared loss.
    """
    x_with_intercept = _add_intercept(x)
    beta_full = reference_model["beta_full"]

    residual = x_with_intercept @ beta_full - y
    psi = 2.0 * residual.reshape(-1, 1) * x_with_intercept
    return psi


def fit_generator(train_x, train_y, config):
    """
    Fit or prepare the synthetic generator.
    """
    x_dim, beta, intercept, noise_std, _, _ = _get_linear_data_params(config)
    generator_state = {
        "x_dim": x_dim,
        "beta": beta,
        "intercept": intercept,
        "noise_std": noise_std,
        "generator_mode": "same_as_real_distribution",
    }
    return generator_state


def generate_candidates(generator_state, n_candidates, rng, config):
    """
    Generate synthetic candidate samples.
    """
    x_dim = int(generator_state["x_dim"])
    beta = np.asarray(generator_state["beta"], dtype=float)
    intercept = float(generator_state["intercept"])
    noise_std = float(generator_state["noise_std"])

    cand_x = rng.randn(n_candidates, x_dim)
    noise = noise_std * rng.randn(n_candidates)
    cand_y = intercept + cand_x @ beta + noise

    return cand_x, cand_y


def fit_ratio_model(train_x, train_y, generator_state, rng, config):
    """
    Fit or prepare the ratio model.
    """
    ratio_state = {
        "ratio_mode": "q_equals_p",
        "clip_log_ratio": float(config["ratio"]["clip_log_ratio"]),
    }
    return ratio_state


def compute_log_ratio(x, y, ratio_state, config):
    """
    Compute the log density ratio.
    """
    return np.zeros(x.shape[0], dtype=float)


def fit_final_model(x, y, sample_weight, config):
    """
    Fit the final weighted linear regression model.
    """
    ridge = float(config["model"]["ridge"])

    return _weighted_linear_regression_fit(
        x=x,
        y=y,
        sample_weight=sample_weight,
        ridge=ridge,
    )


def evaluate_model(model_state, test_x, test_y, config):
    """
    Evaluate the fitted linear regression model using test MSE.
    """
    prediction = _predict_linear_regression(model_state, test_x)
    mse = float(np.mean((prediction - test_y) ** 2))

    return {
        "primary_metric_name": "mse",
        "primary_metric_value": mse,
        "mse": mse,
    }
