"""
Generic mathematical and estimation utilities for the synthetic-sample pipeline.
"""

import math
import numpy as np


def validate_xy(x, y, name_prefix):
    """
    Validate that x and y have compatible shapes.

    Expected:
        x: 2D array of shape (n_samples, x_dim)
        y: 1D array of shape (n_samples,)
    """
    if x.ndim != 2:
        raise ValueError(f"{name_prefix}_x must be a 2D array.")
    if y.ndim != 1:
        raise ValueError(f"{name_prefix}_y must be a 1D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{name_prefix}_x and {name_prefix}_y must have the same number of rows.")


def validate_loaded_data(data_dict):
    """
    Validate the dictionary returned by load_real_data().

    Expected keys:
        - train_x
        - train_y
        - test_x
        - test_y
    """
    train_x = data_dict["train_x"]
    train_y = data_dict["train_y"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]

    validate_xy(train_x, train_y, "train")
    validate_xy(test_x, test_y, "test")

    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train_x and test_x must have the same feature dimension.")


def estimate_vp(real_x, real_y, reference_model, config, compute_psi_fn):
    """
    Estimate Vp from real data.

    The empirical estimator is:
        Vp_hat = (1 / n) * sum_i psi_i psi_i^T
    """
    psi = compute_psi_fn(real_x, real_y, reference_model, config)

    n_samples = psi.shape[0]
    vp = (psi.T @ psi) / float(n_samples)
    return vp


def invert_positive_matrix(matrix, ridge):
    """
    Invert a positive semi-definite matrix after ridge stabilization.
    """
    dim = matrix.shape[0]
    stabilized = matrix + float(ridge) * np.eye(dim)
    inverse = np.linalg.inv(stabilized)
    return inverse


def compute_influence_norm(x, y, reference_model, vp_inv, config, compute_psi_fn):
    """
    Compute sqrt(psi^T Vp^{-1} psi) for each sample.
    """
    psi = compute_psi_fn(x, y, reference_model, config)

    quadratic_values = np.sum((psi @ vp_inv) * psi, axis=1)
    quadratic_values = np.maximum(quadratic_values, 0.0)
    influence_norm = np.sqrt(quadratic_values)
    return influence_norm


def compute_theory_statistics(
    x,
    y,
    reference_model,
    ratio_state,
    vp_inv,
    config,
    compute_log_ratio_fn,
    compute_psi_fn,
):
    """
    Compute the theory quantities used before pi calibration.

    The returned dictionary contains:
        - log_ratio
        - ratio
        - influence
        - score

    The score is:
        ratio * sqrt(psi^T Vp^{-1} psi)
    """
    log_ratio = compute_log_ratio_fn(x, y, ratio_state, config)

    ratio = np.exp(log_ratio)

    influence = compute_influence_norm(
        x=x,
        y=y,
        reference_model=reference_model,
        vp_inv=vp_inv,
        config=config,
        compute_psi_fn=compute_psi_fn,
    )

    score = ratio * influence

    return {
        "log_ratio": log_ratio,
        "ratio": ratio,
        "influence": influence,
        "score": score,
    }


def calibrate_kappa(score, alpha, n_iter=60):
    """
    Calibrate kappa so that the average acceptance probability matches alpha.

    We use:
        pi_i = min(1, score_i / sqrt(kappa))

    This function uses binary search over t = 1 / sqrt(kappa).
    """
    score = np.asarray(score, dtype=float)
    score = np.maximum(score, 0.0)

    if alpha <= 0.0:
        return 1e30

    if alpha >= 1.0:
        return 1e-30

    if np.all(score == 0.0):
        return 1e30

    low = 0.0
    high = 1.0

    for _ in range(60):
        avg_pi = np.mean(np.minimum(1.0, high * score))
        if avg_pi >= alpha:
            break
        high = high * 2.0

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        avg_pi = np.mean(np.minimum(1.0, mid * score))

        if avg_pi < alpha:
            low = mid
        else:
            high = mid

    t_value = high
    kappa = 1.0 / max(t_value * t_value, 1e-30)
    return kappa


def compute_theory_pi(score, alpha):
    """
    Compute theory acceptance probabilities and calibrated kappa.
    """
    kappa = calibrate_kappa(score=score, alpha=alpha)
    pi = np.minimum(1.0, np.asarray(score, dtype=float) / math.sqrt(kappa))
    pi = np.clip(pi, 0.0, 1.0)
    return pi, kappa


def build_uniform_pi(n_candidates, alpha):
    """
    Build uniform acceptance probabilities.
    """
    return np.full(n_candidates, float(alpha), dtype=float)


def sample_with_pi(x, y, pi, ratio, alpha, rng):
    """
    Run rejection sampling and compute synthetic weights.

    Weight formula:
        weight_i = alpha * ratio_i / pi_i

    Returns
    -------
    dict
        Dictionary with accepted samples and metadata.
    """
    pi = np.asarray(pi, dtype=float)
    ratio = np.asarray(ratio, dtype=float)

    pi = np.clip(pi, 0.0, 1.0)

    random_values = rng.rand(x.shape[0])
    accepted_mask = random_values < pi

    accepted_x = x[accepted_mask]
    accepted_y = y[accepted_mask]

    accepted_pi = np.clip(pi[accepted_mask], 1e-12, 1.0)
    accepted_ratio = ratio[accepted_mask]
    accepted_weight = alpha * accepted_ratio / accepted_pi

    return {
        "x": accepted_x,
        "y": accepted_y,
        "weight": accepted_weight,
        "accepted_mask": accepted_mask,
        "accept_rate": float(np.mean(accepted_mask)),
    }


def estimate_vw_from_accepted(
    accepted_x,
    accepted_y,
    accepted_weight,
    reference_model,
    config,
    psi_dim,
    compute_psi_fn,
):
    """
    Estimate Vw from accepted synthetic samples.

    The empirical estimator is:
        Vw_hat = (1 / m) * sum_j w_j^2 psi_j psi_j^T

    If no synthetic sample is accepted, this function returns a matrix
    filled with NaN values.
    """
    m = accepted_x.shape[0]

    if m == 0:
        return np.full((psi_dim, psi_dim), np.nan, dtype=float)

    psi = compute_psi_fn(accepted_x, accepted_y, reference_model, config)

    weighted_psi = accepted_weight.reshape(-1, 1) * psi
    vw = (weighted_psi.T @ weighted_psi) / float(m)
    return vw


def fit_augmented_model(real_x, real_y, synth_x, synth_y, synth_weight, config, fit_final_model_fn):
    """
    Fit the final model on real data plus accepted synthetic data.

    Real samples always receive weight 1.
    Synthetic samples receive the computed synthetic weights.
    """
    if synth_x is None or synth_x.shape[0] == 0:
        return fit_final_model_fn(real_x, real_y, None, config)

    x_all = np.vstack([real_x, synth_x])
    y_all = np.concatenate([real_y, synth_y])
    weights = np.concatenate([
        np.ones(real_x.shape[0], dtype=float),
        synth_weight.astype(float),
    ])

    return fit_final_model_fn(x_all, y_all, weights, config)
