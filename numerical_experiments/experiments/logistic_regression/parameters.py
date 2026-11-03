"""Distributional definitions for the logistic-classification extension."""


from __future__ import annotations
import math
from typing import Any
import numpy as np


BASE = np.asarray([1.0, 0.8, 0.6, 0.4, 0.2], dtype=float)


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def unit_direction() -> np.ndarray:
    """Return v = base / ||base||."""
    return BASE / np.linalg.norm(BASE)


def target_parameters(pi_p: float, a: float = 1.0) -> dict[str, Any]:
    """Return target class means and the implied true logistic parameters."""
    v = unit_direction()
    target_std = 1.5
    beta_true = (2.0 * float(a) / target_std**2) * v
    intercept_true = math.log(float(pi_p) / (1.0 - float(pi_p)))
    return {
        "v": v,
        "mean_y1": float(a) * v,
        "mean_y0": -float(a) * v,
        "std": target_std,
        "intercept_true": intercept_true,
        "beta_true": beta_true,
    }


def proposal_parameters(a: float = 1.0) -> dict[str, Any]:
    """Return proposal class means and common isotropic standard deviation."""
    v = unit_direction()
    offset = 0.5 * np.ones(BASE.size, dtype=float)
    shift = 0.8 * float(a) * v
    return {
        "mean_y1": offset + shift,
        "mean_y0": offset - shift,
        "std": 2.5,
    }


def validate_config(config: dict[str, Any]) -> None:
    """Validate the fixed protocol encoded in an experiment configuration."""
    if int(config["d"]) != 5:
        raise ValueError("The logistic extension fixes d=5.")
    if config["arm"] not in {"rare_positive", "balanced"}:
        raise ValueError("Unknown logistic arm.")
    expected_pi_p = 0.2 if config["arm"] == "rare_positive" else 0.5
    if not np.isclose(float(config["pi_p"]), expected_pi_p):
        raise ValueError("pi_p does not match the named arm.")
    if not np.isclose(float(config["pi_q"]), 0.5):
        raise ValueError("The proposal prior must be 0.5.")
    if not np.isclose(float(config["a"]), 1.0):
        raise ValueError("The experiment fixes a=1.")
    if int(config["n_real"]) != 100:
        raise ValueError("n_real must be 100.")
    if int(config["n_candidates"]) != 6000:
        raise ValueError("n_candidates must be 6000.")
    if int(config["n_test"]) != 10000:
        raise ValueError("n_test must be 10000.")
    if int(config["repetitions"]) != 100:
        raise ValueError("Configurations must retain 100 repetitions.")
    if not np.isclose(float(config["alpha"]), 0.10):
        raise ValueError("alpha must be 0.10.")

    solver = config["solver"]
    expected_solver = {
        "ridge": 1e-6,
        "max_iterations": 100,
        "gradient_tolerance": 1e-8,
    }
    for key, expected in expected_solver.items():
        if not np.isclose(float(solver[key]), float(expected)):
            raise ValueError(f"Solver setting {key} must equal {expected}.")
    if bool(solver["penalize_intercept"]):
        raise ValueError("The intercept must not be penalized.")
    if not bool(solver["backtracking_line_search"]):
        raise ValueError("Backtracking line search is required.")

    target = target_parameters(float(config["pi_p"]), float(config["a"]))
    proposal = proposal_parameters(float(config["a"]))
    checks = [
        (config["target"]["mean_y1"], target["mean_y1"], "target mean_y1"),
        (config["target"]["mean_y0"], target["mean_y0"], "target mean_y0"),
        (config["proposal"]["mean_y1"], proposal["mean_y1"], "proposal mean_y1"),
        (config["proposal"]["mean_y0"], proposal["mean_y0"], "proposal mean_y0"),
        (config["true_logistic"]["beta"], target["beta_true"], "true beta"),
    ]
    for observed, expected, label in checks:
        if not np.allclose(np.asarray(observed, dtype=float), expected, atol=1e-12):
            raise ValueError(f"Config {label} is inconsistent with the factorization.")
    if not np.isclose(
        float(config["true_logistic"]["intercept"]),
        target["intercept_true"],
        atol=1e-12,
    ):
        raise ValueError("Config true intercept is inconsistent with pi_p.")


def _class_means(distribution: dict[str, Any], y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    mean_y1 = np.asarray(distribution["mean_y1"], dtype=float)
    mean_y0 = np.asarray(distribution["mean_y0"], dtype=float)
    return np.where(y[:, None] == 1, mean_y1[None, :], mean_y0[None, :])


def sample_joint(
    n_samples: int,
    prior: float,
    distribution: dict[str, Any],
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample Y first and then X conditional on Y."""
    y = rng.binomial(1, float(prior), size=int(n_samples)).astype(int)
    means = _class_means(distribution, y)
    x = means + float(distribution["std"]) * rng.normal(
        size=(int(n_samples), means.shape[1])
    )
    return x, y


def isotropic_gaussian_logpdf(
    x: np.ndarray,
    means: np.ndarray,
    std: float,
) -> np.ndarray:
    """Log density of independent isotropic Gaussian rows."""
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    variance = float(std) ** 2
    dimension = x.shape[1]
    return (
        -0.5 * dimension * math.log(2.0 * math.pi * variance)
        - 0.5 * np.sum((x - means) ** 2, axis=1) / variance
    )


def bernoulli_logpmf(y: np.ndarray, prior: float) -> np.ndarray:
    """Stable Bernoulli log probability for binary integer labels."""
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, math.log(float(prior)), math.log1p(-float(prior)))


def target_posterior_from_factorization(
    x: np.ndarray,
    pi_p: float,
    a: float = 1.0,
) -> np.ndarray:
    """Compute P_p(Y=1|X=x) directly from the Gaussian factorization."""
    target = target_parameters(pi_p=pi_p, a=a)
    x = np.asarray(x, dtype=float)
    log_joint_1 = (
        math.log(float(pi_p))
        + isotropic_gaussian_logpdf(
            x,
            np.broadcast_to(target["mean_y1"], x.shape),
            target["std"],
        )
    )
    log_joint_0 = (
        math.log1p(-float(pi_p))
        + isotropic_gaussian_logpdf(
            x,
            np.broadcast_to(target["mean_y0"], x.shape),
            target["std"],
        )
    )
    return sigmoid(log_joint_1 - log_joint_0)


def target_posterior_from_logistic(
    x: np.ndarray,
    pi_p: float,
    a: float = 1.0,
) -> np.ndarray:
    """Compute the same target posterior from the implied logistic model."""
    target = target_parameters(pi_p=pi_p, a=a)
    return sigmoid(
        float(target["intercept_true"])
        + np.asarray(x, dtype=float) @ np.asarray(target["beta_true"], dtype=float)
    )


def oracle_log_density_ratio(
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    apply_clipping: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped and raw log p(x,y)/q(x,y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    target = config["target"]
    proposal = config["proposal"]
    target_means = _class_means(target, y)
    proposal_means = _class_means(proposal, y)
    raw = (
        bernoulli_logpmf(y, float(config["pi_p"]))
        - bernoulli_logpmf(y, float(config["pi_q"]))
        + isotropic_gaussian_logpdf(x, target_means, float(target["std"]))
        - isotropic_gaussian_logpdf(x, proposal_means, float(proposal["std"]))
    )
    if not apply_clipping:
        return raw.copy(), raw
    clip_value = float(config["ratio"]["clip_log_ratio"])
    return np.clip(raw, -clip_value, clip_value), raw
