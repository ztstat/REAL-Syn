"""Weighted logistic regression using a fixed Newton/IRLS solver."""


from __future__ import annotations
from typing import Any
import numpy as np
from .parameters import sigmoid


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Prepend an intercept column."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    return np.column_stack([np.ones(x.shape[0], dtype=float), x])


def logistic_objective_gradient_hessian(
    theta: np.ndarray,
    x_bar: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    ridge: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return normalized penalized NLL, gradient, and Hessian.

    The ridge penalty excludes the intercept. Data weights are normalized by
    their sum so every method uses the same objective convention.
    """
    theta = np.asarray(theta, dtype=float)
    x_bar = np.asarray(x_bar, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones(y.size, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if x_bar.ndim != 2 or y.ndim != 1 or weights.ndim != 1:
        raise ValueError("Invalid logistic objective dimensions.")
    if x_bar.shape[0] != y.size or y.size != weights.size:
        raise ValueError("x, y, and weights have incompatible row counts.")
    if theta.shape != (x_bar.shape[1],):
        raise ValueError("theta has the wrong dimension.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Logistic weights must be finite and nonnegative.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Logistic weights must have positive sum.")

    logits = x_bar @ theta
    probabilities = sigmoid(logits)
    data_objective = float(
        np.sum(weights * (np.logaddexp(0.0, logits) - y * logits)) / weight_sum
    )
    penalty_mask = np.ones(theta.size, dtype=float)
    penalty_mask[0] = 0.0
    objective = data_objective + 0.5 * float(ridge) * float(
        np.sum((penalty_mask * theta) ** 2)
    )
    gradient = (
        x_bar.T @ (weights * (probabilities - y)) / weight_sum
        + float(ridge) * penalty_mask * theta
    )
    curvature = weights * probabilities * (1.0 - probabilities) / weight_sum
    hessian = x_bar.T @ (curvature[:, None] * x_bar)
    hessian = hessian + float(ridge) * np.diag(penalty_mask)
    return objective, gradient, hessian


def fit_weighted_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit logistic regression with fixed Newton steps and backtracking."""
    x_bar = add_intercept(x)
    y = np.asarray(y, dtype=float)
    ridge = float(solver_config["ridge"])
    max_iterations = int(solver_config["max_iterations"])
    tolerance = float(solver_config["gradient_tolerance"])
    armijo = float(solver_config.get("armijo_constant", 1e-4))
    shrink = float(solver_config.get("backtracking_shrink", 0.5))
    max_backtracking = int(solver_config.get("max_backtracking_steps", 50))
    theta = np.zeros(x_bar.shape[1], dtype=float)
    status = "maximum_iterations_reached"
    converged = False
    iteration_count = 0

    for iteration in range(max_iterations + 1):
        objective, gradient, hessian = logistic_objective_gradient_hessian(
            theta=theta,
            x_bar=x_bar,
            y=y,
            weights=weights,
            ridge=ridge,
        )
        gradient_norm = float(np.linalg.norm(gradient, ord=2))
        if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
            status = "nonfinite_objective_or_gradient"
            break
        if gradient_norm <= tolerance:
            converged = True
            status = "converged"
            iteration_count = iteration
            break
        if iteration == max_iterations:
            iteration_count = iteration
            break
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            status = "hessian_solve_failed"
            iteration_count = iteration
            break
        directional_derivative = float(gradient @ direction)
        step_size = 1.0
        accepted_step = False
        for _ in range(max_backtracking):
            proposed = theta - step_size * direction
            proposed_objective, _, _ = logistic_objective_gradient_hessian(
                theta=proposed,
                x_bar=x_bar,
                y=y,
                weights=weights,
                ridge=ridge,
            )
            if (
                np.isfinite(proposed_objective)
                and proposed_objective
                <= objective - armijo * step_size * directional_derivative
            ):
                theta = proposed
                accepted_step = True
                break
            step_size *= shrink
        iteration_count = iteration + 1
        if not accepted_step:
            status = "line_search_failed"
            break

    final_objective, final_gradient, final_hessian = (
        logistic_objective_gradient_hessian(
            theta=theta,
            x_bar=x_bar,
            y=y,
            weights=weights,
            ridge=ridge,
        )
    )
    return {
        "theta": theta,
        "converged": bool(converged),
        "status": status,
        "iterations": int(iteration_count),
        "gradient_norm": float(np.linalg.norm(final_gradient, ord=2)),
        "hessian_condition_number": float(np.linalg.cond(final_hessian)),
        "objective": float(final_objective),
        "ridge": ridge,
    }


def predict_probability(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Return fitted P(Y=1|X)."""
    return sigmoid(add_intercept(x) @ np.asarray(model["theta"], dtype=float))


def compute_score_psi(
    x: np.ndarray,
    y: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    """Return psi=(mu_theta(x)-y)*x_bar for every row."""
    x_bar = add_intercept(x)
    probabilities = sigmoid(x_bar @ np.asarray(model["theta"], dtype=float))
    return (probabilities - np.asarray(y, dtype=float))[:, None] * x_bar
