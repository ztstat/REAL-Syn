from __future__ import annotations
import copy
import hashlib
from typing import Any
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from common import pipeline_math as original_math
from common import ols_pipeline as original_pipeline
from common import ols_model as original_model
from common import importance as importance

GRID = (
    ("G1", 0.40, 2.50),
    ("G2", 0.40, 2.25),
    ("G3", 0.40, 2.00),
    ("G4", 0.50, 2.50),
    ("G5", 0.50, 2.25),
    ("G6", 0.50, 2.00),
    ("G7", 0.60, 2.50),
    ("G8", 0.60, 2.25),
    ("G9", 0.60, 2.00),
)

METHODS = (
    "Real-only",
    "Uniform",
    "DRE-only",
    "Influence-only",
    "REAL-Syn",
    "SIR",
)

SELECTION_METHODS = (
    "Uniform",
    "DRE-only",
    "Influence-only",
    "REAL-Syn",
)

METRICS = (
    "test_mse",
    "noiseless_prediction_mse",
    "parameter_squared_error",
)

N_REAL = 40

N_CANDIDATES = 6000

N_TEST = 8000

ALPHA = 0.1

BUDGET = 600

RANDOM_SEED_MODULUS = 2**32 - 1

def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def stream_seed(rep_seed: int, offset: int) -> int:
    return int((int(rep_seed) + int(offset)) % RANDOM_SEED_MODULUS)


def proposal_section(
    source: dict[str, Any], mu_x: float, sigma_x: float
) -> dict[str, Any]:
    section = copy.deepcopy(source["synthetic_candidate"])
    section["x_mean"] = [float(mu_x)] * 5
    section["x_std"] = [float(sigma_x)] * 5
    return section


def linear_gaussian_joint_parameters(
    section: dict[str, Any], dimension: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    x_mean, x_std, beta, intercept, noise_std = (
        original_pipeline.get_linear_gaussian_params(section, dimension)
    )
    x_covariance = np.diag(np.asarray(x_std, dtype=float) ** 2)
    mean = np.concatenate(
        [x_mean, [float(intercept + np.asarray(beta) @ x_mean)]]
    )
    covariance = np.zeros((dimension + 1, dimension + 1), dtype=float)
    covariance[:dimension, :dimension] = x_covariance
    covariance[:dimension, dimension] = x_covariance @ beta
    covariance[dimension, :dimension] = beta @ x_covariance
    covariance[dimension, dimension] = float(
        beta @ x_covariance @ beta + noise_std**2
    )
    return mean, covariance


def gaussian_severity(
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    proposal_mean: np.ndarray,
    proposal_covariance: np.ndarray,
) -> dict[str, Any]:
    p_precision = np.linalg.inv(target_covariance)
    q_precision = np.linalg.inv(proposal_covariance)
    bounded_matrix = 0.5 * (
        p_precision - q_precision + (p_precision - q_precision).T
    )
    bounded_linear = p_precision @ target_mean - q_precision @ proposal_mean
    bounded_eigenvalues, bounded_vectors = np.linalg.eigh(bounded_matrix)
    scale = max(
        1.0,
        float(np.linalg.norm(p_precision, ord=2)),
        float(np.linalg.norm(q_precision, ord=2)),
    )
    tolerance = 1e-10 * scale
    if float(np.min(bounded_eigenvalues)) > tolerance:
        bounded = True
        bounded_reason = "positive_definite_quadratic"
    elif float(np.min(bounded_eigenvalues)) < -tolerance:
        bounded = False
        bounded_reason = "negative_quadratic_direction"
    else:
        null = bounded_vectors[:, np.abs(bounded_eigenvalues) <= tolerance]
        null_linear = (
            0.0
            if null.shape[1] == 0
            else float(np.linalg.norm(null.T @ bounded_linear))
        )
        bounded = null_linear <= tolerance * (
            1.0 + float(np.linalg.norm(bounded_linear))
        )
        bounded_reason = (
            "positive_semidefinite_null_linear_zero"
            if bounded
            else "positive_semidefinite_null_linear_nonzero"
        )

    second_matrix = 0.5 * (
        2.0 * p_precision
        - q_precision
        + (2.0 * p_precision - q_precision).T
    )
    second_eigenvalues = np.linalg.eigvalsh(second_matrix)
    second_finite = float(np.min(second_eigenvalues)) > tolerance
    log_second_moment = None
    population_ress = None
    if second_finite:
        linear = 2.0 * p_precision @ target_mean - q_precision @ proposal_mean
        constant = float(
            -target_mean @ p_precision @ target_mean
            + 0.5 * proposal_mean @ q_precision @ proposal_mean
        )
        sign_q, logdet_q = np.linalg.slogdet(proposal_covariance)
        sign_p, logdet_p = np.linalg.slogdet(target_covariance)
        sign_a, logdet_a = np.linalg.slogdet(second_matrix)
        if min(sign_q, sign_p, sign_a) <= 0:
            raise RuntimeError("Invalid covariance determinant in severity check.")
        log_second_moment = float(
            0.5 * logdet_q
            - logdet_p
            - 0.5 * logdet_a
            + constant
            + 0.5 * linear @ np.linalg.solve(second_matrix, linear)
        )
        population_ress = float(np.exp(-log_second_moment))
    return {
        "bounded": bool(bounded),
        "boundedness_reason": bounded_reason,
        "boundedness_matrix_eigenvalues": bounded_eigenvalues,
        "second_moment_finite": bool(second_finite),
        "second_moment_matrix_eigenvalues": second_eigenvalues,
        "log_second_moment": log_second_moment,
        "population_relative_ess": population_ress,
        "boundedness_tolerance": tolerance,
    }


def analytic_grid(
    config: dict[str, Any], source: dict[str, Any]
) -> list[dict[str, Any]]:
    target_mean, target_covariance = linear_gaussian_joint_parameters(
        source["data"]
    )
    output: list[dict[str, Any]] = []
    for grid_cell, mu_x, sigma_x in GRID:
        proposal = proposal_section(source, mu_x, sigma_x)
        proposal_mean, proposal_covariance = linear_gaussian_joint_parameters(
            proposal
        )
        severity = gaussian_severity(
            target_mean,
            target_covariance,
            proposal_mean,
            proposal_covariance,
        )
        output.append(
            {
                "grid_cell": grid_cell,
                "mu_x": mu_x,
                "sigma_x": sigma_x,
                "c_prop": sigma_x**2,
                "target_mean": target_mean,
                "target_covariance": target_covariance,
                "proposal_mean": proposal_mean,
                "proposal_covariance": proposal_covariance,
                **severity,
            }
        )
    failed = [
        row["grid_cell"] for row in output if not row["second_moment_finite"]
    ]
    if failed:
        raise RuntimeError(
            f"Finite-second-moment gate failed for: {', '.join(failed)}"
        )
    return output


def numerical_log_second_moment_check(
    analytic_row: dict[str, Any], seed: int, samples: int = 256
) -> float:
    p_mean = analytic_row["target_mean"]
    p_covariance = analytic_row["target_covariance"]
    q_mean = analytic_row["proposal_mean"]
    q_covariance = analytic_row["proposal_covariance"]
    p_precision = np.linalg.inv(p_covariance)
    q_precision = np.linalg.inv(q_covariance)
    second_matrix = 2.0 * p_precision - q_precision
    linear = 2.0 * p_precision @ p_mean - q_precision @ q_mean
    tilted_covariance = np.linalg.inv(second_matrix)
    tilted_mean = tilted_covariance @ linear
    rng = np.random.RandomState(int(seed))
    values = rng.multivariate_normal(
        tilted_mean, tilted_covariance, size=int(samples)
    )
    log_weights = (
        2.0
        * multivariate_normal.logpdf(values, mean=p_mean, cov=p_covariance)
        - multivariate_normal.logpdf(values, mean=q_mean, cov=q_covariance)
        - multivariate_normal.logpdf(
            values, mean=tilted_mean, cov=tilted_covariance
        )
    )
    return float(logsumexp(log_weights) - np.log(log_weights.size))


def effective_sample_size(weights: np.ndarray) -> float:
    return importance.effective_sample_size(np.asarray(weights, dtype=float))


def fixed_test_data(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fixed_config = original_pipeline.copy_config_with_seed(
        source,
        original_pipeline.stream_seed(int(source["seed"]), "fixed_data"),
    )
    data = original_pipeline.load_misspecified_real_data(fixed_config)
    return data["test_x"], data["test_y"]


def prepare_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    rep_seed = int(config["seeds"]["base_seed"]) + int(
        config["seeds"]["rep_seed_stride"]
    ) * int(rep)
    rep_config = original_pipeline.copy_config_with_seed(
        source, original_pipeline.stream_seed(rep_seed, "real_data")
    )
    data = original_pipeline.load_misspecified_real_data(rep_config)
    real_x, real_y = data["train_x"], data["train_y"]
    pilot_state = original_model.fit_reference_model(real_x, real_y, rep_config)
    pilot = np.asarray(pilot_state["beta_full"], dtype=float)
    vp = original_math.estimate_vp(
        real_x,
        real_y,
        pilot_state,
        rep_config,
        compute_psi_fn=original_model.compute_psi,
    )
    vp_inverse = original_math.invert_positive_matrix(
        vp, float(source["pipeline"]["vp_ridge"])
    )
    truth = np.asarray(
        [source["data"]["intercept"], *source["data"]["beta"]],
        dtype=float,
    )
    pilot_error = pilot - truth
    pilot_parameter_squared_error = float(pilot_error @ pilot_error)
    vp_condition_number = float(np.linalg.cond(vp))
    if not np.isfinite(vp_condition_number):
        raise RuntimeError("Vp condition number is nonfinite.")
    candidate_seed = original_pipeline.stream_seed(rep_seed, "candidate")
    candidate_rng = np.random.RandomState(candidate_seed)
    base_proposal_x = candidate_rng.randn(N_CANDIDATES, 5)
    base_proposal_noise = candidate_rng.randn(N_CANDIDATES)
    acceptance_seeds = {
        "Uniform": original_pipeline.stream_seed(rep_seed, "sample_uniform"),
        "DRE-only": original_pipeline.stream_seed(
            rep_seed, "sample_ratio_only"
        ),
        "Influence-only": original_pipeline.stream_seed(
            rep_seed, "sample_influence_only"
        ),
        "REAL-Syn": original_pipeline.stream_seed(rep_seed, "sample_theory"),
    }
    base_uniforms = {
        method: np.random.RandomState(seed).random_sample(N_CANDIDATES)
        for method, seed in acceptance_seeds.items()
    }
    sir_seed = stream_seed(
        rep_seed, int(config["seeds"]["sir_resampling_offset"])
    )
    sir_base_uniforms = np.random.RandomState(sir_seed).random_sample(BUDGET)
    prepared = {
        "rep": int(rep),
        "rep_seed": rep_seed,
        "fit_config": rep_config,
        "source": source,
        "real_x": real_x,
        "real_y": real_y,
        "test_x": test_x,
        "test_y": test_y,
        "pilot": pilot,
        "pilot_state": pilot_state,
        "vp": vp,
        "vp_inverse": vp_inverse,
        "pilot_parameter_squared_error": pilot_parameter_squared_error,
        "vp_condition_number": vp_condition_number,
        "base_proposal_x": base_proposal_x,
        "base_proposal_noise": base_proposal_noise,
        "base_uniforms": base_uniforms,
        "sir_base_uniforms": sir_base_uniforms,
        "seeds": {
            "rep_seed": rep_seed,
            "real_data": original_pipeline.stream_seed(rep_seed, "real_data"),
            "candidate_generation": candidate_seed,
            "fixed_test": original_pipeline.stream_seed(
                int(source["seed"]), "fixed_data"
            ),
            "sir_resampling": sir_seed,
            **{
                f"{method}_acceptance": seed
                for method, seed in acceptance_seeds.items()
            },
        },
    }
    prepared["fingerprints"] = {
        "real_data_fingerprint": array_fingerprint(real_x, real_y),
        "test_data_fingerprint": array_fingerprint(test_x, test_y),
        "pilot_fingerprint": array_fingerprint(pilot),
        "vp_fingerprint": array_fingerprint(vp),
        "base_proposal_fingerprint": array_fingerprint(
            base_proposal_x, base_proposal_noise
        ),
        "sir_base_uniform_fingerprint": array_fingerprint(sir_base_uniforms),
        **{
            f"{method}_uniform_fingerprint": array_fingerprint(values)
            for method, values in base_uniforms.items()
        },
    }
    return prepared


def prepare_cell(
    prepared: dict[str, Any],
    grid_cell: str,
    mu_x: float,
    sigma_x: float,
) -> dict[str, Any]:
    source = prepared["source"]
    proposal = proposal_section(source, mu_x, sigma_x)
    candidate_x = mu_x + sigma_x * prepared["base_proposal_x"]
    beta = np.asarray(proposal["beta"], dtype=float)
    candidate_y = (
        float(proposal["intercept"])
        + candidate_x @ beta
        + float(proposal["noise_std"]) * prepared["base_proposal_noise"]
    )
    raw_log_ratio = original_pipeline.linear_gaussian_joint_logpdf(
        candidate_x, candidate_y, source["data"], 5
    ) - original_pipeline.linear_gaussian_joint_logpdf(
        candidate_x, candidate_y, proposal, 5
    )
    joint = np.column_stack([candidate_x, candidate_y])
    target_mean, target_covariance = linear_gaussian_joint_parameters(
        source["data"]
    )
    proposal_mean, proposal_covariance = linear_gaussian_joint_parameters(
        proposal
    )
    generic_raw_log_ratio = multivariate_normal.logpdf(
        joint, mean=target_mean, cov=target_covariance
    ) - multivariate_normal.logpdf(
        joint, mean=proposal_mean, cov=proposal_covariance
    )
    generic_log_ratio_max_abs_difference = float(
        np.max(np.abs(raw_log_ratio - generic_raw_log_ratio))
    )
    if generic_log_ratio_max_abs_difference > 1e-11:
        raise RuntimeError(
            f"{grid_cell} oracle joint log-ratio audit failed."
        )
    clip = float(source["ratio"]["clip_log_ratio"])
    log_ratio = np.clip(raw_log_ratio, -clip, clip)
    ratio = np.exp(log_ratio)
    influence = original_math.compute_influence_norm(
        candidate_x,
        candidate_y,
        prepared["pilot_state"],
        prepared["vp_inverse"],
        prepared["fit_config"],
        original_model.compute_psi,
    )
    for array in (candidate_x, candidate_y, raw_log_ratio, ratio, influence):
        if np.any(~np.isfinite(array)):
            raise RuntimeError(f"{grid_cell} contains a nonfinite value.")
    return {
        "grid_cell": grid_cell,
        "mu_x": mu_x,
        "sigma_x": sigma_x,
        "c_prop": sigma_x**2,
        "proposal": proposal,
        "candidate_x": candidate_x,
        "candidate_y": candidate_y,
        "raw_log_ratio": raw_log_ratio,
        "log_ratio": log_ratio,
        "ratio": ratio,
        "influence": influence,
        "score": ratio * influence,
        "candidate_fingerprint": array_fingerprint(candidate_x, candidate_y),
        "lower_clipping_count": int(np.count_nonzero(raw_log_ratio < -clip)),
        "upper_clipping_count": int(np.count_nonzero(raw_log_ratio > clip)),
        "nonfinite_ratio_count": 0,
        "generic_log_ratio_max_abs_difference": (
            generic_log_ratio_max_abs_difference
        ),
    }


def calibrate_pi(score: np.ndarray) -> tuple[np.ndarray, float]:
    pi, constant = original_pipeline.compute_calibrated_pi(score, ALPHA)
    pi = np.asarray(pi, dtype=float)
    if np.any(~np.isfinite(pi)) or np.any((pi < 0.0) | (pi > 1.0)):
        raise RuntimeError("Invalid inclusion probabilities.")
    if not np.isclose(float(np.mean(pi)), ALPHA, rtol=0.0, atol=2e-12):
        raise RuntimeError("Mean inclusion probability missed alpha.")
    return pi, float(constant)


def empty_sample() -> dict[str, Any]:
    return {
        "x": None,
        "y": None,
        "weights": None,
        "indices": None,
        "pi": None,
        "calibration_constant": None,
        "synthetic_rows_used": None,
        "unique_synthetic_rows_used": None,
        "clipping_activated": None,
        "clipping_threshold": None,
        "clipped_count": None,
        "ess_before_clipping": None,
        "ess_after_clipping": None,
        "sir_total_draws": None,
        "sir_unique_count": None,
        "sir_multiplicity_ess": None,
        "sir_maximum_multiplicity": None,
        "sir_expanded_aggregate_coefficient_difference": None,
    }


def build_method_sample(
    method: str,
    prepared: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    result = empty_sample()
    ratio = cell["ratio"]
    x = cell["candidate_x"]
    y = cell["candidate_y"]
    if method == "Real-only":
        return result
    if method in SELECTION_METHODS:
        if method == "Uniform":
            pi = np.full(N_CANDIDATES, ALPHA)
            constant = ALPHA
        elif method == "DRE-only":
            pi, constant = calibrate_pi(ratio)
        elif method == "Influence-only":
            pi, constant = calibrate_pi(cell["influence"])
        elif method == "REAL-Syn":
            pi, constant = calibrate_pi(ratio * cell["influence"])
        else:
            raise ValueError(f"Unknown selection method: {method}")
        mask = prepared["base_uniforms"][method] < pi
        if not np.any(mask):
            raise RuntimeError(f"{cell['grid_cell']}/{method} empty selection.")
        indices = np.flatnonzero(mask)
        weights = ALPHA * ratio[indices] / np.clip(
            pi[indices], 1e-12, 1.0
        )
        result.update(
            {
                "x": x[indices],
                "y": y[indices],
                "weights": weights,
                "indices": indices,
                "pi": pi,
                "calibration_constant": constant,
                "synthetic_rows_used": int(indices.size),
                "unique_synthetic_rows_used": int(indices.size),
            }
        )
        return result
    if method == "SIR":
        probabilities = ratio / float(np.sum(ratio))
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0
        draws = np.searchsorted(
            cdf, prepared["sir_base_uniforms"], side="right"
        )
        draws = np.minimum(draws, N_CANDIDATES - 1)
        multiplicities = np.bincount(draws, minlength=N_CANDIDATES)
        indices = np.flatnonzero(multiplicities)
        weights = multiplicities[indices].astype(float)
        result.update(
            {
                "x": x[indices],
                "y": y[indices],
                "weights": weights,
                "indices": indices,
                "synthetic_rows_used": BUDGET,
                "unique_synthetic_rows_used": int(indices.size),
                "sir_total_draws": BUDGET,
                "sir_unique_count": int(indices.size),
                "sir_multiplicity_ess": float(
                    BUDGET**2 / np.sum(multiplicities**2)
                ),
                "sir_maximum_multiplicity": int(np.max(multiplicities)),
                "_sir_draws": draws,
            }
        )
        return result
    raise ValueError(f"Unknown method: {method}")


def fit_coefficients(
    prepared: dict[str, Any], sample: dict[str, Any]
) -> np.ndarray:
    if sample["x"] is None:
        return prepared["pilot"].copy()
    model = original_math.fit_augmented_model(
        real_x=prepared["real_x"],
        real_y=prepared["real_y"],
        synth_x=sample["x"],
        synth_y=sample["y"],
        synth_weight=sample["weights"],
        config=prepared["fit_config"],
        fit_final_model_fn=original_model.fit_final_model,
    )
    return np.asarray(model["beta_full"], dtype=float)


def evaluate_coefficients(
    coefficients: np.ndarray, prepared: dict[str, Any]
) -> dict[str, float]:
    design = np.column_stack(
        [np.ones(prepared["test_x"].shape[0]), prepared["test_x"]]
    )
    prediction = design @ coefficients
    target = prepared["source"]["data"]
    true_mean = float(target["intercept"]) + prepared["test_x"] @ np.asarray(
        target["beta"], dtype=float
    )
    truth = np.asarray([target["intercept"], *target["beta"]], dtype=float)
    error = coefficients - truth
    return {
        "test_mse": float(np.mean((prediction - prepared["test_y"]) ** 2)),
        "noiseless_prediction_mse": float(
            np.mean((prediction - true_mean) ** 2)
        ),
        "parameter_squared_error": float(error @ error),
        "slope_parameter_squared_error": float(error[1:] @ error[1:]),
        "intercept_squared_error": float(error[0] ** 2),
    }


def covariance_diagnostic(
    prepared: dict[str, Any],
    cell: dict[str, Any],
    sample: dict[str, Any],
    method: str,
) -> float | None:
    """Return the submitted trace(Vp^-1 Vw)/psi_dim diagnostic."""
    if method == "Real-only":
        return None
    if method == "SIR":
        accepted_x = cell["candidate_x"][sample["_sir_draws"]]
        accepted_y = cell["candidate_y"][sample["_sir_draws"]]
        accepted_weight = np.ones(BUDGET, dtype=float)
    else:
        accepted_x = sample["x"]
        accepted_y = sample["y"]
        accepted_weight = sample["weights"]
    psi_dimension = int(prepared["vp"].shape[0])
    vw = original_math.estimate_vw_from_accepted(
        accepted_x=accepted_x,
        accepted_y=accepted_y,
        accepted_weight=accepted_weight,
        reference_model=prepared["pilot_state"],
        config=prepared["fit_config"],
        psi_dim=psi_dimension,
        compute_psi_fn=original_model.compute_psi,
    )
    value = float(
        np.trace(prepared["vp_inverse"] @ vw) / psi_dimension
    )
    if not np.isfinite(value):
        raise RuntimeError(f"{cell['grid_cell']}/{method} covariance diagnostic.")
    return value


def ratio_diagnostic_row(
    prepared: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    ratio = cell["ratio"]
    quantiles = np.quantile(ratio, [0.5, 0.9, 0.95, 0.99, 0.999])
    return {
        "rep": prepared["rep"],
        "grid_cell": cell["grid_cell"],
        "mu_x": cell["mu_x"],
        "sigma_x": cell["sigma_x"],
        "c_prop": cell["c_prop"],
        "mean_candidate_ratio": float(np.mean(ratio)),
        "ratio_ess": effective_sample_size(ratio),
        "ratio_relative_ess": effective_sample_size(ratio) / N_CANDIDATES,
        "ratio_p50": float(quantiles[0]),
        "ratio_p90": float(quantiles[1]),
        "ratio_p95": float(quantiles[2]),
        "ratio_p99": float(quantiles[3]),
        "ratio_p99p9": float(quantiles[4]),
        "maximum_ratio": float(np.max(ratio)),
        "lower_log_ratio_clipping_count": cell["lower_clipping_count"],
        "upper_log_ratio_clipping_count": cell["upper_clipping_count"],
        "nonfinite_ratio_count": cell["nonfinite_ratio_count"],
        "generic_log_ratio_max_abs_difference": cell[
            "generic_log_ratio_max_abs_difference"
        ],
        "candidate_fingerprint": cell["candidate_fingerprint"],
        **prepared["fingerprints"],
    }


def run_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    prepared = prepare_repetition(config, source, rep, test_x, test_y)
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    samples: dict[tuple[str, str], dict[str, Any]] = {}
    cells: dict[str, dict[str, Any]] = {}
    for grid_cell, mu_x, sigma_x in GRID:
        cell = prepare_cell(prepared, grid_cell, mu_x, sigma_x)
        cells[grid_cell] = cell
        ratio_rows.append(ratio_diagnostic_row(prepared, cell))
        for method in METHODS:
            sample = build_method_sample(method, prepared, cell)
            if method == "SIR":
                aggregate = fit_coefficients(prepared, sample)
                expanded_sample = {
                    "x": cell["candidate_x"][sample["_sir_draws"]],
                    "y": cell["candidate_y"][sample["_sir_draws"]],
                    "weights": np.ones(BUDGET),
                }
                expanded = fit_coefficients(prepared, expanded_sample)
                sample["sir_expanded_aggregate_coefficient_difference"] = float(
                    np.max(np.abs(aggregate - expanded))
                )
                coefficients = aggregate
            else:
                coefficients = fit_coefficients(prepared, sample)
            metrics = evaluate_coefficients(coefficients, prepared)
            covariance_value = covariance_diagnostic(
                prepared, cell, sample, method
            )
            if np.any(~np.isfinite(coefficients)) or any(
                not np.isfinite(value) for value in metrics.values()
            ):
                raise RuntimeError(f"{grid_cell}/{method} nonfinite output.")
            weights = sample["weights"]
            pi = sample["pi"]
            if weights is None:
                total_weight = weight_ess = relative_ess = None
                p95 = p99 = maximum = None
            else:
                total_weight = float(np.sum(weights))
                weight_ess = effective_sample_size(weights)
                relative_ess = weight_ess / int(sample["synthetic_rows_used"])
                p95 = float(np.quantile(weights, 0.95))
                p99 = float(np.quantile(weights, 0.99))
                maximum = float(np.max(weights))
            clipped_count = sample["clipped_count"]
            row = {
                "rep": rep,
                "rep_seed": prepared["rep_seed"],
                "grid_cell": grid_cell,
                "mu_x": mu_x,
                "sigma_x": sigma_x,
                "c_prop": sigma_x**2,
                "method": method,
                "n_real": N_REAL,
                "n_candidates": N_CANDIDATES,
                "n_test": N_TEST,
                "alpha": ALPHA,
                **metrics,
                "per_rep_squared_parameter_error": metrics[
                    "parameter_squared_error"
                ],
                **{
                    f"theta_{index}": float(value)
                    for index, value in enumerate(coefficients)
                },
                "accepted_count": (
                    0
                    if method == "Real-only"
                    else int(sample["synthetic_rows_used"])
                ),
                "draw_count": (
                    int(sample["sir_total_draws"])
                    if method == "SIR"
                    else None
                ),
                "synthetic_rows_used": sample["synthetic_rows_used"],
                "unique_synthetic_rows_used": sample[
                    "unique_synthetic_rows_used"
                ],
                "calibration_constant": sample["calibration_constant"],
                "mean_inclusion_probability": (
                    None if pi is None else float(np.mean(pi))
                ),
                "saturation_fraction": (
                    None
                    if pi is None
                    else float(np.mean(pi >= 1.0 - 1e-12))
                ),
                "minimum_inclusion_probability": (
                    None if pi is None else float(np.min(pi))
                ),
                "maximum_inclusion_probability": (
                    None if pi is None else float(np.max(pi))
                ),
                "candidate_ratio_ess": effective_sample_size(cell["ratio"]),
                "candidate_ratio_relative_ess": (
                    effective_sample_size(cell["ratio"]) / N_CANDIDATES
                ),
                "synthetic_total_weight": total_weight,
                "final_weight_ess": weight_ess,
                "relative_final_weight_ess": relative_ess,
                "p95_final_weight": p95,
                "p99_final_weight": p99,
                "maximum_final_weight": maximum,
                "clipping_activated": sample["clipping_activated"],
                "clipping_threshold": sample["clipping_threshold"],
                "clipped_count": clipped_count,
                "clipped_fraction": (
                    None
                    if clipped_count is None
                    else int(clipped_count) / N_CANDIDATES
                ),
                "ess_before_clipping": sample["ess_before_clipping"],
                "ess_after_clipping": sample["ess_after_clipping"],
                "sir_total_draws": sample["sir_total_draws"],
                "sir_unique_candidate_count": sample["sir_unique_count"],
                "sir_multiplicity_ess": sample["sir_multiplicity_ess"],
                "sir_maximum_multiplicity": sample[
                    "sir_maximum_multiplicity"
                ],
                "sir_expanded_aggregate_coefficient_difference": sample[
                    "sir_expanded_aggregate_coefficient_difference"
                ],
                "covariance_diagnostic": covariance_value,
                "pilot_parameter_squared_error": prepared[
                    "pilot_parameter_squared_error"
                ],
                "vp_condition_number": prepared["vp_condition_number"],
                "lower_log_ratio_clipping_count": cell[
                    "lower_clipping_count"
                ],
                "upper_log_ratio_clipping_count": cell[
                    "upper_clipping_count"
                ],
                "candidate_ratio_nonfinite_count": cell[
                    "nonfinite_ratio_count"
                ],
                "generic_log_ratio_max_abs_difference": cell[
                    "generic_log_ratio_max_abs_difference"
                ],
                "coefficient_nonfinite_count": int(
                    np.count_nonzero(~np.isfinite(coefficients))
                ),
                "metric_nonfinite_count": int(
                    sum(not np.isfinite(value) for value in metrics.values())
                ),
                "covariance_diagnostic_nonfinite_count": int(
                    covariance_value is not None
                    and not np.isfinite(covariance_value)
                ),
                "solver_failure_count": 0,
                "execution_failure_count": 0,
                "candidate_fingerprint": cell["candidate_fingerprint"],
                **prepared["fingerprints"],
            }
            rows.append(row)
            parameter_rows.append(
                {
                    "rep": rep,
                    "grid_cell": grid_cell,
                    "mu_x": mu_x,
                    "sigma_x": sigma_x,
                    "c_prop": sigma_x**2,
                    "rep_seed": prepared["rep_seed"],
                    "method": method,
                    **{
                        f"theta_{index}": float(value)
                        for index, value in enumerate(coefficients)
                    },
                }
            )
            samples[(grid_cell, method)] = sample

    maximum_mean_pi_error = 0.0
    maximum_selected_weight_error = 0.0
    maximum_sir_coefficient_difference = 0.0
    maximum_generic_log_ratio_difference = 0.0
    empty_selection_count = 0
    for grid_cell, _, _ in GRID:
        ratio = cells[grid_cell]["ratio"]
        for method in SELECTION_METHODS:
            sample = samples[(grid_cell, method)]
            indices = sample["indices"]
            empty_selection_count += int(indices.size == 0)
            expected = ALPHA * ratio[indices] / sample["pi"][indices]
            maximum_selected_weight_error = max(
                maximum_selected_weight_error,
                float(np.max(np.abs(sample["weights"] - expected))),
            )
            maximum_mean_pi_error = max(
                maximum_mean_pi_error,
                abs(float(np.mean(sample["pi"])) - ALPHA),
            )
        maximum_sir_coefficient_difference = max(
            maximum_sir_coefficient_difference,
            samples[(grid_cell, "SIR")][
                "sir_expanded_aggregate_coefficient_difference"
            ],
        )
        maximum_generic_log_ratio_difference = max(
            maximum_generic_log_ratio_difference,
            cells[grid_cell]["generic_log_ratio_max_abs_difference"],
        )
    sanity = {
        "rep": rep,
        "maximum_mean_pi_error": maximum_mean_pi_error,
        "maximum_selected_weight_error": maximum_selected_weight_error,
        "maximum_sir_expanded_aggregate_coefficient_difference": (
            maximum_sir_coefficient_difference
        ),
        "maximum_generic_log_ratio_difference": (
            maximum_generic_log_ratio_difference
        ),
        "nonfinite_ratio_count": int(
            sum(cell["nonfinite_ratio_count"] for cell in cells.values())
        ),
        "empty_selection_count": empty_selection_count,
        "grid_cell_count": len(cells),
        "method_count": len(METHODS),
        "common_random_number_fingerprints_saved": True,
        **prepared["fingerprints"],
    }
    return {
        "rows": rows,
        "parameter_rows": parameter_rows,
        "ratio_rows": ratio_rows,
        "sanity_row": sanity,
        "prepared": prepared,
        "cells": cells,
        "samples": samples,
    }


def seed_manifest(
    config: dict[str, Any], source: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    base = int(config["seeds"]["base_seed"])
    stride = int(config["seeds"]["rep_seed_stride"])
    offsets = {
        **original_pipeline.SEED_STREAM_OFFSETS,
        "sir_resampling": int(config["seeds"]["sir_resampling_offset"]),
    }
    return {
        "base_seed": base,
        "rep_seed_stride": stride,
        "fixed_test_seed": original_pipeline.stream_seed(
            int(source["seed"]), "fixed_data"
        ),
        "stream_offsets": offsets,
        "repetitions": [
            {
                "rep": rep,
                "rep_seed": base + stride * rep,
                "streams": {
                    name: stream_seed(base + stride * rep, offset)
                    for name, offset in offsets.items()
                },
            }
            for rep in range(repetitions)
        ],
    }
