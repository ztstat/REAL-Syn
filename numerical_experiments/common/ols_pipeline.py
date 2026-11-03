"""
Main pipeline for simulation.
"""


import copy
import csv
import importlib
import json
import os
import numpy as np
from . import evaluation
from .pipeline_math import (
    build_uniform_pi,
    compute_theory_statistics,
    estimate_vp,
    estimate_vw_from_accepted,
    fit_augmented_model,
    invert_positive_matrix,
    sample_with_pi,
    validate_loaded_data,
    validate_xy,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


RANDOM_SEED_MODULUS = 2 ** 32 - 1


SEED_STREAM_OFFSETS = {
    "fixed_data": 104729,
    "real_data": 1_000_003,
    "ratio_model": 2_000_003,
    "candidate": 3_000_017,
    "sample_theory": 4_000_027,
    "sample_uniform": 5_000_029,
    "sample_ratio_only": 6_000_037,
    "sample_influence_only": 7_000_043,
}


def resolve_project_path(path):
    """
    Resolve a path relative to the project directory unless it is absolute.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


def load_config(config_path):
    """
    Load the experiment configuration from JSON.
    """
    resolved_config_path = resolve_project_path(config_path)
    with open(resolved_config_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_full_config(config_path, output_dir_override=None):
    """
    Build the final configuration after applying command-line overrides.
    """
    config = load_config(config_path)

    if output_dir_override is not None:
        config["output_base_dir"] = output_dir_override

    config["output_base_dir"] = resolve_project_path(config["output_base_dir"])
    return config


def load_model_module(config):
    """
    Load the active model module named in the config.
    """
    return importlib.import_module(config["active_model_module"])


def copy_config_with_seed(config, seed):
    """
    Copy the configuration dictionary and override the seed.
    """
    config_copy = copy.deepcopy(config)
    config_copy["seed"] = int(seed)
    return config_copy


def stream_seed(base_seed, stream_name):
    """
    Derive a reproducible independent seed for a named random stream.
    """
    return int((int(base_seed) + SEED_STREAM_OFFSETS[stream_name]) % RANDOM_SEED_MODULUS)


def extract_primary_metric(metric_dict):
    """
    Extract the primary metric value from an evaluation dictionary.
    """
    if "primary_metric_value" not in metric_dict:
        raise ValueError("evaluate_model() must return 'primary_metric_value'.")
    return float(metric_dict["primary_metric_value"])


def compute_calibrated_pi(score, alpha, n_iter=60):
    """
    Build pi_i = min(1, c * score_i) with mean(pi) calibrated to alpha.
    """
    score = np.maximum(np.asarray(score, dtype=float), 0.0)

    if alpha <= 0.0:
        return np.zeros_like(score), 0.0
    if alpha >= 1.0:
        return np.ones_like(score), np.inf

    low = 0.0
    high = 1.0
    for _ in range(n_iter):
        avg_pi = np.mean(np.minimum(1.0, high * score))
        if avg_pi >= alpha:
            break
        high *= 2.0

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        avg_pi = np.mean(np.minimum(1.0, mid * score))
        if avg_pi < alpha:
            low = mid
        else:
            high = mid

    pi = np.minimum(1.0, high * score)
    return pi, high


def append_result(values, value):
    """
    Append a float value to a result list.
    """
    values.append(float(value))


def build_misspecified_summary_text(
    active_model_module,
    config,
    metric_name,
    n_train,
    n_test,
    x_dim,
    n_candidates,
    alpha_target,
    method_summary,
):
    """
    Build one combined summary for all misspecified methods.
    """
    lines = []
    lines.append("=== Summary over reps ===")
    lines.append(f"active model module       : {active_model_module}")
    lines.append(f"p (feature dim)           : {x_dim}")
    lines.append(f"reps                      : {config['pipeline']['reps']}")
    lines.append(f"n_real                    : {n_train}")
    lines.append(f"n_test                    : {n_test}")
    lines.append(f"M_cand                    : {n_candidates}")
    lines.append(f"target alpha              : {alpha_target}")
    lines.append(f"sigma (noise std)         : {config['data']['noise_std']}")

    true_beta_full = evaluation.try_get_true_line_from_config(config)
    if true_beta_full is not None:
        lines.append(f"beta_true (intercept,slope): {true_beta_full.tolist()}")

    lines.append("")
    for label, values in method_summary.items():
        if values["alpha_hat"] is not None:
            mean_value, se_value = evaluation.mean_se(values["alpha_hat"])
            lines.append(f"accept rate ({label})     : {mean_value:.4f} +/- {se_value:.4f}")

    lines.append("")
    for label, values in method_summary.items():
        if values["eta"] is not None:
            mean_value, se_value = evaluation.mean_se(values["eta"])
            lines.append(f"eta_trace = tr(Vp^-1 Vw)/d ({label}): {mean_value:.4f} +/- {se_value:.4f}")

    lines.append("")
    for label, values in method_summary.items():
        lines.append(f"trace(Cov(beta_hat)) ({label}): {values['trace_cov']:.6f}")

    lines.append("")
    if metric_name.lower() == "mse":
        metric_label = "MSE"
    else:
        metric_label = metric_name

    for label, values in method_summary.items():
        mean_value, se_value = evaluation.mean_se(values["primary"])
        lines.append(f"test {metric_label} ({label}): {mean_value:.4f} +/- {se_value:.4f}")

    return "\n".join(lines)


def save_extended_results_details(path, results, metric_name):
    """
    Save per-repetition details for the extra misspecified baselines.
    """
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "rep",
            "alpha_hat_ratio_only",
            "alpha_hat_influence_only",
            "eta_ratio_only",
            "eta_influence_only",
            f"{metric_name}_ratio_only",
            f"{metric_name}_influence_only",
        ])

        for result in results:
            writer.writerow([
                result["rep"],
                f"{result['alpha_hat_ratio_only']:.8f}",
                f"{result['alpha_hat_influence_only']:.8f}",
                f"{result['eta_ratio_only']:.8f}",
                f"{result['eta_influence_only']:.8f}",
                f"{result['primary_ratio_only']:.8f}",
                f"{result['primary_influence_only']:.8f}",
            ])


def as_vector(value, x_dim):
    """
    Convert scalar or list config values to a length-x_dim vector.
    """
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(x_dim, float(array), dtype=float)
    return array


def get_linear_gaussian_params(section, x_dim):
    """
    Extract independent-X linear Gaussian joint parameters.
    """
    x_mean = as_vector(section.get("x_mean", 0.0), x_dim)
    x_std = as_vector(section.get("x_std", 1.0), x_dim)
    beta = as_vector(section["beta"], x_dim)
    intercept = float(section["intercept"])
    noise_std = float(section["noise_std"])
    return x_mean, x_std, beta, intercept, noise_std


def simulate_linear_gaussian_joint(n_samples, section, x_dim, rng):
    """
    Simulate X ~ N(x_mean, diag(x_std^2)) and Y | X ~ N(intercept + X beta, noise_std^2).
    """
    x_mean, x_std, beta, intercept, noise_std = get_linear_gaussian_params(section, x_dim)
    x = x_mean.reshape(1, -1) + rng.randn(n_samples, x_dim) * x_std.reshape(1, -1)
    y = intercept + x @ beta + noise_std * rng.randn(n_samples)
    return x, y


def normal_logpdf_independent_x(x, x_mean, x_std):
    """
    Log-density for independent Gaussian X coordinates.
    """
    normalized = (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    log_density = -0.5 * (
        normalized ** 2
        + 2.0 * np.log(x_std.reshape(1, -1))
        + np.log(2.0 * np.pi)
    )
    return np.sum(log_density, axis=1)


def linear_gaussian_joint_logpdf(x, y, section, x_dim):
    """
    Log p(x, y) for the independent-X linear Gaussian joint model.
    """
    x_mean, x_std, beta, intercept, noise_std = get_linear_gaussian_params(section, x_dim)
    log_px = normal_logpdf_independent_x(x, x_mean, x_std)
    residual = (y - (intercept + x @ beta)) / noise_std
    log_py_given_x = -0.5 * (
        residual ** 2
        + 2.0 * np.log(noise_std)
        + np.log(2.0 * np.pi)
    )
    return log_px + log_py_given_x


def load_misspecified_real_data(config):
    """
    Load real data from the true p distribution.
    """
    data_config = config["data"]
    seed = int(config["seed"])
    rng = np.random.RandomState(seed)
    x_dim = int(data_config["x_dim"])
    n_train = int(data_config["n_train"])
    n_test = int(data_config["n_test"])

    train_x, train_y = simulate_linear_gaussian_joint(n_train, data_config, x_dim, rng)
    test_x, test_y = simulate_linear_gaussian_joint(n_test, data_config, x_dim, rng)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }


def fit_misspecified_generator(train_x, train_y, config):
    """
    Prepare the synthetic q distribution from config.
    """
    return {
        "x_dim": int(config["data"]["x_dim"]),
        "candidate_config": copy.deepcopy(config["synthetic_candidate"]),
        "generator_mode": "misspecified_linear_gaussian",
    }


def generate_misspecified_candidates(generator_state, n_candidates, rng, config):
    """
    Generate candidates from the synthetic q distribution.
    """
    return simulate_linear_gaussian_joint(
        n_samples=n_candidates,
        section=generator_state["candidate_config"],
        x_dim=int(generator_state["x_dim"]),
        rng=rng,
    )


def fit_misspecified_ratio_model(train_x, train_y, generator_state, rng, config):
    """
    Prepare oracle p/q ratio parameters.
    """
    return {
        "x_dim": int(config["data"]["x_dim"]),
        "p_config": copy.deepcopy(config["data"]),
        "q_config": copy.deepcopy(config["synthetic_candidate"]),
        "clip_log_ratio": float(config["ratio"]["clip_log_ratio"]),
        "ratio_mode": "oracle_misspecified_gaussian",
    }


def compute_misspecified_log_ratio(x, y, ratio_state, config):
    """
    Compute log p(x, y) - log q(x, y) under the configured Gaussian joints.
    """
    x_dim = int(ratio_state["x_dim"])
    log_p = linear_gaussian_joint_logpdf(x, y, ratio_state["p_config"], x_dim)
    log_q = linear_gaussian_joint_logpdf(x, y, ratio_state["q_config"], x_dim)
    log_ratio = log_p - log_q
    clip_value = float(ratio_state["clip_log_ratio"])
    return np.clip(log_ratio, -clip_value, clip_value)


def run(config_path, output_dir_override=None):
    """
    Run the full experiment.
    """
    config = build_full_config(config_path, output_dir_override)
    active_model_module = config["active_model_module"]
    model_module = load_model_module(config)

    load_real_data = load_misspecified_real_data
    fit_reference_model = model_module.fit_reference_model
    compute_psi = model_module.compute_psi
    fit_generator = fit_misspecified_generator
    generate_candidates = generate_misspecified_candidates
    fit_ratio_model = fit_misspecified_ratio_model
    compute_log_ratio = compute_misspecified_log_ratio
    fit_final_model = model_module.fit_final_model
    evaluate_model = model_module.evaluate_model

    output_dir = output_dir_override or evaluation.create_timestamp_output_dir(
        base_dir=config["output_base_dir"],
        experiment_name=config["experiment_name"],
    )
    evaluation.save_json(
        path=f"{output_dir}/config.json",
        obj=config,
    )

    base_seed = int(config["seed"])
    reps = int(config["pipeline"]["reps"])
    n_candidates = int(config["pipeline"]["n_candidates"])
    alpha = float(config["pipeline"]["alpha"])
    rep_seed_stride = int(config["pipeline"]["rep_seed_stride"])
    vp_ridge = float(config["pipeline"]["vp_ridge"])

    # ---------------------------------------------------------------
    # 1. Load a fixed test set once.
    # ---------------------------------------------------------------
    fixed_data_config = copy_config_with_seed(config, stream_seed(base_seed, "fixed_data"))
    fixed_data = load_real_data(fixed_data_config)
    validate_loaded_data(fixed_data)

    test_x = fixed_data["test_x"]
    test_y = fixed_data["test_y"]

    results = []

    primary_real_list = []
    primary_uniform_list = []
    primary_theory_list = []
    primary_ratio_only_list = []
    primary_influence_only_list = []

    alpha_hat_uniform_list = []
    alpha_hat_theory_list = []
    alpha_hat_ratio_only_list = []
    alpha_hat_influence_only_list = []

    eta_uniform_list = []
    eta_theory_list = []
    eta_ratio_only_list = []
    eta_influence_only_list = []

    param_real_list = []
    param_uniform_list = []
    param_theory_list = []
    param_ratio_only_list = []
    param_influence_only_list = []

    metric_name = None
    plot_bundle = None

    # ---------------------------------------------------------------
    # 2. Run the experiment over repetitions.
    # ---------------------------------------------------------------
    for rep in range(reps):

        # Set independent random streams for this repetition.
        rep_seed = base_seed + rep_seed_stride * rep
        rep_config = copy_config_with_seed(config, stream_seed(rep_seed, "real_data"))
        rng_ratio = np.random.RandomState(stream_seed(rep_seed, "ratio_model"))
        rng_candidate = np.random.RandomState(stream_seed(rep_seed, "candidate"))
        rng_sample_theory = np.random.RandomState(stream_seed(rep_seed, "sample_theory"))
        rng_sample_uniform = np.random.RandomState(stream_seed(rep_seed, "sample_uniform"))
        rng_sample_ratio_only = np.random.RandomState(stream_seed(rep_seed, "sample_ratio_only"))
        rng_sample_influence_only = np.random.RandomState(stream_seed(rep_seed, "sample_influence_only"))

        # Training data
        rep_data = load_real_data(rep_config)
        validate_loaded_data(rep_data)

        train_x = rep_data["train_x"]
        train_y = rep_data["train_y"]

        # Fit the model
        reference_model = fit_reference_model(train_x, train_y, rep_config)

        # Estimate V_p
        vp = estimate_vp(
            train_x,
            train_y,
            reference_model,
            rep_config,
            compute_psi_fn=compute_psi,
        )
        vp_inv = invert_positive_matrix(vp, vp_ridge)
        psi_dim = vp.shape[0]

        # Prepare generating synthetic data
        generator_state = fit_generator(train_x, train_y, rep_config)
        ratio_state = fit_ratio_model(train_x, train_y, generator_state, rng_ratio, rep_config)

        cand_x, cand_y = generate_candidates(generator_state, n_candidates, rng_candidate, rep_config)
        validate_xy(cand_x, cand_y, "candidate")

        theory_stats = compute_theory_statistics(
            x=cand_x,
            y=cand_y,
            reference_model=reference_model,
            ratio_state=ratio_state,
            vp_inv=vp_inv,
            config=rep_config,
            compute_log_ratio_fn=compute_log_ratio,
            compute_psi_fn=compute_psi,
        )

        # Compute calibrated acceptance probabilities.
        theory_pi, theory_c = compute_calibrated_pi(
            score=theory_stats["score"],
            alpha=alpha,
        )

        theory_sample = sample_with_pi(
            x=cand_x,
            y=cand_y,
            pi=theory_pi,
            ratio=theory_stats["ratio"],
            alpha=alpha,
            rng=rng_sample_theory,
        )

        uniform_pi = build_uniform_pi(
            n_candidates=n_candidates,
            alpha=alpha,
        )

        uniform_sample = sample_with_pi(
            x=cand_x,
            y=cand_y,
            pi=uniform_pi,
            ratio=theory_stats["ratio"],
            alpha=alpha,
            rng=rng_sample_uniform,
        )

        ratio_only_pi, ratio_only_c = compute_calibrated_pi(
            score=theory_stats["ratio"],
            alpha=alpha,
        )

        ratio_only_sample = sample_with_pi(
            x=cand_x,
            y=cand_y,
            pi=ratio_only_pi,
            ratio=theory_stats["ratio"],
            alpha=alpha,
            rng=rng_sample_ratio_only,
        )

        influence_only_pi, influence_only_c = compute_calibrated_pi(
            score=theory_stats["influence"],
            alpha=alpha,
        )

        influence_only_sample = sample_with_pi(
            x=cand_x,
            y=cand_y,
            pi=influence_only_pi,
            ratio=theory_stats["ratio"],
            alpha=alpha,
            rng=rng_sample_influence_only,
        )

        # Estimate V_w
        vw_uniform = estimate_vw_from_accepted(
            accepted_x=uniform_sample["x"],
            accepted_y=uniform_sample["y"],
            accepted_weight=uniform_sample["weight"],
            reference_model=reference_model,
            config=rep_config,
            psi_dim=psi_dim,
            compute_psi_fn=compute_psi,
        )

        vw_theory = estimate_vw_from_accepted(
            accepted_x=theory_sample["x"],
            accepted_y=theory_sample["y"],
            accepted_weight=theory_sample["weight"],
            reference_model=reference_model,
            config=rep_config,
            psi_dim=psi_dim,
            compute_psi_fn=compute_psi,
        )

        vw_ratio_only = estimate_vw_from_accepted(
            accepted_x=ratio_only_sample["x"],
            accepted_y=ratio_only_sample["y"],
            accepted_weight=ratio_only_sample["weight"],
            reference_model=reference_model,
            config=rep_config,
            psi_dim=psi_dim,
            compute_psi_fn=compute_psi,
        )

        vw_influence_only = estimate_vw_from_accepted(
            accepted_x=influence_only_sample["x"],
            accepted_y=influence_only_sample["y"],
            accepted_weight=influence_only_sample["weight"],
            reference_model=reference_model,
            config=rep_config,
            psi_dim=psi_dim,
            compute_psi_fn=compute_psi,
        )

        if np.all(np.isfinite(vw_uniform)):
            eta_uniform = float(np.trace(vp_inv @ vw_uniform) / psi_dim)
        else:
            eta_uniform = float("nan")

        if np.all(np.isfinite(vw_theory)):
            eta_theory = float(np.trace(vp_inv @ vw_theory) / psi_dim)
        else:
            eta_theory = float("nan")

        if np.all(np.isfinite(vw_ratio_only)):
            eta_ratio_only = float(np.trace(vp_inv @ vw_ratio_only) / psi_dim)
        else:
            eta_ratio_only = float("nan")

        if np.all(np.isfinite(vw_influence_only)):
            eta_influence_only = float(np.trace(vp_inv @ vw_influence_only) / psi_dim)
        else:
            eta_influence_only = float("nan")

        real_model = fit_final_model(train_x, train_y, None, rep_config)
        real_metrics = evaluate_model(real_model, test_x, test_y, rep_config)

        uniform_model = fit_augmented_model(
            real_x=train_x,
            real_y=train_y,
            synth_x=uniform_sample["x"],
            synth_y=uniform_sample["y"],
            synth_weight=uniform_sample["weight"],
            config=rep_config,
            fit_final_model_fn=fit_final_model,
        )
        uniform_metrics = evaluate_model(uniform_model, test_x, test_y, rep_config)

        theory_model = fit_augmented_model(
            real_x=train_x,
            real_y=train_y,
            synth_x=theory_sample["x"],
            synth_y=theory_sample["y"],
            synth_weight=theory_sample["weight"],
            config=rep_config,
            fit_final_model_fn=fit_final_model,
        )
        theory_metrics = evaluate_model(theory_model, test_x, test_y, rep_config)

        ratio_only_model = fit_augmented_model(
            real_x=train_x,
            real_y=train_y,
            synth_x=ratio_only_sample["x"],
            synth_y=ratio_only_sample["y"],
            synth_weight=ratio_only_sample["weight"],
            config=rep_config,
            fit_final_model_fn=fit_final_model,
        )
        ratio_only_metrics = evaluate_model(ratio_only_model, test_x, test_y, rep_config)

        influence_only_model = fit_augmented_model(
            real_x=train_x,
            real_y=train_y,
            synth_x=influence_only_sample["x"],
            synth_y=influence_only_sample["y"],
            synth_weight=influence_only_sample["weight"],
            config=rep_config,
            fit_final_model_fn=fit_final_model,
        )
        influence_only_metrics = evaluate_model(influence_only_model, test_x, test_y, rep_config)

        primary_real = extract_primary_metric(real_metrics)
        primary_uniform = extract_primary_metric(uniform_metrics)
        primary_theory = extract_primary_metric(theory_metrics)
        primary_ratio_only = extract_primary_metric(ratio_only_metrics)
        primary_influence_only = extract_primary_metric(influence_only_metrics)

        if metric_name is None:
            metric_name = real_metrics.get("primary_metric_name", "primary_metric")

        param_real = evaluation.try_extract_parameter_vector(real_model)
        param_uniform = evaluation.try_extract_parameter_vector(uniform_model)
        param_theory = evaluation.try_extract_parameter_vector(theory_model)
        param_ratio_only = evaluation.try_extract_parameter_vector(ratio_only_model)
        param_influence_only = evaluation.try_extract_parameter_vector(influence_only_model)

        results.append({
            "rep": rep,
            "alpha_hat_u": float(uniform_sample["accept_rate"]),
            "alpha_hat_t": float(theory_sample["accept_rate"]),
            "alpha_hat_ratio_only": float(ratio_only_sample["accept_rate"]),
            "alpha_hat_influence_only": float(influence_only_sample["accept_rate"]),
            "eta_u": eta_uniform,
            "eta_t": eta_theory,
            "eta_ratio_only": eta_ratio_only,
            "eta_influence_only": eta_influence_only,
            "primary_r": primary_real,
            "primary_u": primary_uniform,
            "primary_t": primary_theory,
            "primary_ratio_only": primary_ratio_only,
            "primary_influence_only": primary_influence_only,
            "theory_c": theory_c,
            "ratio_only_c": ratio_only_c,
            "influence_only_c": influence_only_c,
        })

        primary_real_list.append(primary_real)
        primary_uniform_list.append(primary_uniform)
        primary_theory_list.append(primary_theory)
        primary_ratio_only_list.append(primary_ratio_only)
        primary_influence_only_list.append(primary_influence_only)

        alpha_hat_uniform_list.append(float(uniform_sample["accept_rate"]))
        alpha_hat_theory_list.append(float(theory_sample["accept_rate"]))
        alpha_hat_ratio_only_list.append(float(ratio_only_sample["accept_rate"]))
        alpha_hat_influence_only_list.append(float(influence_only_sample["accept_rate"]))

        eta_uniform_list.append(eta_uniform)
        eta_theory_list.append(eta_theory)
        eta_ratio_only_list.append(eta_ratio_only)
        eta_influence_only_list.append(eta_influence_only)

        param_real_list.append(param_real)
        param_uniform_list.append(param_uniform)
        param_theory_list.append(param_theory)
        param_ratio_only_list.append(param_ratio_only)
        param_influence_only_list.append(param_influence_only)

        if rep == 0:
            plot_bundle = {
                "real_x": train_x,
                "real_y": train_y,
                "cand_x": cand_x,
                "cand_y": cand_y,
                "accepted_uniform": uniform_sample["accepted_mask"],
                "accepted_theory": theory_sample["accepted_mask"],
                "accepted_ratio_only": ratio_only_sample["accepted_mask"],
                "accepted_influence_only": influence_only_sample["accepted_mask"],
                "log_ratio": theory_stats["log_ratio"],
                "influence": theory_stats["influence"],
                "theory_pi": theory_pi,
                "ratio_only_pi": ratio_only_pi,
                "influence_only_pi": influence_only_pi,
                "score": theory_stats["score"],
            }

    # ---------------------------------------------------------------
    # 3. Aggregate repetition-level results.
    # ---------------------------------------------------------------
    stacked_real = evaluation.stack_parameter_vectors(param_real_list)
    stacked_uniform = evaluation.stack_parameter_vectors(param_uniform_list)
    stacked_theory = evaluation.stack_parameter_vectors(param_theory_list)
    stacked_ratio_only = evaluation.stack_parameter_vectors(param_ratio_only_list)
    stacked_influence_only = evaluation.stack_parameter_vectors(param_influence_only_list)

    trace_cov_real = evaluation.trace_covariance_of_vectors(stacked_real)
    trace_cov_uniform = evaluation.trace_covariance_of_vectors(stacked_uniform)
    trace_cov_theory = evaluation.trace_covariance_of_vectors(stacked_theory)
    trace_cov_ratio_only = evaluation.trace_covariance_of_vectors(stacked_ratio_only)
    trace_cov_influence_only = evaluation.trace_covariance_of_vectors(stacked_influence_only)

    method_summary = {
        "real only": {
            "primary": primary_real_list,
            "alpha_hat": None,
            "eta": None,
            "trace_cov": trace_cov_real,
        },
        "uniform": {
            "primary": primary_uniform_list,
            "alpha_hat": alpha_hat_uniform_list,
            "eta": eta_uniform_list,
            "trace_cov": trace_cov_uniform,
        },
        "density ratio only": {
            "primary": primary_ratio_only_list,
            "alpha_hat": alpha_hat_ratio_only_list,
            "eta": eta_ratio_only_list,
            "trace_cov": trace_cov_ratio_only,
        },
        "influence only": {
            "primary": primary_influence_only_list,
            "alpha_hat": alpha_hat_influence_only_list,
            "eta": eta_influence_only_list,
            "trace_cov": trace_cov_influence_only,
        },
        "Real-Syn": {
            "primary": primary_theory_list,
            "alpha_hat": alpha_hat_theory_list,
            "eta": eta_theory_list,
            "trace_cov": trace_cov_theory,
        },
    }
    # Save full-precision values in addition to the compact text tables.
    from common.io import write_csv
    vectors = {
        "real only": stacked_real, "uniform": stacked_uniform,
        "density ratio only": stacked_ratio_only,
        "influence only": stacked_influence_only, "Real-Syn": stacked_theory,
    }
    summary_rows, metric_rows, coefficient_rows = [], [], []
    for method, values in method_summary.items():
        mean, se = evaluation.mean_se(values["primary"])
        eta_mean, eta_se = (
            evaluation.mean_se(values["eta"])
            if values["eta"] is not None else (None, None)
        )
        summary_rows.append({
            "method": method, "repetitions": reps,
            "mean_test_mse": mean, "mc_se_test_mse": se,
            "coefficient_covariance_trace": values["trace_cov"],
            "mean_weighted_score_diagnostic": eta_mean,
            "mc_se_weighted_score_diagnostic": eta_se,
        })
        for rep in range(reps):
            metric_rows.append({
                "rep": rep, "method": method,
                "test_mse": values["primary"][rep],
                "accept_rate": (values["alpha_hat"][rep]
                                if values["alpha_hat"] is not None else None),
                "weighted_score_diagnostic": (values["eta"][rep]
                                              if values["eta"] is not None else None),
            })
            coefficient_rows.append({
                "rep": rep, "method": method,
                **{f"theta_{j}": float(v) for j, v in enumerate(vectors[method][rep])},
            })
    write_csv(f"{output_dir}/method_summary.csv", summary_rows)
    write_csv(f"{output_dir}/per_rep_metrics.csv", metric_rows)
    write_csv(f"{output_dir}/parameter_vectors.csv", coefficient_rows)
    summary_text = build_misspecified_summary_text(
        active_model_module=active_model_module,
        config=config,
        metric_name=metric_name,
        n_train=int(fixed_data["train_x"].shape[0]),
        n_test=int(fixed_data["test_x"].shape[0]),
        x_dim=int(fixed_data["train_x"].shape[1]),
        n_candidates=n_candidates,
        alpha_target=alpha,
        method_summary=method_summary,
    )

    # ---------------------------------------------------------------
    # 4. Save outputs.
    # ---------------------------------------------------------------
    evaluation.save_text(
        path=f"{output_dir}/results_summary.txt",
        text=summary_text,
    )

    evaluation.save_results_details(
        path=f"{output_dir}/results_details.txt",
        results=results,
        metric_name=metric_name if metric_name is not None else "primary_metric",
        stacked_real=stacked_real,
        stacked_uniform=stacked_uniform,
        stacked_theory=stacked_theory,
    )
    save_extended_results_details(
        path=f"{output_dir}/results_details_extra_baselines.txt",
        results=results,
        metric_name=metric_name,
    )

    evaluation.save_json(
        path=f"{output_dir}/summary_metrics.json",
        obj={
            "active_model_module": active_model_module,
            "metric_name": metric_name,
            "trace_cov_real": trace_cov_real,
            "trace_cov_uniform": trace_cov_uniform,
            "trace_cov_real_syn": trace_cov_theory,
            "trace_cov_ratio_only": trace_cov_ratio_only,
            "trace_cov_influence_only": trace_cov_influence_only,
        },
    )
    evaluation.save_json(
        path=f"{output_dir}/summary_metrics_extra_baselines.json",
        obj={
            "metric_name": metric_name,
            "density_ratio_only": {
                "trace_cov": trace_cov_ratio_only,
                "mean_accept_rate": float(np.mean(alpha_hat_ratio_only_list)),
                "mean_eta": float(np.mean(eta_ratio_only_list)),
                "mean_primary": float(np.mean(primary_ratio_only_list)),
            },
            "influence_only": {
                "trace_cov": trace_cov_influence_only,
                "mean_accept_rate": float(np.mean(alpha_hat_influence_only_list)),
                "mean_eta": float(np.mean(eta_influence_only_list)),
                "mean_primary": float(np.mean(primary_influence_only_list)),
            },
        },
    )

    if config["pipeline"]["save_candidate_csv"] and plot_bundle is not None:
        evaluation.save_candidate_table(
            path=f"{output_dir}/candidate_diagnostics_rep0.csv",
            x=plot_bundle["cand_x"],
            y=plot_bundle["cand_y"],
            log_ratio=plot_bundle["log_ratio"],
            influence=plot_bundle["influence"],
            score=plot_bundle["score"],
            theory_pi=plot_bundle["theory_pi"],
            theory_accept_mask=plot_bundle["accepted_theory"],
            uniform_accept_mask=plot_bundle["accepted_uniform"],
        )

    if config["pipeline"]["save_histograms"] and plot_bundle is not None:
        evaluation.plot_histogram(
            values=plot_bundle["log_ratio"],
            title="Histogram of log density ratio",
            xlabel="log ratio",
            output_path=f"{output_dir}/hist_log_ratio_rep0.png",
        )

        evaluation.plot_histogram(
            values=plot_bundle["influence"],
            title="Histogram of influence norm",
            xlabel="sqrt(psi^T Vp^{-1} psi)",
            output_path=f"{output_dir}/hist_influence_rep0.png",
        )

        evaluation.plot_histogram(
            values=plot_bundle["theory_pi"],
            title="Histogram of Real-Syn acceptance probability",
            xlabel="Real-Syn pi",
            output_path=f"{output_dir}/hist_real_syn_pi_rep0.png",
        )

    if config["pipeline"]["make_xy_band_plots"] and plot_bundle is not None:
        true_beta_full = evaluation.try_get_true_line_from_config(config)

        evaluation.plot_xy_with_band(
            real_x=plot_bundle["real_x"],
            real_y=plot_bundle["real_y"],
            cand_x=plot_bundle["cand_x"],
            cand_y=plot_bundle["cand_y"],
            accepted_mask=plot_bundle["accepted_theory"],
            stacked_parameter_vectors=stacked_theory,
            title="Real-Syn",
            output_path=f"{output_dir}/real_syn_xy_band.png",
            metric_name=metric_name,
            true_beta_full=true_beta_full,
        )

        evaluation.plot_xy_with_band(
            real_x=plot_bundle["real_x"],
            real_y=plot_bundle["real_y"],
            cand_x=plot_bundle["cand_x"],
            cand_y=plot_bundle["cand_y"],
            accepted_mask=plot_bundle["accepted_ratio_only"],
            stacked_parameter_vectors=stacked_ratio_only,
            title="Density Ratio Only",
            output_path=f"{output_dir}/ratio_only_xy_band.png",
            metric_name=metric_name,
            true_beta_full=true_beta_full,
        )

        evaluation.plot_xy_with_band(
            real_x=plot_bundle["real_x"],
            real_y=plot_bundle["real_y"],
            cand_x=plot_bundle["cand_x"],
            cand_y=plot_bundle["cand_y"],
            accepted_mask=plot_bundle["accepted_influence_only"],
            stacked_parameter_vectors=stacked_influence_only,
            title="Influence Only",
            output_path=f"{output_dir}/influence_only_xy_band.png",
            metric_name=metric_name,
            true_beta_full=true_beta_full,
        )

        evaluation.plot_xy_with_band(
            real_x=plot_bundle["real_x"],
            real_y=plot_bundle["real_y"],
            cand_x=plot_bundle["cand_x"],
            cand_y=plot_bundle["cand_y"],
            accepted_mask=plot_bundle["accepted_uniform"],
            stacked_parameter_vectors=stacked_uniform,
            title="Uniform",
            output_path=f"{output_dir}/uniform_xy_band.png",
            metric_name=metric_name,
            true_beta_full=true_beta_full,
        )

    # ---------------------------------------------------------------
    # 5. Print final summary.
    # ---------------------------------------------------------------
    print(summary_text)
    print("")
    print(f"Output directory: {output_dir}")
    print("Expected main files:")
    print("  - config.json")
    print("  - results_summary.txt")
    print("  - results_details.txt")
    print("  - results_details_extra_baselines.txt")
    print("  - summary_metrics.json")
    print("  - summary_metrics_extra_baselines.json")
    print("  - real_syn_xy_band.png")
    print("  - uniform_xy_band.png")
    print("  - ratio_only_xy_band.png")
    print("  - influence_only_xy_band.png")
    print("")
    return output_dir
