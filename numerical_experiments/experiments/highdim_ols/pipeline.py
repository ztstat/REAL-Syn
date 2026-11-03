"""
Main pipeline for simulation.
"""


from common.ols_pipeline import load_model_module, copy_config_with_seed, stream_seed, extract_primary_metric, compute_calibrated_pi, save_extended_results_details, linear_gaussian_joint_logpdf, load_misspecified_real_data, fit_misspecified_generator, generate_misspecified_candidates, fit_misspecified_ratio_model, compute_misspecified_log_ratio
import hashlib
import json
import os
import time
import numpy as np
from . import metrics as dimension_metrics
from common import evaluation
from .parameters import (
    make_dense_beta,
    make_proposal_beta,
    make_sparse_beta,
)
from common.pipeline_math import (
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


def hash_arrays(*arrays):
    """Return a deterministic SHA-256 digest for shared numeric arrays."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


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
            mean_value, se_value = dimension_metrics.mean_se(values["alpha_hat"])
            se_text = "N/A" if se_value is None else f"{se_value:.4f}"
            lines.append(
                f"accept rate ({label})     : {mean_value:.4f} +/- {se_text}"
            )

    lines.append("")
    for label, values in method_summary.items():
        if values["eta"] is not None:
            mean_value, se_value = dimension_metrics.mean_se(values["eta"])
            se_text = "N/A" if se_value is None else f"{se_value:.4f}"
            lines.append(
                f"eta_trace = tr(Vp^-1 Vw)/d ({label}): "
                f"{mean_value:.4f} +/- {se_text}"
            )

    lines.append("")
    for label, values in method_summary.items():
        trace_covariance = values["trace_cov"]
        if trace_covariance is None:
            lines.append(f"trace(Cov(beta_hat)) ({label}): N/A")
        else:
            lines.append(
                f"trace(Cov(beta_hat)) ({label}): {trace_covariance:.6f}"
            )

    lines.append("")
    if metric_name.lower() == "mse":
        metric_label = "MSE"
    else:
        metric_label = metric_name

    for label, values in method_summary.items():
        mean_value, se_value = dimension_metrics.mean_se(values["primary"])
        se_text = "N/A" if se_value is None else f"{se_value:.4f}"
        lines.append(
            f"test {metric_label} ({label}): {mean_value:.4f} +/- {se_text}"
        )

    return "\n".join(lines)


def run(config_path, output_dir_override=None, quick_repetitions=None):
    """
    Run the full experiment.
    """
    config = build_full_config(
        config_path=config_path,
        output_dir_override=output_dir_override,
    )

    planned_repetitions = int(config["pipeline"]["reps"])
    if planned_repetitions != 100:
        raise ValueError("Dimension-extension configurations must retain 100 repetitions.")

    experiment_arm = str(config["experiment_arm"])
    dimension = int(config["dimension"])
    if dimension != int(config["data"]["x_dim"]):
        raise ValueError("config['dimension'] must equal config['data']['x_dim'].")
    if dimension not in (10, 20):
        raise ValueError("Higher-dimensional OLS fixes dimension to d=10 or d=20.")

    n_real = int(config["data"]["n_train"])
    n_candidates = int(config["pipeline"]["n_candidates"])
    if n_real != 200:
        raise ValueError("Higher-dimensional OLS fixes n_real at 200.")
    if n_candidates != 150 * n_real:
        raise ValueError("n_candidates must equal 150 * n_real.")
    expected_accepted_count = float(config["expected_accepted_count"])
    expected_accepted_to_real_ratio = float(
        config["expected_accepted_to_real_ratio"]
    )
    if not np.isclose(expected_accepted_count, 15.0 * n_real):
        raise ValueError("expected_accepted_count must equal 15 * n_real.")
    if not np.isclose(expected_accepted_to_real_ratio, 15.0):
        raise ValueError("expected accepted-to-real ratio must equal 15.")

    if experiment_arm == "primary_sparse":
        expected_beta = make_sparse_beta(dimension)
    elif experiment_arm == "sensitivity_dense_norm_matched":
        expected_beta = make_dense_beta(dimension)
    else:
        raise ValueError(f"Unknown experiment arm: {experiment_arm}")

    target_beta = np.asarray(config["data"]["beta"], dtype=float)
    proposal_beta = np.asarray(config["synthetic_candidate"]["beta"], dtype=float)
    if target_beta.shape != (dimension,) or not np.allclose(target_beta, expected_beta):
        raise ValueError("Configured target beta does not match its declared construction.")
    if proposal_beta.shape != (dimension,) or not np.allclose(
        proposal_beta,
        make_proposal_beta(target_beta),
    ):
        raise ValueError("Configured proposal beta must equal 0.8 * target beta.")

    is_quick = quick_repetitions is not None
    if is_quick:
        if int(quick_repetitions) not in (1, 2):
            raise ValueError("Quick checks require one or two repetitions.")
        config["pipeline"]["reps"] = int(quick_repetitions)
    config["run_kind"] = "quick" if is_quick else "full"

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
    fixed_test_hash = hash_arrays(test_x, test_y)

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
    coefficient_records = []
    extended_metric_rows = []
    diagnostic_rows = []

    # ---------------------------------------------------------------
    # 2. Run the experiment over repetitions.
    # ---------------------------------------------------------------
    for rep in range(reps):
        rep_start_time = time.perf_counter()

        # Set independent random streams for this repetition.
        rep_seed = base_seed + rep_seed_stride * rep
        rep_stream_seeds = {
            name: stream_seed(rep_seed, name)
            for name in SEED_STREAM_OFFSETS
            if name != "fixed_data"
        }
        rep_config = copy_config_with_seed(config, rep_stream_seeds["real_data"])
        rng_ratio = np.random.RandomState(rep_stream_seeds["ratio_model"])
        rng_candidate = np.random.RandomState(rep_stream_seeds["candidate"])
        rng_sample_theory = np.random.RandomState(rep_stream_seeds["sample_theory"])
        rng_sample_uniform = np.random.RandomState(rep_stream_seeds["sample_uniform"])
        rng_sample_ratio_only = np.random.RandomState(rep_stream_seeds["sample_ratio_only"])
        rng_sample_influence_only = np.random.RandomState(
            rep_stream_seeds["sample_influence_only"]
        )

        # Training data
        rep_data = load_real_data(rep_config)
        validate_loaded_data(rep_data)

        train_x = rep_data["train_x"]
        train_y = rep_data["train_y"]
        real_sample_hash = hash_arrays(train_x, train_y)

        # Fit the model
        reference_model = fit_reference_model(train_x, train_y, rep_config)
        pilot_parameter = np.asarray(
            evaluation.try_extract_parameter_vector(reference_model),
            dtype=float,
        )
        if pilot_parameter.shape != (dimension + 1,):
            raise ValueError("Pilot parameter vector has the wrong dimension.")
        pilot_intercept_error = float(
            (pilot_parameter[0] - float(config["data"]["intercept"])) ** 2
        )
        pilot_slope_error = float(
            np.sum((pilot_parameter[1:] - target_beta) ** 2)
        )

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
        vp_diagnostics = dimension_metrics.vp_summary(vp, vp_ridge)

        # Prepare generating synthetic data
        generator_state = fit_generator(train_x, train_y, rep_config)
        ratio_state = fit_ratio_model(train_x, train_y, generator_state, rng_ratio, rep_config)

        cand_x, cand_y = generate_candidates(generator_state, n_candidates, rng_candidate, rep_config)
        validate_xy(cand_x, cand_y, "candidate")
        candidate_pool_hash = hash_arrays(cand_x, cand_y)

        raw_log_p = linear_gaussian_joint_logpdf(
            cand_x,
            cand_y,
            ratio_state["p_config"],
            int(ratio_state["x_dim"]),
        )
        raw_log_q = linear_gaussian_joint_logpdf(
            cand_x,
            cand_y,
            ratio_state["q_config"],
            int(ratio_state["x_dim"]),
        )
        raw_log_ratio = raw_log_p - raw_log_q
        ratio_clip_value = float(ratio_state["clip_log_ratio"])
        ratio_lower_clipping_count = int(
            np.count_nonzero(raw_log_ratio < -ratio_clip_value)
        )
        ratio_upper_clipping_count = int(
            np.count_nonzero(raw_log_ratio > ratio_clip_value)
        )

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
        pilot_hash = hash_arrays(pilot_parameter)
        vp_hash = hash_arrays(vp)
        ratio_hash = hash_arrays(theory_stats["ratio"])
        score_diagnostics = dimension_metrics.score_diagnostics(
            influence=theory_stats["influence"],
            acceptance_score=theory_stats["score"],
        )
        candidate_ratio_diagnostics = dimension_metrics.ratio_summary(
            theory_stats["ratio"]
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

        extension_method_bundle = {
            "real only": {
                "parameter": param_real,
                "primary": primary_real,
                "sample": None,
                "pi": None,
                "eta": None,
                "sample_seed": None,
            },
            "uniform": {
                "parameter": param_uniform,
                "primary": primary_uniform,
                "sample": uniform_sample,
                "pi": uniform_pi,
                "eta": eta_uniform,
                "sample_seed": rep_stream_seeds["sample_uniform"],
            },
            "density ratio only": {
                "parameter": param_ratio_only,
                "primary": primary_ratio_only,
                "sample": ratio_only_sample,
                "pi": ratio_only_pi,
                "eta": eta_ratio_only,
                "sample_seed": rep_stream_seeds["sample_ratio_only"],
            },
            "influence only": {
                "parameter": param_influence_only,
                "primary": primary_influence_only,
                "sample": influence_only_sample,
                "pi": influence_only_pi,
                "eta": eta_influence_only,
                "sample_seed": rep_stream_seeds["sample_influence_only"],
            },
            "Real-Syn": {
                "parameter": param_theory,
                "primary": primary_theory,
                "sample": theory_sample,
                "pi": theory_pi,
                "eta": eta_theory,
                "sample_seed": rep_stream_seeds["sample_theory"],
            },
        }

        for method in dimension_metrics.METHOD_ORDER:
            method_values = extension_method_bundle[method]
            beta_full = np.asarray(method_values["parameter"], dtype=float)
            coefficient_records.append({
                "rep": rep,
                "method": method,
                "beta_full": beta_full.copy(),
            })

            recovery = dimension_metrics.parameter_errors(
                beta_full=beta_full,
                target_beta=target_beta,
                experiment_arm=experiment_arm,
            )
            sample = method_values["sample"]
            if sample is None:
                accepted_count = None
                accept_rate = None
                final_weight_ess = None
                final_weight_relative_ess = None
                final_weight_diagnostics = {
                    "final_weight_p95": None,
                    "final_weight_p99": None,
                    "final_weight_maximum": None,
                }
                pi_saturation_rate = None
                pi_minimum = None
                pi_maximum = None
                synthetic_total_weight = None
            else:
                accepted_count = int(sample["x"].shape[0])
                accept_rate = float(sample["accept_rate"])
                final_weight_ess, final_weight_relative_ess = (
                    dimension_metrics.effective_sample_size(sample["weight"])
                )
                final_weight_diagnostics = dimension_metrics.final_weight_summary(
                    sample["weight"]
                )
                pi_saturation_rate = dimension_metrics.saturation_rate(
                    method_values["pi"]
                )
                pi_minimum = float(np.min(method_values["pi"]))
                pi_maximum = float(np.max(method_values["pi"]))
                synthetic_total_weight = float(np.sum(sample["weight"]))

            extended_metric_rows.append({
                "config_id": config["config_name"],
                "experiment_arm": experiment_arm,
                "dimension": dimension,
                "n_real": n_real,
                "n_candidates": n_candidates,
                "alpha": alpha,
                "rep": rep,
                "rep_seed": rep_seed,
                "method_sample_seed": method_values["sample_seed"],
                "method": method,
                "test_mse": float(method_values["primary"]),
                "excess_mse": float(method_values["primary"] - 0.25),
                "coefficient_covariance_diagnostic": method_values["eta"],
                "accepted_count": accepted_count,
                "accept_rate": accept_rate,
                "acceptance_probability_saturation_rate": pi_saturation_rate,
                "acceptance_probability_minimum": pi_minimum,
                "acceptance_probability_maximum": pi_maximum,
                "final_weight_ess": final_weight_ess,
                "final_weight_relative_ess": final_weight_relative_ess,
                "synthetic_total_weight": synthetic_total_weight,
                **final_weight_diagnostics,
                "parameter_squared_error": recovery["parameter_squared_error"],
                "active_error": recovery["active_error"],
                "nuisance_error": recovery["nuisance_error"],
            })

        nonfinite_arrays = [
            train_x,
            train_y,
            cand_x,
            cand_y,
            vp,
            vp_inv,
            raw_log_ratio,
            theory_stats["log_ratio"],
            theory_stats["ratio"],
            theory_stats["influence"],
            theory_stats["score"],
            theory_pi,
            uniform_pi,
            ratio_only_pi,
            influence_only_pi,
            theory_sample["weight"],
            uniform_sample["weight"],
            ratio_only_sample["weight"],
            influence_only_sample["weight"],
            param_real,
            param_uniform,
            param_theory,
            param_ratio_only,
            param_influence_only,
        ]
        nan_count, inf_count = dimension_metrics.count_nonfinite(nonfinite_arrays)
        rep_runtime_seconds = float(time.perf_counter() - rep_start_time)
        diagnostic_rows.append({
            "config_id": config["config_name"],
            "experiment_arm": experiment_arm,
            "dimension": dimension,
            "n_real": n_real,
            "n_candidates": n_candidates,
            "alpha": alpha,
            "rep": rep,
            "rep_seed": rep_seed,
            "fixed_data_seed": stream_seed(base_seed, "fixed_data"),
            "real_data_seed": rep_stream_seeds["real_data"],
            "ratio_model_seed": rep_stream_seeds["ratio_model"],
            "candidate_seed": rep_stream_seeds["candidate"],
            "sample_theory_seed": rep_stream_seeds["sample_theory"],
            "sample_uniform_seed": rep_stream_seeds["sample_uniform"],
            "sample_ratio_only_seed": rep_stream_seeds["sample_ratio_only"],
            "sample_influence_only_seed": rep_stream_seeds[
                "sample_influence_only"
            ],
            "fixed_test_hash": fixed_test_hash,
            "real_sample_hash": real_sample_hash,
            "candidate_pool_hash": candidate_pool_hash,
            "pilot_hash": pilot_hash,
            "vp_hash": vp_hash,
            "ratio_hash": ratio_hash,
            **candidate_ratio_diagnostics,
            **vp_diagnostics,
            **score_diagnostics,
            "pilot_slope_error": pilot_slope_error,
            "pilot_intercept_error": pilot_intercept_error,
            "vp_ridge_value": vp_ridge,
            "vp_inverse_failure_count": 0,
            "ratio_lower_clipping_count": ratio_lower_clipping_count,
            "ratio_upper_clipping_count": ratio_upper_clipping_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "solver_failure_count": 0,
            "runtime_seconds": rep_runtime_seconds,
        })

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

    if reps < 2:
        # A sample covariance is undefined for a one-repetition run.
        # Record N/A without changing the covariance computation.
        trace_cov_real = None
        trace_cov_uniform = None
        trace_cov_theory = None
        trace_cov_ratio_only = None
        trace_cov_influence_only = None
    else:
        trace_cov_real = evaluation.trace_covariance_of_vectors(stacked_real)
        trace_cov_uniform = evaluation.trace_covariance_of_vectors(stacked_uniform)
        trace_cov_theory = evaluation.trace_covariance_of_vectors(stacked_theory)
        trace_cov_ratio_only = evaluation.trace_covariance_of_vectors(stacked_ratio_only)
        trace_cov_influence_only = evaluation.trace_covariance_of_vectors(
            stacked_influence_only
        )

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

    dimension_metrics.write_coefficient_estimates(
        path=f"{output_dir}/coefficient_estimates.csv",
        records=coefficient_records,
        dimension=dimension,
    )
    dimension_metrics.write_dict_rows(
        path=f"{output_dir}/per_rep_metrics.csv",
        rows=extended_metric_rows,
    )
    dimension_metrics.write_dict_rows(
        path=f"{output_dir}/per_rep_diagnostics.csv",
        rows=diagnostic_rows,
    )

    parameter_summary = dimension_metrics.aggregate_parameter_summary(
        coefficient_records=coefficient_records,
        target_beta=target_beta,
        experiment_arm=experiment_arm,
    )
    trace_covariance_by_method = {
        "real only": trace_cov_real,
        "uniform": trace_cov_uniform,
        "density ratio only": trace_cov_ratio_only,
        "influence only": trace_cov_influence_only,
        "Real-Syn": trace_cov_theory,
    }

    extended_method_summary = {}
    for method in dimension_metrics.METHOD_ORDER:
        method_rows = [
            row for row in extended_metric_rows if row["method"] == method
        ]
        test_mse_mean, test_mse_se = dimension_metrics.mean_se(
            [row["test_mse"] for row in method_rows]
        )
        excess_mse_mean, excess_mse_se = dimension_metrics.mean_se(
            [row["excess_mse"] for row in method_rows]
        )

        accepted_counts = [
            float(row["accepted_count"])
            for row in method_rows
            if row["accepted_count"] is not None
        ]
        final_ess_values = [
            float(row["final_weight_ess"])
            for row in method_rows
            if row["final_weight_ess"] is not None
            and np.isfinite(row["final_weight_ess"])
        ]
        final_relative_ess_values = [
            float(row["final_weight_relative_ess"])
            for row in method_rows
            if row["final_weight_relative_ess"] is not None
            and np.isfinite(row["final_weight_relative_ess"])
        ]
        saturation_values = [
            float(row["acceptance_probability_saturation_rate"])
            for row in method_rows
            if row["acceptance_probability_saturation_rate"] is not None
        ]
        final_weight_p95_values = [
            float(row["final_weight_p95"])
            for row in method_rows
            if row["final_weight_p95"] is not None
        ]
        final_weight_p99_values = [
            float(row["final_weight_p99"])
            for row in method_rows
            if row["final_weight_p99"] is not None
        ]
        final_weight_maximum_values = [
            float(row["final_weight_maximum"])
            for row in method_rows
            if row["final_weight_maximum"] is not None
        ]

        extended_method_summary[method] = {
            "test_mse_mean": test_mse_mean,
            "test_mse_se": test_mse_se,
            "excess_mse_mean": excess_mse_mean,
            "excess_mse_se": excess_mse_se,
            "coefficient_covariance_trace": trace_covariance_by_method[method],
            "accepted_count_mean": (
                float(np.mean(accepted_counts)) if accepted_counts else None
            ),
            "final_weight_ess_mean": (
                float(np.mean(final_ess_values)) if final_ess_values else None
            ),
            "final_weight_relative_ess_mean": (
                float(np.mean(final_relative_ess_values))
                if final_relative_ess_values
                else None
            ),
            "final_weight_p95_mean": (
                float(np.mean(final_weight_p95_values))
                if final_weight_p95_values
                else None
            ),
            "final_weight_p99_mean": (
                float(np.mean(final_weight_p99_values))
                if final_weight_p99_values
                else None
            ),
            "final_weight_maximum_mean": (
                float(np.mean(final_weight_maximum_values))
                if final_weight_maximum_values
                else None
            ),
            "acceptance_probability_saturation_rate_mean": (
                float(np.mean(saturation_values)) if saturation_values else None
            ),
            **parameter_summary[method],
        }

    diagnostic_summary = {}
    for key in diagnostic_rows[0]:
        if key in {
            "config_id",
            "experiment_arm",
            "rep",
            "fixed_test_hash",
            "real_sample_hash",
            "candidate_pool_hash",
            "pilot_hash",
            "vp_hash",
            "ratio_hash",
        }:
            continue
        finite_values = [
            float(row[key])
            for row in diagnostic_rows
            if row[key] is not None and np.isfinite(row[key])
        ]
        diagnostic_summary[f"{key}_mean"] = (
            float(np.mean(finite_values)) if finite_values else None
        )

    dimension_metrics.save_json(
        path=f"{output_dir}/dimension_summary.json",
        obj={
            "run_kind": config["run_kind"],
            "config_name": config["config_name"],
            "experiment_arm": experiment_arm,
            "dimension": dimension,
            "n_real": n_real,
            "n_candidates": n_candidates,
            "d_over_n": float(dimension / n_real),
            "expected_accepted_count": expected_accepted_count,
            "expected_accepted_to_real_ratio": expected_accepted_to_real_ratio,
            "candidate_scaling_rule": "n_candidates = 150 * n_real",
            "planned_repetitions": planned_repetitions,
            "executed_repetitions": reps,
            "target_beta": target_beta.tolist(),
            "proposal_beta": proposal_beta.tolist(),
            "methods": extended_method_summary,
            "diagnostics": diagnostic_summary,
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
    print("  - coefficient_estimates.csv")
    print("  - per_rep_metrics.csv")
    print("  - per_rep_diagnostics.csv")
    print("  - dimension_summary.json")
    print("  - real_syn_xy_band.png")
    print("  - uniform_xy_band.png")
    print("  - ratio_only_xy_band.png")
    print("  - influence_only_xy_band.png")
    print("")
    return output_dir
