"""
Main pipeline for large-scale simulation with synthetic samples.

This file contains the model-independent algorithmic logic:
1. data validation
2. Vp estimation
3. influence norm computation
4. score computation
5. kappa calibration
6. theory pi computation
7. rejection sampling
8. synthetic sample weighting
9. repetition-level experiment loop
10. Summary/detail output generation

This standalone copy defaults to its bundled temporal SHARPs callbacks.
"""

import argparse
import copy
import importlib
import json
import math
import os
import numpy as np

from . import evaluation
from .temporal_artifact_data import (
    inverse_condition_values as _default_inverse_condition_values,
)


# ------------------------------------------------------------------
# Fixed repository default model module.
# Examples:
#   "user_module_linear_regression"
#   "user_module_logistic_regression"
# ------------------------------------------------------------------
DEFAULT_ACTIVE_MODEL_MODULE = f"{__package__}.user_module_temporal"
ACTIVE_MODEL_MODULE = DEFAULT_ACTIVE_MODEL_MODULE


# ------------------------------------------------------------------
# Generic pipeline configuration only.
# Model-specific defaults are loaded from the active user module and
# merged into this base configuration.
# ------------------------------------------------------------------
BASE_CONFIG = {
    "seed": 0,
    "output_base_dir": "var/outputs",
    "experiment_name": "synthetic_sample_experiment",

    "pipeline": {
        "reps": 20,
        "inner_reps": 1,
        "n_candidates": "pool_size",
        "alpha": 0.10,
        "vp_ridge": 1e-10,
        "save_candidate_csv": True,
        "save_histograms": True,
        "make_xy_band_plots": False,
        "compute_analytic_covariance": True,
    },
    "diagnostics": {
        "overlap_warning": {
            "enabled": True,
            "q_mean_r_min": 0.1,
            "q_mean_r_max": 10.0,
            "ess_syn_reweighted_min": 10.0,
        },
    },
}


# ------------------------------------------------------------------
# Dynamically load or switch the active model module.
# ------------------------------------------------------------------
MODEL_MODULE = None
inverse_condition_values = _default_inverse_condition_values


def configure_active_model_module(module_name):
    """
    Load one model callback module and bind the generic pipeline callbacks.

    The default remains `user_module_SHARPs`. Scheduler adapters can select a
    different callback module before building configs or running repetitions.
    """
    global ACTIVE_MODEL_MODULE
    global MODEL_MODULE
    global load_real_data
    global fit_reference_model
    global compute_psi
    global fit_generator
    global generate_candidates
    global fit_ratio_model
    global compute_log_ratio
    global fit_final_model
    global evaluate_model
    global estimate_parameter_covariance_trace
    global inverse_condition_values

    resolved_name = str(module_name).strip()
    if not resolved_name:
        raise ValueError("module_name must be non-empty.")

    model_module = importlib.import_module(resolved_name)
    required_callbacks = (
        "load_real_data",
        "fit_reference_model",
        "compute_psi",
        "fit_generator",
        "generate_candidates",
        "fit_ratio_model",
        "compute_log_ratio",
        "fit_final_model",
        "evaluate_model",
    )
    missing_callbacks = [
        name for name in required_callbacks if not hasattr(model_module, name)
    ]
    if missing_callbacks:
        raise ValueError(
            f"{resolved_name} is missing required callbacks: "
            + ", ".join(missing_callbacks)
        )

    ACTIVE_MODEL_MODULE = resolved_name
    MODEL_MODULE = model_module
    load_real_data = model_module.load_real_data
    fit_reference_model = model_module.fit_reference_model
    compute_psi = model_module.compute_psi
    fit_generator = model_module.fit_generator
    generate_candidates = model_module.generate_candidates
    fit_ratio_model = model_module.fit_ratio_model
    compute_log_ratio = model_module.compute_log_ratio
    fit_final_model = model_module.fit_final_model
    evaluate_model = model_module.evaluate_model
    estimate_parameter_covariance_trace = getattr(
        model_module,
        "estimate_parameter_covariance_trace",
        None,
    )
    inverse_condition_values = getattr(
        model_module,
        "inverse_condition_values",
        _default_inverse_condition_values,
    )
    return model_module


configure_active_model_module(ACTIVE_MODEL_MODULE)


EXTRA_SYNTHETIC_METHODS = [
    {
        "key": "dre_only",
        "suffix": "dre",
        "label": "DRE only",
        "score_key": "ratio",
        "pi_key": "dre_only_pi",
        "accepted_key": "accepted_dre_only",
    },
    {
        "key": "influence_only",
        "suffix": "if",
        "label": "influence only",
        "score_key": "influence",
        "pi_key": "influence_only_pi",
        "accepted_key": "accepted_influence_only",
    },
]


def merge_dicts(base_dict, override_dict):
    """
    Recursively merge two dictionaries.

    Values from override_dict take precedence over values from base_dict.
    """
    merged = copy.deepcopy(base_dict)

    for key, value in override_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def build_full_config():
    """
    Build the final configuration by merging:
        1. generic base config from this pipeline file
        2. model-specific default config from the active user module
    """
    if not hasattr(MODEL_MODULE, "get_default_model_config"):
        raise ValueError(
            f"{ACTIVE_MODEL_MODULE} must define get_default_model_config()."
        )

    model_default_config = MODEL_MODULE.get_default_model_config()
    full_config = merge_dicts(BASE_CONFIG, model_default_config)
    return full_config


def resolve_overlap_warning_config(config):
    """
    Resolve the DRE overlap-warning configuration with generic defaults.
    """
    diagnostics_config = config.get("diagnostics", {})
    if not isinstance(diagnostics_config, dict):
        return copy.deepcopy(BASE_CONFIG["diagnostics"]["overlap_warning"])

    overlap_warning_config = diagnostics_config.get("overlap_warning", {})
    if not isinstance(overlap_warning_config, dict):
        return copy.deepcopy(BASE_CONFIG["diagnostics"]["overlap_warning"])

    return merge_dicts(
        BASE_CONFIG["diagnostics"]["overlap_warning"],
        overlap_warning_config,
    )


def evaluate_overlap_warning(diagnostics, config, rep, outer_rep, inner_rep):
    """
    Flag obviously unhealthy DRE overlap diagnostics for one repetition.
    """
    warning_config = resolve_overlap_warning_config(config)
    q_mean_r_min = float(warning_config["q_mean_r_min"])
    q_mean_r_max = float(warning_config["q_mean_r_max"])
    ess_min = float(warning_config["ess_syn_reweighted_min"])

    has_q_mean_r = "q_mean_r" in diagnostics
    has_ess_syn_reweighted = "ess_syn_reweighted" in diagnostics
    q_mean_r = float(diagnostics.get("q_mean_r", np.nan))
    ess_syn_reweighted = float(diagnostics.get("ess_syn_reweighted", np.nan))

    reasons = []
    if bool(warning_config["enabled"]):
        if has_q_mean_r and not np.isfinite(q_mean_r):
            reasons.append("q_mean_r is non-finite")
        elif has_q_mean_r and q_mean_r < q_mean_r_min:
            reasons.append(
                f"q_mean_r={q_mean_r:.6g} < {q_mean_r_min:.6g}"
            )
        elif has_q_mean_r and q_mean_r > q_mean_r_max:
            reasons.append(
                f"q_mean_r={q_mean_r:.6g} > {q_mean_r_max:.6g}"
            )

        if has_ess_syn_reweighted and not np.isfinite(ess_syn_reweighted):
            reasons.append("ess_syn_reweighted is non-finite")
        elif has_ess_syn_reweighted and ess_syn_reweighted < ess_min:
            reasons.append(
                f"ess_syn_reweighted={ess_syn_reweighted:.6g} < {ess_min:.6g}"
            )

    triggered = len(reasons) > 0
    return {
        "enabled": bool(warning_config["enabled"]),
        "triggered": triggered,
        "rep": int(rep),
        "outer_rep": int(outer_rep),
        "inner_rep": int(inner_rep),
        "q_mean_r": q_mean_r,
        "ess_syn_reweighted": ess_syn_reweighted,
        "thresholds": {
            "q_mean_r_min": q_mean_r_min,
            "q_mean_r_max": q_mean_r_max,
            "ess_syn_reweighted_min": ess_min,
        },
        "reasons": reasons,
        "message": "; ".join(reasons),
    }


def emit_overlap_warning(overlap_warning):
    """
    Print a loud warning when the held-out DRE overlap diagnostics are poor.
    """
    if not overlap_warning.get("triggered", False):
        return

    print(
        "WARNING [DRE overlap]: "
        f"rep {int(overlap_warning['rep']):03d} "
        f"(outer={int(overlap_warning['outer_rep'])}, "
        f"inner={int(overlap_warning['inner_rep'])}) "
        f"{overlap_warning['message']}. "
        "Theory acceptance metrics may be unreliable for this rep.",
        flush=True,
    )


def build_overlap_warning_summary(dre_diagnostics_by_rep):
    """
    Aggregate rep-level overlap warnings into one run-level summary.
    """
    overlap_warnings = []
    for entry in dre_diagnostics_by_rep:
        overlap_warning = entry.get("overlap_warning")
        if overlap_warning is not None:
            overlap_warnings.append(overlap_warning)

    if len(overlap_warnings) == 0:
        return None

    thresholds = overlap_warnings[0].get("thresholds", {})
    q_mean_r_min = float(thresholds.get("q_mean_r_min", np.nan))
    q_mean_r_max = float(thresholds.get("q_mean_r_max", np.nan))
    ess_min = float(thresholds.get("ess_syn_reweighted_min", np.nan))

    triggered = [warning for warning in overlap_warnings if warning.get("triggered", False)]
    q_mean_r_below_min = [
        warning for warning in overlap_warnings
        if np.isfinite(float(warning.get("q_mean_r", np.nan)))
        and float(warning.get("q_mean_r", np.nan)) < q_mean_r_min
    ]
    q_mean_r_above_max = [
        warning for warning in overlap_warnings
        if np.isfinite(float(warning.get("q_mean_r", np.nan)))
        and float(warning.get("q_mean_r", np.nan)) > q_mean_r_max
    ]
    ess_below_min = [
        warning for warning in overlap_warnings
        if np.isfinite(float(warning.get("ess_syn_reweighted", np.nan)))
        and float(warning.get("ess_syn_reweighted", np.nan)) < ess_min
    ]

    finite_q_warnings = [
        warning for warning in overlap_warnings
        if np.isfinite(float(warning.get("q_mean_r", np.nan)))
    ]
    finite_ess_warnings = [
        warning for warning in overlap_warnings
        if np.isfinite(float(warning.get("ess_syn_reweighted", np.nan)))
    ]

    worst_q_mean_r = None
    if len(finite_q_warnings) > 0:
        worst_q_mean_r = min(
            finite_q_warnings,
            key=lambda warning: float(warning["q_mean_r"]),
        )

    worst_ess = None
    if len(finite_ess_warnings) > 0:
        worst_ess = min(
            finite_ess_warnings,
            key=lambda warning: float(warning["ess_syn_reweighted"]),
        )

    return {
        "enabled": bool(overlap_warnings[0].get("enabled", False)),
        "total_reps": int(len(overlap_warnings)),
        "triggered_reps": int(len(triggered)),
        "thresholds": {
            "q_mean_r_min": q_mean_r_min,
            "q_mean_r_max": q_mean_r_max,
            "ess_syn_reweighted_min": ess_min,
        },
        "q_mean_r_below_min_reps": int(len(q_mean_r_below_min)),
        "q_mean_r_above_max_reps": int(len(q_mean_r_above_max)),
        "ess_syn_reweighted_below_min_reps": int(len(ess_below_min)),
        "triggered_rep_indices": [int(warning["rep"]) for warning in triggered],
        "worst_q_mean_r": worst_q_mean_r,
        "worst_ess_syn_reweighted": worst_ess,
    }


def parse_cli_args(argv=None):
    """
    Parse optional command-line overrides for the pipeline run.
    """
    parser = argparse.ArgumentParser(description="Run the synthetic-sample pipeline.")
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override the number of outer repetitions to run locally.",
    )
    parser.add_argument(
        "--inner-reps",
        type=int,
        default=None,
        help="Override the number of inner DRE/sampling repetitions per outer split.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override the target synthetic acceptance rate in (0, 1).",
    )
    parser.add_argument(
        "--condition-normalization-mode",
        choices=["zscore", "log", "log1p"],
        default=None,
        help=(
            "Override the SHARPs regression condition normalization mode. "
            "Choices: zscore, log, log1p."
        ),
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config, cli_args):
    """
    Apply supported command-line overrides onto a merged config.
    """
    if cli_args.reps is not None:
        reps = int(cli_args.reps)
        if reps <= 0:
            raise ValueError("--reps must be positive.")
        config["pipeline"]["reps"] = reps

    if cli_args.inner_reps is not None:
        inner_reps = int(cli_args.inner_reps)
        if inner_reps <= 0:
            raise ValueError("--inner-reps must be positive.")
        config["pipeline"]["inner_reps"] = inner_reps

    if cli_args.alpha is not None:
        alpha = float(cli_args.alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("--alpha must lie strictly between 0 and 1.")
        config["pipeline"]["alpha"] = alpha

    if cli_args.condition_normalization_mode is not None:
        config["artifacts"]["condition_normalization_mode"] = str(
            cli_args.condition_normalization_mode
        )

    return config


def resolve_n_candidates(generator_state, config):
    """
    Resolve the repetition candidate budget from config and generator state.
    """
    configured_value = config["pipeline"]["n_candidates"]

    if isinstance(configured_value, str):
        if configured_value != "pool_size":
            raise ValueError(
                "config['pipeline']['n_candidates'] must be a positive integer "
                "or the string 'pool_size'."
            )
        return int(np.asarray(generator_state["q_prop_x"]).shape[0])

    n_candidates = int(configured_value)
    if n_candidates <= 0:
        raise ValueError("config['pipeline']['n_candidates'] must be positive.")
    return n_candidates


def unpack_generated_candidates(generated_candidates):
    """
    Normalize model-module candidate outputs to `(x, y, source_index_or_none)`.
    """
    if not isinstance(generated_candidates, (tuple, list)):
        raise TypeError("generate_candidates() must return a tuple or list.")

    if len(generated_candidates) == 2:
        cand_x, cand_y = generated_candidates
        candidate_source_index = None
    elif len(generated_candidates) == 3:
        cand_x, cand_y, candidate_source_index = generated_candidates
    else:
        raise ValueError(
            "generate_candidates() must return (cand_x, cand_y) "
            "or (cand_x, cand_y, candidate_source_index)."
        )

    cand_x = np.asarray(cand_x, dtype=float)
    cand_y = np.asarray(cand_y, dtype=float)

    if candidate_source_index is None:
        return cand_x, cand_y, None

    candidate_source_index = np.asarray(candidate_source_index, dtype=int)
    if candidate_source_index.ndim != 1:
        raise ValueError("candidate_source_index must be a 1D array.")
    if candidate_source_index.shape[0] != cand_x.shape[0]:
        raise ValueError("candidate_source_index must match the candidate row count.")

    return cand_x, cand_y, candidate_source_index


def get_output_group_for_model(active_model_module):
    """
    Map a model module name to the output subdirectory used for its runs.
    """
    if active_model_module in {
        "user_module_linear_regression",
        "user_module_SHARPs",
        "temporal_experiment.user_module_temporal",
    }:
        if active_model_module == "temporal_experiment.user_module_temporal":
            return "temporal_linear_regression"
        return "linear_regression"

    if active_model_module in {
        "user_module_logistic_regression",
        "user_module_SHARPs_logistic",
    }:
        return "logistic_regression"

    return active_model_module


def ensure_directory(path):
    """
    Create a directory if needed and return the same path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_figures_dir(output_dir):
    """
    Return the run-local directory used for figure outputs.
    """
    return ensure_directory(os.path.join(output_dir, "figures"))


def get_per_rep_figures_dir(figures_dir):
    """
    Return the subdirectory used for plots that vary by repetition.
    """
    return ensure_directory(os.path.join(figures_dir, "per_rep"))


def save_rep0_diagnostics(output_dir, figures_dir, plot_bundle, config):
    """
    Save rep-0 diagnostics that are not expected to vary across repetitions.
    """
    if plot_bundle is None:
        return

    if config["pipeline"]["save_candidate_csv"]:
        save_candidate_diagnostics_table(
            path=f"{output_dir}/candidate_diagnostics_rep0.csv",
            plot_bundle=plot_bundle,
        )

    if config["pipeline"]["save_histograms"]:
        evaluation.plot_histogram(
            values=plot_bundle["influence"],
            title="Histogram of influence norm",
            xlabel="sqrt(psi^T Vp^{-1} psi)",
            output_path=f"{figures_dir}/hist_influence_rep0.png",
        )


def save_per_rep_diagnostics(per_rep_figures_dir, plot_bundle, rep):
    """
    Save the rep-varying histogram and boxplot diagnostics for one repetition.
    """
    rep = int(rep)

    evaluation.plot_histogram(
        values=plot_bundle["log_ratio"],
        title="Histogram of log density ratio",
        xlabel="log ratio",
        output_path=f"{per_rep_figures_dir}/hist_log_ratio_rep{rep}.png",
    )

    evaluation.plot_histogram(
        values=plot_bundle["theory_pi"],
        title="Histogram of theory acceptance probability",
        xlabel="theory pi",
        output_path=f"{per_rep_figures_dir}/hist_theory_pi_rep{rep}.png",
        log_y_scale=True,
    )

    evaluation.plot_boxplot_by_binned_scalar(
        x_values=plot_bundle["theory_pi"],
        y_values=plot_bundle["cand_y_plot"],
        title="Synthetic log10(flux) by theory acceptance probability",
        xlabel="theory pi bin",
        ylabel=plot_bundle["cand_y_plot_label"],
        output_path=f"{per_rep_figures_dir}/box_theory_pi_vs_log10_flux_rep{rep}.png",
        n_bins=10,
    )


def save_candidate_diagnostics_table(path, plot_bundle):
    """
    Save a candidate-level diagnostics CSV for one repetition.
    """
    dre_only_pi = plot_bundle.get("dre_only_pi")
    dre_only_accept_mask = plot_bundle.get("accepted_dre_only")
    influence_only_pi = plot_bundle.get("influence_only_pi")
    influence_only_accept_mask = plot_bundle.get("accepted_influence_only")
    evaluation.save_candidate_table(
        path=path,
        x=plot_bundle["cand_x"],
        y=plot_bundle["cand_y"],
        log_ratio=plot_bundle["log_ratio"],
        influence=plot_bundle["influence"],
        score=plot_bundle["score"],
        theory_pi=plot_bundle["theory_pi"],
        theory_accept_mask=plot_bundle["accepted_theory"],
        uniform_accept_mask=plot_bundle["accepted_uniform"],
        dre_only_pi=dre_only_pi,
        dre_only_accept_mask=dre_only_accept_mask,
        influence_only_pi=influence_only_pi,
        influence_only_accept_mask=influence_only_accept_mask,
    )


def save_candidate_acceptance_table(path, plot_bundle):
    """
    Save a minimal candidate-acceptance CSV for one repetition.
    """
    candidate_source_index = plot_bundle.get("candidate_source_index")
    if candidate_source_index is None:
        return

    evaluation.save_candidate_acceptance_table(
        path=path,
        candidate_source_index=candidate_source_index,
        theory_accept_mask=plot_bundle["accepted_theory"],
        uniform_accept_mask=plot_bundle["accepted_uniform"],
        dre_only_accept_mask=plot_bundle.get("accepted_dre_only"),
        influence_only_accept_mask=plot_bundle.get("accepted_influence_only"),
    )


def get_candidate_acceptance_filename(rep):
    """
    Return the barebones acceptance-tracking filename for one repetition.
    """
    return f"candidate_acceptance_rep{int(rep)}.csv"


def save_acceptance_matrix_from_paths(
    output_path,
    rep_acceptance_paths,
    q_prop_pool_size,
    acceptance_key,
):
    """
    Build and save one top-level acceptance matrix from rep CSV paths.
    """
    if q_prop_pool_size is None:
        return False

    existence_flags = [os.path.exists(path) for path in rep_acceptance_paths]
    if not any(existence_flags):
        return False
    if not all(existence_flags):
        missing_paths = [
            path for path, exists in zip(rep_acceptance_paths, existence_flags)
            if not exists
        ]
        raise FileNotFoundError(
            "Missing rep acceptance CSVs required to build acceptance matrix: "
            + ", ".join(missing_paths)
        )

    rep_acceptance_entries = []
    for rep, path in enumerate(rep_acceptance_paths):
        acceptance_table = evaluation.load_candidate_acceptance_table(path)
        if acceptance_key not in acceptance_table:
            return False
        rep_acceptance_entries.append(
            {
                "rep_label": f"rep_{int(rep):03d}",
                "q_prop_index": acceptance_table["q_prop_index"],
                "accepted": acceptance_table[acceptance_key],
            }
        )

    evaluation.save_acceptance_matrix(
        path=output_path,
        q_prop_pool_size=q_prop_pool_size,
        rep_acceptance_entries=rep_acceptance_entries,
    )
    return True


def save_local_rep_outputs(rep_output_dir, plot_bundle, rep, config):
    """
    Save per-repetition metrics plots and rep-0 candidate diagnostics inside one rep directory.
    """
    if plot_bundle is None:
        return

    rep = int(rep)
    rep_figures_dir = ensure_directory(os.path.join(rep_output_dir, "figures"))

    if config["pipeline"]["save_candidate_csv"]:
        save_candidate_acceptance_table(
            path=os.path.join(rep_output_dir, get_candidate_acceptance_filename(rep)),
            plot_bundle=plot_bundle,
        )
        if rep == 0:
            save_candidate_diagnostics_table(
                path=os.path.join(rep_output_dir, f"candidate_diagnostics_rep{rep}.csv"),
                plot_bundle=plot_bundle,
            )

    if config["pipeline"]["save_histograms"]:
        evaluation.plot_histogram(
            values=plot_bundle["influence"],
            title="Histogram of influence norm",
            xlabel="sqrt(psi^T Vp^{-1} psi)",
            output_path=os.path.join(rep_figures_dir, f"hist_influence_rep{rep}.png"),
        )
        save_per_rep_diagnostics(rep_figures_dir, plot_bundle, rep)


def save_xy_band_inputs(path, plot_bundle):
    """
    Save the rep-local arrays needed to recreate XY band plots later.
    """
    if plot_bundle is None:
        return

    arrays = {
        "real_x": plot_bundle["real_x"],
        "real_y": plot_bundle["real_y"],
        "cand_x": plot_bundle["cand_x"],
        "cand_y": plot_bundle["cand_y"],
        "accepted_uniform": plot_bundle["accepted_uniform"],
        "accepted_theory": plot_bundle["accepted_theory"],
    }
    if plot_bundle.get("accepted_dre_only") is not None:
        arrays["accepted_dre_only"] = plot_bundle["accepted_dre_only"]
    if plot_bundle.get("accepted_influence_only") is not None:
        arrays["accepted_influence_only"] = plot_bundle["accepted_influence_only"]
    np.savez_compressed(path, **arrays)


def compute_relative_metric_delta(baseline_value, method_value):
    """
    Compute the relative difference `(method - baseline) / baseline`.
    """
    baseline_value = float(baseline_value)
    method_value = float(method_value)
    if abs(baseline_value) <= 1e-12:
        return float("nan")
    return float((method_value - baseline_value) / baseline_value)


def estimate_model_covariance_trace(model_state, x, y, sample_weight, config):
    """
    Estimate the analytic regression covariance trace for one fitted model.
    """
    if not bool(
        config.get("pipeline", {}).get(
            "compute_analytic_covariance",
            True,
        )
    ):
        return float("nan")
    if estimate_parameter_covariance_trace is None:
        return float("nan")
    return float(
        estimate_parameter_covariance_trace(
            model_state=model_state,
            x=x,
            y=y,
            sample_weight=sample_weight,
            config=config,
        )
    )


def build_augmented_dataset(real_x, real_y, synth_x, synth_y, synth_weight):
    """
    Build one downstream regression dataset with optional weighted synthetic rows.
    """
    if synth_x is None or synth_x.shape[0] == 0:
        return real_x, real_y, None

    x_all = np.vstack([real_x, synth_x])
    y_all = np.concatenate([real_y, synth_y])
    weights = np.concatenate([
        np.ones(real_x.shape[0], dtype=float),
        synth_weight.astype(float),
    ])
    return x_all, y_all, weights


def load_xy_band_inputs(path):
    """
    Load the rep-local arrays needed to recreate XY band plots later.
    """
    with np.load(path, allow_pickle=False) as data:
        bundle = {
            "real_x": data["real_x"],
            "real_y": data["real_y"],
            "cand_x": data["cand_x"],
            "cand_y": data["cand_y"],
            "accepted_uniform": data["accepted_uniform"].astype(bool),
            "accepted_theory": data["accepted_theory"].astype(bool),
        }
        if "accepted_dre_only" in data:
            accepted_dre_only = data["accepted_dre_only"]
            if accepted_dre_only.shape == bundle["accepted_theory"].shape:
                bundle["accepted_dre_only"] = accepted_dre_only.astype(bool)
        if "accepted_influence_only" in data:
            accepted_influence_only = data["accepted_influence_only"]
            if accepted_influence_only.shape == bundle["accepted_theory"].shape:
                bundle["accepted_influence_only"] = accepted_influence_only.astype(bool)
        return bundle


def validate_xy(x, y, name_prefix):
    """
    Validate that x and y have compatible shapes.

    Expected:
        x: 2D array of shape (n_samples, x_dim)
        y: 1D array of shape (n_samples,)
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name_prefix}_x must be a numpy array.")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"{name_prefix}_y must be a numpy array.")
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
    required_keys = ["train_x", "train_y", "test_x", "test_y"]
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"load_real_data() must return key '{key}'.")

    train_x = data_dict["train_x"]
    train_y = data_dict["train_y"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]

    validate_xy(train_x, train_y, "train")
    validate_xy(test_x, test_y, "test")

    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train_x and test_x must have the same feature dimension.")

    ratio_keys = {"ratio_train_x", "ratio_train_y"}
    present_ratio_keys = ratio_keys.intersection(data_dict)
    if present_ratio_keys and present_ratio_keys != ratio_keys:
        raise ValueError(
            "load_real_data() must return both ratio_train_x and "
            "ratio_train_y when either is provided."
        )
    if present_ratio_keys:
        ratio_train_x = data_dict["ratio_train_x"]
        ratio_train_y = data_dict["ratio_train_y"]
        validate_xy(ratio_train_x, ratio_train_y, "ratio_train")
        if ratio_train_x.shape[1] != train_x.shape[1]:
            raise ValueError(
                "ratio_train_x and train_x must have the same feature "
                "dimension."
            )


def copy_config_with_seed(config, seed):
    """
    Copy the configuration dictionary and override the split seed.
    """
    config_copy = copy.deepcopy(config)
    config_copy["seed"] = int(seed)
    return config_copy


def copy_config_with_run_seeds(config, split_seed, dre_seed):
    """
    Copy the configuration dictionary and override the split and DRE seeds.
    """
    config_copy = copy_config_with_seed(config, split_seed)
    config_copy["dre_seed"] = int(dre_seed)
    return config_copy


def load_summary_data(config):
    """
    Load and validate one seeded train/test split used for summary dimensions.
    """
    summary_data_config = copy_config_with_seed(config, int(config["seed"]))
    summary_data = load_real_data(summary_data_config)
    validate_loaded_data(summary_data)
    return summary_data


def derive_child_seeds(parent_seed, count):
    """
    Derive deterministic child seeds from one parent seed.
    """
    seed_sequence = np.random.SeedSequence(int(parent_seed))
    child_sequences = seed_sequence.spawn(int(count))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in child_sequences
    ]


def build_rep_plan(global_seed, outer_reps, inner_reps):
    """
    Build the flat run plan induced by nested outer and inner repetitions.
    """
    outer_reps = int(outer_reps)
    inner_reps = int(inner_reps)
    if outer_reps <= 0:
        raise ValueError("outer_reps must be positive.")
    if inner_reps <= 0:
        raise ValueError("inner_reps must be positive.")

    outer_seeds = derive_child_seeds(global_seed, outer_reps)
    rep_plan = []
    flat_rep = 0
    for outer_rep, split_seed in enumerate(outer_seeds):
        inner_seeds = derive_child_seeds(split_seed, inner_reps)
        for inner_rep, dre_seed in enumerate(inner_seeds):
            rep_plan.append(
                {
                    "rep": int(flat_rep),
                    "outer_rep": int(outer_rep),
                    "inner_rep": int(inner_rep),
                    "split_seed": int(split_seed),
                    "dre_seed": int(dre_seed),
                }
            )
            flat_rep += 1
    return rep_plan


def summarize_data_split(summary_data):
    """
    Extract seeded dataset dimensions used in aggregate summaries.
    """
    summary = {
        "n_train": int(summary_data["train_x"].shape[0]),
        "n_test": int(summary_data["test_x"].shape[0]),
        "x_dim": int(summary_data["train_x"].shape[1]),
    }
    if "ratio_train_x" in summary_data:
        summary["n_ratio_train"] = int(
            summary_data["ratio_train_x"].shape[0]
        )
    return summary


def build_candidate_plot_bundle(
    train_x,
    train_y,
    cand_x,
    cand_y,
    candidate_source_index,
    generator_state,
    theory_stats,
    theory_pi,
    uniform_sample,
    theory_sample,
    extra_method_plot_data=None,
):
    """
    Build the candidate diagnostics bundle used for plots and CSV outputs.
    """
    cand_y_plot = cand_y
    cand_y_plot_label = "y"

    if (
        "condition_log_mean" in generator_state
        and "condition_log_std" in generator_state
    ):
        cand_y_plot = (
            cand_y * float(generator_state["condition_log_std"])
            + float(generator_state["condition_log_mean"])
        )
        cand_y_plot_label = generator_state.get(
            "condition_plot_label",
            cand_y_plot_label,
        )
        condition_inverse_transform = generator_state.get(
            "condition_inverse_transform",
        )
        if condition_inverse_transform is not None:
            raw_flux = inverse_condition_values(
                cand_y_plot,
                condition_inverse_transform,
            )
            if np.all(raw_flux > 0.0):
                cand_y_plot = np.log10(raw_flux)
                cand_y_plot_label = "log10_flux"

    bundle = {
        "real_x": train_x,
        "real_y": train_y,
        "cand_x": cand_x,
        "cand_y": cand_y,
        "candidate_source_index": candidate_source_index,
        "cand_y_plot": cand_y_plot,
        "cand_y_plot_label": cand_y_plot_label,
        "accepted_uniform": uniform_sample["accepted_mask"],
        "accepted_theory": theory_sample["accepted_mask"],
        "log_ratio": theory_stats["log_ratio"],
        "influence": theory_stats["influence"],
        "theory_pi": theory_pi,
        "score": theory_stats["score"],
    }
    for method_data in extra_method_plot_data or []:
        bundle[method_data["pi_key"]] = method_data["pi"]
        bundle[method_data["accepted_key"]] = method_data["sample"]["accepted_mask"]
    return bundle


def run_single_rep(rep, outer_rep, inner_rep, split_seed, dre_seed, config):
    """
    Run one repetition of the pipeline and return its metrics and diagnostics.
    """
    rep = int(rep)
    outer_rep = int(outer_rep)
    inner_rep = int(inner_rep)
    split_seed = int(split_seed)
    dre_seed = int(dre_seed)

    rep_config = copy_config_with_run_seeds(
        config,
        split_seed=split_seed,
        dre_seed=dre_seed,
    )
    rng_rep = np.random.RandomState(dre_seed)

    rep_data = load_real_data(rep_config)
    validate_loaded_data(rep_data)

    train_x = rep_data["train_x"]
    train_y = rep_data["train_y"]
    test_x = rep_data["test_x"]
    test_y = rep_data["test_y"]
    ratio_train_x = rep_data.get("ratio_train_x", train_x)
    ratio_train_y = rep_data.get("ratio_train_y", train_y)

    reference_model = fit_reference_model(train_x, train_y, rep_config)

    vp = estimate_vp(train_x, train_y, reference_model, rep_config)
    vp_inv = invert_positive_matrix(vp, float(config["pipeline"]["vp_ridge"]))
    psi_dim = vp.shape[0]

    generator_state = fit_generator(train_x, train_y, rep_config)
    n_candidates = resolve_n_candidates(generator_state, rep_config)

    ratio_state = fit_ratio_model(
        ratio_train_x,
        ratio_train_y,
        generator_state,
        rng_rep,
        rep_config,
    )
    overlap_warning = evaluate_overlap_warning(
        diagnostics=ratio_state["diagnostics"],
        config=rep_config,
        rep=rep,
        outer_rep=outer_rep,
        inner_rep=inner_rep,
    )
    emit_overlap_warning(overlap_warning)

    cand_x, cand_y, candidate_source_index = unpack_generated_candidates(
        generate_candidates(generator_state, n_candidates, rng_rep, rep_config)
    )
    validate_xy(cand_x, cand_y, "candidate")

    theory_stats = compute_theory_statistics(
        x=cand_x,
        y=cand_y,
        reference_model=reference_model,
        ratio_state=ratio_state,
        vp_inv=vp_inv,
        config=rep_config,
    )

    theory_pi, theory_kappa = compute_theory_pi(
        score=theory_stats["score"],
        alpha=float(config["pipeline"]["alpha"]),
    )

    theory_sample = sample_with_pi(
        x=cand_x,
        y=cand_y,
        pi=theory_pi,
        ratio=theory_stats["ratio"],
        alpha=float(config["pipeline"]["alpha"]),
        rng=rng_rep,
    )

    uniform_sample = sample_with_pi(
        x=cand_x,
        y=cand_y,
        pi=build_uniform_pi(n_candidates=n_candidates, alpha=float(config["pipeline"]["alpha"])),
        ratio=theory_stats["ratio"],
        alpha=float(config["pipeline"]["alpha"]),
        rng=rng_rep,
    )
    extra_method_outputs = {}
    for method_spec in EXTRA_SYNTHETIC_METHODS:
        method_pi, method_kappa = compute_theory_pi(
            score=theory_stats[method_spec["score_key"]],
            alpha=float(config["pipeline"]["alpha"]),
        )
        method_sample = sample_with_pi(
            x=cand_x,
            y=cand_y,
            pi=method_pi,
            ratio=theory_stats["ratio"],
            alpha=float(config["pipeline"]["alpha"]),
            rng=rng_rep,
        )
        extra_method_outputs[method_spec["key"]] = {
            "spec": method_spec,
            "pi": method_pi,
            "kappa": method_kappa,
            "sample": method_sample,
        }

    vw_uniform = estimate_vw_from_accepted(
        accepted_x=uniform_sample["x"],
        accepted_y=uniform_sample["y"],
        accepted_weight=uniform_sample["weight"],
        reference_model=reference_model,
        config=rep_config,
        psi_dim=psi_dim,
    )
    vw_theory = estimate_vw_from_accepted(
        accepted_x=theory_sample["x"],
        accepted_y=theory_sample["y"],
        accepted_weight=theory_sample["weight"],
        reference_model=reference_model,
        config=rep_config,
        psi_dim=psi_dim,
    )

    if np.all(np.isfinite(vw_uniform)):
        eta_uniform = float(np.trace(vp_inv @ vw_uniform) / psi_dim)
    else:
        eta_uniform = float("nan")

    if np.all(np.isfinite(vw_theory)):
        eta_theory = float(np.trace(vp_inv @ vw_theory) / psi_dim)
    else:
        eta_theory = float("nan")

    for method_key, method_output in extra_method_outputs.items():
        method_sample = method_output["sample"]
        method_vw = estimate_vw_from_accepted(
            accepted_x=method_sample["x"],
            accepted_y=method_sample["y"],
            accepted_weight=method_sample["weight"],
            reference_model=reference_model,
            config=rep_config,
            psi_dim=psi_dim,
        )
        if np.all(np.isfinite(method_vw)):
            method_eta = float(np.trace(vp_inv @ method_vw) / psi_dim)
        else:
            method_eta = float("nan")
        extra_method_outputs[method_key]["vw"] = method_vw
        extra_method_outputs[method_key]["eta"] = method_eta

    real_model = fit_final_model(train_x, train_y, None, rep_config)
    analytic_cov_trace_real = estimate_model_covariance_trace(
        model_state=real_model,
        x=train_x,
        y=train_y,
        sample_weight=None,
        config=rep_config,
    )
    real_metrics = evaluate_model(real_model, test_x, test_y, rep_config)

    uniform_x_all, uniform_y_all, uniform_weights = build_augmented_dataset(
        real_x=train_x,
        real_y=train_y,
        synth_x=uniform_sample["x"],
        synth_y=uniform_sample["y"],
        synth_weight=uniform_sample["weight"],
    )
    uniform_model = fit_augmented_model(
        real_x=train_x,
        real_y=train_y,
        synth_x=uniform_sample["x"],
        synth_y=uniform_sample["y"],
        synth_weight=uniform_sample["weight"],
        config=rep_config,
    )
    analytic_cov_trace_uniform = estimate_model_covariance_trace(
        model_state=uniform_model,
        x=uniform_x_all,
        y=uniform_y_all,
        sample_weight=uniform_weights,
        config=rep_config,
    )
    uniform_metrics = evaluate_model(uniform_model, test_x, test_y, rep_config)

    theory_x_all, theory_y_all, theory_weights = build_augmented_dataset(
        real_x=train_x,
        real_y=train_y,
        synth_x=theory_sample["x"],
        synth_y=theory_sample["y"],
        synth_weight=theory_sample["weight"],
    )
    theory_model = fit_augmented_model(
        real_x=train_x,
        real_y=train_y,
        synth_x=theory_sample["x"],
        synth_y=theory_sample["y"],
        synth_weight=theory_sample["weight"],
        config=rep_config,
    )
    analytic_cov_trace_theory = estimate_model_covariance_trace(
        model_state=theory_model,
        x=theory_x_all,
        y=theory_y_all,
        sample_weight=theory_weights,
        config=rep_config,
    )
    theory_metrics = evaluate_model(theory_model, test_x, test_y, rep_config)

    for method_key, method_output in extra_method_outputs.items():
        method_sample = method_output["sample"]
        method_x_all, method_y_all, method_weights = build_augmented_dataset(
            real_x=train_x,
            real_y=train_y,
            synth_x=method_sample["x"],
            synth_y=method_sample["y"],
            synth_weight=method_sample["weight"],
        )
        method_model = fit_augmented_model(
            real_x=train_x,
            real_y=train_y,
            synth_x=method_sample["x"],
            synth_y=method_sample["y"],
            synth_weight=method_sample["weight"],
            config=rep_config,
        )
        method_analytic_cov_trace = estimate_model_covariance_trace(
            model_state=method_model,
            x=method_x_all,
            y=method_y_all,
            sample_weight=method_weights,
            config=rep_config,
        )
        method_metrics = evaluate_model(method_model, test_x, test_y, rep_config)
        extra_method_outputs[method_key]["model"] = method_model
        extra_method_outputs[method_key]["analytic_cov_trace"] = (
            method_analytic_cov_trace
        )
        extra_method_outputs[method_key]["metrics"] = method_metrics

    metric_name = real_metrics.get("primary_metric_name", "primary_metric")
    reported_metric_names = get_reported_metric_names(real_metrics)
    reported_test_set_metric_names = (
        get_reported_test_set_metric_names(real_metrics)
    )
    reported_metric_context = get_reported_metric_context(real_metrics)

    primary_real = extract_primary_metric(real_metrics)
    primary_uniform = extract_primary_metric(uniform_metrics)
    primary_theory = extract_primary_metric(theory_metrics)
    for method_output in extra_method_outputs.values():
        method_output["primary"] = extract_primary_metric(method_output["metrics"])
    delta_mse_uniform_rel = float("nan")
    delta_mse_theory_rel = float("nan")
    delta_mse_extra_rel = {
        method_key: float("nan")
        for method_key in extra_method_outputs
    }
    if metric_name is not None and metric_name.lower() == "mse":
        delta_mse_uniform_rel = compute_relative_metric_delta(
            primary_real,
            primary_uniform,
        )
        delta_mse_theory_rel = compute_relative_metric_delta(
            primary_real,
            primary_theory,
        )
        for method_key, method_output in extra_method_outputs.items():
            delta_mse_extra_rel[method_key] = compute_relative_metric_delta(
                primary_real,
                method_output["primary"],
            )

    rep_result = {
        "rep": rep,
        "outer_rep": outer_rep,
        "inner_rep": inner_rep,
        "split_seed": split_seed,
        "dre_seed": dre_seed,
        "alpha_hat_u": float(uniform_sample["accept_rate"]),
        "alpha_hat_t": float(theory_sample["accept_rate"]),
        "eta_u": eta_uniform,
        "eta_t": eta_theory,
        "analytic_cov_trace_r": analytic_cov_trace_real,
        "analytic_cov_trace_u": analytic_cov_trace_uniform,
        "analytic_cov_trace_t": analytic_cov_trace_theory,
        "primary_r": primary_real,
        "primary_u": primary_uniform,
        "primary_t": primary_theory,
        "delta_mse_u_rel": delta_mse_uniform_rel,
        "delta_mse_t_rel": delta_mse_theory_rel,
        "theory_kappa": theory_kappa,
    }
    for method_key, method_output in extra_method_outputs.items():
        method_suffix = method_output["spec"]["suffix"]
        rep_result[f"alpha_hat_{method_suffix}"] = float(
            method_output["sample"]["accept_rate"]
        )
        rep_result[f"eta_{method_suffix}"] = float(method_output["eta"])
        rep_result[f"analytic_cov_trace_{method_suffix}"] = float(
            method_output["analytic_cov_trace"]
        )
        rep_result[f"primary_{method_suffix}"] = float(method_output["primary"])
        rep_result[f"delta_mse_{method_suffix}_rel"] = float(
            delta_mse_extra_rel[method_key]
        )
        rep_result[f"{method_suffix}_kappa"] = float(method_output["kappa"])

    reported_metrics = {}
    for metric_name_extra in reported_metric_names:
        real_value = float(real_metrics[metric_name_extra])
        uniform_value = float(uniform_metrics[metric_name_extra])
        theory_value = float(theory_metrics[metric_name_extra])

        metric_values = {
            "real": real_value,
            "uniform": uniform_value,
        }
        rep_result[f"{metric_name_extra}_r"] = real_value
        rep_result[f"{metric_name_extra}_u"] = uniform_value
        for method_key, method_output in extra_method_outputs.items():
            method_suffix = method_output["spec"]["suffix"]
            method_value = float(method_output["metrics"][metric_name_extra])
            rep_result[f"{metric_name_extra}_{method_suffix}"] = method_value
            metric_values[method_key] = method_value
        rep_result[f"{metric_name_extra}_t"] = theory_value
        metric_values["theory"] = theory_value

        reported_metrics[metric_name_extra] = metric_values

    for metric_name_extra in reported_test_set_metric_names:
        metric_value = float(real_metrics[metric_name_extra])
        rep_result[metric_name_extra] = metric_value
        reported_metrics[metric_name_extra] = {
            "test_set": metric_value,
        }

    plot_bundle = None
    if (
        config["pipeline"]["save_histograms"]
        or config["pipeline"]["save_candidate_csv"]
        or rep == 0
    ):
        plot_bundle = build_candidate_plot_bundle(
            train_x=train_x,
            train_y=train_y,
            cand_x=cand_x,
            cand_y=cand_y,
            candidate_source_index=candidate_source_index,
            generator_state=generator_state,
            theory_stats=theory_stats,
            theory_pi=theory_pi,
            uniform_sample=uniform_sample,
            theory_sample=theory_sample,
            extra_method_plot_data=[
                {
                    "pi_key": method_output["spec"]["pi_key"],
                    "accepted_key": method_output["spec"]["accepted_key"],
                    "pi": method_output["pi"],
                    "sample": method_output["sample"],
                }
                for method_output in extra_method_outputs.values()
            ],
        )

    return {
        "rep": rep,
        "outer_rep": outer_rep,
        "inner_rep": inner_rep,
        "rep_seed": dre_seed,
        "split_seed": split_seed,
        "dre_seed": dre_seed,
        "metric_name": metric_name,
        "reported_metric_context": reported_metric_context,
        "rep_result": rep_result,
        "reported_metrics": reported_metrics,
        "dre_diagnostics": {
            "rep": rep,
            "outer_rep": outer_rep,
            "inner_rep": inner_rep,
            "split_seed": split_seed,
            "dre_seed": dre_seed,
            "diagnostics": ratio_state["diagnostics"],
            "overlap_warning": overlap_warning,
            "split_sizes": ratio_state["split_sizes"],
            "q_prop_description": ratio_state["q_prop_description"],
            "q_prop_source_path": ratio_state["q_prop_source_path"],
        },
        "parameter_vectors": {
            "real": evaluation.try_extract_parameter_vector(real_model),
            "uniform": evaluation.try_extract_parameter_vector(uniform_model),
            "dre_only": evaluation.try_extract_parameter_vector(
                extra_method_outputs["dre_only"]["model"]
            ),
            "influence_only": evaluation.try_extract_parameter_vector(
                extra_method_outputs["influence_only"]["model"]
            ),
            "theory": evaluation.try_extract_parameter_vector(theory_model),
        },
        "resolved_n_candidates": int(n_candidates),
        "q_prop_pool_size": int(generator_state.get("q_prop_pool_size", n_candidates)),
        "plot_bundle": plot_bundle,
    }


def _serialize_parameter_vector(vector):
    """
    Convert an optional parameter vector into a JSON-safe payload.
    """
    if vector is None:
        return None
    return np.asarray(vector, dtype=float).tolist()


def _deserialize_parameter_vector(vector):
    """
    Convert a JSON-loaded parameter vector back into a numpy array.
    """
    if vector is None:
        return None
    return np.asarray(vector, dtype=float)


def build_rep_payload(rep_output):
    """
    Build the JSON payload saved for one rep job.
    """
    return {
        "rep": int(rep_output["rep"]),
        "outer_rep": int(rep_output["outer_rep"]),
        "inner_rep": int(rep_output["inner_rep"]),
        "rep_seed": int(rep_output["rep_seed"]),
        "split_seed": int(rep_output["split_seed"]),
        "dre_seed": int(rep_output["dre_seed"]),
        "metric_name": rep_output["metric_name"],
        "reported_metric_context": rep_output["reported_metric_context"],
        "rep_result": rep_output["rep_result"],
        "reported_metrics": rep_output["reported_metrics"],
        "dre_diagnostics": rep_output["dre_diagnostics"],
        "parameter_vectors": {
            name: _serialize_parameter_vector(vector)
            for name, vector in rep_output["parameter_vectors"].items()
        },
        "resolved_n_candidates": int(rep_output["resolved_n_candidates"]),
        "q_prop_pool_size": (
            int(rep_output["q_prop_pool_size"])
            if rep_output.get("q_prop_pool_size") is not None
            else None
        ),
    }


def save_rep_payload(path, rep_output):
    """
    Save one rep payload as JSON.
    """
    evaluation.save_json(path, build_rep_payload(rep_output))


def load_rep_payload(path):
    """
    Load one rep payload JSON and restore parameter vectors as arrays.
    """
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return {
        "rep": int(payload["rep"]),
        "outer_rep": int(payload["outer_rep"]),
        "inner_rep": int(payload["inner_rep"]),
        "rep_seed": int(payload["rep_seed"]),
        "split_seed": int(payload["split_seed"]),
        "dre_seed": int(payload["dre_seed"]),
        "metric_name": payload["metric_name"],
        "reported_metric_context": payload["reported_metric_context"],
        "rep_result": payload["rep_result"],
        "reported_metrics": payload["reported_metrics"],
        "dre_diagnostics": payload["dre_diagnostics"],
        "parameter_vectors": {
            name: _deserialize_parameter_vector(vector)
            for name, vector in payload["parameter_vectors"].items()
        },
        "resolved_n_candidates": int(payload["resolved_n_candidates"]),
        "q_prop_pool_size": (
            int(payload["q_prop_pool_size"])
            if payload.get("q_prop_pool_size") is not None
            else None
        ),
    }


def build_summary_metrics_payload(
    active_model_module,
    metric_name,
    trace_cov_real,
    trace_cov_uniform,
    trace_cov_theory,
    mean_analytic_cov_trace_real,
    mean_analytic_cov_trace_uniform,
    mean_analytic_cov_trace_theory,
    mean_delta_mse_uniform_rel,
    sd_delta_mse_uniform_rel,
    mean_delta_mse_theory_rel,
    sd_delta_mse_theory_rel,
    dre_diagnostics_by_rep,
    reported_metric_series_by_name,
    reported_metric_context,
    extra_method_summaries=None,
    overlap_warning_summary=None,
):
    """
    Build the JSON summary payload saved at the end of the run.
    """
    payload = {
        "active_model_module": active_model_module,
        "metric_name": metric_name,
        "trace_cov_real": trace_cov_real,
        "trace_cov_uniform": trace_cov_uniform,
        "trace_cov_theory": trace_cov_theory,
        "empirical_trace_cov_across_reps_real": trace_cov_real,
        "empirical_trace_cov_across_reps_uniform": trace_cov_uniform,
        "empirical_trace_cov_across_reps_theory": trace_cov_theory,
        "mean_analytic_cov_trace_real": mean_analytic_cov_trace_real,
        "mean_analytic_cov_trace_uniform": mean_analytic_cov_trace_uniform,
        "mean_analytic_cov_trace_theory": mean_analytic_cov_trace_theory,
        "mean_delta_mse_uniform_rel": mean_delta_mse_uniform_rel,
        "sd_delta_mse_uniform_rel": sd_delta_mse_uniform_rel,
        "mean_delta_mse_theory_rel": mean_delta_mse_theory_rel,
        "sd_delta_mse_theory_rel": sd_delta_mse_theory_rel,
        "dre_diagnostics_by_rep": dre_diagnostics_by_rep,
        "reported_metric_series_by_name": reported_metric_series_by_name,
        "reported_metric_context": reported_metric_context,
        "overlap_warning_summary": overlap_warning_summary,
    }
    extra_method_payloads = []
    for method_summary in extra_method_summaries or []:
        method_key = str(method_summary["key"])
        method_payload = {
            "key": method_key,
            "label": str(method_summary["label"]),
            "suffix": str(method_summary["suffix"]),
            "trace_cov": float(method_summary["trace_cov"]),
            "mean_analytic_cov_trace": evaluation.mean_se(
                method_summary["analytic_cov_trace_values"]
            )[0],
            "mean_delta_mse_rel": evaluation.mean_sd(
                method_summary["delta_mse_rel_values"]
            )[0],
            "sd_delta_mse_rel": evaluation.mean_sd(
                method_summary["delta_mse_rel_values"]
            )[1],
        }
        extra_method_payloads.append(method_payload)
        payload[f"trace_cov_{method_key}"] = method_payload["trace_cov"]
        payload[f"empirical_trace_cov_across_reps_{method_key}"] = (
            method_payload["trace_cov"]
        )
        payload[f"mean_analytic_cov_trace_{method_key}"] = method_payload[
            "mean_analytic_cov_trace"
        ]
        payload[f"mean_delta_mse_{method_key}_rel"] = method_payload[
            "mean_delta_mse_rel"
        ]
        payload[f"sd_delta_mse_{method_key}_rel"] = method_payload[
            "sd_delta_mse_rel"
        ]
    payload["extra_method_summaries"] = extra_method_payloads
    return payload


def build_aggregate_state(rep_outputs):
    """
    Aggregate repetition outputs into the summary inputs used by the pipeline.
    """
    if len(rep_outputs) == 0:
        raise ValueError("rep_outputs must be non-empty.")

    ordered_outputs = sorted(rep_outputs, key=lambda item: int(item["rep"]))

    metric_name = ordered_outputs[0]["metric_name"]
    reported_metric_context = ordered_outputs[0]["reported_metric_context"]
    reported_metric_names = list(ordered_outputs[0]["reported_metrics"].keys())

    reported_metric_method_keys_by_name = {
        name: list(ordered_outputs[0]["reported_metrics"][name].keys())
        for name in reported_metric_names
    }
    reported_metric_series_by_name = {
        name: {
            method_key: []
            for method_key in reported_metric_method_keys_by_name[name]
        }
        for name in reported_metric_names
    }

    results = []
    dre_diagnostics_by_rep = []
    primary_real_values = []
    primary_uniform_values = []
    primary_theory_values = []
    alpha_uniform_values = []
    alpha_theory_values = []
    eta_uniform_values = []
    eta_theory_values = []
    analytic_cov_trace_real_values = []
    analytic_cov_trace_uniform_values = []
    analytic_cov_trace_theory_values = []
    delta_mse_uniform_rel_values = []
    delta_mse_theory_rel_values = []
    param_real_list = []
    param_uniform_list = []
    param_theory_list = []
    extra_method_states = {}
    first_rep_result = ordered_outputs[0]["rep_result"]
    for method_spec in EXTRA_SYNTHETIC_METHODS:
        suffix = method_spec["suffix"]
        if f"primary_{suffix}" not in first_rep_result:
            continue
        extra_method_states[method_spec["key"]] = {
            "key": method_spec["key"],
            "suffix": suffix,
            "label": method_spec["label"],
            "primary_values": [],
            "alpha_values": [],
            "eta_values": [],
            "analytic_cov_trace_values": [],
            "delta_mse_rel_values": [],
            "param_list": [],
        }
    resolved_n_candidates = None

    for rep_output in ordered_outputs:
        rep_result = rep_output["rep_result"]
        results.append(rep_result)
        dre_diagnostics_by_rep.append(rep_output["dre_diagnostics"])

        primary_real_values.append(float(rep_result["primary_r"]))
        primary_uniform_values.append(float(rep_result["primary_u"]))
        primary_theory_values.append(float(rep_result["primary_t"]))
        alpha_uniform_values.append(float(rep_result["alpha_hat_u"]))
        alpha_theory_values.append(float(rep_result["alpha_hat_t"]))
        eta_uniform_values.append(float(rep_result["eta_u"]))
        eta_theory_values.append(float(rep_result["eta_t"]))
        analytic_cov_trace_real_values.append(float(rep_result.get("analytic_cov_trace_r", np.nan)))
        analytic_cov_trace_uniform_values.append(float(rep_result.get("analytic_cov_trace_u", np.nan)))
        analytic_cov_trace_theory_values.append(float(rep_result.get("analytic_cov_trace_t", np.nan)))
        delta_mse_uniform_rel_values.append(float(rep_result.get("delta_mse_u_rel", np.nan)))
        delta_mse_theory_rel_values.append(float(rep_result.get("delta_mse_t_rel", np.nan)))

        param_real_list.append(rep_output["parameter_vectors"]["real"])
        param_uniform_list.append(rep_output["parameter_vectors"]["uniform"])
        param_theory_list.append(rep_output["parameter_vectors"]["theory"])
        for method_key, method_state in extra_method_states.items():
            suffix = method_state["suffix"]
            method_state["primary_values"].append(float(rep_result[f"primary_{suffix}"]))
            method_state["alpha_values"].append(float(rep_result[f"alpha_hat_{suffix}"]))
            method_state["eta_values"].append(float(rep_result[f"eta_{suffix}"]))
            method_state["analytic_cov_trace_values"].append(
                float(rep_result[f"analytic_cov_trace_{suffix}"])
            )
            method_state["delta_mse_rel_values"].append(
                float(rep_result[f"delta_mse_{suffix}_rel"])
            )
            method_state["param_list"].append(
                rep_output["parameter_vectors"].get(method_key)
            )

        rep_n_candidates = int(rep_output["resolved_n_candidates"])
        if resolved_n_candidates is None:
            resolved_n_candidates = rep_n_candidates
        elif resolved_n_candidates != rep_n_candidates:
            raise ValueError("Resolved n_candidates changed across repetitions.")

        for metric_name_extra, metric_values in rep_output["reported_metrics"].items():
            for method_key in reported_metric_series_by_name[metric_name_extra]:
                reported_metric_series_by_name[metric_name_extra][method_key].append(
                    float(metric_values[method_key])
                )

    stacked_real = evaluation.stack_parameter_vectors(param_real_list)
    stacked_uniform = evaluation.stack_parameter_vectors(param_uniform_list)
    stacked_theory = evaluation.stack_parameter_vectors(param_theory_list)
    extra_method_summaries = []
    for method_state in extra_method_states.values():
        stacked_method = evaluation.stack_parameter_vectors(method_state["param_list"])
        method_state["stacked"] = stacked_method
        method_state["trace_cov"] = evaluation.trace_covariance_of_vectors(
            stacked_method
        )
        extra_method_summaries.append(
            {
                "key": method_state["key"],
                "suffix": method_state["suffix"],
                "label": method_state["label"],
                "primary_values": method_state["primary_values"],
                "alpha_values": method_state["alpha_values"],
                "eta_values": method_state["eta_values"],
                "analytic_cov_trace_values": method_state[
                    "analytic_cov_trace_values"
                ],
                "delta_mse_rel_values": method_state["delta_mse_rel_values"],
                "trace_cov": method_state["trace_cov"],
            }
        )

    return {
        "metric_name": metric_name,
        "results": results,
        "dre_diagnostics_by_rep": dre_diagnostics_by_rep,
        "overlap_warning_summary": build_overlap_warning_summary(dre_diagnostics_by_rep),
        "reported_metric_context": reported_metric_context,
        "reported_metric_series_by_name": reported_metric_series_by_name,
        "primary_real_values": primary_real_values,
        "primary_uniform_values": primary_uniform_values,
        "primary_theory_values": primary_theory_values,
        "alpha_uniform_values": alpha_uniform_values,
        "alpha_theory_values": alpha_theory_values,
        "eta_uniform_values": eta_uniform_values,
        "eta_theory_values": eta_theory_values,
        "analytic_cov_trace_real_values": analytic_cov_trace_real_values,
        "analytic_cov_trace_uniform_values": analytic_cov_trace_uniform_values,
        "analytic_cov_trace_theory_values": analytic_cov_trace_theory_values,
        "delta_mse_uniform_rel_values": delta_mse_uniform_rel_values,
        "delta_mse_theory_rel_values": delta_mse_theory_rel_values,
        "extra_method_summaries": extra_method_summaries,
        "extra_method_states": extra_method_states,
        "stacked_real": stacked_real,
        "stacked_uniform": stacked_uniform,
        "stacked_theory": stacked_theory,
        "trace_cov_real": evaluation.trace_covariance_of_vectors(stacked_real),
        "trace_cov_uniform": evaluation.trace_covariance_of_vectors(stacked_uniform),
        "trace_cov_theory": evaluation.trace_covariance_of_vectors(stacked_theory),
        "resolved_n_candidates": resolved_n_candidates,
    }


def get_reported_metric_names(metric_dict):
    """
    Return the downstream metrics that should be surfaced in saved outputs.
    """
    preferred_metric_names = [
        "m_class_tail_mse",
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "tss",
        "hss",
        "train_positive_rate",
        "test_positive_rate",
    ]
    return [
        name for name in preferred_metric_names
        if name in metric_dict
    ]


def get_reported_test_set_metric_names(metric_dict):
    """
    Return evaluation metrics that describe the shared held-out subset.
    """
    preferred_metric_names = [
        "m_class_tail_test_count",
        "m_class_tail_test_fraction",
    ]
    return [
        name for name in preferred_metric_names
        if name in metric_dict
    ]


def write_aggregate_outputs(output_dir, figures_dir, config, data_summary, aggregate_state, plot_bundle=None):
    """
    Write the run-level summary files and any aggregate figures.
    """
    metric_name = aggregate_state["metric_name"]

    summary_text = evaluation.build_summary_text(
        active_model_module=ACTIVE_MODEL_MODULE,
        config=config,
        metric_name=metric_name,
        n_train=int(data_summary["n_train"]),
        n_test=int(data_summary["n_test"]),
        x_dim=int(data_summary["x_dim"]),
        n_candidates=int(aggregate_state["resolved_n_candidates"]),
        alpha_target=float(config["pipeline"]["alpha"]),
        alpha_uniform_values=aggregate_state["alpha_uniform_values"],
        alpha_theory_values=aggregate_state["alpha_theory_values"],
        eta_uniform_values=aggregate_state["eta_uniform_values"],
        eta_theory_values=aggregate_state["eta_theory_values"],
        primary_real_values=aggregate_state["primary_real_values"],
        primary_uniform_values=aggregate_state["primary_uniform_values"],
        primary_theory_values=aggregate_state["primary_theory_values"],
        trace_cov_real=aggregate_state["trace_cov_real"],
        trace_cov_uniform=aggregate_state["trace_cov_uniform"],
        trace_cov_theory=aggregate_state["trace_cov_theory"],
        analytic_cov_trace_real_values=aggregate_state["analytic_cov_trace_real_values"],
        analytic_cov_trace_uniform_values=aggregate_state["analytic_cov_trace_uniform_values"],
        analytic_cov_trace_theory_values=aggregate_state["analytic_cov_trace_theory_values"],
        delta_mse_uniform_rel_values=aggregate_state["delta_mse_uniform_rel_values"],
        delta_mse_theory_rel_values=aggregate_state["delta_mse_theory_rel_values"],
        extra_method_summaries=aggregate_state.get("extra_method_summaries", []),
        extra_metric_series_by_name=aggregate_state["reported_metric_series_by_name"],
        context_metric_values=aggregate_state["reported_metric_context"],
        overlap_warning_summary=aggregate_state.get("overlap_warning_summary"),
    )

    evaluation.save_text(
        path=f"{output_dir}/results_summary.txt",
        text=summary_text,
    )

    evaluation.save_results_details(
        path=f"{output_dir}/results_details.csv",
        results=aggregate_state["results"],
        metric_name=metric_name if metric_name is not None else "primary_metric",
        stacked_real=aggregate_state["stacked_real"],
        stacked_uniform=aggregate_state["stacked_uniform"],
        stacked_theory=aggregate_state["stacked_theory"],
    )

    evaluation.save_json(
        path=f"{output_dir}/summary_metrics.json",
        obj=build_summary_metrics_payload(
            active_model_module=ACTIVE_MODEL_MODULE,
            metric_name=metric_name,
            trace_cov_real=aggregate_state["trace_cov_real"],
            trace_cov_uniform=aggregate_state["trace_cov_uniform"],
            trace_cov_theory=aggregate_state["trace_cov_theory"],
            mean_analytic_cov_trace_real=evaluation.mean_se(
                aggregate_state["analytic_cov_trace_real_values"]
            )[0],
            mean_analytic_cov_trace_uniform=evaluation.mean_se(
                aggregate_state["analytic_cov_trace_uniform_values"]
            )[0],
            mean_analytic_cov_trace_theory=evaluation.mean_se(
                aggregate_state["analytic_cov_trace_theory_values"]
            )[0],
            mean_delta_mse_uniform_rel=evaluation.mean_sd(
                aggregate_state["delta_mse_uniform_rel_values"]
            )[0],
            sd_delta_mse_uniform_rel=evaluation.mean_sd(
                aggregate_state["delta_mse_uniform_rel_values"]
            )[1],
            mean_delta_mse_theory_rel=evaluation.mean_sd(
                aggregate_state["delta_mse_theory_rel_values"]
            )[0],
            sd_delta_mse_theory_rel=evaluation.mean_sd(
                aggregate_state["delta_mse_theory_rel_values"]
            )[1],
            dre_diagnostics_by_rep=aggregate_state["dre_diagnostics_by_rep"],
            reported_metric_series_by_name=aggregate_state["reported_metric_series_by_name"],
            reported_metric_context=aggregate_state["reported_metric_context"],
            extra_method_summaries=aggregate_state.get("extra_method_summaries", []),
            overlap_warning_summary=aggregate_state.get("overlap_warning_summary"),
        ),
    )

    if config["pipeline"]["make_xy_band_plots"] and plot_bundle is not None:
        true_beta_full = evaluation.try_get_true_line_from_config(config)

        evaluation.plot_xy_with_band(
            real_x=plot_bundle["real_x"],
            real_y=plot_bundle["real_y"],
            cand_x=plot_bundle["cand_x"],
            cand_y=plot_bundle["cand_y"],
            accepted_mask=plot_bundle["accepted_theory"],
            stacked_parameter_vectors=aggregate_state["stacked_theory"],
            title="Theory acceptance: real (black), synthetic rejected (blue), accepted (red), 95% band",
            output_path=f"{figures_dir}/theory_xy_band.png",
            metric_name=metric_name,
            true_beta_full=true_beta_full,
        )

        for method_summary in aggregate_state.get("extra_method_summaries", []):
            method_key = method_summary["key"]
            accepted_key = None
            for method_spec in EXTRA_SYNTHETIC_METHODS:
                if method_spec["key"] == method_key:
                    accepted_key = method_spec["accepted_key"]
                    break
            method_state = aggregate_state.get("extra_method_states", {}).get(method_key)
            if (
                accepted_key is None
                or accepted_key not in plot_bundle
                or method_state is None
            ):
                continue
            evaluation.plot_xy_with_band(
                real_x=plot_bundle["real_x"],
                real_y=plot_bundle["real_y"],
                cand_x=plot_bundle["cand_x"],
                cand_y=plot_bundle["cand_y"],
                accepted_mask=plot_bundle[accepted_key],
                stacked_parameter_vectors=method_state["stacked"],
                title=(
                    f"{method_summary['label']} acceptance: real (black), "
                    "synthetic rejected (blue), accepted (red), 95% band"
                ),
                output_path=f"{figures_dir}/{method_key}_xy_band.png",
                metric_name=metric_name,
                true_beta_full=true_beta_full,
            )

        evaluation.plot_xy_with_band(
            real_x=plot_bundle["real_x"],
            real_y=plot_bundle["real_y"],
            cand_x=plot_bundle["cand_x"],
            cand_y=plot_bundle["cand_y"],
            accepted_mask=plot_bundle["accepted_uniform"],
            stacked_parameter_vectors=aggregate_state["stacked_uniform"],
            title="Uniform acceptance: real (black), synthetic rejected (blue), accepted (red), 95% band",
            output_path=f"{figures_dir}/uniform_xy_band.png",
            metric_name=metric_name,
            true_beta_full=true_beta_full,
        )

    return summary_text


def get_reported_metric_context(metric_dict):
    """
    Return static downstream-task context values for saved outputs.
    """
    preferred_context_names = [
        "m_class_tail_flux_threshold_raw",
        "m_class_tail_flux_threshold_log10",
        "m_class_tail_flux_threshold_standardized",
        "m_class_tail_threshold_inclusive",
        "positive_class_name",
        "positive_flux_threshold_raw",
        "prediction_threshold",
    ]
    return {
        name: metric_dict[name]
        for name in preferred_context_names
        if name in metric_dict
    }


def extract_primary_metric(metric_dict):
    """
    Extract the primary metric value from an evaluation dictionary.
    """
    if "primary_metric_value" not in metric_dict:
        raise ValueError("evaluate_model() must return 'primary_metric_value'.")
    return float(metric_dict["primary_metric_value"])


def estimate_vp(real_x, real_y, reference_model, config):
    """
    Estimate Vp from real data.

    The empirical estimator is:
        Vp_hat = (1 / n) * sum_i psi_i psi_i^T
    """
    validate_xy(real_x, real_y, "real")

    psi = compute_psi(real_x, real_y, reference_model, config)

    if not isinstance(psi, np.ndarray):
        raise TypeError("compute_psi() must return a numpy array.")
    if psi.ndim != 2:
        raise ValueError("compute_psi() must return a 2D array.")
    if psi.shape[0] != real_x.shape[0]:
        raise ValueError("The number of psi rows must match the number of samples.")

    n_samples = psi.shape[0]
    vp = (psi.T @ psi) / float(n_samples)
    return vp


def invert_positive_matrix(matrix, ridge):
    """
    Invert a positive semi-definite matrix after ridge stabilization.
    """
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    dim = matrix.shape[0]
    stabilized = matrix + float(ridge) * np.eye(dim)
    inverse = np.linalg.inv(stabilized)
    return inverse


def compute_influence_norm(x, y, reference_model, vp_inv, config):
    """
    Compute sqrt(psi^T Vp^{-1} psi) for each sample.
    """
    validate_xy(x, y, "candidate")
    psi = compute_psi(x, y, reference_model, config)

    if psi.ndim != 2:
        raise ValueError("compute_psi() must return a 2D array.")
    if psi.shape[1] != vp_inv.shape[0]:
        raise ValueError("Psi dimension does not match Vp inverse dimension.")

    quadratic_values = np.sum((psi @ vp_inv) * psi, axis=1)
    quadratic_values = np.maximum(quadratic_values, 0.0)
    influence_norm = np.sqrt(quadratic_values)
    return influence_norm


def compute_theory_statistics(x, y, reference_model, ratio_state, vp_inv, config):
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
    validate_xy(x, y, "candidate")

    log_ratio = compute_log_ratio(x, y, ratio_state, config)

    if not isinstance(log_ratio, np.ndarray):
        raise TypeError("compute_log_ratio() must return a numpy array.")
    if log_ratio.ndim != 1:
        raise ValueError("compute_log_ratio() must return a 1D array.")
    if log_ratio.shape[0] != x.shape[0]:
        raise ValueError("The length of log_ratio must match the number of candidates.")

    ratio = np.exp(log_ratio)

    influence = compute_influence_norm(
        x=x,
        y=y,
        reference_model=reference_model,
        vp_inv=vp_inv,
        config=config,
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
    validate_xy(x, y, "candidate")

    pi = np.asarray(pi, dtype=float)
    ratio = np.asarray(ratio, dtype=float)

    if pi.ndim != 1:
        raise ValueError("pi must be a 1D array.")
    if ratio.ndim != 1:
        raise ValueError("ratio must be a 1D array.")
    if pi.shape[0] != x.shape[0]:
        raise ValueError("pi length must match number of candidates.")
    if ratio.shape[0] != x.shape[0]:
        raise ValueError("ratio length must match number of candidates.")

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


def estimate_vw_from_accepted(accepted_x, accepted_y, accepted_weight, reference_model, config, psi_dim):
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

    psi = compute_psi(accepted_x, accepted_y, reference_model, config)

    if psi.ndim != 2:
        raise ValueError("compute_psi() must return a 2D array.")

    weighted_psi = accepted_weight.reshape(-1, 1) * psi
    vw = (weighted_psi.T @ weighted_psi) / float(m)
    return vw


def fit_augmented_model(real_x, real_y, synth_x, synth_y, synth_weight, config):
    """
    Fit the final model on real data plus accepted synthetic data.

    Real samples always receive weight 1.
    Synthetic samples receive the computed synthetic weights.
    """
    x_all, y_all, weights = build_augmented_dataset(
        real_x=real_x,
        real_y=real_y,
        synth_x=synth_x,
        synth_y=synth_y,
        synth_weight=synth_weight,
    )
    return fit_final_model(x_all, y_all, weights, config)


def main(argv=None):
    """
    Run the full experiment.
    """
    cli_args = parse_cli_args(argv)
    config = apply_cli_overrides(build_full_config(), cli_args)

    output_base_dir = os.path.join(
        config["output_base_dir"],
        get_output_group_for_model(ACTIVE_MODEL_MODULE),
    )

    output_dir = evaluation.create_timestamp_output_dir(
        base_dir=output_base_dir,
        experiment_name=config["experiment_name"],
    )
    figures_dir = get_figures_dir(output_dir)
    per_rep_figures_dir = get_per_rep_figures_dir(figures_dir)

    base_seed = int(config["seed"])
    outer_reps = int(config["pipeline"]["reps"])
    inner_reps = int(config["pipeline"]["inner_reps"])
    rep_plan = build_rep_plan(base_seed, outer_reps, inner_reps)

    # ---------------------------------------------------------------
    # 1. Load one outer-seeded split to summarize shared train/test dimensions.
    # ---------------------------------------------------------------
    summary_config = copy_config_with_run_seeds(
        config,
        split_seed=int(rep_plan[0]["split_seed"]),
        dre_seed=int(rep_plan[0]["dre_seed"]),
    )
    data_summary = summarize_data_split(load_summary_data(summary_config))

    total_runs = len(rep_plan)
    rep_outputs = []
    plot_bundle = None

    # ---------------------------------------------------------------
    # 2. Run the experiment over repetitions.
    # ---------------------------------------------------------------
    for rep_entry in rep_plan:
        rep = int(rep_entry["rep"])
        print(
            "[run_pipeline] starting run "
            f"{rep + 1}/{total_runs} "
            f"(outer={rep_entry['outer_rep'] + 1}/{outer_reps}, "
            f"inner={rep_entry['inner_rep'] + 1}/{inner_reps})"
        )
        rep_output = run_single_rep(
            rep=rep_entry["rep"],
            outer_rep=rep_entry["outer_rep"],
            inner_rep=rep_entry["inner_rep"],
            split_seed=rep_entry["split_seed"],
            dre_seed=rep_entry["dre_seed"],
            config=config,
        )
        rep_outputs.append(rep_output)

        if config["pipeline"]["save_candidate_csv"] and rep_output["plot_bundle"] is not None:
            save_candidate_acceptance_table(
                path=os.path.join(output_dir, get_candidate_acceptance_filename(rep)),
                plot_bundle=rep_output["plot_bundle"],
            )

        if config["pipeline"]["save_histograms"] and rep_output["plot_bundle"] is not None:
            save_per_rep_diagnostics(
                per_rep_figures_dir,
                rep_output["plot_bundle"],
                rep,
            )

        if rep == 0:
            plot_bundle = rep_output["plot_bundle"]
            save_rep0_diagnostics(output_dir, figures_dir, plot_bundle, config)

        print(
            "[run_pipeline] finished run "
            f"{rep + 1}/{total_runs} "
            f"(outer={rep_entry['outer_rep'] + 1}/{outer_reps}, "
            f"inner={rep_entry['inner_rep'] + 1}/{inner_reps}, "
            f"alpha_u={rep_output['rep_result']['alpha_hat_u']:.4f}, "
            f"alpha_t={rep_output['rep_result']['alpha_hat_t']:.4f}, "
            f"primary_t={rep_output['rep_result']['primary_t']:.6f})"
        )

    # ---------------------------------------------------------------
    # 3. Aggregate repetition-level results.
    # ---------------------------------------------------------------
    aggregate_state = build_aggregate_state(rep_outputs)

    # ---------------------------------------------------------------
    # 4. Save outputs.
    # ---------------------------------------------------------------
    if config["pipeline"]["save_candidate_csv"] and len(rep_outputs) > 0:
        rep_acceptance_paths = [
            os.path.join(output_dir, get_candidate_acceptance_filename(rep))
            for rep in range(total_runs)
        ]
        save_acceptance_matrix_from_paths(
            output_path=os.path.join(output_dir, "theory_acceptance_by_rep.csv"),
            rep_acceptance_paths=rep_acceptance_paths,
            q_prop_pool_size=rep_outputs[0].get("q_prop_pool_size"),
            acceptance_key="theory_accepted",
        )
        save_acceptance_matrix_from_paths(
            output_path=os.path.join(output_dir, "uniform_acceptance_by_rep.csv"),
            rep_acceptance_paths=rep_acceptance_paths,
            q_prop_pool_size=rep_outputs[0].get("q_prop_pool_size"),
            acceptance_key="uniform_accepted",
        )
        for method_spec in EXTRA_SYNTHETIC_METHODS:
            save_acceptance_matrix_from_paths(
                output_path=os.path.join(
                    output_dir,
                    f"{method_spec['key']}_acceptance_by_rep.csv",
                ),
                rep_acceptance_paths=rep_acceptance_paths,
                q_prop_pool_size=rep_outputs[0].get("q_prop_pool_size"),
                acceptance_key=f"{method_spec['key']}_accepted",
            )

    summary_text = write_aggregate_outputs(
        output_dir=output_dir,
        figures_dir=figures_dir,
        config=config,
        data_summary=data_summary,
        aggregate_state=aggregate_state,
        plot_bundle=plot_bundle,
    )

    # ---------------------------------------------------------------
    # 5. Print final summary.
    # ---------------------------------------------------------------
    print(summary_text)
    print("")
    print(f"Output directory: {output_dir}")
    print("Expected main files:")
    print("  - results_summary.txt")
    print("  - results_details.csv")
    print("  - summary_metrics.json")
    print("  - candidate_diagnostics_rep0.csv")
    print("  - theory_acceptance_by_rep.csv")
    print("  - uniform_acceptance_by_rep.csv")
    print("  - dre_only_acceptance_by_rep.csv")
    print("  - influence_only_acceptance_by_rep.csv")
    print("  - figures/")
    print("    * hist_influence_rep0.png")
    print("    * per_rep/hist_log_ratio_rep0.png")
    print("    * per_rep/hist_theory_pi_rep0.png")
    print("    * per_rep/box_theory_pi_vs_log10_flux_rep0.png")
    print("    * theory_xy_band.png")
    print("    * uniform_xy_band.png")
    print("")


if __name__ == "__main__":
    main()
