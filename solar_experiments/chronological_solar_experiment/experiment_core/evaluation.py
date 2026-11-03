"""
Evaluation and output utilities for the synthetic-sample pipeline.

This file is responsible for:
1. creating a timestamped output directory
2. computing summary statistics
3. saving summary and detail files
4. plotting figures when possible
5. plotting generic diagnostic histograms
"""

import os
import csv
import json
import math
from datetime import datetime

import numpy as np

from .portable_metadata import save_json as save_portable_json

HAS_PLT = False
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def ensure_dir(path):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def create_timestamp_output_dir(base_dir, experiment_name):
    """
    Create an output directory with a timestamped name.

    Returns
    -------
    str
        Full path of the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    ensure_dir(output_dir)
    return output_dir


def mean_se(values):
    """
    Compute the mean and standard error of a numeric list.
    """
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))

    if values.shape[0] <= 1:
        return mean_value, float("nan")

    se_value = float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))
    return mean_value, se_value


def mean_sd(values):
    """
    Compute the mean and sample standard deviation of a numeric list.
    """
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))

    if values.shape[0] <= 1:
        return mean_value, float("nan")

    sd_value = float(np.std(values, ddof=1))
    return mean_value, sd_value


def format_detail_value(value):
    """
    Format a scalar detail value for CSV output.
    """
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def format_scientific_value(value, precision=2):
    """
    Format a scalar value in uppercase scientific notation.
    """
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{precision}E}"


def display_metric_name(metric_name):
    """
    Convert internal metric keys to human-readable labels.
    """
    mapping = {
        "m_class_tail_mse": "M-class Tail MSE",
        "m_class_tail_test_count": "M-class Tail Test Count",
        "m_class_tail_test_fraction": "M-class Tail Test Fraction",
        "m_class_tail_flux_threshold_raw": (
            "M-class Tail Flux Threshold (raw W/m^2)"
        ),
        "m_class_tail_flux_threshold_log10": (
            "M-class Tail Flux Threshold (log10)"
        ),
        "m_class_tail_flux_threshold_standardized": (
            "M-class Tail Flux Threshold (standardized)"
        ),
        "m_class_tail_threshold_inclusive": (
            "M-class Tail Threshold Is Inclusive"
        ),
        "roc_auc": "ROC AUC",
        "average_precision": "Average Precision",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "tss": "True Skill Statistic",
        "hss": "Heidke Skill Score",
        "train_positive_rate": "Train Positive Rate",
        "test_positive_rate": "Test Positive Rate",
    }
    return mapping.get(metric_name, str(metric_name))


def save_text(path, text):
    """
    Save a plain text file.
    """
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(text + "\n")


def save_json(path, obj):
    """
    Save a Python object as formatted JSON.
    """
    save_portable_json(path, make_json_ready(obj))


def make_json_ready(obj):
    """
    Recursively convert numpy-backed objects into JSON-safe Python values.
    """
    if isinstance(obj, dict):
        return {
            str(key): make_json_ready(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_ready(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def save_candidate_table(
    path,
    x,
    y,
    log_ratio,
    influence,
    score,
    theory_pi,
    theory_accept_mask,
    uniform_accept_mask,
    dre_only_pi=None,
    dre_only_accept_mask=None,
    influence_only_pi=None,
    influence_only_accept_mask=None,
):
    """
    Save candidate-level diagnostics as a CSV file.
    """
    x_dim = x.shape[1]

    header = []
    for idx in range(x_dim):
        header.append(f"x_{idx}")

    header.extend([
        "y",
        "log_ratio",
        "influence",
        "score",
        "theory_pi",
        "theory_accepted",
        "uniform_accepted",
    ])
    has_dre_only = dre_only_pi is not None and dre_only_accept_mask is not None
    has_influence_only = (
        influence_only_pi is not None and influence_only_accept_mask is not None
    )
    if has_dre_only:
        header.extend(["dre_only_pi", "dre_only_accepted"])
    if has_influence_only:
        header.extend(["influence_only_pi", "influence_only_accepted"])

    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)

        for idx in range(x.shape[0]):
            row = list(x[idx])
            row.extend([
                float(y[idx]),
                float(log_ratio[idx]),
                float(influence[idx]),
                float(score[idx]),
                float(theory_pi[idx]),
                int(theory_accept_mask[idx]),
                int(uniform_accept_mask[idx]),
            ])
            if has_dre_only:
                row.extend([
                    float(dre_only_pi[idx]),
                    int(dre_only_accept_mask[idx]),
                ])
            if has_influence_only:
                row.extend([
                    float(influence_only_pi[idx]),
                    int(influence_only_accept_mask[idx]),
                ])
            writer.writerow(row)


def save_candidate_acceptance_table(
    path,
    candidate_source_index,
    theory_accept_mask,
    uniform_accept_mask,
    dre_only_accept_mask=None,
    influence_only_accept_mask=None,
):
    """
    Save minimal candidate-acceptance diagnostics for one repetition.
    """
    candidate_source_index = np.asarray(candidate_source_index, dtype=int)
    theory_accept_mask = np.asarray(theory_accept_mask, dtype=bool)
    uniform_accept_mask = np.asarray(uniform_accept_mask, dtype=bool)
    optional_masks = []

    n_rows = candidate_source_index.shape[0]
    if candidate_source_index.ndim != 1:
        raise ValueError("candidate_source_index must be a 1D array.")
    if theory_accept_mask.shape != (n_rows,):
        raise ValueError("theory_accept_mask must match candidate_source_index.")
    if uniform_accept_mask.shape != (n_rows,):
        raise ValueError("uniform_accept_mask must match candidate_source_index.")
    if dre_only_accept_mask is not None:
        dre_only_accept_mask = np.asarray(dre_only_accept_mask, dtype=bool)
        if dre_only_accept_mask.shape != (n_rows,):
            raise ValueError("dre_only_accept_mask must match candidate_source_index.")
        optional_masks.append(("dre_only_accepted", dre_only_accept_mask))
    if influence_only_accept_mask is not None:
        influence_only_accept_mask = np.asarray(influence_only_accept_mask, dtype=bool)
        if influence_only_accept_mask.shape != (n_rows,):
            raise ValueError(
                "influence_only_accept_mask must match candidate_source_index."
            )
        optional_masks.append(("influence_only_accepted", influence_only_accept_mask))

    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "q_prop_index",
            "theory_accepted",
            "uniform_accepted",
            *[name for name, _ in optional_masks],
        ])
        for idx in range(n_rows):
            row = [
                int(candidate_source_index[idx]),
                int(theory_accept_mask[idx]),
                int(uniform_accept_mask[idx]),
            ]
            row.extend(int(mask[idx]) for _, mask in optional_masks)
            writer.writerow(row)


def load_candidate_acceptance_table(path):
    """
    Load one minimal candidate-acceptance table from CSV.
    """
    q_prop_index = []
    accepted_columns = {}

    with open(path, "r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required_columns = {"q_prop_index", "theory_accepted", "uniform_accepted"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "candidate acceptance CSV must contain at least "
                "q_prop_index, theory_accepted, uniform_accepted."
            )
        acceptance_columns = [
            column_name for column_name in reader.fieldnames
            if column_name != "q_prop_index"
        ]
        accepted_columns = {column_name: [] for column_name in acceptance_columns}

        for row in reader:
            q_prop_index.append(int(row["q_prop_index"]))
            for column_name in acceptance_columns:
                accepted_columns[column_name].append(int(row[column_name]))

    table = {
        "q_prop_index": np.asarray(q_prop_index, dtype=int),
    }
    for column_name, values in accepted_columns.items():
        table[column_name] = np.asarray(values, dtype=np.uint8)
    return table


def save_acceptance_matrix(path, q_prop_pool_size, rep_acceptance_entries):
    """
    Save a wide acceptance matrix keyed by q_prop row index.
    """
    q_prop_pool_size = int(q_prop_pool_size)
    if q_prop_pool_size <= 0:
        raise ValueError("q_prop_pool_size must be positive.")
    if len(rep_acceptance_entries) == 0:
        raise ValueError("rep_acceptance_entries must be non-empty.")

    rep_labels = [str(entry["rep_label"]) for entry in rep_acceptance_entries]
    matrix = np.zeros((q_prop_pool_size, len(rep_acceptance_entries)), dtype=np.uint8)

    for col_idx, entry in enumerate(rep_acceptance_entries):
        q_prop_index = np.asarray(entry["q_prop_index"], dtype=int)
        accepted = np.asarray(entry["accepted"], dtype=np.uint8)

        if q_prop_index.ndim != 1:
            raise ValueError("q_prop_index must be 1D for every rep.")
        if accepted.shape != q_prop_index.shape:
            raise ValueError("accepted must match q_prop_index for every rep.")
        if np.any(q_prop_index < 0) or np.any(q_prop_index >= q_prop_pool_size):
            raise ValueError("q_prop_index values must lie within q_prop_pool_size.")

        matrix[q_prop_index, col_idx] = accepted

    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["q_prop_index"] + rep_labels)
        for row_idx in range(q_prop_pool_size):
            writer.writerow([row_idx] + matrix[row_idx].astype(int).tolist())


def try_extract_parameter_vector(model_state):
    """
    Try to extract a one-dimensional parameter vector from a model state.

    This helper supports old-style regression summaries and generic models
    that return a parameter vector in a simple dictionary or array form.
    """
    if isinstance(model_state, np.ndarray) and model_state.ndim == 1:
        return model_state.astype(float)

    if isinstance(model_state, dict):
        candidate_keys = [
            "beta_full",
            "coef",
            "coefficients",
            "weights",
            "params",
        ]
        for key in candidate_keys:
            if key in model_state:
                value = np.asarray(model_state[key], dtype=float)
                if value.ndim == 1:
                    return value

    return None


def stack_parameter_vectors(parameter_vector_list):
    """
    Stack a list of parameter vectors if all entries are available and
    have the same dimension.

    Returns
    -------
    np.ndarray or None
        Stacked matrix of shape (n_reps, param_dim), or None if stacking
        is not possible.
    """
    cleaned = []
    for vector in parameter_vector_list:
        if vector is None:
            return None
        cleaned.append(np.asarray(vector, dtype=float))

    if len(cleaned) == 0:
        return None

    first_dim = cleaned[0].shape[0]
    for vector in cleaned:
        if vector.ndim != 1 or vector.shape[0] != first_dim:
            return None

    return np.stack(cleaned, axis=0)


def trace_covariance_of_vectors(stacked_vectors):
    """
    Compute trace(Cov(parameter_vector)) across repetitions.

    If stacking is not possible, this function returns NaN.
    """
    if stacked_vectors is None:
        return float("nan")

    if stacked_vectors.shape[0] <= 1:
        return float("nan")

    covariance = np.cov(stacked_vectors.T, ddof=1)
    return float(np.trace(covariance))


def plot_histogram(
    values,
    title,
    xlabel,
    output_path,
    bins=50,
    log_x_scale=False,
    log_y_scale=False,
):
    """
    Save a histogram plot.

    If matplotlib is not available, this function does nothing.
    """
    if not HAS_PLT:
        return

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    plt.figure(figsize=(8, 5))
    if log_x_scale:
        positive_values = values[values > 0.0]
        if positive_values.size == 0:
            return

        if positive_values.min() < positive_values.max():
            histogram_bins = np.logspace(
                np.log10(float(positive_values.min())),
                np.log10(float(positive_values.max())),
                int(bins) + 1,
            )
        else:
            center = float(positive_values.min())
            histogram_bins = np.array([center / 10.0, center * 10.0], dtype=float)

        plt.hist(positive_values, bins=histogram_bins, alpha=0.8, edgecolor="black")
        plt.xscale("log")
    else:
        plt.hist(values, bins=bins, alpha=0.8, edgecolor="black")
    if log_y_scale:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def prepare_binned_boxplot_data(x_values, y_values, n_bins=10, x_min=0.0, x_max=1.0):
    """
    Bin scalar x-values and return non-empty groups for a boxplot.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if x_values.ndim != 1:
        raise ValueError("x_values must be a 1D array.")
    if y_values.ndim != 1:
        raise ValueError("y_values must be a 1D array.")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x_values and y_values must have the same length.")
    if x_values.shape[0] == 0:
        raise ValueError("x_values and y_values must be non-empty.")
    if np.any(~np.isfinite(x_values)):
        raise ValueError("x_values must be finite.")
    if np.any(~np.isfinite(y_values)):
        raise ValueError("y_values must be finite.")

    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    x_min = float(x_min)
    x_max = float(x_max)
    if not x_min < x_max:
        raise ValueError("x_min must be strictly smaller than x_max.")

    bin_edges = np.linspace(x_min, x_max, n_bins + 1, dtype=float)
    clipped_x = np.clip(x_values, x_min, x_max)
    bin_indices = np.searchsorted(bin_edges, clipped_x, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    labels = []
    grouped_values = []

    for idx in range(n_bins):
        mask = bin_indices == idx
        if not np.any(mask):
            continue

        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        right_bracket = "]" if idx == n_bins - 1 else ")"
        labels.append(f"[{left:.2f}, {right:.2f}{right_bracket}\n(n={int(mask.sum())})")
        grouped_values.append(y_values[mask])

    return {
        "labels": labels,
        "grouped_values": grouped_values,
    }


def plot_boxplot_by_binned_scalar(
    x_values,
    y_values,
    title,
    xlabel,
    ylabel,
    output_path,
    n_bins=10,
    x_min=0.0,
    x_max=1.0,
):
    """
    Save a boxplot of y-values grouped by binned scalar x-values.
    """
    if not HAS_PLT:
        return

    boxplot_data = prepare_binned_boxplot_data(
        x_values=x_values,
        y_values=y_values,
        n_bins=n_bins,
        x_min=x_min,
        x_max=x_max,
    )
    grouped_values = boxplot_data["grouped_values"]
    labels = boxplot_data["labels"]

    if len(grouped_values) == 0:
        return

    plt.figure(figsize=(11, 5.5))
    boxplot = plt.boxplot(
        grouped_values,
        patch_artist=True,
        showfliers=False,
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#8fb9ff")
        patch.set_alpha(0.8)

    plt.xticks(
        ticks=np.arange(1, len(labels) + 1),
        labels=labels,
        rotation=35,
        ha="right",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def try_get_true_line_from_config(config):
    """
    Try to build the true line coefficients for the 1D linear regression case.

    Returns
    -------
    np.ndarray or None
        Array [intercept, slope] if x_dim == 1 and the configuration provides
        a single slope coefficient. Otherwise returns None.
    """
    try:
        x_dim = int(config["data"]["x_dim"])
        beta = np.asarray(config["data"]["beta"], dtype=float)
        intercept = float(config["data"]["intercept"])

        if x_dim == 1 and beta.shape[0] == 1:
            return np.array([intercept, beta[0]], dtype=float)
    except Exception:
        pass

    return None


def plot_xy_with_band(
    real_x,
    real_y,
    cand_x,
    cand_y,
    accepted_mask,
    stacked_parameter_vectors,
    title,
    output_path,
    metric_name,
    true_beta_full=None,
):
    """
    Save the 1D X vs Y plot in the style of the old regression script.

    If the model looks like a 1D regression model, this function includes:
        - real data in black
        - synthetic rejected points in blue
        - synthetic accepted points in red
        - the 95% band over all fitted lines across repetitions
        - the mean fitted line
        - the true line if available

    If the required information is not available, the function falls back
    to a scatter-only version with the same colors.
    """
    if not HAS_PLT:
        return

    use_band = (
        metric_name is not None
        and metric_name.lower() == "mse"
        and stacked_parameter_vectors is not None
        and stacked_parameter_vectors.ndim == 2
        and stacked_parameter_vectors.shape[1] >= 2
        and real_x.shape[1] == 1
        and cand_x.shape[1] == 1
    )

    xr = real_x[:, 0]
    xc = cand_x[:, 0]

    accepted_mask = accepted_mask.astype(bool)
    rejected_mask = ~accepted_mask

    plt.figure(figsize=(9, 6))

    if use_band:
        x_min = float(min(xr.min(), xc.min()))
        x_max = float(max(xr.max(), xc.max()))
        pad = 0.05 * (x_max - x_min + 1e-12)
        x_min -= pad
        x_max += pad

        x_grid = np.linspace(x_min, x_max, 300)

        b0 = stacked_parameter_vectors[:, 0].reshape(-1, 1)
        b1 = stacked_parameter_vectors[:, 1].reshape(-1, 1)

        y_grid = b0 + b1 * x_grid.reshape(1, -1)

        lo = np.percentile(y_grid, 2.5, axis=0)
        hi = np.percentile(y_grid, 97.5, axis=0)
        mean_line = np.mean(y_grid, axis=0)

        plt.fill_between(x_grid, lo, hi, alpha=0.35, label="95% band (over reps)")
        plt.plot(x_grid, lo, linewidth=1.2, alpha=0.9, label="2.5%/97.5% bounds")
        plt.plot(x_grid, hi, linewidth=1.2, alpha=0.9)
        plt.plot(x_grid, mean_line, linewidth=2.0, label="mean fitted line")

        if true_beta_full is not None and true_beta_full.shape[0] >= 2:
            true_line = true_beta_full[0] + true_beta_full[1] * x_grid
            plt.plot(x_grid, true_line, linestyle="--", linewidth=1.5, alpha=0.8, label="true line")

    plt.scatter(xr, real_y, s=22, c="k", alpha=0.85, label="real data")
    plt.scatter(xc[rejected_mask], cand_y[rejected_mask], s=8, c="tab:blue", alpha=0.20, label="synthetic rejected")
    plt.scatter(xc[accepted_mask], cand_y[accepted_mask], s=8, c="tab:red", alpha=0.20, label="synthetic accepted")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_summary_text(
    active_model_module,
    config,
    metric_name,
    n_train,
    n_test,
    x_dim,
    n_candidates,
    alpha_target,
    alpha_uniform_values,
    alpha_theory_values,
    eta_uniform_values,
    eta_theory_values,
    primary_real_values,
    primary_uniform_values,
    primary_theory_values,
    trace_cov_real,
    trace_cov_uniform,
    trace_cov_theory,
    analytic_cov_trace_real_values,
    analytic_cov_trace_uniform_values,
    analytic_cov_trace_theory_values,
    delta_mse_uniform_rel_values=None,
    delta_mse_theory_rel_values=None,
    extra_method_summaries=None,
    extra_metric_series_by_name=None,
    context_metric_values=None,
    overlap_warning_summary=None,
):
    """
    Build the summary text in the style of the old regression script.
    """
    extra_method_summaries = list(extra_method_summaries or [])
    mean_alpha_u, se_alpha_u = mean_se(alpha_uniform_values)
    mean_alpha_t, se_alpha_t = mean_se(alpha_theory_values)

    mean_eta_u, se_eta_u = mean_se(eta_uniform_values)
    mean_eta_t, se_eta_t = mean_se(eta_theory_values)

    mean_primary_r, se_primary_r = mean_se(primary_real_values)
    mean_primary_u, se_primary_u = mean_se(primary_uniform_values)
    mean_primary_t, se_primary_t = mean_se(primary_theory_values)
    mean_analytic_cov_r, se_analytic_cov_r = mean_se(analytic_cov_trace_real_values)
    mean_analytic_cov_u, se_analytic_cov_u = mean_se(analytic_cov_trace_uniform_values)
    mean_analytic_cov_t, se_analytic_cov_t = mean_se(analytic_cov_trace_theory_values)
    outer_reps = int(config["pipeline"]["reps"])
    inner_reps = int(config["pipeline"].get("inner_reps", 1))
    pooled_runs = outer_reps * inner_reps

    summary_lines = []
    summary_lines.append("=== Summary over pooled runs ===")
    summary_lines.append(f"active model module       : {active_model_module}")
    summary_lines.append(f"p (feature dim)           : {x_dim}")
    summary_lines.append(f"outer reps                : {outer_reps}")
    summary_lines.append(f"inner reps per outer      : {inner_reps}")
    summary_lines.append(f"pooled runs               : {pooled_runs}")
    summary_lines.append(f"n_real                    : {n_train}")
    summary_lines.append(f"n_test                    : {n_test}")
    summary_lines.append(f"M_cand                    : {n_candidates}")
    summary_lines.append(f"target alpha              : {alpha_target}")

    if "noise_std" in config["data"]:
        summary_lines.append(f"sigma (noise std)         : {config['data']['noise_std']}")

    true_beta_full = try_get_true_line_from_config(config)
    if true_beta_full is not None:
        summary_lines.append(f"beta_true (intercept,slope): {true_beta_full.tolist()}")

    summary_lines.append("")
    summary_lines.append(f"accept rate (uniform)     : {mean_alpha_u:.4f} ± {se_alpha_u:.4f}")
    for method_summary in extra_method_summaries:
        mean_alpha_extra, se_alpha_extra = mean_se(method_summary["alpha_values"])
        summary_lines.append(
            f"accept rate ({method_summary['label']}): "
            f"{mean_alpha_extra:.4f} ± {se_alpha_extra:.4f}"
        )
    summary_lines.append(f"accept rate (theory)      : {mean_alpha_t:.4f} ± {se_alpha_t:.4f}")
    summary_lines.append("")
    summary_lines.append(f"eta_trace = tr(Vp^-1 Vw)/d  (uniform): {mean_eta_u:.4f} ± {se_eta_u:.4f}")
    for method_summary in extra_method_summaries:
        mean_eta_extra, se_eta_extra = mean_se(method_summary["eta_values"])
        summary_lines.append(
            f"eta_trace = tr(Vp^-1 Vw)/d  ({method_summary['label']}): "
            f"{mean_eta_extra:.4f} ± {se_eta_extra:.4f}"
        )
    summary_lines.append(f"eta_trace = tr(Vp^-1 Vw)/d  (theory) : {mean_eta_t:.4f} ± {se_eta_t:.4f}")
    summary_lines.append("")

    if (
        overlap_warning_summary is not None
        and overlap_warning_summary.get("enabled", False)
        and int(overlap_warning_summary.get("triggered_reps", 0)) > 0
    ):
        thresholds = overlap_warning_summary["thresholds"]
        summary_lines.append("WARNING: DRE overlap health check triggered.")
        summary_lines.append(
            "warning criteria          : "
            f"q_mean_r outside [{thresholds['q_mean_r_min']:.4g}, "
            f"{thresholds['q_mean_r_max']:.4g}] or "
            f"ess_syn_reweighted < {thresholds['ess_syn_reweighted_min']:.4g}"
        )
        summary_lines.append(
            "flagged reps              : "
            f"{int(overlap_warning_summary['triggered_reps'])} / "
            f"{int(overlap_warning_summary['total_reps'])}"
        )
        summary_lines.append(
            "q_mean_r below min reps   : "
            f"{int(overlap_warning_summary['q_mean_r_below_min_reps'])}"
        )
        summary_lines.append(
            "q_mean_r above max reps   : "
            f"{int(overlap_warning_summary['q_mean_r_above_max_reps'])}"
        )
        summary_lines.append(
            "ESS below min reps        : "
            f"{int(overlap_warning_summary['ess_syn_reweighted_below_min_reps'])}"
        )

        worst_q_mean_r = overlap_warning_summary.get("worst_q_mean_r")
        if worst_q_mean_r is not None:
            summary_lines.append(
                "worst q_mean_r            : "
                f"rep {int(worst_q_mean_r['rep']):03d} = "
                f"{float(worst_q_mean_r['q_mean_r']):.6g}"
            )

        worst_ess = overlap_warning_summary.get("worst_ess_syn_reweighted")
        if worst_ess is not None:
            summary_lines.append(
                "worst ess_syn_reweighted  : "
                f"rep {int(worst_ess['rep']):03d} = "
                f"{float(worst_ess['ess_syn_reweighted']):.6g}"
            )

        summary_lines.append(
            "Theory acceptance diagnostics may be unreliable in flagged reps."
        )
        summary_lines.append("")

    if not math.isnan(trace_cov_real) or not math.isnan(trace_cov_uniform) or not math.isnan(trace_cov_theory):
        summary_lines.append(
            f"empirical trace(Cov(beta_hat across reps)) (real only): {trace_cov_real:.6f}"
        )
        summary_lines.append(
            f"empirical trace(Cov(beta_hat across reps)) (uniform): {trace_cov_uniform:.6f}"
        )
        for method_summary in extra_method_summaries:
            summary_lines.append(
                "empirical trace(Cov(beta_hat across reps)) "
                f"({method_summary['label']}): {float(method_summary['trace_cov']):.6f}"
            )
        summary_lines.append(
            f"empirical trace(Cov(beta_hat across reps)) (theory) : {trace_cov_theory:.6f}"
        )
    else:
        summary_lines.append(
            f"empirical trace(Cov(param_vector across reps)) (real only): {trace_cov_real}"
        )
        summary_lines.append(
            f"empirical trace(Cov(param_vector across reps)) (uniform): {trace_cov_uniform}"
        )
        for method_summary in extra_method_summaries:
            summary_lines.append(
                "empirical trace(Cov(param_vector across reps)) "
                f"({method_summary['label']}): {method_summary['trace_cov']}"
            )
        summary_lines.append(
            f"empirical trace(Cov(param_vector across reps)) (theory) : {trace_cov_theory}"
        )

    summary_lines.append(
        "mean analytic trace(Cov_hat(beta_hat | rep)) uses OLS for unweighted fits "
        "and sigma^2 (X'WX)^-1 for weighted fits."
    )
    summary_lines.append(
        f"mean analytic trace(Cov_hat(beta_hat | rep)) (real only): {mean_analytic_cov_r:.6f} ± {se_analytic_cov_r:.6f}"
    )
    summary_lines.append(
        f"mean analytic trace(Cov_hat(beta_hat | rep)) (uniform): {mean_analytic_cov_u:.6f} ± {se_analytic_cov_u:.6f}"
    )
    for method_summary in extra_method_summaries:
        mean_analytic_extra, se_analytic_extra = mean_se(
            method_summary["analytic_cov_trace_values"]
        )
        summary_lines.append(
            "mean analytic trace(Cov_hat(beta_hat | rep)) "
            f"({method_summary['label']}): "
            f"{mean_analytic_extra:.6f} ± {se_analytic_extra:.6f}"
        )
    summary_lines.append(
        f"mean analytic trace(Cov_hat(beta_hat | rep)) (theory) : {mean_analytic_cov_t:.6f} ± {se_analytic_cov_t:.6f}"
    )

    summary_lines.append("")

    if metric_name is not None and metric_name.lower() == "mse":
        summary_lines.append(f"test MSE (real only): {mean_primary_r:.4f} ± {se_primary_r:.4f}")
        summary_lines.append(f"test MSE (uniform): {mean_primary_u:.4f} ± {se_primary_u:.4f}")
        for method_summary in extra_method_summaries:
            mean_primary_extra, se_primary_extra = mean_se(
                method_summary["primary_values"]
            )
            summary_lines.append(
                f"test MSE ({method_summary['label']}): "
                f"{mean_primary_extra:.4f} ± {se_primary_extra:.4f}"
            )
        summary_lines.append(f"test MSE (theory) : {mean_primary_t:.4f} ± {se_primary_t:.4f}")
        if delta_mse_uniform_rel_values is not None and delta_mse_theory_rel_values is not None:
            mean_delta_u, sd_delta_u = mean_sd(delta_mse_uniform_rel_values)
            mean_delta_t, sd_delta_t = mean_sd(delta_mse_theory_rel_values)
            summary_lines.append("")
            summary_lines.append(
                "relative delta test MSE vs real is defined per rep as "
                "(MSE_method - MSE_real) / MSE_real"
            )
            summary_lines.append(
                "relative delta test MSE vs real (uniform): "
                f"{format_scientific_value(mean_delta_u, precision=2)} ± "
                f"{format_scientific_value(sd_delta_u, precision=2)} [sd]"
            )
            for method_summary in extra_method_summaries:
                mean_delta_extra, sd_delta_extra = mean_sd(
                    method_summary["delta_mse_rel_values"]
                )
                summary_lines.append(
                    "relative delta test MSE vs real "
                    f"({method_summary['label']}): "
                    f"{format_scientific_value(mean_delta_extra, precision=2)} ± "
                    f"{format_scientific_value(sd_delta_extra, precision=2)} [sd]"
                )
            summary_lines.append(
                "relative delta test MSE vs real (theory) : "
                f"{format_scientific_value(mean_delta_t, precision=2)} ± "
                f"{format_scientific_value(sd_delta_t, precision=2)} [sd]"
            )
    else:
        summary_lines.append(f"test {metric_name} (real only): {mean_primary_r:.4f} ± {se_primary_r:.4f}")
        summary_lines.append(f"test {metric_name} (uniform): {mean_primary_u:.4f} ± {se_primary_u:.4f}")
        for method_summary in extra_method_summaries:
            mean_primary_extra, se_primary_extra = mean_se(
                method_summary["primary_values"]
            )
            summary_lines.append(
                f"test {metric_name} ({method_summary['label']}): "
                f"{mean_primary_extra:.4f} ± {se_primary_extra:.4f}"
            )
        summary_lines.append(f"test {metric_name} (theory) : {mean_primary_t:.4f} ± {se_primary_t:.4f}")

    if context_metric_values:
        summary_lines.append("")
        for key, value in context_metric_values.items():
            label = display_metric_name(key)
            summary_lines.append(f"{label}: {value}")

    if extra_metric_series_by_name:
        summary_lines.append("")
        for metric_key, metric_series in extra_metric_series_by_name.items():
            label = display_metric_name(metric_key)
            metric_method_labels = {
                "test_set": "test set",
                "real": "real only",
                "uniform": "uniform",
                "dre_only": "DRE only",
                "influence_only": "influence only",
                "theory": "theory",
            }
            ordered_metric_methods = [
                "test_set",
                "real",
                "uniform",
                "dre_only",
                "influence_only",
                "theory",
            ]
            for method_key in ordered_metric_methods:
                if method_key not in metric_series:
                    continue
                mean_value, se_value = mean_se(metric_series[method_key])
                summary_lines.append(
                    f"{label} ({metric_method_labels[method_key]}): "
                    f"{mean_value:.4f} ± {se_value:.4f}"
                )

    return "\n".join(summary_lines)


def save_results_details(path, results, metric_name, stacked_real, stacked_uniform, stacked_theory):
    """
    Save the per-repetition detail file.

    If the model looks like a 1D regression model with two parameters
    [intercept, slope], this function writes an old-style detail file.
    Otherwise, it falls back to a simpler generic format.
    """
    use_regression_like_format = False

    if (
        stacked_real is not None
        and stacked_uniform is not None
        and stacked_theory is not None
        and stacked_real.shape[1] >= 2
        and stacked_uniform.shape[1] >= 2
        and stacked_theory.shape[1] >= 2
        and metric_name.lower() == "mse"
    ):
        use_regression_like_format = True

    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        detail_headers = [
            header for header in ["rep", "outer_rep", "inner_rep", "split_seed", "dre_seed"]
            if results and header in results[0]
        ]
        optional_method_specs = [
            ("dre", "dre_only"),
            ("if", "influence_only"),
        ]
        optional_method_specs = [
            (suffix, label)
            for suffix, label in optional_method_specs
            if results and f"primary_{suffix}" in results[0]
        ]
        supplemental_headers = sorted(
            key
            for key in (results[0] if results else {})
            if key.startswith("m_class_tail_")
        )

        if use_regression_like_format:
            writer.writerow([
                *detail_headers,
                "alpha_hat_u",
                "alpha_hat_t",
                *[
                    f"alpha_hat_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "eta_u",
                "eta_t",
                *[
                    f"eta_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "analytic_cov_trace_r",
                "analytic_cov_trace_u",
                "analytic_cov_trace_t",
                *[
                    f"analytic_cov_trace_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "mse_r",
                "mse_u",
                "mse_t",
                *[
                    f"mse_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "delta_mse_u_rel",
                "delta_mse_t_rel",
                *[
                    f"delta_mse_{suffix}_rel"
                    for suffix, _ in optional_method_specs
                ],
                "b0_r",
                "b1_r",
                "b0_u",
                "b1_u",
                "b0_t",
                "b1_t",
                *supplemental_headers,
            ])

            for idx, result in enumerate(results):
                writer.writerow([
                    *[result[header] for header in detail_headers],
                    f"{result['alpha_hat_u']:.8f}",
                    f"{result['alpha_hat_t']:.8f}",
                    *[
                        f"{result[f'alpha_hat_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['eta_u']:.8f}",
                    f"{result['eta_t']:.8f}",
                    *[
                        f"{result[f'eta_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['analytic_cov_trace_r']:.8f}",
                    f"{result['analytic_cov_trace_u']:.8f}",
                    f"{result['analytic_cov_trace_t']:.8f}",
                    *[
                        f"{result[f'analytic_cov_trace_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['primary_r']:.8f}",
                    f"{result['primary_u']:.8f}",
                    f"{result['primary_t']:.8f}",
                    *[
                        f"{result[f'primary_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['delta_mse_u_rel']:.8f}",
                    f"{result['delta_mse_t_rel']:.8f}",
                    *[
                        f"{result[f'delta_mse_{suffix}_rel']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{stacked_real[idx, 0]:.8f}",
                    f"{stacked_real[idx, 1]:.8f}",
                    f"{stacked_uniform[idx, 0]:.8f}",
                    f"{stacked_uniform[idx, 1]:.8f}",
                    f"{stacked_theory[idx, 0]:.8f}",
                    f"{stacked_theory[idx, 1]:.8f}",
                    *[
                        format_detail_value(result[key])
                        for key in supplemental_headers
                    ],
                ])
        else:
            base_headers = [
                *detail_headers,
                "alpha_hat_u",
                "alpha_hat_t",
                *[
                    f"alpha_hat_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "eta_u",
                "eta_t",
                *[
                    f"eta_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "analytic_cov_trace_r",
                "analytic_cov_trace_u",
                "analytic_cov_trace_t",
                *[
                    f"analytic_cov_trace_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
                "delta_mse_u_rel",
                "delta_mse_t_rel",
                *[
                    f"delta_mse_{suffix}_rel"
                    for suffix, _ in optional_method_specs
                ],
                f"{metric_name}_r",
                f"{metric_name}_u",
                f"{metric_name}_t",
                *[
                    f"{metric_name}_{suffix}"
                    for suffix, _ in optional_method_specs
                ],
            ]
            excluded_keys = {
                "rep",
                "outer_rep",
                "inner_rep",
                "split_seed",
                "dre_seed",
                "alpha_hat_u",
                "alpha_hat_t",
                "eta_u",
                "eta_t",
                "analytic_cov_trace_r",
                "analytic_cov_trace_u",
                "analytic_cov_trace_t",
                "delta_mse_u_rel",
                "delta_mse_t_rel",
                "primary_r",
                "primary_u",
                "primary_t",
            }
            for suffix, _ in optional_method_specs:
                excluded_keys.update(
                    {
                        f"alpha_hat_{suffix}",
                        f"eta_{suffix}",
                        f"analytic_cov_trace_{suffix}",
                        f"delta_mse_{suffix}_rel",
                        f"primary_{suffix}",
                    }
                )
            extra_headers = sorted(
                key for key in results[0].keys()
                if key not in excluded_keys
            ) if results else []

            writer.writerow([
                *base_headers,
                *extra_headers,
            ])

            for result in results:
                writer.writerow([
                    *[result[header] for header in detail_headers],
                    f"{result['alpha_hat_u']:.8f}",
                    f"{result['alpha_hat_t']:.8f}",
                    *[
                        f"{result[f'alpha_hat_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['eta_u']:.8f}",
                    f"{result['eta_t']:.8f}",
                    *[
                        f"{result[f'eta_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['analytic_cov_trace_r']:.8f}",
                    f"{result['analytic_cov_trace_u']:.8f}",
                    f"{result['analytic_cov_trace_t']:.8f}",
                    *[
                        f"{result[f'analytic_cov_trace_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['delta_mse_u_rel']:.8f}",
                    f"{result['delta_mse_t_rel']:.8f}",
                    *[
                        f"{result[f'delta_mse_{suffix}_rel']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    f"{result['primary_r']:.8f}",
                    f"{result['primary_u']:.8f}",
                    f"{result['primary_t']:.8f}",
                    *[
                        f"{result[f'primary_{suffix}']:.8f}"
                        for suffix, _ in optional_method_specs
                    ],
                    *[format_detail_value(result[key]) for key in extra_headers],
                ])
