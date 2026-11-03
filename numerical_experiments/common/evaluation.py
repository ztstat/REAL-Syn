"""
Evaluation and output utilities for the synthetic-sample pipeline.

This file is responsible for:
1. creating a timestamped output directory
2. computing summary statistics
3. saving summary and detail files
4. plotting figures when possible
5. plotting generic diagrams
"""

import os
import csv
import json
import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


PUBLICATION_DPI = 400
PUBLICATION_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
}


def apply_publication_style():
    """
    Apply a compact, consistent matplotlib style for paper-ready PNGs.
    """
    plt.rcParams.update(PUBLICATION_STYLE)


def format_publication_axis(ax):
    """
    Use light gridlines and clean spines for publication-style plots.
    """
    ax.grid(True, color="#E5E7EB", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")
    ax.tick_params(colors="#111827", width=0.8)


def subsample_indices(mask, max_points, seed):
    """
    Deterministically subsample plotted points without changing computation.
    """
    indices = np.flatnonzero(mask)
    if indices.shape[0] <= max_points:
        return indices

    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def set_quantile_axis_limits(
    ax,
    x_values,
    y_values,
    x_lower=0.005,
    x_upper=0.995,
    y_lower=0.015,
    y_upper=0.985,
    pad_fraction=0.035,
):
    """
    Set plot limits from central quantiles so the main point cloud fills the panel.
    """
    x_low, x_high = np.quantile(x_values, [x_lower, x_upper])
    y_low, y_high = np.quantile(y_values, [y_lower, y_upper])

    x_pad = pad_fraction * (x_high - x_low + 1e-12)
    y_pad = pad_fraction * (y_high - y_low + 1e-12)

    ax.set_xlim(float(x_low - x_pad), float(x_high + x_pad))
    ax.set_ylim(float(y_low - y_pad), float(y_high + y_pad))


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
    se_value = float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))
    return mean_value, se_value


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
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2)


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
            writer.writerow(row)


def try_extract_parameter_vector(model_state):
    """
    Try to extract a one-dimensional parameter vector from a model state.
    """
    if isinstance(model_state, np.ndarray) and model_state.ndim == 1:
        return model_state.astype(float)

    return np.asarray(model_state["beta_full"], dtype=float)


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
    stacked = np.asarray(parameter_vector_list, dtype=float)
    if stacked.ndim != 2:
        raise ValueError("Parameter vectors must all be 1D and have the same length.")

    return stacked


def trace_covariance_of_vectors(stacked_vectors):
    """
    Compute trace(Cov(parameter_vector)) across repetitions.
    """
    covariance = np.atleast_2d(np.cov(stacked_vectors.T, ddof=1))
    return float(np.trace(covariance))


def plot_histogram(values, title, xlabel, output_path, bins=50):
    """
    Save a histogram plot.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(
        values,
        bins=bins,
        color="#0072B2",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    format_publication_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    plt.close(fig)


def try_get_true_line_from_config(config):
    """
    Try to build the true line coefficients for the 1D linear regression case.

    Returns
    -------
    np.ndarray or None
        Array [intercept, slope] if x_dim == 1 and the configuration provides
        a single slope coefficient. Otherwise returns None.
    """
    x_dim = int(config["data"]["x_dim"])
    beta = np.asarray(config["data"]["beta"], dtype=float)
    intercept = float(config["data"]["intercept"])

    if x_dim == 1 and beta.shape[0] == 1:
        return np.array([intercept, beta[0]], dtype=float)

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

    If the model is not one-dimensional, the function falls back to a
    scatter-only version with the same colors.
    """

    use_band = (
        metric_name.lower() == "mse"
        and stacked_parameter_vectors.shape[1] >= 2
        and real_x.shape[1] == 1
        and cand_x.shape[1] == 1
    )

    xr = real_x[:, 0]
    xc = cand_x[:, 0]

    accepted_mask = accepted_mask.astype(bool)
    rejected_mask = ~accepted_mask

    apply_publication_style()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

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

        ax.fill_between(
            x_grid,
            lo,
            hi,
            color="#E69F00",
            alpha=0.18,
            linewidth=0,
            label="95% band",
            zorder=1,
        )
        ax.plot(x_grid, lo, color="#E69F00", linewidth=1.0, alpha=0.8, zorder=2)
        ax.plot(x_grid, hi, color="#E69F00", linewidth=1.0, alpha=0.8, zorder=2)
        ax.plot(x_grid, mean_line, color="#D55E00", linewidth=1.8, label="mean fit", zorder=3)

        if true_beta_full is not None:
            true_line = true_beta_full[0] + true_beta_full[1] * x_grid
            ax.plot(
                x_grid,
                true_line,
                color="#111827",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label="true line",
                zorder=3,
            )

    rejected_idx = subsample_indices(rejected_mask, max_points=8000, seed=17)
    accepted_idx = subsample_indices(accepted_mask, max_points=7000, seed=23)

    ax.scatter(
        xc[rejected_idx],
        cand_y[rejected_idx],
        s=7,
        c="#1F77B4",
        alpha=0.28,
        linewidths=0,
        label="synthetic rejected",
        zorder=0,
    )
    ax.scatter(
        xc[accepted_idx],
        cand_y[accepted_idx],
        s=8,
        c="#D62728",
        alpha=0.40,
        linewidths=0,
        label="synthetic accepted",
        zorder=1,
    )
    ax.scatter(
        xr,
        real_y,
        s=18,
        c="#111827",
        alpha=0.88,
        linewidths=0,
        label="real data",
        zorder=4,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, pad=8)
    set_quantile_axis_limits(
        ax,
        x_values=np.concatenate([xr, xc]),
        y_values=np.concatenate([real_y, cand_y]),
    )
    format_publication_axis(ax)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D1D5DB",
        borderpad=0.5,
        handlelength=1.8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    plt.close(fig)


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
):
    """
    Build the summary text in the style of the old regression script.
    """
    mean_alpha_u, se_alpha_u = mean_se(alpha_uniform_values)
    mean_alpha_t, se_alpha_t = mean_se(alpha_theory_values)

    mean_eta_u, se_eta_u = mean_se(eta_uniform_values)
    mean_eta_t, se_eta_t = mean_se(eta_theory_values)

    mean_primary_r, se_primary_r = mean_se(primary_real_values)
    mean_primary_u, se_primary_u = mean_se(primary_uniform_values)
    mean_primary_t, se_primary_t = mean_se(primary_theory_values)

    summary_lines = []
    summary_lines.append("=== Summary over reps ===")
    summary_lines.append(f"active model module       : {active_model_module}")
    summary_lines.append(f"p (feature dim)           : {x_dim}")
    summary_lines.append(f"reps                      : {config['pipeline']['reps']}")
    summary_lines.append(f"n_real                    : {n_train}")
    summary_lines.append(f"n_test                    : {n_test}")
    summary_lines.append(f"M_cand                    : {n_candidates}")
    summary_lines.append(f"target alpha              : {alpha_target}")

    summary_lines.append(f"sigma (noise std)         : {config['data']['noise_std']}")

    true_beta_full = try_get_true_line_from_config(config)
    if true_beta_full is not None:
        summary_lines.append(f"beta_true (intercept,slope): {true_beta_full.tolist()}")

    summary_lines.append("")
    summary_lines.append(f"accept rate (uniform)     : {mean_alpha_u:.4f} ± {se_alpha_u:.4f}")
    summary_lines.append(f"accept rate (theory)      : {mean_alpha_t:.4f} ± {se_alpha_t:.4f}")
    summary_lines.append("")
    summary_lines.append(f"eta_trace = tr(Vp^-1 Vw)/d  (uniform): {mean_eta_u:.4f} ± {se_eta_u:.4f}")
    summary_lines.append(f"eta_trace = tr(Vp^-1 Vw)/d  (theory) : {mean_eta_t:.4f} ± {se_eta_t:.4f}")
    summary_lines.append("")

    summary_lines.append(f"trace(Cov(beta_hat)) (real only): {trace_cov_real:.6f}")
    summary_lines.append(f"trace(Cov(beta_hat)) (uniform): {trace_cov_uniform:.6f}")
    summary_lines.append(f"trace(Cov(beta_hat)) (theory) : {trace_cov_theory:.6f}")

    summary_lines.append("")

    if metric_name.lower() == "mse":
        summary_lines.append(f"test MSE (real only): {mean_primary_r:.4f} ± {se_primary_r:.4f}")
        summary_lines.append(f"test MSE (uniform): {mean_primary_u:.4f} ± {se_primary_u:.4f}")
        summary_lines.append(f"test MSE (theory) : {mean_primary_t:.4f} ± {se_primary_t:.4f}")
    else:
        summary_lines.append(f"test {metric_name} (real only): {mean_primary_r:.4f} ± {se_primary_r:.4f}")
        summary_lines.append(f"test {metric_name} (uniform): {mean_primary_u:.4f} ± {se_primary_u:.4f}")
        summary_lines.append(f"test {metric_name} (theory) : {mean_primary_t:.4f} ± {se_primary_t:.4f}")

    return "\n".join(summary_lines)


def save_results_details(path, results, metric_name, stacked_real, stacked_uniform, stacked_theory):
    """
    Save the per-repetition detail file.
    """
    use_regression_like_format = (
        metric_name.lower() == "mse"
        and stacked_real.shape[1] >= 2
        and stacked_uniform.shape[1] >= 2
        and stacked_theory.shape[1] >= 2
    )

    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)

        if use_regression_like_format:
            writer.writerow([
                "rep",
                "alpha_hat_u",
                "alpha_hat_t",
                "eta_u",
                "eta_t",
                "mse_r",
                "mse_u",
                "mse_t",
                "b0_r",
                "b1_r",
                "b0_u",
                "b1_u",
                "b0_t",
                "b1_t",
            ])

            for idx, result in enumerate(results):
                writer.writerow([
                    result["rep"],
                    f"{result['alpha_hat_u']:.8f}",
                    f"{result['alpha_hat_t']:.8f}",
                    f"{result['eta_u']:.8f}",
                    f"{result['eta_t']:.8f}",
                    f"{result['primary_r']:.8f}",
                    f"{result['primary_u']:.8f}",
                    f"{result['primary_t']:.8f}",
                    f"{stacked_real[idx, 0]:.8f}",
                    f"{stacked_real[idx, 1]:.8f}",
                    f"{stacked_uniform[idx, 0]:.8f}",
                    f"{stacked_uniform[idx, 1]:.8f}",
                    f"{stacked_theory[idx, 0]:.8f}",
                    f"{stacked_theory[idx, 1]:.8f}",
                ])
        else:
            writer.writerow([
                "rep",
                "alpha_hat_u",
                "alpha_hat_t",
                "eta_u",
                "eta_t",
                f"{metric_name}_r",
                f"{metric_name}_u",
                f"{metric_name}_t",
            ])

            for result in results:
                writer.writerow([
                    result["rep"],
                    f"{result['alpha_hat_u']:.8f}",
                    f"{result['alpha_hat_t']:.8f}",
                    f"{result['eta_u']:.8f}",
                    f"{result['eta_t']:.8f}",
                    f"{result['primary_r']:.8f}",
                    f"{result['primary_u']:.8f}",
                    f"{result['primary_t']:.8f}",
                ])
