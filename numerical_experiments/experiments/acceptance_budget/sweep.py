"""
Run an alpha-budget sweep for the misspecified Real-Syn experiment.
"""


import csv
import json
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from common import evaluation


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


METHODS = [
    "real only",
    "uniform",
    "density ratio only",
    "influence only",
    "Real-Syn",
]


METHOD_STYLE = {
    "real only": {"color": "#111827", "marker": "o", "linestyle": "--"},
    "uniform": {"color": "#4B5563", "marker": "s", "linestyle": "-"},
    "density ratio only": {"color": "#1F77B4", "marker": "^", "linestyle": "-"},
    "influence only": {"color": "#2CA02C", "marker": "D", "linestyle": "-"},
    "Real-Syn": {"color": "#D62728", "marker": "o", "linestyle": "-"},
}


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


def load_json(path):
    with open(resolve_project_path(path), "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2)


def alpha_token(alpha):
    return f"{alpha:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_alpha_grid(args, config):
    if args.alphas is not None:
        return [float(value.strip()) for value in args.alphas.split(",") if value.strip()]

    if "alpha_sweep" in config and "alphas" in config["alpha_sweep"]:
        return [float(value) for value in config["alpha_sweep"]["alphas"]]

    return [0.05, 0.10, 0.20, 0.30, 0.50]


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def numeric_column(rows, column):
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def find_metric_column(rows, suffix):
    columns = rows[0].keys()
    for column in columns:
        if column.lower().endswith(suffix):
            return column
    raise KeyError(f"Could not find metric column ending with {suffix}.")


def mean_se_or_blank(values):
    if values is None:
        return "", ""
    mean_value, se_value = evaluation.mean_se(values)
    return mean_value, se_value


def get_child_n_candidates(base_config, alpha):
    """
    Return the candidate pool size for this alpha.
    """
    sweep_config = base_config.get("alpha_sweep", {})
    mode = sweep_config.get("candidate_pool_mode", "fixed_expected_accept_count")

    if mode == "fixed_expected_accept_count":
        default_target = (
            float(base_config["pipeline"]["alpha"])
            * int(base_config["pipeline"]["n_candidates"])
        )
        target_accepted = float(
            sweep_config.get("target_expected_accept_count", default_target)
        )
        return int(round(target_accepted / float(alpha)))

    raise ValueError(f"Unknown candidate_pool_mode: {mode}")


def build_child_config(base_config, alpha, sweep_output_dir):
    config = deepcopy(base_config)
    config["pipeline"]["alpha"] = float(alpha)
    config["pipeline"]["n_candidates"] = get_child_n_candidates(base_config, alpha)
    config["pipeline"]["save_candidate_csv"] = False
    config["pipeline"]["save_histograms"] = False
    config["pipeline"]["make_xy_band_plots"] = False
    config["output_base_dir"] = sweep_output_dir
    config["experiment_name"] = f"alpha_{alpha_token(alpha)}"
    return config


def collect_alpha_rows(alpha, output_dir):
    child_config = load_json(os.path.join(output_dir, "config.json"))
    summary = load_json(os.path.join(output_dir, "summary_metrics.json"))
    details = read_csv_rows(os.path.join(output_dir, "results_details.txt"))
    extra_details = read_csv_rows(os.path.join(output_dir, "results_details_extra_baselines.txt"))

    metric_name = summary["metric_name"]
    n_candidates = int(child_config["pipeline"]["n_candidates"])
    expected_accept_count = float(alpha) * n_candidates
    metric_r = f"{metric_name}_r"
    metric_u = f"{metric_name}_u"
    metric_t = f"{metric_name}_t"
    metric_ratio = f"{metric_name}_ratio_only"
    metric_influence = f"{metric_name}_influence_only"

    per_method = {
        "real only": {
            "accept": None,
            "eta": None,
            "trace_cov": summary["trace_cov_real"],
            "metric": numeric_column(details, metric_r),
        },
        "uniform": {
            "accept": numeric_column(details, "alpha_hat_u"),
            "eta": numeric_column(details, "eta_u"),
            "trace_cov": summary["trace_cov_uniform"],
            "metric": numeric_column(details, metric_u),
        },
        "density ratio only": {
            "accept": numeric_column(extra_details, "alpha_hat_ratio_only"),
            "eta": numeric_column(extra_details, "eta_ratio_only"),
            "trace_cov": summary["trace_cov_ratio_only"],
            "metric": numeric_column(extra_details, metric_ratio),
        },
        "influence only": {
            "accept": numeric_column(extra_details, "alpha_hat_influence_only"),
            "eta": numeric_column(extra_details, "eta_influence_only"),
            "trace_cov": summary["trace_cov_influence_only"],
            "metric": numeric_column(extra_details, metric_influence),
        },
        "Real-Syn": {
            "accept": numeric_column(details, "alpha_hat_t"),
            "eta": numeric_column(details, "eta_t"),
            "trace_cov": summary["trace_cov_real_syn"],
            "metric": numeric_column(details, metric_t),
        },
    }

    rows = []
    for method in METHODS:
        values = per_method[method]
        accept_mean, accept_se = mean_se_or_blank(values["accept"])
        eta_mean, eta_se = mean_se_or_blank(values["eta"])
        metric_mean, metric_se = evaluation.mean_se(values["metric"])
        rows.append({
            "alpha": float(alpha),
            "n_candidates": n_candidates,
            "expected_accept_count": expected_accept_count,
            "method": method,
            "accept_rate_mean": accept_mean,
            "accept_rate_se": accept_se,
            "eta_trace_mean": eta_mean,
            "eta_trace_se": eta_se,
            "trace_cov_beta": float(values["trace_cov"]),
            f"test_{metric_name}_mean": metric_mean,
            f"test_{metric_name}_se": metric_se,
            "child_output_dir": output_dir,
        })
    return rows, metric_name


def save_summary_csv(path, rows, metric_name):
    fieldnames = [
        "alpha",
        "n_candidates",
        "expected_accept_count",
        "method",
        "accept_rate_mean",
        "accept_rate_se",
        "eta_trace_mean",
        "eta_trace_se",
        "trace_cov_beta",
        f"test_{metric_name}_mean",
        f"test_{metric_name}_se",
        "child_output_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_text(path, rows, metric_name):
    lines = ["=== Alpha sweep summary ===", ""]
    for alpha in sorted({row["alpha"] for row in rows}):
        first_row = next(item for item in rows if item["alpha"] == alpha)
        lines.append(
            f"alpha = {alpha:.2f} "
            f"(M_cand={first_row['n_candidates']}, "
            f"E[accepted]={first_row['expected_accept_count']:.1f})"
        )
        for method in METHODS:
            row = next(item for item in rows if item["alpha"] == alpha and item["method"] == method)
            lines.append(
                f"  {method:18s} trace_cov={row['trace_cov_beta']:.6f}, "
                f"test_{metric_name}={row[f'test_{metric_name}_mean']:.4f} "
                f"+/- {row[f'test_{metric_name}_se']:.4f}"
            )
        lines.append("")
    evaluation.save_text(path, "\n".join(lines).rstrip())


def plot_metric(rows, metric_key, ylabel, output_path, include_real=True):
    evaluation.apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    methods = METHODS if include_real else [method for method in METHODS if method != "real only"]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows = sorted(method_rows, key=lambda row: row["alpha"])
        x_values = [row["alpha"] for row in method_rows]
        y_values = [row[metric_key] for row in method_rows]
        style = METHOD_STYLE[method]
        ax.plot(
            x_values,
            y_values,
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5.0,
        )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    evaluation.format_publication_axis(ax)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D1D5DB",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=evaluation.PUBLICATION_DPI, bbox_inches="tight")
    plt.close(fig)


def run_sweep(
    config_path,
    alphas_arg=None,
    output_dir=None,
    forced_candidate_pool_mode=None,
    quick=False,
):
    """
    Run one alpha sweep from a base config.
    """
    base_config = load_json(config_path)
    if quick:
        base_config["pipeline"]["reps"] = 2

    if forced_candidate_pool_mode is not None:
        base_config.setdefault("alpha_sweep", {})
        base_config["alpha_sweep"]["candidate_pool_mode"] = forced_candidate_pool_mode

    class Args:
        pass

    args = Args()
    args.alphas = alphas_arg
    alpha_grid = parse_alpha_grid(args, base_config)

    output_base_dir = output_dir if output_dir is not None else base_config["output_base_dir"]
    output_base_dir = resolve_project_path(output_base_dir)
    experiment_name = base_config.get("alpha_sweep", {}).get("experiment_name", "alpha_sweep_d2")
    sweep_output_dir = output_dir or evaluation.create_timestamp_output_dir(output_base_dir, experiment_name)

    save_json(
        os.path.join(sweep_output_dir, "alpha_sweep_config.json"),
        {
            "base_config_path": resolve_project_path(config_path),
            "alpha_grid": alpha_grid,
            "base_config": base_config,
        },
    )

    all_rows = []
    metric_name = None
    child_outputs = []

    for alpha in alpha_grid:
        child_config = build_child_config(base_config, alpha, sweep_output_dir)
        child_config_path = os.path.join(sweep_output_dir, f"config_alpha_{alpha_token(alpha)}.json")
        save_json(child_config_path, child_config)

        print(f"Running alpha={alpha:.3f} ...")
        child_output_dir, _ = run_child_pipeline(child_config_path)
        child_outputs.append(child_output_dir)

        alpha_rows, current_metric_name = collect_alpha_rows(alpha, child_output_dir)
        all_rows.extend(alpha_rows)
        metric_name = current_metric_name

    save_summary_csv(
        path=os.path.join(sweep_output_dir, "alpha_sweep_summary.csv"),
        rows=all_rows,
        metric_name=metric_name,
    )
    save_summary_text(
        path=os.path.join(sweep_output_dir, "alpha_sweep_summary.txt"),
        rows=all_rows,
        metric_name=metric_name,
    )

    plot_metric(
        rows=all_rows,
        metric_key="trace_cov_beta",
        ylabel=r"$\mathrm{tr}\{\widehat{\mathrm{Cov}}(\widehat{\beta})\}$",
        output_path=os.path.join(sweep_output_dir, "alpha_sweep_trace_cov.png"),
    )
    plot_metric(
        rows=all_rows,
        metric_key=f"test_{metric_name}_mean",
        ylabel=f"test {metric_name.upper()}",
        output_path=os.path.join(sweep_output_dir, "alpha_sweep_test_mse.png"),
    )

    print("")
    print(f"Alpha sweep output directory: {sweep_output_dir}")
    print("Expected main files:")
    print("  - alpha_sweep_config.json")
    print("  - alpha_sweep_summary.csv")
    print("  - alpha_sweep_summary.txt")
    print("  - alpha_sweep_trace_cov.png")
    print("  - alpha_sweep_test_mse.png")
    print("Per-alpha output directories:")
    for child_output in child_outputs:
        print(f"  - {child_output}")

    return sweep_output_dir

def run_child_pipeline(config_path):
    from common.ols_pipeline import run
    return run(config_path), ""
