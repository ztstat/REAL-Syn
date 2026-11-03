"""
Aggregate one SHARPs train/candidate-budget grid experiment into tidy summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import evaluation
from .portable_metadata import portable_values, runtime_paths


GRID_MANIFEST_FILENAME = "experiment_manifest.json"
SPLIT_CHILDREN_FILENAME = "split_children.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
REP_PAYLOAD_FILENAME = "rep_metrics.json"


def load_json(path):
    """
    Load one UTF-8 JSON file.
    """
    with open(path, "r", encoding="utf-8") as file_obj:
        return runtime_paths(json.load(file_obj), Path(path).resolve().parent)


def format_multiplier_label(multiplier):
    """
    Format one candidate-train multiplier as a filesystem-safe label.
    """
    text = f"{float(multiplier):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def candidate_multiplier_column_label(multiplier):
    """
    Return the candidate-budget suffix used in wide CSV outputs.
    """
    return f"cand_{format_multiplier_label(multiplier)}x_train"


def load_results_details(path):
    """
    Load one results-details CSV into a list of dictionaries.
    """
    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def get_float_series(rows, column_name):
    """
    Extract one numeric CSV column as a numpy array.
    """
    return np.asarray([float(row[column_name]) for row in rows], dtype=float)


def add_m_class_tail_summary(row, summary_metrics):
    """
    Add optional chronological M-class tail metrics to one grid row.
    """
    reported = summary_metrics.get("reported_metric_series_by_name") or {}
    tail_series = reported.get("m_class_tail_mse")
    if tail_series is None:
        return

    method_output_names = {
        "real": "real",
        "uniform": "uniform",
        "dre_only": "dre_only",
        "influence_only": "influence_only",
        "theory": "theory",
    }
    missing_methods = set(method_output_names).difference(tail_series)
    if missing_methods:
        raise ValueError(
            "M-class tail MSE is missing method series: "
            + ", ".join(sorted(missing_methods))
        )
    for method_key, output_name in method_output_names.items():
        values = np.asarray(tail_series[method_key], dtype=float)
        mean_value, se_value = evaluation.mean_se(values)
        row[f"mean_m_class_tail_mse_{output_name}"] = mean_value
        row[f"se_m_class_tail_mse_{output_name}"] = se_value

    tail_count_series = reported.get("m_class_tail_test_count", {}).get(
        "test_set"
    )
    if tail_count_series is not None:
        counts = np.asarray(tail_count_series, dtype=float)
        row["mean_m_class_tail_test_count"] = float(np.mean(counts))
        row["min_m_class_tail_test_count"] = int(np.min(counts))
        row["max_m_class_tail_test_count"] = int(np.max(counts))

    tail_fraction_series = reported.get(
        "m_class_tail_test_fraction", {}
    ).get("test_set")
    if tail_fraction_series is not None:
        fractions = np.asarray(tail_fraction_series, dtype=float)
        mean_fraction, se_fraction = evaluation.mean_se(fractions)
        row["mean_m_class_tail_test_fraction"] = mean_fraction
        row["se_m_class_tail_test_fraction"] = se_fraction

    context = summary_metrics.get("reported_metric_context") or {}
    for context_key in (
        "m_class_tail_flux_threshold_raw",
        "m_class_tail_flux_threshold_log10",
        "m_class_tail_flux_threshold_standardized",
        "m_class_tail_threshold_inclusive",
    ):
        if context_key in context:
            row[context_key] = context[context_key]


def load_resolved_n_candidates(run_dir):
    """
    Read the resolved candidate budget from rep 0.
    """
    rep0_payload_path = Path(run_dir) / "reps" / "rep_000" / REP_PAYLOAD_FILENAME
    return int(load_json(rep0_payload_path)["resolved_n_candidates"])


def summarize_child_run(child_entry):
    """
    Build one long-form summary row for a completed
    `(train_fraction, candidate_train_multiplier)` run.
    """
    run_dir = Path(child_entry["run_dir"])
    summary_metrics = load_json(run_dir / "summary_metrics.json")
    run_manifest = load_json(run_dir / RUN_MANIFEST_FILENAME)
    results_details = load_results_details(run_dir / "results_details.csv")

    if len(results_details) == 0:
        raise ValueError(f"results_details.csv is empty for run: {run_dir}")

    mse_real = get_float_series(results_details, "mse_r")
    mse_uniform = get_float_series(results_details, "mse_u")
    mse_theory = get_float_series(results_details, "mse_t")
    mse_dre_only = (
        get_float_series(results_details, "mse_dre")
        if "mse_dre" in results_details[0]
        else None
    )
    mse_influence_only = (
        get_float_series(results_details, "mse_if")
        if "mse_if" in results_details[0]
        else None
    )
    alpha_uniform = get_float_series(results_details, "alpha_hat_u")
    alpha_theory = get_float_series(results_details, "alpha_hat_t")
    alpha_dre_only = (
        get_float_series(results_details, "alpha_hat_dre")
        if "alpha_hat_dre" in results_details[0]
        else None
    )
    alpha_influence_only = (
        get_float_series(results_details, "alpha_hat_if")
        if "alpha_hat_if" in results_details[0]
        else None
    )
    eta_uniform = get_float_series(results_details, "eta_u")
    eta_theory = get_float_series(results_details, "eta_t")
    eta_dre_only = (
        get_float_series(results_details, "eta_dre")
        if "eta_dre" in results_details[0]
        else None
    )
    eta_influence_only = (
        get_float_series(results_details, "eta_if")
        if "eta_if" in results_details[0]
        else None
    )

    data_summary = run_manifest["data_summary"]
    config = run_manifest["config"]
    overlap_warning_summary = summary_metrics.get("overlap_warning_summary") or {}

    row = {
        "split_index": int(child_entry["split_index"]),
        "train_fraction": float(child_entry["train_fraction"]),
        "test_fraction": float(child_entry["test_fraction"]),
        "candidate_train_multiplier": float(child_entry["candidate_train_multiplier"]),
        "candidate_train_pct": 100.0 * float(child_entry["candidate_train_multiplier"]),
        "target_n_candidates": int(child_entry["target_n_candidates"]),
        "run_dir": str(run_dir),
        "outer_reps": int(config["pipeline"]["reps"]),
        "inner_reps": int(config["pipeline"]["inner_reps"]),
        "pooled_runs": int(run_manifest["expected_reps"]),
        "alpha_target": float(config["pipeline"]["alpha"]),
        "resolved_n_candidates": load_resolved_n_candidates(run_dir),
        "n_train": int(data_summary["n_train"]),
        "n_test": int(data_summary["n_test"]),
        "x_dim": int(data_summary["x_dim"]),
        "metric_name": str(summary_metrics["metric_name"]),
        "mean_test_mse_real": evaluation.mean_se(mse_real)[0],
        "se_test_mse_real": evaluation.mean_se(mse_real)[1],
        "mean_test_mse_uniform": evaluation.mean_se(mse_uniform)[0],
        "se_test_mse_uniform": evaluation.mean_se(mse_uniform)[1],
        "mean_test_mse_theory": evaluation.mean_se(mse_theory)[0],
        "se_test_mse_theory": evaluation.mean_se(mse_theory)[1],
        "mean_accept_rate_uniform": evaluation.mean_se(alpha_uniform)[0],
        "se_accept_rate_uniform": evaluation.mean_se(alpha_uniform)[1],
        "mean_accept_rate_theory": evaluation.mean_se(alpha_theory)[0],
        "se_accept_rate_theory": evaluation.mean_se(alpha_theory)[1],
        "mean_eta_uniform": evaluation.mean_se(eta_uniform)[0],
        "se_eta_uniform": evaluation.mean_se(eta_uniform)[1],
        "mean_eta_theory": evaluation.mean_se(eta_theory)[0],
        "se_eta_theory": evaluation.mean_se(eta_theory)[1],
        "empirical_trace_cov_real": float(summary_metrics["trace_cov_real"]),
        "empirical_trace_cov_uniform": float(summary_metrics["trace_cov_uniform"]),
        "empirical_trace_cov_theory": float(summary_metrics["trace_cov_theory"]),
        "mean_analytic_cov_trace_real": float(summary_metrics["mean_analytic_cov_trace_real"]),
        "mean_analytic_cov_trace_uniform": float(summary_metrics["mean_analytic_cov_trace_uniform"]),
        "mean_analytic_cov_trace_theory": float(summary_metrics["mean_analytic_cov_trace_theory"]),
        "mean_delta_mse_uniform_rel": float(summary_metrics["mean_delta_mse_uniform_rel"]),
        "sd_delta_mse_uniform_rel": float(summary_metrics["sd_delta_mse_uniform_rel"]),
        "mean_delta_mse_theory_rel": float(summary_metrics["mean_delta_mse_theory_rel"]),
        "sd_delta_mse_theory_rel": float(summary_metrics["sd_delta_mse_theory_rel"]),
        "overlap_triggered_reps": int(overlap_warning_summary.get("triggered_reps", 0)),
        "overlap_total_reps": int(overlap_warning_summary.get("total_reps", 0)),
    }
    if mse_dre_only is not None:
        row["mean_test_mse_dre_only"] = evaluation.mean_se(mse_dre_only)[0]
        row["se_test_mse_dre_only"] = evaluation.mean_se(mse_dre_only)[1]
    if mse_influence_only is not None:
        row["mean_test_mse_influence_only"] = evaluation.mean_se(
            mse_influence_only
        )[0]
        row["se_test_mse_influence_only"] = evaluation.mean_se(
            mse_influence_only
        )[1]
    if alpha_dre_only is not None:
        row["mean_accept_rate_dre_only"] = evaluation.mean_se(alpha_dre_only)[0]
        row["se_accept_rate_dre_only"] = evaluation.mean_se(alpha_dre_only)[1]
    if alpha_influence_only is not None:
        row["mean_accept_rate_influence_only"] = evaluation.mean_se(
            alpha_influence_only
        )[0]
        row["se_accept_rate_influence_only"] = evaluation.mean_se(
            alpha_influence_only
        )[1]
    if eta_dre_only is not None:
        row["mean_eta_dre_only"] = evaluation.mean_se(eta_dre_only)[0]
        row["se_eta_dre_only"] = evaluation.mean_se(eta_dre_only)[1]
    if eta_influence_only is not None:
        row["mean_eta_influence_only"] = evaluation.mean_se(eta_influence_only)[0]
        row["se_eta_influence_only"] = evaluation.mean_se(eta_influence_only)[1]

    optional_summary_metric_keys = [
        "trace_cov_dre_only",
        "trace_cov_influence_only",
        "mean_analytic_cov_trace_dre_only",
        "mean_analytic_cov_trace_influence_only",
        "mean_delta_mse_dre_only_rel",
        "sd_delta_mse_dre_only_rel",
        "mean_delta_mse_influence_only_rel",
        "sd_delta_mse_influence_only_rel",
    ]
    optional_row_key_map = {
        "trace_cov_dre_only": "empirical_trace_cov_dre_only",
        "trace_cov_influence_only": "empirical_trace_cov_influence_only",
    }
    for summary_key in optional_summary_metric_keys:
        if summary_key in summary_metrics:
            row[optional_row_key_map.get(summary_key, summary_key)] = float(
                summary_metrics[summary_key]
            )
    add_m_class_tail_summary(row, summary_metrics)
    return row


def write_csv(path, rows):
    """
    Save a list of dictionaries to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as file_obj:
            file_obj.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(portable_values(rows, path.parent))


def build_metric_pivot_rows(
    summary_rows,
    real_key,
    method_key_specs,
):
    """
    Build a wide train-fraction-by-candidate-budget table for one metric family.
    """
    grouped_rows = {}
    candidate_train_multipliers = sorted(
        {float(row["candidate_train_multiplier"]) for row in summary_rows}
    )
    for row in summary_rows:
        grouped_rows.setdefault(float(row["train_fraction"]), []).append(row)

    pivot_rows = []
    for train_fraction in sorted(grouped_rows):
        rows_for_train = sorted(
            grouped_rows[train_fraction],
            key=lambda row: float(row["candidate_train_multiplier"]),
        )
        first_row = rows_for_train[0]
        pivot_row = {
            "train_fraction": float(first_row["train_fraction"]),
            "test_fraction": float(first_row["test_fraction"]),
            "n_train": int(first_row["n_train"]),
            "n_test": int(first_row["n_test"]),
        }
        if real_key is not None:
            pivot_row["real_only"] = float(first_row[real_key])

        by_multiplier = {
            float(row["candidate_train_multiplier"]): row
            for row in rows_for_train
        }
        for candidate_train_multiplier in candidate_train_multipliers:
            syn_row = by_multiplier[candidate_train_multiplier]
            syn_label = candidate_multiplier_column_label(candidate_train_multiplier)
            for method_label, metric_key in method_key_specs:
                if metric_key is not None and metric_key in syn_row:
                    pivot_row[f"{method_label}_{syn_label}"] = float(
                        syn_row[metric_key]
                    )
        pivot_rows.append(pivot_row)

    return pivot_rows


def analyze_grid_experiment(experiment_dir, output_dir=None):
    """
    Analyze one completed SHARPs grid experiment and save combined summaries.
    """
    experiment_dir = Path(experiment_dir).resolve()
    grid_manifest = load_json(experiment_dir / GRID_MANIFEST_FILENAME)
    if output_dir is None:
        output_dir = experiment_dir / "analysis"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for split_entry in grid_manifest["split_plan"]:
        split_dir = Path(split_entry["split_dir"])
        split_children = load_json(split_dir / SPLIT_CHILDREN_FILENAME)
        for child_entry in split_children["child_runs"]:
            summary_rows.append(summarize_child_run(child_entry))

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            float(row["train_fraction"]),
            float(row["candidate_train_multiplier"]),
        ),
    )

    write_csv(output_dir / "grid_summary.csv", summary_rows)
    evaluation.save_json(
        str(output_dir / "grid_summary.json"),
        {
            "experiment_dir": str(experiment_dir),
            "train_fractions": grid_manifest["train_fractions"],
            "candidate_train_multipliers": grid_manifest["candidate_train_multipliers"],
            "n_combinations": int(len(summary_rows)),
            "rows": summary_rows,
        },
    )

    pivot_specs = [
        (
            "grid_test_mse_pivot.csv",
            "mean_test_mse_real",
            [
                ("uniform", "mean_test_mse_uniform"),
                ("dre_only", "mean_test_mse_dre_only"),
                ("influence_only", "mean_test_mse_influence_only"),
                ("theory", "mean_test_mse_theory"),
            ],
        ),
        (
            "grid_empirical_trace_cov_pivot.csv",
            "empirical_trace_cov_real",
            [
                ("uniform", "empirical_trace_cov_uniform"),
                ("dre_only", "empirical_trace_cov_dre_only"),
                ("influence_only", "empirical_trace_cov_influence_only"),
                ("theory", "empirical_trace_cov_theory"),
            ],
        ),
        (
            "grid_analytic_trace_cov_pivot.csv",
            "mean_analytic_cov_trace_real",
            [
                ("uniform", "mean_analytic_cov_trace_uniform"),
                ("dre_only", "mean_analytic_cov_trace_dre_only"),
                ("influence_only", "mean_analytic_cov_trace_influence_only"),
                ("theory", "mean_analytic_cov_trace_theory"),
            ],
        ),
        (
            "grid_delta_mse_rel_pivot.csv",
            None,
            [
                ("uniform", "mean_delta_mse_uniform_rel"),
                ("dre_only", "mean_delta_mse_dre_only_rel"),
                ("influence_only", "mean_delta_mse_influence_only_rel"),
                ("theory", "mean_delta_mse_theory_rel"),
            ],
        ),
        (
            "grid_eta_pivot.csv",
            None,
            [
                ("uniform", "mean_eta_uniform"),
                ("dre_only", "mean_eta_dre_only"),
                ("influence_only", "mean_eta_influence_only"),
                ("theory", "mean_eta_theory"),
            ],
        ),
        (
            "grid_accept_rate_pivot.csv",
            None,
            [
                ("uniform", "mean_accept_rate_uniform"),
                ("dre_only", "mean_accept_rate_dre_only"),
                ("influence_only", "mean_accept_rate_influence_only"),
                ("theory", "mean_accept_rate_theory"),
            ],
        ),
    ]
    if (
        summary_rows
        and "mean_m_class_tail_mse_real" in summary_rows[0]
    ):
        pivot_specs.insert(
            1,
            (
                "grid_m_class_tail_mse_pivot.csv",
                "mean_m_class_tail_mse_real",
                [
                    (
                        "uniform",
                        "mean_m_class_tail_mse_uniform",
                    ),
                    (
                        "dre_only",
                        "mean_m_class_tail_mse_dre_only",
                    ),
                    (
                        "influence_only",
                        "mean_m_class_tail_mse_influence_only",
                    ),
                    (
                        "theory",
                        "mean_m_class_tail_mse_theory",
                    ),
                ],
            ),
        )
    for filename, real_key, method_key_specs in pivot_specs:
        write_csv(
            output_dir / filename,
            build_metric_pivot_rows(
                summary_rows=summary_rows,
                real_key=real_key,
                method_key_specs=method_key_specs,
            ),
        )

    return {
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "n_combinations": int(len(summary_rows)),
    }


def build_parser():
    """
    Build the CLI parser for direct post-hoc grid analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze one completed SHARPs grid experiment."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv=None):
    """
    Run the requested SHARPs grid analysis.
    """
    args = build_parser().parse_args(argv)
    result = analyze_grid_experiment(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
    )
    print(f"Analyzed grid experiment: {result['experiment_dir']}")
    print(f"Analysis output directory: {result['output_dir']}")
    print(f"Combined combinations: {result['n_combinations']}")


if __name__ == "__main__":
    main()
