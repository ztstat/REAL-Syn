"""Full-test and M1.0+ heatmaps and tables from saved chronological results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import (
        BoundaryNorm,
        ListedColormap,
        LogNorm,
        Normalize,
        TwoSlopeNorm,
    )
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependency. This script requires numpy, pandas, "
        "and matplotlib in the active Python environment."
    ) from exc


METHODS = ("uniform", "dre_only", "influence_only", "theory")
PAIRWISE_METHODS = ("dre_only", "influence_only", "theory")
ALL_METHODS = ("real", *METHODS)

METHOD_LABELS = {
    "uniform": "Uniform",
    "dre_only": "DRE-only",
    "influence_only": "Influence-only",
    "theory": "REAL-Syn",
}
METHOD_SHORT_LABELS = {
    "uniform": "U",
    "dre_only": "DRE",
    "influence_only": "IF",
    "theory": "REAL-Syn",
}
ALL_METHOD_LABELS = {"real": "Real-only", **METHOD_LABELS}
ALL_METHOD_SHORT_LABELS = {"real": "R", **METHOD_SHORT_LABELS}

METHOD_COLORS = {
    "real": "#4C78A8",
    "uniform": "#FFFFFF",
    "dre_only": "#C0C0C0",
    "influence_only": "#000000",
    "theory": "#B54A4A",
}

TAIL_MEAN_COLUMN = "mean_m_class_tail_mse_{method}"
TAIL_SE_COLUMN = "se_m_class_tail_mse_{method}"
TAIL_SD_COLUMN = "sd_m_class_tail_mse_{method}"


def required_columns() -> list[str]:
    columns = [
        "train_fraction",
        "candidate_train_multiplier",
        "pooled_runs",
        "mean_test_mse_real",
        "se_test_mse_real",
        "empirical_trace_cov_real",
        "mean_analytic_cov_trace_real",
    ]
    for method in METHODS:
        columns.extend(
            [
                f"mean_test_mse_{method}",
                f"se_test_mse_{method}",
                f"mean_delta_mse_{method}_rel",
                f"sd_delta_mse_{method}_rel",
                f"empirical_trace_cov_{method}",
                f"mean_analytic_cov_trace_{method}",
                f"mean_accept_rate_{method}",
                f"mean_eta_{method}",
            ]
        )
    for method in ALL_METHODS:
        columns.extend(
            [
                TAIL_MEAN_COLUMN.format(method=method),
                TAIL_SE_COLUMN.format(method=method),
            ]
        )
    columns.extend(
        [
            "mean_m_class_tail_test_count",
            "min_m_class_tail_test_count",
            "max_m_class_tail_test_count",
            "mean_m_class_tail_test_fraction",
            "se_m_class_tail_test_fraction",
            "m_class_tail_flux_threshold_raw",
            "m_class_tail_flux_threshold_log10",
            "m_class_tail_flux_threshold_standardized",
            "m_class_tail_threshold_inclusive",
        ]
    )
    return columns


def load_grid(analysis_dir: Path) -> tuple[pd.DataFrame, dict]:
    summary_path = analysis_dir / "grid_summary.csv"
    metadata_path = analysis_dir / "grid_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing grid summary: {summary_path}")

    grid = pd.read_csv(summary_path)
    missing = [column for column in required_columns() if column not in grid.columns]
    if missing:
        raise KeyError(f"Grid summary is missing required columns: {missing}")

    grid_keys = ["train_fraction", "candidate_train_multiplier"]
    duplicate_mask = grid.duplicated(grid_keys, keep=False)
    if duplicate_mask.any():
        duplicates = grid.loc[duplicate_mask, grid_keys].drop_duplicates()
        raise ValueError(
            "Expected one aggregate row per grid cell; duplicate keys found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    metadata: dict = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
    validate_tail_metric_contract(grid)
    return grid, metadata


def validate_tail_metric_contract(grid: pd.DataFrame) -> None:
    if (grid["pooled_runs"] <= 0).any():
        raise ValueError("pooled_runs must be positive in every grid cell.")

    tail_counts = grid["mean_m_class_tail_test_count"].to_numpy(
        dtype=float
    )
    if np.any(~np.isfinite(tail_counts)) or np.any(tail_counts < 0.0):
        raise ValueError("M-class tail counts must be finite and nonnegative.")

    positive_tail_mask = tail_counts > 0.0
    for method in ALL_METHODS:
        mean_values = grid[
            TAIL_MEAN_COLUMN.format(method=method)
        ].to_numpy(dtype=float)
        se_values = grid[
            TAIL_SE_COLUMN.format(method=method)
        ].to_numpy(dtype=float)
        if np.any(~np.isfinite(mean_values[positive_tail_mask])):
            raise ValueError(
                f"M-class tail MSE is non-finite for {method} despite a "
                "nonempty tail subset."
            )
        if np.any(~np.isfinite(se_values[positive_tail_mask])) or np.any(
            se_values[positive_tail_mask] < 0.0
        ):
            raise ValueError(
                f"M-class tail MSE SE is invalid for {method}."
            )

    raw_thresholds = grid[
        "m_class_tail_flux_threshold_raw"
    ].dropna().unique()
    log10_thresholds = grid[
        "m_class_tail_flux_threshold_log10"
    ].dropna().unique()
    standardized_thresholds = grid[
        "m_class_tail_flux_threshold_standardized"
    ].dropna().unique()
    if not (
        len(raw_thresholds)
        == len(log10_thresholds)
        == len(standardized_thresholds)
        == 1
    ):
        raise ValueError(
            "Expected one shared M-class tail threshold across the grid."
        )
    raw_threshold = float(raw_thresholds[0])
    log10_threshold = float(log10_thresholds[0])
    if raw_threshold <= 0.0 or not np.isclose(
        np.log10(raw_threshold), log10_threshold
    ):
        raise ValueError("Raw and log10 M-class thresholds disagree.")
    if not np.isfinite(float(standardized_thresholds[0])):
        raise ValueError("Standardized M-class threshold must be finite.")

    inclusive_values = {
        str(value).strip().lower()
        for value in grid["m_class_tail_threshold_inclusive"].unique()
    }
    if inclusive_values != {"true"}:
        raise ValueError(
            "This visualization expects the inclusive M1.0 tail boundary."
        )


def prepare_plot_data(grid: pd.DataFrame) -> pd.DataFrame:
    plot_df = grid.copy()
    for method in ALL_METHODS:
        # The grid analyzer saves
        # SE = sample_SD / sqrt(pooled_runs), so invert that relation here.
        plot_df[f"sd_test_mse_{method}"] = (
            plot_df[f"se_test_mse_{method}"]
            * np.sqrt(plot_df["pooled_runs"])
        )
        plot_df[TAIL_SD_COLUMN.format(method=method)] = (
            plot_df[TAIL_SE_COLUMN.format(method=method)]
            * np.sqrt(plot_df["pooled_runs"])
        )

    for method in METHODS:
        plot_df[f"mse_change_pct_{method}"] = (
            100.0 * plot_df[f"mean_delta_mse_{method}_rel"]
        )
        plot_df[f"sd_mse_change_pct_{method}"] = (
            100.0 * plot_df[f"sd_delta_mse_{method}_rel"]
        )
        plot_df[f"mse_vs_uniform_pct_{method}"] = 100.0 * (
            plot_df[f"mean_test_mse_{method}"]
            - plot_df["mean_test_mse_uniform"]
        ) / plot_df["mean_test_mse_uniform"]
        plot_df[f"empirical_trace_cov_diff_{method}"] = (
            plot_df[f"empirical_trace_cov_{method}"]
            - plot_df["empirical_trace_cov_real"]
        )
        plot_df[f"analytic_trace_cov_diff_{method}"] = (
            plot_df[f"mean_analytic_cov_trace_{method}"]
            - plot_df["mean_analytic_cov_trace_real"]
        )
        plot_df[f"tail_mse_change_pct_{method}"] = 100.0 * (
            plot_df[TAIL_MEAN_COLUMN.format(method=method)]
            - plot_df[TAIL_MEAN_COLUMN.format(method="real")]
        ) / plot_df[TAIL_MEAN_COLUMN.format(method="real")]
        plot_df[f"tail_mse_vs_uniform_pct_{method}"] = 100.0 * (
            plot_df[TAIL_MEAN_COLUMN.format(method=method)]
            - plot_df[TAIL_MEAN_COLUMN.format(method="uniform")]
        ) / plot_df[TAIL_MEAN_COLUMN.format(method="uniform")]

    plot_df["mse_change_pct_real"] = 0.0
    plot_df["sd_mse_change_pct_real"] = 0.0
    plot_df["mse_vs_uniform_pct_real"] = 100.0 * (
        plot_df["mean_test_mse_real"] - plot_df["mean_test_mse_uniform"]
    ) / plot_df["mean_test_mse_uniform"]
    plot_df["empirical_trace_cov_diff_real"] = 0.0
    plot_df["analytic_trace_cov_diff_real"] = 0.0
    plot_df["tail_mse_change_pct_real"] = 0.0
    plot_df["tail_mse_vs_uniform_pct_real"] = 100.0 * (
        plot_df[TAIL_MEAN_COLUMN.format(method="real")]
        - plot_df[TAIL_MEAN_COLUMN.format(method="uniform")]
    ) / plot_df[TAIL_MEAN_COLUMN.format(method="uniform")]

    synthetic_columns = [f"mean_test_mse_{method}" for method in METHODS]
    best_synthetic_column = plot_df[synthetic_columns].idxmin(axis=1)
    plot_df["best_synthetic_method"] = best_synthetic_column.str.replace(
        "mean_test_mse_", "", regex=False
    )
    plot_df["best_synthetic_method_label"] = plot_df[
        "best_synthetic_method"
    ].map(METHOD_LABELS)

    tail_synthetic_columns = [
        TAIL_MEAN_COLUMN.format(method=method) for method in METHODS
    ]
    best_tail_synthetic_column = plot_df[tail_synthetic_columns].idxmin(
        axis=1
    )
    plot_df["best_tail_synthetic_method"] = (
        best_tail_synthetic_column.str.replace(
            "mean_m_class_tail_mse_", "", regex=False
        )
    )
    plot_df["best_tail_synthetic_method_label"] = plot_df[
        "best_tail_synthetic_method"
    ].map(METHOD_LABELS)

    tail_all_columns = [
        TAIL_MEAN_COLUMN.format(method=method) for method in ALL_METHODS
    ]
    best_tail_column = plot_df[tail_all_columns].idxmin(axis=1)
    plot_df["best_tail_method"] = best_tail_column.str.replace(
        "mean_m_class_tail_mse_", "", regex=False
    )
    plot_df["best_tail_method_label"] = plot_df[
        "best_tail_method"
    ].map(ALL_METHOD_LABELS)

    guarded_results = plot_df.apply(choose_guarded_best_method, axis=1)
    return pd.concat([plot_df, guarded_results], axis=1)


def finite_method_values(
    row: pd.Series, column_by_method: dict[str, str]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for method, column in column_by_method.items():
        value = row[column]
        if pd.notna(value) and np.isfinite(value):
            values[method] = float(value)
    return values


def mse_delta_interval_covers_zero(
    row: pd.Series, synthetic_method: str
) -> bool:
    mean_delta = row[f"mean_delta_mse_{synthetic_method}_rel"]
    sd_delta = row[f"sd_delta_mse_{synthetic_method}_rel"]
    if not np.isfinite(mean_delta) or not np.isfinite(sd_delta):
        return True
    return (mean_delta - sd_delta) <= 0.0 <= (mean_delta + sd_delta)


def choose_guarded_best_method(row: pd.Series) -> pd.Series:
    mse_columns = {
        method: f"mean_test_mse_{method}" for method in ALL_METHODS
    }
    # Real-only is fixed across repetitions, so its empirical covariance is
    # zero by construction. Keep it eligible only for the MSE comparison.
    covariance_columns = {
        method: f"empirical_trace_cov_{method}" for method in METHODS
    }
    mse_values = finite_method_values(row, mse_columns)
    covariance_values = finite_method_values(row, covariance_columns)
    synthetic_mse_values = {
        method: value for method, value in mse_values.items() if method in METHODS
    }
    if not mse_values or not synthetic_mse_values:
        raise ValueError("A grid cell has no finite test-MSE values.")
    if not covariance_values:
        raise ValueError(
            "A grid cell has no finite synthetic empirical covariance values."
        )

    best_test_mse_method = min(mse_values, key=mse_values.get)
    best_synthetic_mse_method = min(
        synthetic_mse_values, key=synthetic_mse_values.get
    )
    comparison_method = (
        best_test_mse_method
        if best_test_mse_method != "real"
        else best_synthetic_mse_method
    )
    mean_delta = row[f"mean_delta_mse_{comparison_method}_rel"]
    sd_delta = row[f"sd_delta_mse_{comparison_method}_rel"]
    covers_zero = mse_delta_interval_covers_zero(row, comparison_method)

    if covers_zero:
        selected_method = min(covariance_values, key=covariance_values.get)
        selection_basis = "empirical_trace_cov"
    else:
        selected_method = best_test_mse_method
        selection_basis = "test_mse"

    best_covariance_method = min(
        covariance_values, key=covariance_values.get
    )
    return pd.Series(
        {
            "best_method": selected_method,
            "best_method_label": ALL_METHOD_LABELS[selected_method],
            "best_method_selection_basis": selection_basis,
            "best_test_mse_method": best_test_mse_method,
            "best_test_mse_method_label": ALL_METHOD_LABELS[
                best_test_mse_method
            ],
            "best_mse_comparison_method": comparison_method,
            "best_mse_comparison_method_label": METHOD_LABELS[
                comparison_method
            ],
            "best_mse_delta_pct": 100.0 * mean_delta,
            "best_mse_sd_pct": 100.0 * sd_delta,
            "best_mse_interval_covers_zero": covers_zero,
            "best_empirical_cov_method": best_covariance_method,
            "best_empirical_cov_method_label": ALL_METHOD_LABELS[
                best_covariance_method
            ],
        }
    )


class HeatmapContext:
    def __init__(self, plot_df: pd.DataFrame):
        self.plot_df = plot_df
        self.train_order = sorted(plot_df["train_fraction"].unique())
        self.candidate_order = sorted(
            plot_df["candidate_train_multiplier"].unique()
        )
        self.train_labels = [f"{value:.2f}" for value in self.train_order]
        self.candidate_labels = [
            f"{100.0 * value:.0f}%" for value in self.candidate_order
        ]

    def matrix(self, value_column: str) -> pd.DataFrame:
        return (
            self.plot_df.pivot(
                index="candidate_train_multiplier",
                columns="train_fraction",
                values=value_column,
            )
            .reindex(index=self.candidate_order, columns=self.train_order)
            .set_axis(self.candidate_labels, axis="index")
            .set_axis(self.train_labels, axis="columns")
        )


def symmetric_vmax_from_mean_sd(
    value_matrices: Sequence[pd.DataFrame],
    sd_matrices: Sequence[pd.DataFrame | None] | None = None,
) -> float:
    if sd_matrices is None:
        sd_matrices = [None] * len(value_matrices)

    bounds = []
    for value_matrix, sd_matrix in zip(value_matrices, sd_matrices):
        values = value_matrix.to_numpy(dtype=float)
        if sd_matrix is None:
            lower = values
            upper = values
        else:
            standard_deviations = sd_matrix.to_numpy(dtype=float)
            lower = values - standard_deviations
            upper = values + standard_deviations
        bounds.extend([lower.ravel(), upper.ravel()])

    finite_bounds = np.concatenate(bounds)
    finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
    if finite_bounds.size == 0:
        raise ValueError("No finite heatmap values found.")
    limit = float(np.max(np.abs(finite_bounds)))
    return limit if limit > 0.0 else 1.0


def symmetric_vmax(value_matrices: Sequence[pd.DataFrame]) -> float:
    return symmetric_vmax_from_mean_sd(value_matrices)


def copy_cmap(name: str, missing_color: str = "#E6E6E6"):
    color_map = plt.get_cmap(name).copy()
    color_map.set_bad(missing_color)
    return color_map


def text_color_for_value(image, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.5 else "black"


def style_heatmap_axis(
    axis,
    matrix: pd.DataFrame,
    title: str,
) -> None:
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_xticklabels(matrix.columns)
    axis.set_yticks(np.arange(matrix.shape[0]))
    axis.set_yticklabels(matrix.index)
    axis.invert_yaxis()
    axis.set_xlabel("Downstream chronological training-set proportion")
    axis.set_ylabel(r"Candidate multiplier (% of $n_{\mathrm{train}}$)")
    axis.set_title(title)
    axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)


def annotate_heatmap(
    axis,
    matrix: pd.DataFrame,
    image,
    formatter: Callable[[float, int, int], str],
    *,
    fontsize: float = 8.0,
    mark_missing: bool = False,
) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iat[row_index, column_index]
            if pd.notna(value) and np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    formatter(float(value), row_index, column_index).replace(
                        "±", r"$\pm$"
                    ),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color_for_value(image, float(value)),
                )
            elif mark_missing:
                axis.text(
                    column_index,
                    row_index,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="#777777",
                )


def save_figure(
    figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> None:
    for output_format in formats:
        output_path = output_dir / f"{stem}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(output_path, **save_kwargs)
        print(f"Saved {output_path}")
    plt.close(figure)


def experiment_caption(grid: pd.DataFrame) -> str:
    parts = ["Strict chronological DRE/downstream grid"]
    if {"n_train", "n_test"}.issubset(grid.columns):
        total_sizes = (grid["n_train"] + grid["n_test"]).dropna().unique()
        if len(total_sizes) == 1:
            parts.append(f"N={int(total_sizes[0]):,}")
    if "pooled_runs" in grid.columns:
        pooled_runs = grid["pooled_runs"].dropna().unique()
        if len(pooled_runs) == 1:
            parts.append(f"{int(pooled_runs[0])} pooled runs/cell")
    if "alpha_target" in grid.columns:
        alpha_values = grid["alpha_target"].dropna().unique()
        if len(alpha_values) == 1:
            parts.append(f"target α={float(alpha_values[0]):g}")
    return " | ".join(parts)


def tail_experiment_caption(grid: pd.DataFrame) -> str:
    caption = experiment_caption(grid)
    threshold_values = grid[
        "m_class_tail_flux_threshold_raw"
    ].dropna().unique()
    count_min = int(grid["min_m_class_tail_test_count"].min())
    count_max = int(grid["max_m_class_tail_test_count"].max())
    if len(threshold_values) == 1:
        threshold_text = f"flux ≥ {float(threshold_values[0]):g} W/m²"
    else:
        threshold_text = "M1.0-or-higher flux"
    count_text = (
        f"{count_min} tail rows/cell"
        if count_min == count_max
        else f"{count_min}–{count_max} tail rows/cell"
    )
    return f"{caption} | {threshold_text} | {count_text}"


def plot_relative_mse_vs_real(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    matrices = [
        context.matrix(f"mse_change_pct_{method}") for method in METHODS
    ]
    sd_matrices = [
        context.matrix(f"sd_mse_change_pct_{method}") for method in METHODS
    ]
    limit = symmetric_vmax_from_mean_sd(matrices, sd_matrices)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    color_map = copy_cmap("RdBu_r")

    figure, axes = plt.subplots(
        2, 2, figsize=(16, 10), constrained_layout=True
    )
    for axis, method, matrix, sd_matrix in zip(
        axes.ravel(), METHODS, matrices, sd_matrices
    ):
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap=color_map,
            norm=norm,
            aspect="auto",
        )
        style_heatmap_axis(
            axis, matrix, f"{METHOD_LABELS[method]} vs Real-only"
        )
        annotate_heatmap(
            axis,
            matrix,
            image,
            lambda value, row, column: (
                f"{value:+.2f}\n±{sd_matrix.iat[row, column]:.2f}"
            ),
            fontsize=7.4,
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92
    )
    colorbar.set_label("Mean relative test-MSE change (%)")
    figure.suptitle("Relative Test MSE vs Real-only", fontsize=15)
    save_figure(
        figure,
        output_dir,
        "relative_mse_change_all_methods",
        formats,
        dpi,
    )


def plot_relative_mse_vs_uniform(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    matrices = [
        context.matrix(f"mse_vs_uniform_pct_{method}")
        for method in PAIRWISE_METHODS
    ]
    limit = symmetric_vmax(matrices)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    color_map = copy_cmap("RdBu_r")

    figure, axes = plt.subplots(
        1, 3, figsize=(18, 5.3), constrained_layout=True
    )
    for axis, method, matrix in zip(axes, PAIRWISE_METHODS, matrices):
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap=color_map,
            norm=norm,
            aspect="auto",
        )
        style_heatmap_axis(
            axis, matrix, f"{METHOD_LABELS[method]} - Uniform"
        )
        annotate_heatmap(
            axis,
            matrix,
            image,
            lambda value, row, column: f"{value:+.2f}",
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92
    )
    colorbar.set_label("Mean test-MSE difference vs Uniform (%)")
    figure.suptitle("Pairwise Test MSE vs Uniform", fontsize=15)
    save_figure(
        figure,
        output_dir,
        "relative_mse_change_vs_uniform",
        formats,
        dpi,
    )


def plot_relative_tail_mse_vs_real(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    matrices = [
        context.matrix(f"tail_mse_change_pct_{method}")
        for method in METHODS
    ]
    limit = symmetric_vmax(matrices)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    color_map = copy_cmap("RdBu_r")

    figure, axes = plt.subplots(
        2, 2, figsize=(16, 10), constrained_layout=True
    )
    for axis, method, matrix in zip(axes.ravel(), METHODS, matrices):
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap=color_map,
            norm=norm,
            aspect="auto",
        )
        style_heatmap_axis(
            axis, matrix, f"{METHOD_LABELS[method]} vs Real-only"
        )
        annotate_heatmap(
            axis,
            matrix,
            image,
            lambda value, row, column: f"{value:+.2f}",
            fontsize=7.4,
            mark_missing=True,
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92
    )
    colorbar.set_label(
        "Relative difference in mean M-class tail MSE vs Real-only (%)"
    )
    figure.suptitle(
        "Relative M-class Tail MSE vs Real-only",
        fontsize=15,
    )
    save_figure(
        figure,
        output_dir,
        "relative_m_class_tail_mse_change_all_methods",
        formats,
        dpi,
    )


def plot_relative_tail_mse_vs_uniform(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    matrices = [
        context.matrix(f"tail_mse_vs_uniform_pct_{method}")
        for method in PAIRWISE_METHODS
    ]
    limit = symmetric_vmax(matrices)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    color_map = copy_cmap("RdBu_r")

    figure, axes = plt.subplots(
        1, 3, figsize=(18, 5.3), constrained_layout=True
    )
    for axis, method, matrix in zip(axes, PAIRWISE_METHODS, matrices):
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap=color_map,
            norm=norm,
            aspect="auto",
        )
        style_heatmap_axis(
            axis, matrix, f"{METHOD_LABELS[method]} - Uniform"
        )
        annotate_heatmap(
            axis,
            matrix,
            image,
            lambda value, row, column: f"{value:+.2f}",
            mark_missing=True,
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92
    )
    colorbar.set_label(
        "Relative difference in mean M-class tail MSE vs Uniform (%)"
    )
    figure.suptitle(
        "Pairwise M-class Tail MSE vs Uniform",
        fontsize=15,
    )
    save_figure(
        figure,
        output_dir,
        "relative_m_class_tail_mse_change_vs_uniform",
        formats,
        dpi,
    )


def plot_covariance_differences(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    plot_specs = (
        (
            "empirical_trace_cov_diff",
            "Empirical Trace(Cov) Difference vs Real-only",
            "Empirical trace(Cov) difference vs Real-only",
            "empirical_trace_cov_diff_all_methods",
            "{:+.1f}",
        ),
        (
            "analytic_trace_cov_diff",
            "Analytic Trace(Cov) Difference vs Real-only",
            "Mean analytic trace(Cov) difference vs Real-only",
            "analytic_trace_cov_diff_all_methods",
            "{:+.2f}",
        ),
    )
    color_map = copy_cmap("RdBu_r")
    for column_stem, title, colorbar_label, file_stem, number_format in plot_specs:
        matrices = [
            context.matrix(f"{column_stem}_{method}") for method in METHODS
        ]
        limit = symmetric_vmax(matrices)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        figure, axes = plt.subplots(
            2, 2, figsize=(16, 10), constrained_layout=True
        )
        for axis, method, matrix in zip(axes.ravel(), METHODS, matrices):
            image = axis.imshow(
                np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                cmap=color_map,
                norm=norm,
                aspect="auto",
            )
            style_heatmap_axis(
                axis, matrix, f"{METHOD_LABELS[method]} - Real-only"
            )
            annotate_heatmap(
                axis,
                matrix,
                image,
                lambda value, row, column, fmt=number_format: fmt.format(value),
            )

        colorbar = figure.colorbar(
            image, ax=axes.ravel().tolist(), shrink=0.92
        )
        colorbar.set_label(colorbar_label)
        figure.suptitle(title, fontsize=15)
        save_figure(figure, output_dir, file_stem, formats, dpi)


def plot_eta(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    matrices = [context.matrix(f"mean_eta_{method}") for method in METHODS]
    finite_positive = np.concatenate(
        [matrix.to_numpy(dtype=float).ravel() for matrix in matrices]
    )
    finite_positive = finite_positive[
        np.isfinite(finite_positive) & (finite_positive > 0.0)
    ]
    if finite_positive.size == 0:
        print("Skipping eta plot: no finite positive eta values.")
        return

    norm = LogNorm(
        vmin=float(np.min(finite_positive)),
        vmax=float(np.max(finite_positive)),
    )
    color_map = copy_cmap("magma_r")
    figure, axes = plt.subplots(
        2, 2, figsize=(16, 10), constrained_layout=True
    )
    for axis, method, matrix in zip(axes.ravel(), METHODS, matrices):
        values = matrix.to_numpy(dtype=float)
        masked_values = np.ma.masked_where(
            ~np.isfinite(values) | (values <= 0.0), values
        )
        image = axis.imshow(
            masked_values, cmap=color_map, norm=norm, aspect="auto"
        )
        style_heatmap_axis(axis, matrix, METHOD_LABELS[method])
        annotate_heatmap(
            axis,
            matrix,
            image,
            lambda value, row, column: f"{value:.1e}",
            fontsize=7.2,
            mark_missing=True,
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92
    )
    colorbar.set_label(r"Mean $\eta$ (log color scale)")
    figure.suptitle(
        r"$\eta$ Trace Ratio (smaller is better; NA = undefined)",
        fontsize=15,
    )
    save_figure(figure, output_dir, "eta_all_methods", formats, dpi)


def plot_acceptance_rates(
    context: HeatmapContext,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    matrices = [
        100.0 * context.matrix(f"mean_accept_rate_{method}")
        for method in METHODS
    ]
    finite_values = np.concatenate(
        [matrix.to_numpy(dtype=float).ravel() for matrix in matrices]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        print("Skipping acceptance-rate plot: no finite values.")
        return
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if lower == upper:
        upper = lower + 1.0
    norm = Normalize(vmin=lower, vmax=upper)
    color_map = copy_cmap("viridis")

    figure, axes = plt.subplots(
        2, 2, figsize=(16, 10), constrained_layout=True
    )
    for axis, method, matrix in zip(axes.ravel(), METHODS, matrices):
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            cmap=color_map,
            norm=norm,
            aspect="auto",
        )
        style_heatmap_axis(axis, matrix, METHOD_LABELS[method])
        annotate_heatmap(
            axis,
            matrix,
            image,
            lambda value, row, column: f"{value:.2f}",
        )

    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.92
    )
    colorbar.set_label("Mean acceptance rate (%)")
    figure.suptitle("Acceptance-rate Heatmaps", fontsize=15)
    save_figure(figure, output_dir, "accept_rate_all_methods", formats, dpi)


def plot_categorical_method_map(
    context: HeatmapContext,
    *,
    value_column: str,
    methods: Sequence[str],
    title: str,
    panel_title: str,
    file_stem: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    method_to_code = {method: index for index, method in enumerate(methods)}
    code_to_method = {index: method for method, index in method_to_code.items()}
    codes = context.plot_df[value_column].map(method_to_code)
    temporary = context.plot_df.assign(_method_code=codes)
    temporary_context = HeatmapContext(temporary)
    matrix = temporary_context.matrix("_method_code")

    colors = [METHOD_COLORS[method] for method in methods]
    color_map = ListedColormap(colors)
    color_map.set_bad("#E6E6E6")
    norm = BoundaryNorm(
        np.arange(-0.5, len(methods) + 0.5, 1.0), color_map.N
    )
    figure, axis = plt.subplots(
        figsize=(10.8, 5.6), constrained_layout=True
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
        cmap=color_map,
        norm=norm,
        aspect="auto",
    )
    style_heatmap_axis(axis, matrix, panel_title)
    # Grey borders keep adjacent white cells distinct from the background.
    axis.grid(which="minor", color="#999999", linewidth=0.7)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            code_value = matrix.iat[row_index, column_index]
            if pd.notna(code_value):
                method = code_to_method[int(code_value)]
                axis.text(
                    column_index,
                    row_index,
                    ALL_METHOD_SHORT_LABELS[method],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="normal",
                    color=text_color_for_value(image, float(code_value)),
                )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.92,
        ticks=np.arange(len(methods)),
    )
    colorbar.ax.set_yticklabels(
        [ALL_METHOD_LABELS[method] for method in methods]
    )
    figure.suptitle(title, fontsize=14)
    save_figure(figure, output_dir, file_stem, formats, dpi)


def export_tables(plot_df: pd.DataFrame, output_dir: Path) -> None:
    winner_columns = [
        "train_fraction",
        "candidate_train_multiplier",
        "best_method",
        "best_method_label",
        "best_method_selection_basis",
        "best_test_mse_method",
        "best_test_mse_method_label",
        "best_mse_comparison_method",
        "best_mse_comparison_method_label",
        "best_mse_delta_pct",
        "best_mse_sd_pct",
        "best_mse_interval_covers_zero",
        "best_empirical_cov_method",
        "best_empirical_cov_method_label",
        "best_synthetic_method",
        "best_synthetic_method_label",
    ]
    for method in ALL_METHODS:
        winner_columns.extend(
            [
                f"mean_test_mse_{method}",
                f"empirical_trace_cov_{method}",
            ]
        )
    for method in METHODS:
        winner_columns.append(f"mse_change_pct_{method}")

    winner_table = plot_df[winner_columns].sort_values(
        ["train_fraction", "candidate_train_multiplier"]
    )
    winner_path = output_dir / "best_method_by_grid_cell.csv"
    winner_table.to_csv(winner_path, index=False)
    print(f"Saved {winner_path}")

    synthetic_table = plot_df[
        [
            "train_fraction",
            "candidate_train_multiplier",
            "best_synthetic_method",
            "best_synthetic_method_label",
            *[f"mean_test_mse_{method}" for method in METHODS],
        ]
    ].sort_values(["train_fraction", "candidate_train_multiplier"])
    synthetic_path = output_dir / "best_synthetic_method_by_grid_cell.csv"
    synthetic_table.to_csv(synthetic_path, index=False)
    print(f"Saved {synthetic_path}")

    preferred_source_columns = [
        "split_index",
        "train_fraction",
        "candidate_train_multiplier",
        "candidate_train_pct",
        "resolved_n_candidates",
        "n_train",
        "n_test",
        "pooled_runs",
        "alpha_target",
        *winner_columns[2:],
    ]
    for method in ALL_METHODS:
        preferred_source_columns.extend(
            [
                f"mse_change_pct_{method}",
                f"sd_mse_change_pct_{method}",
                f"mse_vs_uniform_pct_{method}",
                f"empirical_trace_cov_diff_{method}",
                f"analytic_trace_cov_diff_{method}",
            ]
        )
        if method != "real":
            preferred_source_columns.extend(
                [
                    f"mean_eta_{method}",
                    f"mean_accept_rate_{method}",
                ]
            )

    source_columns = []
    seen = set()
    for column in preferred_source_columns:
        if column in plot_df.columns and column not in seen:
            source_columns.append(column)
            seen.add(column)
    source_path = (
        output_dir / "strict_chronological_heatmap_source_data.csv"
    )
    plot_df[source_columns].sort_values(
        ["train_fraction", "candidate_train_multiplier"]
    ).to_csv(source_path, index=False)
    print(f"Saved {source_path}")


def export_tail_tables(plot_df: pd.DataFrame, output_dir: Path) -> None:
    identity_columns = [
        "train_fraction",
        "candidate_train_multiplier",
        "n_train",
        "n_test",
        "pooled_runs",
    ]
    tail_context_columns = [
        "mean_m_class_tail_test_count",
        "min_m_class_tail_test_count",
        "max_m_class_tail_test_count",
        "mean_m_class_tail_test_fraction",
        "se_m_class_tail_test_fraction",
        "m_class_tail_flux_threshold_raw",
        "m_class_tail_flux_threshold_log10",
        "m_class_tail_flux_threshold_standardized",
        "m_class_tail_threshold_inclusive",
    ]
    method_metric_columns = []
    for method in ALL_METHODS:
        method_metric_columns.extend(
            [
                TAIL_MEAN_COLUMN.format(method=method),
                TAIL_SE_COLUMN.format(method=method),
                TAIL_SD_COLUMN.format(method=method),
                f"tail_mse_change_pct_{method}",
                f"tail_mse_vs_uniform_pct_{method}",
            ]
        )

    winner_columns = [
        *identity_columns,
        "best_tail_method",
        "best_tail_method_label",
        "best_tail_synthetic_method",
        "best_tail_synthetic_method_label",
        *tail_context_columns,
        *method_metric_columns,
    ]
    winner_table = plot_df[winner_columns].sort_values(
        ["train_fraction", "candidate_train_multiplier"]
    )
    winner_path = output_dir / "best_m_class_tail_method_by_grid_cell.csv"
    winner_table.to_csv(winner_path, index=False)
    print(f"Saved {winner_path}")

    synthetic_columns = [
        *identity_columns,
        "best_tail_synthetic_method",
        "best_tail_synthetic_method_label",
        *[
            TAIL_MEAN_COLUMN.format(method=method)
            for method in METHODS
        ],
    ]
    synthetic_table = plot_df[synthetic_columns].sort_values(
        ["train_fraction", "candidate_train_multiplier"]
    )
    synthetic_path = (
        output_dir / "best_synthetic_m_class_tail_method_by_grid_cell.csv"
    )
    synthetic_table.to_csv(synthetic_path, index=False)
    print(f"Saved {synthetic_path}")

    source_columns = [
        column
        for column in [
            "split_index",
            *identity_columns,
            "candidate_train_pct",
            "resolved_n_candidates",
            "alpha_target",
            "best_tail_method",
            "best_tail_method_label",
            "best_tail_synthetic_method",
            "best_tail_synthetic_method_label",
            *tail_context_columns,
            *method_metric_columns,
        ]
        if column in plot_df.columns
    ]
    source_path = (
        output_dir / "strict_chronological_tail_heatmap_source_data.csv"
    )
    plot_df[source_columns].sort_values(
        ["train_fraction", "candidate_train_multiplier"]
    ).to_csv(source_path, index=False)
    print(f"Saved {source_path}")


def format_test_mse_mean_sd(mean_value: float, sd_value: float) -> str:
    if not np.isfinite(mean_value) or not np.isfinite(sd_value):
        return "NA"
    return f"{mean_value:.4f} ± {sd_value:.4f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def format_empirical_trace_covariance(
    value: float,
    *,
    is_column_minimum: bool,
) -> str:
    if not np.isfinite(value):
        return "NA"
    rendered = f"{value:.4e}"
    return f"**{rendered}**" if is_column_minimum else rendered


def finite_minimum(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    return min(finite_values) if finite_values else float("nan")


def export_test_mse_mean_sd_markdown(
    context: HeatmapContext,
    caption: str,
    output_dir: Path,
) -> None:
    mean_matrices = {
        method: context.matrix(f"mean_test_mse_{method}")
        for method in ALL_METHODS
    }
    sd_matrices = {
        method: context.matrix(f"sd_test_mse_{method}")
        for method in ALL_METHODS
    }
    candidate_headers = [
        f"{multiplier:g}× n_train"
        for multiplier in context.candidate_order
    ]

    lines = [
        "# Strict Chronological Held-out Test MSE Tables",
        "",
        f"**Experiment:** {caption}",
        "",
        (
            "Every result is **mean test MSE ± sample SD** across the pooled "
            "runs. Sample SD is reconstructed exactly as "
            "`saved SE × sqrt(pooled_runs)`."
        ),
        "",
        (
            "The acceptance target is fixed at **α = 0.05** for every grid "
            "cell. The varying second grid axis below is the synthetic "
            "candidate multiplier, not α."
        ),
        "",
        (
            "The Real-only row is unchanged across candidate multipliers "
            "because that baseline does not use synthetic candidates."
        ),
        "",
        "## Grouped by downstream chronological training proportion",
        "",
        (
            "Each table fixes one downstream chronological training "
            "proportion. Rows are methods and columns are candidate "
            "multipliers."
        ),
        "",
    ]

    for train_index, train_fraction in enumerate(context.train_order):
        rows = []
        for method in ALL_METHODS:
            mean_matrix = mean_matrices[method]
            sd_matrix = sd_matrices[method]
            cells = [
                format_test_mse_mean_sd(
                    float(mean_matrix.iat[candidate_index, train_index]),
                    float(sd_matrix.iat[candidate_index, train_index]),
                )
                for candidate_index in range(len(context.candidate_order))
            ]
            rows.append([ALL_METHOD_LABELS[method], *cells])
        lines.extend(
            [
                f"### Training proportion = {train_fraction:.2f}",
                "",
                markdown_table(["Method", *candidate_headers], rows),
                "",
            ]
        )

    lines.extend(
        [
            "## Grouped by candidate multiplier",
            "",
            (
                "Each table fixes one candidate multiplier. Rows are methods "
                "and columns are downstream chronological training "
                "proportions."
            ),
            "",
        ]
    )
    for candidate_index, candidate_multiplier in enumerate(
        context.candidate_order
    ):
        rows = []
        for method in ALL_METHODS:
            mean_matrix = mean_matrices[method]
            sd_matrix = sd_matrices[method]
            cells = [
                format_test_mse_mean_sd(
                    float(mean_matrix.iat[candidate_index, train_index]),
                    float(sd_matrix.iat[candidate_index, train_index]),
                )
                for train_index in range(len(context.train_order))
            ]
            rows.append([ALL_METHOD_LABELS[method], *cells])
        lines.extend(
            [
                (
                    "### Candidate multiplier = "
                    f"{candidate_multiplier:g}× n_train"
                ),
                "",
                markdown_table(["Method", *context.train_labels], rows),
                "",
            ]
        )

    markdown_path = output_dir / "test_mse_mean_sd_tables.md"
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved {markdown_path}")


def export_empirical_trace_covariance_markdown(
    context: HeatmapContext,
    caption: str,
    output_dir: Path,
) -> None:
    covariance_matrices = {
        method: context.matrix(f"empirical_trace_cov_{method}")
        for method in ALL_METHODS
    }
    candidate_headers = [
        f"{multiplier:g}× n_train"
        for multiplier in context.candidate_order
    ]

    lines = [
        "# Strict Chronological Empirical Trace-Covariance Tables",
        "",
        f"**Experiment:** {caption}",
        "",
        (
            "Each entry is the empirical `trace(Cov(beta_hat))` computed "
            "from the full fitted linear-regression parameter vectors "
            "(intercept included) across all pooled runs in that grid cell. "
            "The covariance uses the sample normalization `ddof = 1`."
        ),
        "",
        (
            "This trace is itself one across-run variability statistic, "
            "not a mean of per-run covariance traces, so no `± SD` is "
            "attached. Smaller values indicate less total coefficient "
            "variation across repetitions."
        ),
        "",
        (
            "Boldface indicates the smallest empirical trace covariance "
            "among all methods in each column. Under this fixed strict "
            "chronological split, Real-only uses the same training rows in "
            "every repetition and is therefore expected to be numerically "
            "near zero."
        ),
        "",
        (
            "The acceptance target is fixed at **α = 0.05** for every grid "
            "cell. The varying second grid axis is the synthetic candidate "
            "multiplier, not α."
        ),
        "",
        "## Grouped by downstream chronological training proportion",
        "",
        (
            "Each table fixes one downstream chronological training "
            "proportion. Rows are methods and columns are candidate "
            "multipliers."
        ),
        "",
    ]

    for train_index, train_fraction in enumerate(context.train_order):
        column_minima = [
            finite_minimum(
                covariance_matrices[method].iat[
                    candidate_index, train_index
                ]
                for method in ALL_METHODS
            )
            for candidate_index in range(len(context.candidate_order))
        ]

        rows = []
        for method in ALL_METHODS:
            matrix = covariance_matrices[method]
            cells = []
            for candidate_index in range(len(context.candidate_order)):
                value = float(matrix.iat[candidate_index, train_index])
                cells.append(
                    format_empirical_trace_covariance(
                        value,
                        is_column_minimum=(
                            np.isfinite(value)
                            and np.isclose(
                                value,
                                column_minima[candidate_index],
                                rtol=1e-12,
                                atol=0.0,
                            )
                        ),
                    )
                )
            rows.append([ALL_METHOD_LABELS[method], *cells])
        lines.extend(
            [
                f"### Training proportion = {train_fraction:.2f}",
                "",
                markdown_table(["Method", *candidate_headers], rows),
                "",
            ]
        )

    lines.extend(
        [
            "## Grouped by candidate multiplier",
            "",
            (
                "Each table fixes one candidate multiplier. Rows are "
                "methods and columns are downstream chronological training "
                "proportions."
            ),
            "",
        ]
    )
    for candidate_index, candidate_multiplier in enumerate(
        context.candidate_order
    ):
        column_minima = [
            finite_minimum(
                covariance_matrices[method].iat[
                    candidate_index, train_index
                ]
                for method in ALL_METHODS
            )
            for train_index in range(len(context.train_order))
        ]

        rows = []
        for method in ALL_METHODS:
            matrix = covariance_matrices[method]
            cells = []
            for train_index in range(len(context.train_order)):
                value = float(matrix.iat[candidate_index, train_index])
                cells.append(
                    format_empirical_trace_covariance(
                        value,
                        is_column_minimum=(
                            np.isfinite(value)
                            and np.isclose(
                                value,
                                column_minima[train_index],
                                rtol=1e-12,
                                atol=0.0,
                            )
                        ),
                    )
                )
            rows.append([ALL_METHOD_LABELS[method], *cells])
        lines.extend(
            [
                (
                    "### Candidate multiplier = "
                    f"{candidate_multiplier:g}× n_train"
                ),
                "",
                markdown_table(["Method", *context.train_labels], rows),
                "",
            ]
        )

    markdown_path = output_dir / "empirical_trace_covariance_tables.md"
    markdown_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Saved {markdown_path}")


def export_tail_mse_mean_sd_markdown(
    context: HeatmapContext,
    caption: str,
    output_dir: Path,
) -> None:
    mean_matrices = {
        method: context.matrix(TAIL_MEAN_COLUMN.format(method=method))
        for method in ALL_METHODS
    }
    sd_matrices = {
        method: context.matrix(TAIL_SD_COLUMN.format(method=method))
        for method in ALL_METHODS
    }
    candidate_headers = [
        f"{multiplier:g}× n_train"
        for multiplier in context.candidate_order
    ]

    grid = context.plot_df
    raw_threshold = float(
        grid["m_class_tail_flux_threshold_raw"].dropna().iloc[0]
    )
    log10_threshold = float(
        grid["m_class_tail_flux_threshold_log10"].dropna().iloc[0]
    )
    standardized_threshold = float(
        grid[
            "m_class_tail_flux_threshold_standardized"
        ].dropna().iloc[0]
    )
    count_min = int(grid["min_m_class_tail_test_count"].min())
    count_max = int(grid["max_m_class_tail_test_count"].max())

    lines = [
        "# Strict Chronological M-class Tail Test MSE Tables",
        "",
        f"**Experiment:** {caption}",
        "",
        (
            "The tail contains held-out observations with raw endpoint flux "
            f"at or above `{raw_threshold:g} W/m²` (M1.0+). The equivalent "
            f"cutoffs are `log10(flux) >= {log10_threshold:g}` and "
            f"standardized target `>= {standardized_threshold:.6f}`."
        ),
        "",
        (
            f"Tail test sets contain {count_min}–{count_max} observations "
            "across downstream training proportions. Tail membership is "
            "shared by all methods within a grid cell."
        ),
        "",
        (
            "Every result is **mean M-class tail test MSE ± sample SD** "
            "across the pooled runs, in squared standardized-log10 target "
            "units. Sample SD is reconstructed exactly as "
            "`saved SE × sqrt(pooled_runs)`."
        ),
        "",
        (
            "The acceptance target is fixed at **α = 0.05** for every grid "
            "cell. The varying second grid axis below is the synthetic "
            "candidate multiplier, not α."
        ),
        "",
        (
            "The Real-only row is unchanged across candidate multipliers "
            "because that baseline does not use synthetic candidates."
        ),
        "",
        "## Grouped by downstream chronological training proportion",
        "",
        (
            "Each table fixes one downstream chronological training "
            "proportion. Rows are methods and columns are candidate "
            "multipliers."
        ),
        "",
    ]

    for train_index, train_fraction in enumerate(context.train_order):
        rows = []
        for method in ALL_METHODS:
            mean_matrix = mean_matrices[method]
            sd_matrix = sd_matrices[method]
            cells = [
                format_test_mse_mean_sd(
                    float(mean_matrix.iat[candidate_index, train_index]),
                    float(sd_matrix.iat[candidate_index, train_index]),
                )
                for candidate_index in range(len(context.candidate_order))
            ]
            rows.append([ALL_METHOD_LABELS[method], *cells])
        lines.extend(
            [
                f"### Training proportion = {train_fraction:.2f}",
                "",
                markdown_table(["Method", *candidate_headers], rows),
                "",
            ]
        )

    lines.extend(
        [
            "## Grouped by candidate multiplier",
            "",
            (
                "Each table fixes one candidate multiplier. Rows are methods "
                "and columns are downstream chronological training "
                "proportions."
            ),
            "",
        ]
    )
    for candidate_index, candidate_multiplier in enumerate(
        context.candidate_order
    ):
        rows = []
        for method in ALL_METHODS:
            mean_matrix = mean_matrices[method]
            sd_matrix = sd_matrices[method]
            cells = [
                format_test_mse_mean_sd(
                    float(mean_matrix.iat[candidate_index, train_index]),
                    float(sd_matrix.iat[candidate_index, train_index]),
                )
                for train_index in range(len(context.train_order))
            ]
            rows.append([ALL_METHOD_LABELS[method], *cells])
        lines.extend(
            [
                (
                    "### Candidate multiplier = "
                    f"{candidate_multiplier:g}× n_train"
                ),
                "",
                markdown_table(["Method", *context.train_labels], rows),
                "",
            ]
        )

    markdown_path = output_dir / "m_class_tail_mse_mean_sd_tables.md"
    markdown_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Saved {markdown_path}")


def configure_plot_style(dpi: int, font_style: str | None = None) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 10,
            "font.family": "serif",
            "text.usetex": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titlepad": 8,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if font_style is not None:
        font_family, math_font = {
            "stix": ("STIXGeneral", "stix"),
            "computer-modern": ("cmr10", "cm"),
        }[font_style]
        # Both families ship with Matplotlib, so no TeX installation is needed.
        plt.rcParams.update(
            {
                "font.serif": [font_family],
                "mathtext.fontset": math_font,
                "axes.formatter.use_mathtext": True,
                "axes.unicode_minus": False,
            }
        )


def render_heatmaps(
    analysis_dir: Path, output_root: Path,
    formats: Sequence[str] = ("png", "pdf"), dpi: int = 300,
    font_style: str | None = None,
) -> pd.DataFrame:
    """Render the existing heatmap suite and return its aggregate input table."""
    all_output_dir = output_root / "all"
    tail_output_dir = output_root / "tails"
    all_output_dir.mkdir(parents=True, exist_ok=True)
    tail_output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style(dpi, font_style)

    grid, metadata = load_grid(analysis_dir)
    plot_df = prepare_plot_data(grid)
    context = HeatmapContext(plot_df)
    caption = experiment_caption(grid)
    tail_caption = tail_experiment_caption(grid)

    print(f"Analysis directory: {analysis_dir}")
    print(f"All-test output directory: {all_output_dir}")
    print(f"Tail output directory: {tail_output_dir}")
    print(f"Grid cells: {len(grid):,}")
    print(f"Train fractions: {len(context.train_order)}")
    print(f"Candidate multipliers: {len(context.candidate_order)}")
    if metadata.get("experiment_dir"):
        print(f"Source experiment: {metadata['experiment_dir']}")

    plot_relative_mse_vs_real(
        context, all_output_dir, formats, dpi
    )
    plot_relative_mse_vs_uniform(
        context, all_output_dir, formats, dpi
    )
    plot_covariance_differences(
        context, all_output_dir, formats, dpi
    )
    plot_eta(context, all_output_dir, formats, dpi)
    plot_acceptance_rates(
        context, all_output_dir, formats, dpi
    )
    plot_categorical_method_map(
        context,
        value_column="best_synthetic_method",
        methods=METHODS,
        title="Best Synthetic Method by Mean Test MSE",
        panel_title="Lowest mean held-out test MSE",
        file_stem="best_synthetic_method_by_test_mse",
        output_dir=all_output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_categorical_method_map(
        context,
        value_column="best_method",
        methods=ALL_METHODS,
        title="Best Method Heatmap",
        panel_title=(
            r"Lowest MSE if relative $\Delta\mathrm{MSE}\pm1\,\mathrm{SD}$ excludes 0;"
            "\n"
            "otherwise lowest empirical covariance among synthetic methods"
        ),
        file_stem="best_method_by_guarded_mse_or_empirical_cov",
        output_dir=all_output_dir,
        formats=formats,
        dpi=dpi,
    )
    export_tables(plot_df, all_output_dir)
    export_test_mse_mean_sd_markdown(context, caption, all_output_dir)
    export_empirical_trace_covariance_markdown(
        context, caption, all_output_dir
    )

    plot_relative_tail_mse_vs_real(
        context,
        tail_output_dir,
        formats,
        dpi,
    )
    plot_relative_tail_mse_vs_uniform(
        context,
        tail_output_dir,
        formats,
        dpi,
    )
    plot_categorical_method_map(
        context,
        value_column="best_tail_synthetic_method",
        methods=METHODS,
        title="Best Synthetic Method by Mean M-class Tail MSE",
        panel_title="Lowest mean held-out M1.0+ tail MSE",
        file_stem="best_synthetic_method_by_m_class_tail_mse",
        output_dir=tail_output_dir,
        formats=formats,
        dpi=dpi,
    )
    plot_categorical_method_map(
        context,
        value_column="best_tail_method",
        methods=ALL_METHODS,
        title="Best Method by Mean M-class Tail MSE",
        panel_title="Lowest mean held-out M1.0+ tail MSE",
        file_stem="best_method_by_m_class_tail_mse",
        output_dir=tail_output_dir,
        formats=formats,
        dpi=dpi,
    )
    export_tail_tables(plot_df, tail_output_dir)
    export_tail_mse_mean_sd_markdown(context, tail_caption, tail_output_dir)
    return grid
