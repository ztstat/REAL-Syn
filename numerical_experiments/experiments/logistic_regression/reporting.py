"""Paired logistic contrasts with normal-approximation intervals."""
from __future__ import annotations
from typing import Any
from . import metrics as logistic_metrics


def build_paired(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        keys = sorted(
            {(row["baseline"], row["metric"]) for row in arm["contrasts"]}
        )
        for baseline, metric in keys:
            selected = [
                row
                for row in arm["contrasts"]
                if row["baseline"] == baseline and row["metric"] == metric
            ]
            values = [float(row["difference"]) for row in selected]
            output = {
                "arm": arm["config"]["arm"],
                "contrast": f"REAL-Syn minus {baseline}",
                "baseline": baseline,
                "metric": metric,
                "favorable_direction": selected[0]["favorable_direction"],
                "paired_repetitions": len(selected),
            }
            summary = logistic_metrics.mean_sd_se_ci(values)
            output.update(
                {
                    "mean_difference": summary["mean"],
                    "sd": summary["sd"],
                    "se": summary["se"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                }
            )
            rows.append(output)
    return rows
