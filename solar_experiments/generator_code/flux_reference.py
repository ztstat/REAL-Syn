"""Flux proposal fitted exclusively from generator training observations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAINING_PROTOCOL = "chronological_training_only"
TRAINING_SCOPE = "generator_training_only"


def assert_training_dates(
    path: str | Path,
    *,
    datetime_column: str = "history_window_end",
    train_end_year: int = 2021,
    max_rows: int | None = None,
) -> np.ndarray:
    """Check only timestamps before loading any feature or flux values."""
    frame = pd.read_csv(path, usecols=[datetime_column], nrows=max_rows)
    dates = pd.to_datetime(frame[datetime_column], errors="coerce", utc=True)
    if dates.isna().any() or dates.empty:
        raise ValueError("Training CSV must contain valid endpoint timestamps.")
    years = dates.dt.year.to_numpy(dtype=np.int64)
    if np.any(years > train_end_year):
        raise ValueError(
            f"Expected a training-only CSV with endpoint years <= {train_end_year}; "
            "later rows are reserved for the solar experiment. "
            "Pass the previously saved chronological train.csv."
        )
    return years


def require_training_reference(summary: dict[str, Any], train_end_year: int) -> None:
    source = summary.get("source", {})
    if source.get("scope") != TRAINING_SCOPE:
        raise ValueError("Flux reference must be fitted from generator training data only.")
    if int(source.get("endpoint_year_max", train_end_year + 1)) > train_end_year:
        raise ValueError("Flux reference includes endpoint years after the training cutoff.")


def finite_positive_flux(values: np.ndarray) -> np.ndarray:
    flux = np.asarray(values, dtype=np.float64)
    return flux[np.isfinite(flux) & (flux > 0.0)]


def compact_condition_reference(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep proposal parameters and training provenance, with filenames only."""
    source = summary.get("source", {})
    compact_source = {
        key: source[key]
        for key in (
            "scope", "flux_column", "endpoint_year_min", "endpoint_year_max",
            "cutoff_year_inclusive",
        )
        if key in source
    }
    if "files" in source:
        compact_source["files"] = [
            str(path).replace("\\", "/").rsplit("/", 1)[-1]
            for path in source["files"]
        ]
    return {
        "format_version": 1,
        "condition_transform": summary["condition_transform"],
        "source": compact_source,
        "flux_quantile": summary["flux_quantile"],
        "bulk_support": {
            key: summary["bulk_support"][key]
            for key in ("lower", "upper", "bulk_count", "full_count")
            if key in summary["bulk_support"]
        },
        "tail_fit": {
            key: summary["tail_fit"][key]
            for key in ("xmin", "xmax", "alpha_hat", "tail_count", "full_count")
            if key in summary["tail_fit"]
        },
    }


def fit_condition_reference(
    flux: np.ndarray,
    *,
    quantile: float,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("flux quantile must lie strictly between zero and one.")
    flux_values = finite_positive_flux(flux)
    if flux_values.size < 10:
        raise ValueError("At least 10 positive finite flux values are required.")

    threshold = float(np.quantile(flux_values, float(quantile)))
    bulk = flux_values[flux_values < threshold]
    tail = flux_values[flux_values >= threshold]
    if bulk.size < 2 or tail.size < 2:
        raise ValueError("Condition reference needs at least two bulk and tail rows.")
    denominator = float(np.sum(np.log(tail / threshold)))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Pareto tail fit has a non-positive denominator.")
    alpha_hat = float(tail.size / denominator)
    return compact_condition_reference({
        "condition_transform": "log10",
        "source": source or {},
        "flux_quantile": float(quantile),
        "bulk_support": {
            "lower": float(np.min(bulk)),
            "upper": threshold,
            "bulk_count": int(bulk.size),
            "full_count": int(flux_values.size),
        },
        "tail_fit": {
            "xmin": threshold,
            "xmax": float(np.max(tail)),
            "alpha_hat": alpha_hat,
            "tail_count": int(tail.size),
            "full_count": int(flux_values.size),
        },
    })

