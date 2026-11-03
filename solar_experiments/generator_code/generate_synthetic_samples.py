#!/usr/bin/env python3
"""Generate 60-frame SHARPs histories from the copied temporal OT-CFM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

if __package__:
    from .model import TemporalConditionalOTFlow
    from .flux_reference import (
        TRAINING_PROTOCOL, TRAINING_SCOPE, assert_training_dates,
        compact_condition_reference, fit_condition_reference, require_training_reference,
    )
else:
    from model import TemporalConditionalOTFlow
    from flux_reference import (
        TRAINING_PROTOCOL, TRAINING_SCOPE, assert_training_dates,
        compact_condition_reference, fit_condition_reference, require_training_reference,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_LABEL = "chronological_split"
TRAINED_DIR = SCRIPT_DIR / "artifacts" / "chronological_training_only_seed3097"
DEFAULT_SAMPLE_COUNT = 100_000
DEFAULT_SAMPLES_DIR = SCRIPT_DIR / "samples"
MODEL_TYPE = "temporal_conditional_ot_flow_matching"
BOUNDARY_MATCHED = "boundary_matched"
EXPLICIT_LOG10 = "explicit_log10"
SPLIT_STRATEGY_LABELS = {
    "random": "random_split",
    "chronological": "chronological_split",
    "harp": "ar_split",
    "harp_by_year": "ar_split",
}
SOURCE_CODE_MAP = {
    "bulk_uniform": 0,
    "truncated_pareto": 1,
    "explicit_log10_flux": 2,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate flattened 60-frame SHARPs histories from the copied temporal "
            "OT-CFM. Every raw endpoint flux is transformed with base-10 logarithm "
            "and standardized using the checkpoint's training-only statistics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Explicit training-only checkpoint. By default load model_epoch_075.pt "
            "from artifacts/chronological_training_only_seed3097/."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional matching training metadata. Automatically uses matching "
            "training_metadata.json beside the selected checkpoint when present."
        ),
    )
    parser.add_argument(
        "--split-label",
        default=DEFAULT_SPLIT_LABEL,
        help=(
            "Output identity; defaults to chronological_split. Supplied metadata "
            "must agree with this label."
        ),
    )
    parser.add_argument(
        "--condition-mode",
        choices=(BOUNDARY_MATCHED, EXPLICIT_LOG10),
        default=BOUNDARY_MATCHED,
        help=(
            "boundary_matched draws a uniform bulk plus truncated-Pareto tail; "
            "explicit_log10 samples at repeated --log10-flux values."
        ),
    )
    parser.add_argument(
        "--condition-summary",
        type=Path,
        default=None,
        help=(
            "Optional training-only flux-reference JSON. Defaults to the reference "
            "embedded in the selected checkpoint."
        ),
    )
    parser.add_argument(
        "--condition-reference-csv",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional training-only CSV containing endpoint flux and timestamps. "
            "When supplied, the q-threshold and Pareto tail are refit."
        ),
    )
    parser.add_argument("--flux-column", default="flux")
    parser.add_argument("--datetime-column", default="history_window_end")
    parser.add_argument("--flux-quantile", type=float, default=0.995)
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=None,
        help=(
            "Optional tail-mixture override. By default the bulk and tail densities "
            "are matched continuously at the quantile threshold."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=(
            "Rows to generate in boundary_matched mode; defaults to 100000 "
            "to match the saved chronological sampling run."
        ),
    )
    parser.add_argument(
        "--log10-flux",
        type=float,
        action="append",
        default=None,
        help=(
            "Already-transformed base-10 endpoint flux for explicit_log10 mode. "
            "Repeat for multiple conditions."
        ),
    )
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3097)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rk4-steps", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--sample-name", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "samples/<run-name>/epoch_<N>/<split-label>/<sample-name>/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the split-labeled CSV and JSON outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the checkpoint/metadata contract and exit without sampling.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved.name}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved.name}.")
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved.name}")
    payload = torch.load(
        resolved,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary payload.")
    return payload


def resolve_checkpoint_path(requested: Path | None) -> Path:
    path = requested if requested is not None else TRAINED_DIR / "model_epoch_075.pt"
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Training-only checkpoint not found: {resolved.name}. Run train_generator.py first."
        )
    return resolved


def resolve_metadata_path(
    checkpoint_path: Path,
    requested_metadata: Path | None,
) -> Path | None:
    if requested_metadata is not None:
        return requested_metadata.expanduser().resolve()
    metadata_path = checkpoint_path.with_name("training_metadata.json")
    return metadata_path if metadata_path.is_file() else None


def resolve_split_label(checkpoint_path: Path, requested_label: str | None) -> str:
    return sanitize_sample_name(requested_label or DEFAULT_SPLIT_LABEL)


def validate_split_identity(
    split_label: str,
    metadata: dict[str, Any] | None,
) -> None:
    if metadata is None:
        return
    strategy = str(metadata.get("split", {}).get("strategy", "")).lower()
    expected_label = SPLIT_STRATEGY_LABELS.get(strategy)
    if expected_label is not None and split_label != expected_label:
        raise ValueError(
            f"Metadata split strategy {strategy!r} requires "
            f"--split-label {expected_label!r}, found {split_label!r}."
        )


def _finite_positive(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive.")
    return result


def validate_artifact_contract(
    checkpoint: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError(
            "This workflow requires a chronological training-only checkpoint. "
            "Retrain with train_generator.py; historical validation-selected weights "
            "are not used by the new experiment."
        )
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            "Expected checkpoint model_type "
            f"{MODEL_TYPE!r}, found {checkpoint.get('model_type')!r}."
        )
    required = {
        "model_state_dict",
        "model_config",
        "schema",
        "normalization",
        "sampling_defaults",
        "epoch",
        "train_end_year",
        "condition_reference",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is missing required keys: " + ", ".join(sorted(missing))
        )
    if int(checkpoint["epoch"]) <= 0:
        raise ValueError("Checkpoint epoch must be positive.")
    require_training_reference(
        checkpoint["condition_reference"], int(checkpoint["train_end_year"]),
    )

    model_config = checkpoint["model_config"]
    schema = checkpoint["schema"]
    normalization = checkpoint["normalization"]
    frame_indices = tuple(int(value) for value in schema["frame_indices"])
    base_features = tuple(str(value) for value in schema["base_features"])
    feature_columns = tuple(str(value) for value in schema["feature_columns"])
    if frame_indices != tuple(range(len(frame_indices))):
        raise ValueError("Checkpoint frame indices must be contiguous from zero.")
    expected_columns = tuple(
        f"frame_{frame_index:03d}__{feature_name}"
        for frame_index in frame_indices
        for feature_name in base_features
    )
    if feature_columns != expected_columns:
        raise ValueError(
            "Checkpoint feature columns are not in frame-major/base-feature order."
        )
    if int(model_config["x_dim"]) != len(feature_columns):
        raise ValueError("model_config.x_dim does not match the checkpoint schema.")
    if int(model_config["condition_dim"]) != 1:
        raise ValueError("This sampler requires a scalar endpoint-flux condition.")

    condition_stats = normalization.get("condition", {})
    if condition_stats.get("source_column") != schema.get("flux_column"):
        raise ValueError("Condition source must match schema.flux_column.")
    if condition_stats.get("transform") != "log10":
        raise ValueError(
            "Temporal checkpoint must declare normalization.condition.transform='log10'."
        )
    condition_mean = float(condition_stats["mean"])
    condition_scale = _finite_positive(
        condition_stats["scale"],
        label="normalization.condition.scale",
    )
    if not math.isfinite(condition_mean):
        raise ValueError("normalization.condition.mean must be finite.")

    feature_statistics = normalization.get("feature_statistics", {})
    if set(feature_statistics) != set(base_features):
        raise ValueError(
            "Feature-normalization keys do not match schema.base_features."
        )
    for feature_name in base_features:
        stats = feature_statistics[feature_name]
        if not math.isfinite(float(stats["mean"])):
            raise ValueError(f"Non-finite mean for feature {feature_name}.")
        _finite_positive(
            stats["scale"],
            label=f"normalization scale for {feature_name}",
        )

    if metadata is not None:
        for key in (
            "model_type", "model_config", "schema", "normalization",
            "training_protocol", "train_end_year", "condition_reference",
        ):
            if metadata.get(key) != checkpoint.get(key):
                raise ValueError(
                    f"Copied training metadata disagrees with checkpoint key {key!r}."
                )

    return {
        "frame_indices": frame_indices,
        "frame_count": len(frame_indices),
        "base_features": base_features,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "flux_column": str(schema["flux_column"]),
        "condition_mean": condition_mean,
        "condition_scale": condition_scale,
    }


def build_model(checkpoint: dict[str, Any]) -> TemporalConditionalOTFlow:
    model = TemporalConditionalOTFlow(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def configure_numerics(device: torch.device) -> None:
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_flux_reference(
    paths: Sequence[Path],
    *,
    flux_column: str,
    quantile: float,
    train_end_year: int,
    datetime_column: str = "history_window_end",
) -> dict[str, Any]:
    arrays: list[np.ndarray] = []
    observed_years: list[int] = []
    filenames: list[str] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Condition reference CSV not found: {resolved.name}")
        years = assert_training_dates(
            resolved, datetime_column=datetime_column, train_end_year=train_end_year,
        )
        observed_years.extend([int(years.min()), int(years.max())])
        frame = pd.read_csv(resolved, usecols=[flux_column])
        arrays.append(frame[flux_column].to_numpy(dtype=np.float64, copy=True))
        filenames.append(resolved.name)
    return fit_condition_reference(
        np.concatenate(arrays),
        quantile=quantile,
        source={
            "scope": TRAINING_SCOPE,
            "endpoint_year_min": min(observed_years),
            "endpoint_year_max": max(observed_years),
            "cutoff_year_inclusive": train_end_year,
            "files": filenames,
            "flux_column": flux_column,
        },
    )


def validate_condition_reference(
    summary: dict[str, Any],
    *,
    quantile: float,
) -> dict[str, Any]:
    if summary.get("condition_transform") != "log10":
        raise ValueError("Condition reference must declare condition_transform='log10'.")
    recorded_quantile = float(summary["flux_quantile"])
    if not math.isclose(recorded_quantile, quantile, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Precomputed condition summary uses flux_quantile="
            f"{recorded_quantile}; provide --condition-reference-csv to refit "
            f"at flux_quantile={quantile}."
        )
    bulk = summary["bulk_support"]
    tail = summary["tail_fit"]
    lower = _finite_positive(bulk["lower"], label="bulk lower support")
    upper = _finite_positive(bulk["upper"], label="bulk upper support")
    xmin = _finite_positive(tail["xmin"], label="tail xmin")
    xmax = _finite_positive(tail["xmax"], label="tail xmax")
    alpha = _finite_positive(tail["alpha_hat"], label="Pareto alpha")
    if not math.isclose(upper, xmin, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Bulk upper support must equal the Pareto tail threshold.")
    if not lower < xmin < xmax:
        raise ValueError("Expected bulk lower < tail threshold < tail maximum.")
    if alpha <= 0.0:
        raise ValueError("Pareto alpha must be positive.")
    return compact_condition_reference(summary)


def compute_boundary_matched_tail_fraction(
    *,
    bulk_low: float,
    tail_xmin: float,
    tail_xmax: float,
    alpha: float,
) -> dict[str, float]:
    if not bulk_low < tail_xmin < tail_xmax:
        raise ValueError("Expected bulk_low < tail_xmin < tail_xmax.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    bulk_density = 1.0 / (tail_xmin - bulk_low)
    truncation_mass = 1.0 - (tail_xmin / tail_xmax) ** alpha
    if truncation_mass <= 0.0:
        raise ValueError("Truncated Pareto mass must be positive.")
    tail_density = alpha / (tail_xmin * truncation_mass)
    tail_fraction = bulk_density / (bulk_density + tail_density)
    return {
        "bulk_density": float(bulk_density),
        "tail_density_at_threshold": float(tail_density),
        "pareto_truncation_mass": float(truncation_mass),
        "tail_fraction": float(tail_fraction),
        "bulk_fraction": float(1.0 - tail_fraction),
    }


def sample_truncated_pareto(
    rng: np.random.RandomState,
    sample_count: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float,
) -> np.ndarray:
    if sample_count < 0:
        raise ValueError("sample_count cannot be negative.")
    if sample_count == 0:
        return np.empty(0, dtype=np.float64)
    if alpha <= 0.0 or xmin <= 0.0 or xmax <= xmin:
        raise ValueError("Invalid truncated Pareto parameters.")
    uniform = rng.uniform(0.0, 1.0, size=sample_count)
    truncation_mass = 1.0 - (xmin / xmax) ** alpha
    return xmin / np.power(
        1.0 - uniform * truncation_mass,
        1.0 / alpha,
    )


def sample_boundary_matched_flux(
    *,
    sample_count: int,
    reference: dict[str, Any],
    seed: int,
    tail_fraction_override: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    bulk = reference["bulk_support"]
    tail = reference["tail_fit"]
    boundary = compute_boundary_matched_tail_fraction(
        bulk_low=float(bulk["lower"]),
        tail_xmin=float(tail["xmin"]),
        tail_xmax=float(tail["xmax"]),
        alpha=float(tail["alpha_hat"]),
    )
    if tail_fraction_override is None:
        tail_fraction = float(boundary["tail_fraction"])
        tail_fraction_mode = "boundary_matched"
    else:
        tail_fraction = float(tail_fraction_override)
        if not 0.0 <= tail_fraction <= 1.0:
            raise ValueError("tail-fraction must lie in [0, 1].")
        tail_fraction_mode = "manual_override"

    tail_count = min(
        max(int(round(sample_count * tail_fraction)), 0),
        sample_count,
    )
    bulk_count = sample_count - tail_count
    rng = np.random.RandomState(int(seed) + 24_680)
    bulk_flux = rng.uniform(
        float(bulk["lower"]),
        float(tail["xmin"]),
        size=bulk_count,
    )
    tail_flux = sample_truncated_pareto(
        rng,
        tail_count,
        alpha=float(tail["alpha_hat"]),
        xmin=float(tail["xmin"]),
        xmax=float(tail["xmax"]),
    )
    raw_flux = np.concatenate([bulk_flux, tail_flux])
    source_codes = np.concatenate(
        [
            np.full(
                bulk_count,
                SOURCE_CODE_MAP["bulk_uniform"],
                dtype=np.int8,
            ),
            np.full(
                tail_count,
                SOURCE_CODE_MAP["truncated_pareto"],
                dtype=np.int8,
            ),
        ]
    )
    permutation = rng.permutation(sample_count)
    raw_flux = raw_flux[permutation]
    source_codes = source_codes[permutation]
    details = {
        "tail_fraction_mode": tail_fraction_mode,
        "tail_fraction_used": tail_fraction,
    }
    return raw_flux, source_codes, details


def build_explicit_flux(
    log10_flux_values: Sequence[float],
    *,
    samples_per_condition: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if samples_per_condition <= 0:
        raise ValueError("samples-per-condition must be positive.")
    values = np.asarray(log10_flux_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            "explicit_log10 mode requires finite repeated --log10-flux values."
        )
    repeated = np.repeat(values, samples_per_condition)
    raw_flux = np.power(10.0, repeated)
    if not np.isfinite(raw_flux).all() or np.any(raw_flux <= 0.0):
        raise ValueError("Explicit log10 flux values overflowed raw-flux space.")
    source_codes = np.full(
        raw_flux.size,
        SOURCE_CODE_MAP["explicit_log10_flux"],
        dtype=np.int8,
    )
    details = {
        "requested_log10_flux": values.tolist(),
        "samples_per_condition": int(samples_per_condition),
    }
    return raw_flux, source_codes, details


def standardize_log10_flux(
    raw_flux: np.ndarray,
    normalization: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    condition = normalization["condition"]
    if condition.get("transform") != "log10":
        raise ValueError("Checkpoint condition transform is not log10.")
    raw = np.asarray(raw_flux, dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise ValueError("Raw flux must be finite and strictly positive.")
    log10_flux = np.log10(raw)
    standardized = (
        log10_flux - float(condition["mean"])
    ) / float(condition["scale"])
    return log10_flux, standardized


def invert_normalized_histories(
    normalized: np.ndarray,
    *,
    contract: dict[str, Any],
    normalization: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != contract["feature_count"]:
        raise ValueError(
            "Normalized histories must have shape "
            f"[rows, {contract['feature_count']}]."
        )
    temporal = values.reshape(
        values.shape[0],
        contract["frame_count"],
        len(contract["base_features"]),
    )
    statistics = normalization["feature_statistics"]
    with np.errstate(over="raise", invalid="raise"):
        for feature_index, feature_name in enumerate(contract["base_features"]):
            stats = statistics[feature_name]
            transformed = (
                temporal[:, :, feature_index] * float(stats["scale"])
                + float(stats["mean"])
            )
            temporal[:, :, feature_index] = (
                np.sign(transformed) * np.expm1(np.abs(transformed))
            )
    if not np.isfinite(values).all():
        raise FloatingPointError("Inverse-transformed histories are non-finite.")
    return values


def sanitize_sample_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("sample-name must contain an alphanumeric character.")
    return sanitized


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(float(quantile) * 1000.0)):03d}"


def default_sample_name(
    *,
    mode: str,
    sample_count: int,
    seed: int,
    quantile: float,
) -> str:
    if mode == BOUNDARY_MATCHED:
        prefix = f"boundary_matched_{quantile_tag(quantile)}"
    else:
        prefix = "explicit_log10"
    return f"{prefix}_n{sample_count}_seed{seed}"


def split_output_paths(
    output_dir: Path,
    split_label: str,
) -> tuple[Path, Path]:
    safe_label = sanitize_sample_name(split_label)
    return (
        output_dir / f"synthetic_samples_{safe_label}.csv",
        output_dir / f"sample_summary_{safe_label}.json",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def write_generated_samples(
    *,
    model: TemporalConditionalOTFlow,
    checkpoint: dict[str, Any],
    contract: dict[str, Any],
    raw_flux: np.ndarray,
    log10_flux: np.ndarray,
    standardized_condition: np.ndarray,
    source_codes: np.ndarray,
    batch_size: int,
    rk4_steps: int,
    seed: int,
    device: torch.device,
    output_csv: Path,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if rk4_steps <= 0:
        raise ValueError("rk4-steps must be positive.")
    label_lookup = {
        SOURCE_CODE_MAP["bulk_uniform"]: "bulk_uniform",
        SOURCE_CODE_MAP["truncated_pareto"]: "truncated_pareto",
        SOURCE_CODE_MAP["explicit_log10_flux"]: "explicit_log10_flux",
    }
    model.to(device)
    model.eval()
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed) + 1_000)
    temporary_path = output_csv.with_name(
        f".{output_csv.name}.{os.getpid()}.tmp"
    )
    temporary_path.unlink(missing_ok=True)
    try:
        for start in range(0, raw_flux.size, batch_size):
            stop = min(start + batch_size, raw_flux.size)
            condition = torch.as_tensor(
                standardized_condition[start:stop, None],
                device=device,
                dtype=torch.float32,
            )
            noise = torch.randn(
                (stop - start, int(model.x_dim)),
                generator=cpu_generator,
                device="cpu",
                dtype=torch.float32,
            ).to(device)
            generated = model.integrate_rk4(
                noise,
                condition,
                num_steps=rk4_steps,
            )
            raw_histories = invert_normalized_histories(
                generated.cpu().numpy(),
                contract=contract,
                normalization=checkpoint["normalization"],
            )
            output = pd.DataFrame(
                raw_histories,
                columns=contract["feature_columns"],
            )
            batch_codes = source_codes[start:stop]
            output.insert(
                0,
                "sample_id",
                np.arange(start, stop, dtype=np.int64),
            )
            output.insert(
                1,
                "generation_source_code",
                batch_codes.astype(np.int64),
            )
            output.insert(
                2,
                "generation_source_label",
                [label_lookup[int(code)] for code in batch_codes],
            )
            output.insert(
                3,
                "condition_log10_flux",
                log10_flux[start:stop],
            )
            output.insert(
                4,
                "condition_standardized_log10_flux",
                standardized_condition[start:stop],
            )
            output.insert(
                5,
                contract["flux_column"],
                raw_flux[start:stop],
            )
            output.to_csv(
                temporary_path,
                mode="w" if start == 0 else "a",
                header=start == 0,
                index=False,
            )
        os.replace(temporary_path, output_csv)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def _ensure_outputs_available(paths: Sequence[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(path.name for path in existing)
        )


def run(args: argparse.Namespace) -> Path | None:
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    metadata_path = resolve_metadata_path(checkpoint_path, args.metadata)
    split_label = resolve_split_label(checkpoint_path, args.split_label)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = load_json(metadata_path) if metadata_path is not None else None
    contract = validate_artifact_contract(checkpoint, metadata)
    validate_split_identity(split_label, metadata)
    model = build_model(checkpoint)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"checkpoint={checkpoint_path.name}", flush=True)

    if args.validate_only:
        print(
            "validated "
            f"split_label={split_label} "
            f"model_type={MODEL_TYPE} frames={contract['frame_count']} "
            f"features={contract['feature_count']} parameters={parameter_count} "
            "condition=standardized_log10_endpoint_flux",
            flush=True,
        )
        return None

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if not 0.0 < float(args.flux_quantile) < 1.0:
        raise ValueError("flux-quantile must lie strictly between zero and one.")
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    configure_numerics(device)

    condition_reference: dict[str, Any] | None = None
    if args.condition_mode == BOUNDARY_MATCHED:
        if args.log10_flux:
            raise ValueError(
                "--log10-flux only applies to --condition-mode explicit_log10."
            )
        if args.condition_reference_csv:
            condition_reference = load_flux_reference(
                args.condition_reference_csv,
                flux_column=str(args.flux_column),
                quantile=float(args.flux_quantile),
                train_end_year=int(checkpoint["train_end_year"]),
                datetime_column=args.datetime_column,
            )
        else:
            condition_reference = (
                load_json(args.condition_summary)
                if args.condition_summary is not None
                else checkpoint.get("condition_reference")
            )
            if condition_reference is None:
                raise ValueError("Checkpoint has no training-only flux reference.")
        require_training_reference(condition_reference, int(checkpoint["train_end_year"]))
        condition_reference = validate_condition_reference(
            condition_reference, quantile=float(args.flux_quantile),
        )
        sample_count = (
            int(args.n_samples)
            if args.n_samples is not None
            else DEFAULT_SAMPLE_COUNT
        )
        raw_flux, source_codes, condition_details = sample_boundary_matched_flux(
            sample_count=sample_count,
            reference=condition_reference,
            seed=int(args.seed),
            tail_fraction_override=args.tail_fraction,
        )
    else:
        if args.n_samples is not None:
            raise ValueError(
                "--n-samples only applies to --condition-mode boundary_matched."
            )
        if args.tail_fraction is not None or args.condition_reference_csv:
            raise ValueError(
                "Tail/reference options only apply to boundary_matched mode."
            )
        raw_flux, source_codes, condition_details = build_explicit_flux(
            args.log10_flux or [],
            samples_per_condition=int(args.samples_per_condition),
        )
        sample_count = int(raw_flux.size)

    log10_flux, standardized_condition = standardize_log10_flux(
        raw_flux,
        checkpoint["normalization"],
    )
    rk4_steps = (
        int(args.rk4_steps)
        if args.rk4_steps is not None
        else int(checkpoint["sampling_defaults"]["rk4_steps"])
    )
    if rk4_steps <= 0:
        raise ValueError("rk4-steps must be positive.")

    sample_name = (
        sanitize_sample_name(args.sample_name)
        if args.sample_name is not None
        else default_sample_name(
            mode=args.condition_mode,
            sample_count=sample_count,
            seed=int(args.seed),
            quantile=float(args.flux_quantile),
        )
    )
    checkpoint_source = checkpoint_path.parent.name
    checkpoint_epoch = int(checkpoint["epoch"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            DEFAULT_SAMPLES_DIR / checkpoint_source / f"epoch_{checkpoint_epoch:03d}" / split_label / sample_name
        ).resolve()
    )
    output_csv, output_summary = split_output_paths(output_dir, split_label)
    _ensure_outputs_available(
        (output_csv, output_summary),
        overwrite=bool(args.overwrite),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_generated_samples(
        model=model,
        checkpoint=checkpoint,
        contract=contract,
        raw_flux=raw_flux,
        log10_flux=log10_flux,
        standardized_condition=standardized_condition,
        source_codes=source_codes,
        batch_size=int(args.batch_size),
        rk4_steps=rk4_steps,
        seed=int(args.seed),
        device=device,
        output_csv=output_csv,
    )

    source_counts = {
        label: int(np.sum(source_codes == code))
        for label, code in SOURCE_CODE_MAP.items()
        if np.any(source_codes == code)
    }
    summary: dict[str, Any] = {
        "format_version": 2,
        "split_label": split_label,
        "output_csv": output_csv.name,
        "checkpoint": {
            "epoch": checkpoint_epoch,
            "training_protocol": checkpoint["training_protocol"],
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        },
        "conditioning": {
            "mode": args.condition_mode,
            "source_column": contract["flux_column"],
            "transform": "log10",
            "standardization_mean": contract["condition_mean"],
            "standardization_scale": contract["condition_scale"],
            "details": condition_details,
            "reference": condition_reference,
        },
        "sample_count": sample_count,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "rk4_steps": rk4_steps,
        "generation_source_code_map": SOURCE_CODE_MAP,
        "generation_source_counts": source_counts,
    }
    atomic_json_dump(summary, output_summary)
    print(f"Wrote {output_csv.name}", flush=True)
    print(f"Wrote {output_summary.name}", flush=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
