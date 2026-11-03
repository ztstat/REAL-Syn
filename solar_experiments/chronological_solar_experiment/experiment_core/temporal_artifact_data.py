from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    SPLIT_SPECS,
    TRAINING_ONLY_PROTOCOL,
    canonical_split_label,
    load_json,
    load_training_only_manifest,
)
from .portable_metadata import runtime_paths, save_json


CACHE_FORMAT_VERSION = 3
LAST_FRAME_FEATURE_VIEW = "last_frame"
FULL_VIDEO_FEATURE_VIEW = "full_video"
FEATURE_VIEWS = {
    LAST_FRAME_FEATURE_VIEW,
    FULL_VIDEO_FEATURE_VIEW,
}
# Backward-compatible name used by the existing endpoint experiments.
FEATURE_VIEW = LAST_FRAME_FEATURE_VIEW
NORMALIZED_CONDITION_NAME = "log10_flux"
CONDITION_INVERSE_TRANSFORM = "power10"
DEFAULT_CHUNK_SIZE = 5_000
RANDOM_DOWNSTREAM_PROTOCOL = "random_downstream"
RANDOM_DRE_CHRONOLOGICAL_DOWNSTREAM_PROTOCOL = (
    "random_dre_chronological_downstream"
)
STRICT_CHRONOLOGICAL_DRE_DOWNSTREAM_PROTOCOL = (
    "strict_chronological_dre_downstream"
)
FIXED_STRICT_CHRONOLOGICAL_VIDEO_PROTOCOL = (
    "fixed_strict_chronological_video"
)
DEFAULT_DRE_REAL_FRACTION = 0.80


def canonical_feature_view(value: str) -> str:
    feature_view = str(value).strip().lower()
    if feature_view not in FEATURE_VIEWS:
        raise ValueError(
            "feature_view must be one of: "
            + ", ".join(sorted(FEATURE_VIEWS))
        )
    return feature_view


def signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("Temporal SHARPs features must be finite.")
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_condition_values(
    values: np.ndarray,
    mode: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("Condition values must be finite.")
    if str(mode) != CONDITION_INVERSE_TRANSFORM:
        raise ValueError(
            "Temporal condition inverse mode must be "
            f"{CONDITION_INVERSE_TRANSFORM!r}."
        )
    with np.errstate(over="raise", invalid="raise"):
        raw_flux = np.power(10.0, values)
    if np.any(~np.isfinite(raw_flux)) or np.any(raw_flux <= 0.0):
        raise FloatingPointError(
            "Inverse base-10 condition transform produced invalid flux."
        )
    return raw_flux


def strip_log_prefix(name: str) -> str:
    return str(name)


def _finite_positive_scale(value: Any, name: str) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return scale


def load_temporal_training_contract(
    metadata_path: str | Path,
    *,
    split_label: str,
    feature_view: str = FEATURE_VIEW,
    real_pool_rows: int | None = None,
) -> dict[str, Any]:
    label = canonical_split_label(split_label)
    feature_view = canonical_feature_view(feature_view)
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {path}")
    metadata = load_json(path)
    schema = metadata.get("schema", {})
    normalization = metadata.get("normalization", {})
    model_config = metadata.get("model_config", {})

    actual_strategy = str(
        metadata.get("split", {}).get("strategy", "")
    ).lower()
    expected_strategy = str(SPLIT_SPECS[label]["training_strategy"])
    if actual_strategy != expected_strategy:
        raise ValueError(
            f"{label} expects split strategy {expected_strategy!r}, found "
            f"{actual_strategy!r}."
        )

    frame_indices = tuple(int(value) for value in schema["frame_indices"])
    base_features = tuple(str(value) for value in schema["base_features"])
    feature_columns = tuple(str(value) for value in schema["feature_columns"])
    if frame_indices != tuple(range(60)):
        raise ValueError("Temporal regression requires frame indices 0 through 59.")
    if len(base_features) != 20:
        raise ValueError("Temporal regression requires 20 base SHARPs features.")
    expected_feature_columns = tuple(
        f"frame_{frame_index:03d}__{feature}"
        for frame_index in frame_indices
        for feature in base_features
    )
    if feature_columns != expected_feature_columns:
        raise ValueError(
            "Temporal metadata feature columns are not frame-major."
        )
    if int(model_config["x_dim"]) != 1_200:
        raise ValueError("Temporal model_config.x_dim must equal 1200.")

    condition = normalization.get("condition", {})
    if condition.get("source_column") != schema.get("flux_column"):
        raise ValueError("Condition source and schema flux column disagree.")
    if condition.get("transform") != "log10":
        raise ValueError("Temporal regression condition must use log10(flux).")
    condition_mean = float(condition["mean"])
    condition_scale = _finite_positive_scale(
        condition["scale"],
        "condition scale",
    )
    if not np.isfinite(condition_mean):
        raise ValueError("Condition mean must be finite.")

    feature_statistics = normalization.get("feature_statistics", {})
    if set(feature_statistics) != set(base_features):
        raise ValueError(
            "Feature normalization statistics do not match base features."
        )
    base_feature_means = []
    base_feature_scales = []
    for feature in base_features:
        stats = feature_statistics[feature]
        mean = float(stats["mean"])
        if not np.isfinite(mean):
            raise ValueError(f"Feature mean for {feature} must be finite.")
        base_feature_means.append(mean)
        base_feature_scales.append(
            _finite_positive_scale(
                stats["scale"],
                f"feature scale for {feature}",
            )
        )

    last_frame_index = frame_indices[-1]
    if feature_view == LAST_FRAME_FEATURE_VIEW:
        regression_feature_columns = tuple(
            f"frame_{last_frame_index:03d}__{feature}"
            for feature in base_features
        )
        feature_means = np.asarray(
            base_feature_means,
            dtype=np.float64,
        )
        feature_scales = np.asarray(
            base_feature_scales,
            dtype=np.float64,
        )
        selected_frame_index: int | None = last_frame_index
    else:
        regression_feature_columns = feature_columns
        feature_means = np.tile(
            np.asarray(base_feature_means, dtype=np.float64),
            len(frame_indices),
        )
        feature_scales = np.tile(
            np.asarray(base_feature_scales, dtype=np.float64),
            len(frame_indices),
        )
        selected_frame_index = None
    datetime_column = str(schema.get("datetime_column", "")).strip()
    if not datetime_column:
        raise ValueError(
            "Temporal metadata schema must define datetime_column."
        )
    if metadata.get("training_protocol") == TRAINING_ONLY_PROTOCOL:
        if real_pool_rows is None:
            raise ValueError(
                "Training-only metadata requires a separate downstream real_pool_rows count."
            )
        real_pool_rows = int(real_pool_rows)
    else:
        real_pool_rows = int(metadata["split"]["validation_rows"])
    if real_pool_rows < 2:
        raise ValueError("Temporal real holdout must contain at least two rows.")

    return {
        "metadata": metadata,
        "metadata_path": path,
        "split_label": label,
        "training_strategy": expected_strategy,
        "feature_view": feature_view,
        "frame_index": selected_frame_index,
        "frame_indices": frame_indices,
        "base_features": base_features,
        "regression_feature_columns": regression_feature_columns,
        "flux_column": str(schema["flux_column"]),
        "datetime_column": datetime_column,
        "feature_means": feature_means,
        "feature_scales": feature_scales,
        "condition_mean": condition_mean,
        "condition_scale": condition_scale,
        "real_pool_rows": real_pool_rows,
        "generator_train_end_year": (
            int(metadata["train_end_year"])
            if metadata.get("training_protocol") == TRAINING_ONLY_PROTOCOL
            else None
        ),
    }


def normalize_temporal_frame(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = list(contract["regression_feature_columns"])
    flux_column = str(contract["flux_column"])
    missing = set(feature_columns + [flux_column]).difference(frame.columns)
    if missing:
        raise ValueError(
            "Temporal CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    raw_x = frame[feature_columns].to_numpy(dtype=np.float64, copy=True)
    raw_flux = frame[flux_column].to_numpy(dtype=np.float64, copy=True)
    if np.any(~np.isfinite(raw_flux)) or np.any(raw_flux <= 0.0):
        raise ValueError("Endpoint flux must be finite and strictly positive.")

    transformed_x = signed_log1p(raw_x)
    normalized_x = (
        transformed_x - contract["feature_means"][None, :]
    ) / contract["feature_scales"][None, :]
    normalized_y = (
        np.log10(raw_flux) - float(contract["condition_mean"])
    ) / float(contract["condition_scale"])
    if np.any(~np.isfinite(normalized_x)):
        raise ValueError("Normalized temporal features are non-finite.")
    if np.any(~np.isfinite(normalized_y)):
        raise ValueError("Normalized temporal targets are non-finite.")
    return normalized_x, normalized_y


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_fingerprint(
    artifacts: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    signatures = {
        "training_metadata": _source_signature(
            Path(artifacts["training_metadata_path"])
        ),
        "real_pool_csv": _source_signature(
            Path(artifacts["real_pool_csv_path"])
        ),
        "synthetic_samples_csv": _source_signature(
            Path(artifacts["synthetic_samples_path"])
        ),
        "synthetic_sample_summary": _source_signature(
            Path(artifacts["synthetic_sample_summary_path"])
        ),
    }
    if contract["generator_train_end_year"] is not None:
        signatures["artifact_manifest"] = _source_signature(
            Path(artifacts["real_pool_csv_path"]).with_name(
                ARTIFACT_MANIFEST_FILENAME
            )
        )
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "split_label": artifacts["split_label"],
        "feature_view": contract["feature_view"],
        "regression_feature_columns": list(
            contract["regression_feature_columns"]
        ),
        "datetime_column": contract["datetime_column"],
        "condition_mean": contract["condition_mean"],
        "condition_scale": contract["condition_scale"],
        "source_signatures": signatures,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20], signatures


def _iter_csv_chunks(
    path: Path,
    contract: dict[str, Any],
    *,
    chunk_size: int,
    validate_saved_condition: bool,
    include_datetime: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    use_columns = list(contract["regression_feature_columns"]) + [
        contract["flux_column"]
    ]
    header = pd.read_csv(path, nrows=0)
    saved_condition_column = "condition_standardized_log10_flux"
    has_saved_condition = saved_condition_column in header.columns
    if validate_saved_condition and not has_saved_condition:
        raise ValueError(
            "Synthetic temporal CSV must include "
            f"{saved_condition_column!r}."
        )
    if validate_saved_condition:
        use_columns.append(saved_condition_column)
    if include_datetime:
        use_columns.append(contract["datetime_column"])

    chunk_reader = pd.read_csv(
        path,
        usecols=use_columns,
        chunksize=int(chunk_size),
    )
    with chunk_reader:
        for chunk in chunk_reader:
            x_values, y_values = normalize_temporal_frame(chunk, contract)
            if validate_saved_condition:
                saved = chunk[saved_condition_column].to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
                if not np.allclose(
                    saved,
                    y_values,
                    rtol=0.0,
                    atol=1e-8,
                ):
                    max_error = float(np.max(np.abs(saved - y_values)))
                    raise ValueError(
                        "Synthetic standardized log10 condition disagrees "
                        "with the selected split metadata "
                        f"(max error {max_error:.6g})."
                    )
            timestamp_values = None
            if include_datetime:
                parsed_timestamps = pd.to_datetime(
                    chunk[contract["datetime_column"]],
                    errors="raise",
                    utc=True,
                )
                if parsed_timestamps.isna().any():
                    raise ValueError(
                        "Temporal real-pool timestamps must not be missing."
                    )
                cutoff = contract["generator_train_end_year"]
                if (
                    cutoff is not None
                    and (parsed_timestamps.dt.year <= cutoff).any()
                ):
                    raise ValueError(
                        "Downstream real pool contains generator-training years."
                    )
                timestamp_values = parsed_timestamps.array.asi8.copy()
            yield x_values, y_values, timestamp_values


def _write_cached_arrays(
    *,
    csv_path: Path,
    contract: dict[str, Any],
    expected_rows: int,
    x_path: Path,
    y_path: Path,
    time_path: Path | None,
    chunk_size: int,
    validate_saved_condition: bool,
) -> None:
    expected_rows = int(expected_rows)
    x_dim = len(contract["regression_feature_columns"])
    x_memmap = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float64,
        shape=(expected_rows, x_dim),
    )
    y_memmap = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float64,
        shape=(expected_rows,),
    )
    time_memmap = None
    if time_path is not None:
        time_memmap = np.lib.format.open_memmap(
            time_path,
            mode="w+",
            dtype=np.int64,
            shape=(expected_rows,),
        )
    row_offset = 0
    for chunk_x, chunk_y, chunk_time in _iter_csv_chunks(
        csv_path,
        contract,
        chunk_size=chunk_size,
        validate_saved_condition=validate_saved_condition,
        include_datetime=time_path is not None,
    ):
        next_offset = row_offset + int(chunk_x.shape[0])
        if next_offset > expected_rows:
            raise ValueError(
                f"{csv_path} contains more than {expected_rows} rows."
            )
        x_memmap[row_offset:next_offset] = chunk_x
        y_memmap[row_offset:next_offset] = chunk_y
        if time_memmap is not None:
            if chunk_time is None:
                raise RuntimeError("Timestamp cache chunk is missing.")
            time_memmap[row_offset:next_offset] = chunk_time
        row_offset = next_offset
    x_memmap.flush()
    y_memmap.flush()
    if time_memmap is not None:
        time_memmap.flush()
    del x_memmap
    del y_memmap
    if time_memmap is not None:
        del time_memmap
    if row_offset != expected_rows:
        raise ValueError(
            f"{csv_path} contains {row_offset} rows; expected {expected_rows}."
        )


def _load_validated_source_contract(
    artifacts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    bundle_manifest = load_training_only_manifest(artifacts)
    contract = load_temporal_training_contract(
        artifacts["training_metadata_path"],
        split_label=artifacts["split_label"],
        feature_view=artifacts.get("feature_view", FEATURE_VIEW),
        real_pool_rows=(
            int(bundle_manifest["real_pool_rows"])
            if bundle_manifest is not None
            else None
        ),
    )
    sample_summary = load_json(
        Path(artifacts["synthetic_sample_summary_path"])
    )
    synthetic_rows = int(sample_summary["sample_count"])
    if synthetic_rows <= 0:
        raise ValueError("Synthetic sample_count must be positive.")
    summary_conditioning = sample_summary["conditioning"]
    summary_condition_mean = float(
        summary_conditioning["standardization_mean"]
    )
    summary_condition_scale = _finite_positive_scale(
        summary_conditioning["standardization_scale"],
        "synthetic summary condition scale",
    )
    if not np.isclose(
        summary_condition_mean,
        contract["condition_mean"],
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        summary_condition_scale,
        contract["condition_scale"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Synthetic summary condition standardization disagrees with "
            "the selected split metadata."
        )
    return contract, sample_summary, synthetic_rows


def _cache_manifest_path_from_contract(
    artifacts: dict[str, Any],
    contract: dict[str, Any],
) -> Path:
    fingerprint, _ = _cache_fingerprint(artifacts, contract)
    cache_root = Path(artifacts["cache_root"]).expanduser().resolve()
    return (
        cache_root
        / artifacts["split_label"]
        / fingerprint
        / "cache_manifest.json"
    )


def expected_temporal_cache_manifest_path(
    artifacts: dict[str, Any],
) -> Path:
    contract, _, _ = _load_validated_source_contract(artifacts)
    return _cache_manifest_path_from_contract(artifacts, contract)


def load_temporal_source_summary(
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    contract, _, synthetic_rows = _load_validated_source_contract(artifacts)
    cache_manifest_path = _cache_manifest_path_from_contract(
        artifacts,
        contract,
    )
    return {
        "n_real": int(contract["real_pool_rows"]),
        "n_synthetic": int(synthetic_rows),
        "x_dim": len(contract["regression_feature_columns"]),
        "split_label": contract["split_label"],
        "training_strategy": contract["training_strategy"],
        "feature_view": contract["feature_view"],
        "cache_manifest_path": str(cache_manifest_path),
    }


def load_cache_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Temporal cache manifest not found: {manifest_path}"
        )
    manifest = runtime_paths(load_json(manifest_path), manifest_path.parent)
    if int(manifest.get("cache_format_version", -1)) != CACHE_FORMAT_VERSION:
        raise ValueError("Unsupported temporal cache format version.")
    for relative_path in manifest["arrays"].values():
        array_path = manifest_path.parent / relative_path
        if not array_path.is_file():
            raise FileNotFoundError(
                f"Temporal cache array not found: {array_path}"
            )
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def prepare_temporal_cache(
    artifacts: dict[str, Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    contract, _, synthetic_rows = _load_validated_source_contract(
        artifacts
    )
    fingerprint, source_signatures = _cache_fingerprint(
        artifacts,
        contract,
    )
    cache_root = Path(artifacts["cache_root"]).expanduser().resolve()
    split_cache_root = cache_root / artifacts["split_label"]
    final_cache_dir = split_cache_root / fingerprint
    final_manifest_path = final_cache_dir / "cache_manifest.json"
    if final_manifest_path.is_file():
        load_cache_manifest(final_manifest_path)
        return final_manifest_path

    load_training_only_manifest(artifacts, verify_csv_hashes=True)
    split_cache_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{fingerprint}.",
            dir=split_cache_root,
        )
    )
    try:
        _write_cached_arrays(
            csv_path=Path(artifacts["real_pool_csv_path"]),
            contract=contract,
            expected_rows=contract["real_pool_rows"],
            x_path=temporary_dir / "real_pool_x.npy",
            y_path=temporary_dir / "real_pool_y.npy",
            time_path=temporary_dir / "real_pool_time.npy",
            chunk_size=chunk_size,
            validate_saved_condition=False,
        )
        _write_cached_arrays(
            csv_path=Path(artifacts["synthetic_samples_path"]),
            contract=contract,
            expected_rows=synthetic_rows,
            x_path=temporary_dir / "q_prop_x.npy",
            y_path=temporary_dir / "q_prop_y.npy",
            time_path=None,
            chunk_size=chunk_size,
            validate_saved_condition=True,
        )
        manifest = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "split_label": artifacts["split_label"],
            "training_strategy": artifacts["training_strategy"],
            "feature_view": contract["feature_view"],
            "frame_index": contract["frame_index"],
            "frame_indices": list(contract["frame_indices"]),
            "feature_names": list(
                contract["regression_feature_columns"]
            ),
            "datetime_column": contract["datetime_column"],
            "normalized_condition_name": NORMALIZED_CONDITION_NAME,
            "condition_inverse_transform": CONDITION_INVERSE_TRANSFORM,
            "condition_mean": float(contract["condition_mean"]),
            "condition_scale": float(contract["condition_scale"]),
            "real_pool_rows": int(contract["real_pool_rows"]),
            "synthetic_rows": synthetic_rows,
            "x_dim": len(contract["regression_feature_columns"]),
            "arrays": {
                "real_pool_x": "real_pool_x.npy",
                "real_pool_y": "real_pool_y.npy",
                "real_pool_time": "real_pool_time.npy",
                "q_prop_x": "q_prop_x.npy",
                "q_prop_y": "q_prop_y.npy",
            },
            "source_signatures": {
                key: {"path": value["path"], "size_bytes": value["size_bytes"]}
                for key, value in source_signatures.items()
            },
        }
        temporary_manifest = temporary_dir / "cache_manifest.json"
        save_json(temporary_manifest, manifest)
        if final_cache_dir.exists():
            shutil.rmtree(temporary_dir)
        else:
            os.replace(temporary_dir, final_cache_dir)
        load_cache_manifest(final_manifest_path)
        return final_manifest_path
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def resolve_train_test_counts(
    n_rows: int,
    *,
    test_fraction: float,
) -> dict[str, int]:
    n_rows = int(n_rows)
    test_fraction = float(test_fraction)
    if n_rows < 2:
        raise ValueError("Need at least two real rows.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one.")
    n_test = max(1, int(round(test_fraction * n_rows)))
    n_test = min(n_test, n_rows - 1)
    return {"n_train": n_rows - n_test, "n_test": n_test}


def resolve_chronological_protocol_counts(
    n_rows: int,
    *,
    downstream_train_fraction: float,
    dre_real_fraction: float = DEFAULT_DRE_REAL_FRACTION,
) -> dict[str, int]:
    n_rows = int(n_rows)
    downstream_train_fraction = float(downstream_train_fraction)
    dre_real_fraction = float(dre_real_fraction)
    if n_rows < 3:
        raise ValueError(
            "Need at least three real rows for separate DRE, downstream "
            "training, and downstream test pools."
        )
    if not 0.0 < downstream_train_fraction < 1.0:
        raise ValueError(
            "downstream_train_fraction must lie strictly between zero "
            "and one."
        )
    if not 0.0 < dre_real_fraction < 1.0:
        raise ValueError(
            "dre_real_fraction must lie strictly between zero and one."
        )

    n_ratio_train = int(round(dre_real_fraction * n_rows))
    n_ratio_train = min(max(1, n_ratio_train), n_rows - 2)
    n_downstream_pool = n_rows - n_ratio_train
    n_train = int(round(downstream_train_fraction * n_downstream_pool))
    n_train = min(max(1, n_train), n_downstream_pool - 1)
    return {
        "n_real_pool": n_rows,
        "n_ratio_train": n_ratio_train,
        "n_downstream_pool": n_downstream_pool,
        "n_train": n_train,
        "n_test": n_downstream_pool - n_train,
    }


def resolve_fixed_strict_chronological_counts(
    n_rows: int,
    *,
    dre_real_fraction: float,
    train_size: int,
    test_size: int | None = None,
) -> dict[str, int]:
    """Resolve fixed train/test counts after an earliest-time DRE block.

    When ``test_size`` is omitted, every downstream row after the fixed
    training prefix is assigned to the test set. Supplying ``test_size``
    retains the optional middle temporal buffer used by earlier callers.
    """
    n_rows = int(n_rows)
    dre_real_fraction = float(dre_real_fraction)
    train_size = int(train_size)
    if n_rows < 3:
        raise ValueError(
            "Need at least three real rows for fixed chronological DRE, "
            "training, and testing."
        )
    if not 0.0 < dre_real_fraction < 1.0:
        raise ValueError(
            "dre_real_fraction must lie strictly between zero and one."
        )
    if train_size <= 0:
        raise ValueError("Fixed chronological train size must be positive.")

    n_ratio_train = int(round(dre_real_fraction * n_rows))
    n_ratio_train = min(max(1, n_ratio_train), n_rows - 2)
    n_downstream_pool = n_rows - n_ratio_train
    if train_size >= n_downstream_pool:
        raise ValueError(
            "Fixed chronological train size must leave at least one "
            f"downstream test row: {train_size} >= {n_downstream_pool}."
        )
    resolved_test_size = (
        n_downstream_pool - train_size
        if test_size is None
        else int(test_size)
    )
    if resolved_test_size <= 0:
        raise ValueError("Fixed chronological test size must be positive.")
    if train_size + resolved_test_size > n_downstream_pool:
        raise ValueError(
            "Fixed chronological train/test sizes exceed the downstream "
            f"pool: {train_size} + {resolved_test_size} > "
            f"{n_downstream_pool}."
        )
    return {
        "n_real_pool": n_rows,
        "n_ratio_train": n_ratio_train,
        "n_downstream_pool": n_downstream_pool,
        "n_train": train_size,
        "n_unused": n_downstream_pool - train_size - resolved_test_size,
        "n_test": resolved_test_size,
    }


def split_real_pool_arrays(
    real_x: np.ndarray,
    real_y: np.ndarray,
    *,
    test_fraction: float,
    split_seed: int,
) -> dict[str, np.ndarray]:
    real_x = np.asarray(real_x, dtype=np.float64)
    real_y = np.asarray(real_y, dtype=np.float64)
    if real_x.ndim != 2 or real_y.ndim != 1:
        raise ValueError("Real cache arrays must have shapes [rows, x_dim] and [rows].")
    if real_x.shape[0] != real_y.shape[0]:
        raise ValueError("Real cache row counts disagree.")
    counts = resolve_train_test_counts(
        real_x.shape[0],
        test_fraction=test_fraction,
    )
    indices = np.random.RandomState(int(split_seed)).permutation(
        real_x.shape[0]
    )
    test_indices = indices[: counts["n_test"]]
    train_indices = indices[counts["n_test"] :]
    return {
        "train_x": real_x[train_indices].copy(),
        "train_y": real_y[train_indices].copy(),
        "test_x": real_x[test_indices].copy(),
        "test_y": real_y[test_indices].copy(),
        "train_indices": train_indices.copy(),
        "test_indices": test_indices.copy(),
    }


def split_random_dre_chronological_downstream_arrays(
    real_x: np.ndarray,
    real_y: np.ndarray,
    real_time: np.ndarray,
    *,
    downstream_train_fraction: float,
    dre_real_fraction: float,
    split_seed: int,
) -> dict[str, np.ndarray]:
    real_x = np.asarray(real_x, dtype=np.float64)
    real_y = np.asarray(real_y, dtype=np.float64)
    real_time = np.asarray(real_time, dtype=np.int64)
    if real_x.ndim != 2 or real_y.ndim != 1:
        raise ValueError(
            "Real cache arrays must have shapes [rows, x_dim] and [rows]."
        )
    if real_time.ndim != 1:
        raise ValueError("Real timestamp cache must have shape [rows].")
    if not (
        real_x.shape[0] == real_y.shape[0] == real_time.shape[0]
    ):
        raise ValueError("Real feature, target, and timestamp rows disagree.")

    counts = resolve_chronological_protocol_counts(
        real_x.shape[0],
        downstream_train_fraction=downstream_train_fraction,
        dre_real_fraction=dre_real_fraction,
    )
    randomized_indices = np.random.RandomState(int(split_seed)).permutation(
        real_x.shape[0]
    )
    ratio_train_indices = randomized_indices[
        : counts["n_ratio_train"]
    ]
    downstream_indices = randomized_indices[
        counts["n_ratio_train"] :
    ]
    chronological_order = np.lexsort(
        (
            downstream_indices,
            real_time[downstream_indices],
        )
    )
    ordered_downstream_indices = downstream_indices[chronological_order]
    train_indices = ordered_downstream_indices[: counts["n_train"]]
    test_indices = ordered_downstream_indices[counts["n_train"] :]

    return {
        "ratio_train_x": real_x[ratio_train_indices].copy(),
        "ratio_train_y": real_y[ratio_train_indices].copy(),
        "train_x": real_x[train_indices].copy(),
        "train_y": real_y[train_indices].copy(),
        "test_x": real_x[test_indices].copy(),
        "test_y": real_y[test_indices].copy(),
        "ratio_train_indices": ratio_train_indices.copy(),
        "train_indices": train_indices.copy(),
        "test_indices": test_indices.copy(),
        "train_time": real_time[train_indices].copy(),
        "test_time": real_time[test_indices].copy(),
    }


def split_strict_chronological_dre_downstream_arrays(
    real_x: np.ndarray,
    real_y: np.ndarray,
    real_time: np.ndarray,
    *,
    downstream_train_fraction: float,
    dre_real_fraction: float,
) -> dict[str, np.ndarray]:
    """
    Put DRE, downstream training, and downstream testing in time order.
    """
    real_x = np.asarray(real_x, dtype=np.float64)
    real_y = np.asarray(real_y, dtype=np.float64)
    real_time = np.asarray(real_time, dtype=np.int64)
    if real_x.ndim != 2 or real_y.ndim != 1:
        raise ValueError(
            "Real cache arrays must have shapes [rows, x_dim] and [rows]."
        )
    if real_time.ndim != 1:
        raise ValueError("Real timestamp cache must have shape [rows].")
    if not (
        real_x.shape[0] == real_y.shape[0] == real_time.shape[0]
    ):
        raise ValueError("Real feature, target, and timestamp rows disagree.")

    counts = resolve_chronological_protocol_counts(
        real_x.shape[0],
        downstream_train_fraction=downstream_train_fraction,
        dre_real_fraction=dre_real_fraction,
    )
    source_indices = np.arange(real_x.shape[0], dtype=int)
    chronological_order = np.lexsort((source_indices, real_time))
    ordered_indices = source_indices[chronological_order]
    ratio_train_indices = ordered_indices[: counts["n_ratio_train"]]
    downstream_indices = ordered_indices[counts["n_ratio_train"] :]
    train_indices = downstream_indices[: counts["n_train"]]
    test_indices = downstream_indices[counts["n_train"] :]

    return {
        "ratio_train_x": real_x[ratio_train_indices].copy(),
        "ratio_train_y": real_y[ratio_train_indices].copy(),
        "train_x": real_x[train_indices].copy(),
        "train_y": real_y[train_indices].copy(),
        "test_x": real_x[test_indices].copy(),
        "test_y": real_y[test_indices].copy(),
        "ratio_train_indices": ratio_train_indices.copy(),
        "train_indices": train_indices.copy(),
        "test_indices": test_indices.copy(),
        "ratio_train_time": real_time[ratio_train_indices].copy(),
        "train_time": real_time[train_indices].copy(),
        "test_time": real_time[test_indices].copy(),
    }


def split_fixed_strict_chronological_arrays(
    real_x: np.ndarray,
    real_y: np.ndarray,
    real_time: np.ndarray,
    *,
    dre_real_fraction: float,
    train_size: int,
    test_size: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Build fixed, disjoint DRE/train/buffer/test blocks in time order.
    """
    real_x = np.asarray(real_x, dtype=np.float64)
    real_y = np.asarray(real_y, dtype=np.float64)
    real_time = np.asarray(real_time, dtype=np.int64)
    if real_x.ndim != 2 or real_y.ndim != 1:
        raise ValueError(
            "Real cache arrays must have shapes [rows, x_dim] and [rows]."
        )
    if real_time.ndim != 1:
        raise ValueError("Real timestamp cache must have shape [rows].")
    if not (
        real_x.shape[0] == real_y.shape[0] == real_time.shape[0]
    ):
        raise ValueError("Real feature, target, and timestamp rows disagree.")

    counts = resolve_fixed_strict_chronological_counts(
        real_x.shape[0],
        dre_real_fraction=dre_real_fraction,
        train_size=train_size,
        test_size=test_size,
    )
    source_indices = np.arange(real_x.shape[0], dtype=int)
    chronological_order = np.lexsort((source_indices, real_time))
    ordered_indices = source_indices[chronological_order]
    ratio_train_indices = ordered_indices[: counts["n_ratio_train"]]
    downstream_indices = ordered_indices[counts["n_ratio_train"] :]
    train_indices = downstream_indices[: counts["n_train"]]
    test_indices = downstream_indices[-counts["n_test"] :]
    unused_indices = downstream_indices[
        counts["n_train"] : counts["n_downstream_pool"] - counts["n_test"]
    ]

    return {
        "ratio_train_x": real_x[ratio_train_indices].copy(),
        "ratio_train_y": real_y[ratio_train_indices].copy(),
        "train_x": real_x[train_indices].copy(),
        "train_y": real_y[train_indices].copy(),
        "test_x": real_x[test_indices].copy(),
        "test_y": real_y[test_indices].copy(),
        "ratio_train_indices": ratio_train_indices.copy(),
        "train_indices": train_indices.copy(),
        "unused_indices": unused_indices.copy(),
        "test_indices": test_indices.copy(),
        "ratio_train_time": real_time[ratio_train_indices].copy(),
        "train_time": real_time[train_indices].copy(),
        "unused_time": real_time[unused_indices].copy(),
        "test_time": real_time[test_indices].copy(),
    }


def load_temporal_artifact_bundle(
    artifact_config: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "split_label",
        "temporal_cache_manifest_path",
        "pipeline_test_fraction",
        "pipeline_split_seed",
    }
    missing = required.difference(artifact_config)
    if missing:
        raise ValueError(
            "Temporal artifact config is missing keys: "
            + ", ".join(sorted(missing))
        )
    label = canonical_split_label(artifact_config["split_label"])
    manifest = load_cache_manifest(
        artifact_config["temporal_cache_manifest_path"]
    )
    if manifest["split_label"] != label:
        raise ValueError(
            "Temporal cache split label does not match artifact config."
        )
    requested_feature_view = canonical_feature_view(
        artifact_config.get("feature_view", manifest["feature_view"])
    )
    if requested_feature_view != manifest["feature_view"]:
        raise ValueError(
            "Temporal cache feature view does not match artifact config."
        )
    cache_dir = Path(manifest["_manifest_path"]).parent
    arrays = manifest["arrays"]
    real_pool_x = np.load(
        cache_dir / arrays["real_pool_x"],
        mmap_mode="r",
    )
    real_pool_y = np.load(
        cache_dir / arrays["real_pool_y"],
        mmap_mode="r",
    )
    q_prop_x = np.load(
        cache_dir / arrays["q_prop_x"],
        mmap_mode="r",
    )
    q_prop_y = np.load(
        cache_dir / arrays["q_prop_y"],
        mmap_mode="r",
    )
    real_pool_time = np.load(
        cache_dir / arrays["real_pool_time"],
        mmap_mode="r",
    )
    if real_pool_x.shape != (
        int(manifest["real_pool_rows"]),
        int(manifest["x_dim"]),
    ):
        raise ValueError("Cached real feature shape disagrees with manifest.")
    if q_prop_x.shape != (
        int(manifest["synthetic_rows"]),
        int(manifest["x_dim"]),
    ):
        raise ValueError("Cached proposal feature shape disagrees with manifest.")
    if real_pool_y.shape != (int(manifest["real_pool_rows"]),):
        raise ValueError("Cached real target shape disagrees with manifest.")
    if q_prop_y.shape != (int(manifest["synthetic_rows"]),):
        raise ValueError("Cached proposal target shape disagrees with manifest.")
    if real_pool_time.shape != (int(manifest["real_pool_rows"]),):
        raise ValueError("Cached real timestamp shape disagrees with manifest.")

    split_protocol = str(
        artifact_config.get(
            "temporal_split_protocol",
            RANDOM_DOWNSTREAM_PROTOCOL,
        )
    )
    if split_protocol == RANDOM_DOWNSTREAM_PROTOCOL:
        split_bundle = split_real_pool_arrays(
            real_pool_x,
            real_pool_y,
            test_fraction=float(
                artifact_config["pipeline_test_fraction"]
            ),
            split_seed=int(artifact_config["pipeline_split_seed"]),
        )
        dre_real_fraction = None
    elif (
        split_protocol
        == RANDOM_DRE_CHRONOLOGICAL_DOWNSTREAM_PROTOCOL
    ):
        dre_real_fraction = float(
            artifact_config.get(
                "dre_real_fraction",
                DEFAULT_DRE_REAL_FRACTION,
            )
        )
        split_bundle = (
            split_random_dre_chronological_downstream_arrays(
                real_pool_x,
                real_pool_y,
                real_pool_time,
                downstream_train_fraction=(
                    1.0
                    - float(
                        artifact_config["pipeline_test_fraction"]
                    )
                ),
                dre_real_fraction=dre_real_fraction,
                split_seed=int(
                    artifact_config["pipeline_split_seed"]
                ),
            )
        )
    elif (
        split_protocol
        == STRICT_CHRONOLOGICAL_DRE_DOWNSTREAM_PROTOCOL
    ):
        dre_real_fraction = float(
            artifact_config.get(
                "dre_real_fraction",
                DEFAULT_DRE_REAL_FRACTION,
            )
        )
        split_bundle = split_strict_chronological_dre_downstream_arrays(
            real_pool_x,
            real_pool_y,
            real_pool_time,
            downstream_train_fraction=(
                1.0
                - float(artifact_config["pipeline_test_fraction"])
            ),
            dre_real_fraction=dre_real_fraction,
        )
    elif split_protocol == FIXED_STRICT_CHRONOLOGICAL_VIDEO_PROTOCOL:
        dre_real_fraction = float(
            artifact_config.get(
                "dre_real_fraction",
                DEFAULT_DRE_REAL_FRACTION,
            )
        )
        try:
            fixed_train_size = int(artifact_config["fixed_train_size"])
        except KeyError as exc:
            raise ValueError(
                "Fixed temporal-video protocol requires fixed_train_size."
            ) from exc
        configured_test_size = artifact_config.get("fixed_test_size")
        fixed_test_size = (
            None
            if configured_test_size is None
            else int(configured_test_size)
        )
        split_bundle = split_fixed_strict_chronological_arrays(
            real_pool_x,
            real_pool_y,
            real_pool_time,
            dre_real_fraction=dre_real_fraction,
            train_size=fixed_train_size,
            test_size=fixed_test_size,
        )
    else:
        raise ValueError(
            "Unsupported temporal_split_protocol: "
            f"{split_protocol!r}."
        )
    feature_names = [str(value) for value in manifest["feature_names"]]
    condition_name = str(manifest["normalized_condition_name"])
    compatibility_metadata = {
        "features": feature_names,
        "condition": condition_name,
        "normalization_stats": {
            condition_name: {
                "mean": float(manifest["condition_mean"]),
                "std": float(manifest["condition_scale"]),
            }
        },
        "condition_normalization_mode": "log10",
    }
    result = {
        "metadata": compatibility_metadata,
        "normalized_feature_names": feature_names,
        "normalized_condition_name": condition_name,
        "condition_normalization_mode": "log10",
        "condition_inverse_transform": CONDITION_INVERSE_TRANSFORM,
        "condition_plot_label": "log10_flux",
        "train_x": split_bundle["train_x"],
        "train_y": split_bundle["train_y"],
        "test_x": split_bundle["test_x"],
        "test_y": split_bundle["test_y"],
        "q_prop_x": q_prop_x,
        "q_prop_y": q_prop_y,
        "real_pool_x": real_pool_x,
        "real_pool_y": real_pool_y,
        "real_pool_time": real_pool_time,
        "pipeline_test_fraction": float(
            artifact_config["pipeline_test_fraction"]
        ),
        "pipeline_split_seed": int(
            artifact_config["pipeline_split_seed"]
        ),
        "train_indices": split_bundle["train_indices"],
        "test_indices": split_bundle["test_indices"],
        "temporal_split_protocol": split_protocol,
        "dre_real_fraction": dre_real_fraction,
        "training_metadata_path": manifest["source_signatures"][
            "training_metadata"
        ]["path"],
        "real_pool_csv_path": manifest["source_signatures"][
            "real_pool_csv"
        ]["path"],
        "q_prop_joint_csv_path": manifest["source_signatures"][
            "synthetic_samples_csv"
        ]["path"],
        "temporal_cache_manifest_path": manifest["_manifest_path"],
        "split_label": label,
        "feature_view": manifest["feature_view"],
    }
    for optional_key in (
        "ratio_train_x",
        "ratio_train_y",
        "ratio_train_indices",
        "ratio_train_time",
        "train_time",
        "unused_indices",
        "unused_time",
        "test_time",
    ):
        if optional_key in split_bundle:
            result[optional_key] = split_bundle[optional_key]
    return result


def load_temporal_real_pool_summary(
    cache_manifest_path: str | Path,
    *,
    test_fraction: float,
) -> dict[str, Any]:
    manifest = load_cache_manifest(cache_manifest_path)
    counts = resolve_train_test_counts(
        int(manifest["real_pool_rows"]),
        test_fraction=test_fraction,
    )
    return {
        "n_real": int(manifest["real_pool_rows"]),
        "n_train": int(counts["n_train"]),
        "n_test": int(counts["n_test"]),
        "x_dim": int(manifest["x_dim"]),
        "split_label": manifest["split_label"],
        "feature_view": manifest["feature_view"],
        "cache_manifest_path": manifest["_manifest_path"],
    }
