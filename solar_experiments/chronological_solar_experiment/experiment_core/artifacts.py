from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SAMPLING_ROOT = REPO_ROOT / "data"
REAL_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"
CHRONOLOGICAL_DATA_ROOT = REAL_DATA_ROOT
TRAINING_ONLY_PROTOCOL = "chronological_training_only"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
BUNDLE_PATH_KEYS = (
    "training_metadata_path",
    "real_pool_csv_path",
    "synthetic_samples_path",
    "synthetic_sample_summary_path",
)

SPLIT_SPECS = {
    "chronological_split": {
        "training_strategy": "chronological",
        "metadata_path": CHRONOLOGICAL_DATA_ROOT / "training_metadata.json",
        "real_pool_path": CHRONOLOGICAL_DATA_ROOT / "test_chronological.csv",
    },
}


def canonical_split_label(value: str) -> str:
    label = str(value).strip().lower()
    if label not in SPLIT_SPECS:
        raise ValueError(
            "split_label must be one of: "
            + ", ".join(sorted(SPLIT_SPECS))
        )
    return label


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def default_metadata_path(split_label: str) -> Path:
    label = canonical_split_label(split_label)
    return Path(SPLIT_SPECS[label]["metadata_path"]).resolve()


def default_real_pool_path(split_label: str) -> Path:
    label = canonical_split_label(split_label)
    return Path(SPLIT_SPECS[label]["real_pool_path"]).resolve()


def discover_synthetic_samples_path(split_label: str) -> Path:
    label = canonical_split_label(split_label)
    if label == "chronological_split":
        return (
            CHRONOLOGICAL_DATA_ROOT
            / "synthetic_samples_chronological_split.csv"
        ).resolve()
    split_dir = SAMPLING_ROOT / "samples" / label
    matches = sorted(
        split_dir.glob(f"*/synthetic_samples_{label}.csv")
    )
    if not matches:
        raise FileNotFoundError(
            f"No synthetic sample CSV found for {label} under {split_dir}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple synthetic sample CSVs found for {label}; pass an "
            "explicit --synthetic-samples-path: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_only_pair(
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Check generator provenance without consulting downstream observations."""
    if metadata.get("training_protocol") != TRAINING_ONLY_PROTOCOL:
        raise ValueError("Expected chronological_training_only generator metadata.")
    split = metadata["split"]
    cutoff = int(metadata["train_end_year"])
    if (
        split.get("strategy") != "chronological"
        or int(split["cutoff_year_inclusive"]) != cutoff
        or "validation_rows" in split
        or any(int(year) > cutoff for year in split["train_years"])
    ):
        raise ValueError("Generator metadata violates the training-only year cutoff.")
    epoch = int(metadata["completed_epochs"])
    checkpoint = summary.get("checkpoint", {})
    if (
        epoch <= 0
        or metadata.get("checkpoint_selection") != "final_epoch"
        or metadata.get("checkpoint_epochs") != [epoch]
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("training_protocol") != TRAINING_ONLY_PROTOCOL
    ):
        raise ValueError(
            "Synthetic samples must come from the final training-only epoch."
        )
    reference = metadata.get("condition_reference", {})
    source = reference.get("source", {})
    conditioning = summary.get("conditioning", {})
    if (
        source.get("scope") != "generator_training_only"
        or source.get("cutoff_year_inclusive") != cutoff
        or int(source.get("endpoint_year_max", cutoff + 1)) > cutoff
        or conditioning.get("reference") != reference
        or conditioning.get("mode") != "boundary_matched"
    ):
        raise ValueError(
            "Synthetic flux reference must match the generator training-only fit."
        )


def load_training_only_manifest(
    artifacts: dict[str, Any],
    *,
    verify_csv_hashes: bool = False,
) -> dict[str, Any] | None:
    """
    Read downstream row count separately from the generator's training metadata.
    """
    metadata = load_json(Path(artifacts["training_metadata_path"]))
    if metadata.get("training_protocol") != TRAINING_ONLY_PROTOCOL:
        return None
    summary = load_json(Path(artifacts["synthetic_sample_summary_path"]))
    validate_training_only_pair(metadata, summary)
    manifest_path = Path(artifacts["real_pool_csv_path"]).with_name(
        ARTIFACT_MANIFEST_FILENAME
    )
    manifest = load_json(manifest_path)
    if manifest.get("format_version") != 1 or int(manifest["real_pool_rows"]) < 2:
        raise ValueError(
            "Invalid training-only artifact manifest or real-pool row count."
        )
    for key in BUNDLE_PATH_KEYS:
        path = Path(artifacts[key])
        entry = manifest["files"][key]
        if (
            entry["name"] != path.name
            or int(entry["size_bytes"]) != path.stat().st_size
        ):
            raise ValueError(
                f"Artifact manifest filename/size mismatch for {path.name}."
            )
        if verify_csv_hashes or path.suffix == ".json":
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Artifact manifest checksum mismatch for {path.name}.")
    return manifest


def expected_sample_summary_path(
    synthetic_samples_path: Path,
    split_label: str,
) -> Path:
    label = canonical_split_label(split_label)
    return synthetic_samples_path.with_name(f"sample_summary_{label}.json")


def resolve_split_artifacts(
    *,
    split_label: str,
    training_metadata_path: str | Path | None = None,
    real_pool_csv_path: str | Path | None = None,
    synthetic_samples_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    require_exists: bool = True,
) -> dict[str, Any]:
    label = canonical_split_label(split_label)
    metadata_path = (
        resolve_repo_path(training_metadata_path)
        if training_metadata_path is not None
        else default_metadata_path(label)
    )
    real_pool_path = (
        resolve_repo_path(real_pool_csv_path)
        if real_pool_csv_path is not None
        else default_real_pool_path(label)
    )
    synthetic_path = (
        resolve_repo_path(synthetic_samples_path)
        if synthetic_samples_path is not None
        else discover_synthetic_samples_path(label)
    )
    resolved_cache_root = (
        resolve_repo_path(cache_root)
        if cache_root is not None
        else DEFAULT_CACHE_ROOT.resolve()
    )
    summary_path = expected_sample_summary_path(synthetic_path, label)

    paths_to_check = {
        "training metadata": metadata_path,
        "real holdout CSV": real_pool_path,
        "synthetic sample CSV": synthetic_path,
        "synthetic sample summary": summary_path,
    }
    if require_exists:
        missing = [
            f"{description}: {path}"
            for description, path in paths_to_check.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing temporal experiment artifacts: " + "; ".join(missing)
            )

    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        if (
            label == "chronological_split"
            and metadata_path == default_metadata_path(label)
            and (
                metadata.get("training_protocol") != TRAINING_ONLY_PROTOCOL
                or metadata.get("completed_epochs") != 75
            )
        ):
            raise ValueError(
                "Default chronological artifacts require the training-only epoch-75 run."
            )
        actual_strategy = str(
            metadata.get("split", {}).get("strategy", "")
        ).lower()
        expected_strategy = str(
            SPLIT_SPECS[label]["training_strategy"]
        )
        if actual_strategy != expected_strategy:
            raise ValueError(
                f"{label} requires metadata split strategy "
                f"{expected_strategy!r}, found {actual_strategy!r}."
            )

    if summary_path.is_file():
        summary = load_json(summary_path)
        if summary.get("split_label") != label:
            raise ValueError(
                f"Synthetic summary split_label must be {label!r}, found "
                f"{summary.get('split_label')!r}."
            )
        conditioning = summary.get("conditioning", {})
        if conditioning.get("source_column") != "flux":
            raise ValueError(
                "Synthetic summary conditioning source_column must be "
                "'flux'."
            )
        if conditioning.get("transform") != "log10":
            raise ValueError(
                "Synthetic summary conditioning transform must be 'log10'."
            )

    resolved = {
        "split_label": label,
        "training_strategy": SPLIT_SPECS[label]["training_strategy"],
        "training_metadata_path": metadata_path,
        "real_pool_csv_path": real_pool_path,
        "synthetic_samples_path": synthetic_path,
        "synthetic_sample_summary_path": summary_path,
        "cache_root": resolved_cache_root,
    }
    if require_exists and load_training_only_manifest(resolved) is not None:
        resolved["artifact_manifest_path"] = real_pool_path.with_name(
            ARTIFACT_MANIFEST_FILENAME
        )
    return resolved


def jsonable_artifact_paths(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in artifacts.items()
    }
