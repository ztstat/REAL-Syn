from __future__ import annotations

import csv
import math
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SHARPS_BASE_FEATURES = [
    "USFLUX",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZD",
    "TOTUSJZ",
    "MEANALP",
    "MEANJZH",
    "TOTUSJH",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "TOTPOT",
    "MEANSHR",
    "SHRGT45",
    "SIZE",
    "SIZE_ACR",
    "NACR",
    "NPIX",
]

SPLIT_STRATEGIES = (
    "random",
    "chronological",
    "harp",
    "harp_by_year",
)
_FRAME_COLUMN_RE = re.compile(r"^frame_(\d+)__(.+)$")


@dataclass(frozen=True)
class TemporalSchema:
    frame_indices: tuple[int, ...]
    base_features: tuple[str, ...]
    feature_columns: tuple[str, ...]
    flux_column: str
    latest_frame_flux_column: str
    datetime_column: str
    harp_column: str

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_indices": list(self.frame_indices),
            "frame_count": self.frame_count,
            "base_features": list(self.base_features),
            "feature_columns": list(self.feature_columns),
            "feature_count": self.feature_count,
            "flux_column": self.flux_column,
            "latest_frame_flux_column": self.latest_frame_flux_column,
            "datetime_column": self.datetime_column,
            "harp_column": self.harp_column,
        }


@dataclass
class RawTemporalData:
    features: np.ndarray
    flux: np.ndarray
    years: np.ndarray
    harp_numbers: np.ndarray
    source_row_indices: np.ndarray
    rows_read: int
    rows_dropped: int


@dataclass(frozen=True)
class SplitResult:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    details: dict[str, Any]


@dataclass
class PreparedTemporalData:
    features: np.ndarray
    conditions: np.ndarray
    normalization: dict[str, Any]


class IndexedTemporalDataset(Dataset):
    """Zero-copy indexed view over the full normalized temporal arrays."""

    def __init__(
        self,
        features: np.ndarray,
        conditions: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        x = np.asarray(features, dtype=np.float32)
        c = np.asarray(conditions, dtype=np.float32)
        idx = np.asarray(indices, dtype=np.int64)
        if x.ndim != 2:
            raise ValueError(f"features must have shape [rows, features], got {x.shape}.")
        if c.ndim == 1:
            c = c[:, None]
        if c.ndim != 2 or c.shape[0] != x.shape[0]:
            raise ValueError("conditions must have shape [rows, condition_dim].")
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional array.")
        if idx.min() < 0 or idx.max() >= x.shape[0]:
            raise IndexError("indices contain a row outside the feature array.")

        self._features = torch.from_numpy(x)
        self._conditions = torch.from_numpy(c)
        self._indices = torch.from_numpy(idx)

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._indices[item]
        return self._features[row], self._conditions[row]

    @property
    def x_dim(self) -> int:
        return int(self._features.shape[1])

    @property
    def condition_dim(self) -> int:
        return int(self._conditions.shape[1])


def inspect_temporal_schema(
    csv_path: str | Path,
    *,
    base_features: Sequence[str] = SHARPS_BASE_FEATURES,
    flux_column: str = "flux",
    datetime_column: str = "history_window_end",
    harp_column: str = "HARPNUM",
) -> TemporalSchema:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Temporal CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"CSV has no header: {path}")
    if len(header) != len(set(header)):
        raise ValueError("CSV header contains duplicate column names.")

    required_metadata = {flux_column, datetime_column, harp_column}
    missing_metadata = required_metadata.difference(header)
    if missing_metadata:
        raise ValueError(
            "CSV is missing required metadata columns: "
            + ", ".join(sorted(missing_metadata))
        )

    frames_by_feature: dict[str, set[int]] = {
        name: set() for name in base_features
    }
    for column in header:
        match = _FRAME_COLUMN_RE.match(column)
        if match is None:
            continue
        frame_index = int(match.group(1))
        field_name = match.group(2)
        if field_name in frames_by_feature:
            frames_by_feature[field_name].add(frame_index)

    all_frame_indices = sorted(
        set().union(*(indices for indices in frames_by_feature.values()))
    )
    if not all_frame_indices:
        raise ValueError("No frame_NNN__<SHARPs feature> columns were found.")
    expected_indices = list(
        range(all_frame_indices[0], all_frame_indices[-1] + 1)
    )
    if all_frame_indices != expected_indices or all_frame_indices[0] != 0:
        raise ValueError(
            "Temporal frame indices must be contiguous and start at frame_000."
        )

    for feature_name, frame_indices in frames_by_feature.items():
        if sorted(frame_indices) != all_frame_indices:
            missing = sorted(set(all_frame_indices).difference(frame_indices))
            raise ValueError(
                f"Feature {feature_name!r} is missing temporal frames: {missing}"
            )

    feature_columns = tuple(
        f"frame_{frame_index:03d}__{feature_name}"
        for frame_index in all_frame_indices
        for feature_name in base_features
    )
    absent = [column for column in feature_columns if column not in header]
    if absent:
        raise ValueError(
            "CSV frame padding does not match frame_NNN convention; "
            f"first missing column: {absent[0]}"
        )
    latest_frame_flux_column = (
        f"frame_{all_frame_indices[-1]:03d}__{flux_column}"
    )
    if latest_frame_flux_column not in header:
        raise ValueError(
            "CSV is missing the latest-frame flux column "
            f"{latest_frame_flux_column!r}."
        )

    return TemporalSchema(
        frame_indices=tuple(all_frame_indices),
        base_features=tuple(base_features),
        feature_columns=feature_columns,
        flux_column=flux_column,
        latest_frame_flux_column=latest_frame_flux_column,
        datetime_column=datetime_column,
        harp_column=harp_column,
    )


def load_temporal_csv(
    csv_path: str | Path,
    schema: TemporalSchema,
    *,
    max_rows: int | None = None,
) -> RawTemporalData:
    if max_rows is not None and max_rows < 2:
        raise ValueError("max_rows must be at least 2 when supplied.")

    selected_columns = [
        schema.harp_column,
        schema.datetime_column,
        *schema.feature_columns,
        schema.latest_frame_flux_column,
        schema.flux_column,
    ]
    float_columns = {
        column: np.float32 for column in schema.feature_columns
    }
    float_columns[schema.latest_frame_flux_column] = np.float64
    float_columns[schema.flux_column] = np.float64
    frame = pd.read_csv(
        Path(csv_path),
        usecols=selected_columns,
        dtype=float_columns,
        nrows=max_rows,
        low_memory=False,
    )
    rows_read = len(frame)
    if rows_read < 2:
        raise ValueError("Temporal CSV must contain at least two data rows.")

    features = frame.loc[:, schema.feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    flux = frame[schema.flux_column].to_numpy(dtype=np.float64, copy=True)
    latest_frame_flux = frame[schema.latest_frame_flux_column].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    datetimes = pd.to_datetime(
        frame[schema.datetime_column],
        errors="coerce",
        utc=True,
    )
    harp_numeric = pd.to_numeric(frame[schema.harp_column], errors="coerce")
    source_row_indices = np.arange(rows_read, dtype=np.int64)

    valid = np.isfinite(features).all(axis=1)
    valid &= np.isfinite(flux) & (flux > 0.0)
    valid &= np.isfinite(latest_frame_flux)
    valid &= datetimes.notna().to_numpy()
    valid &= harp_numeric.notna().to_numpy()
    rows_dropped = int((~valid).sum())
    if not valid.any():
        raise ValueError(
            "No complete rows remain after requiring finite SHARPs values, "
            "positive finite flux, datetime, and HARPNUM."
        )
    if not np.allclose(
        flux[valid],
        latest_frame_flux[valid],
        rtol=1e-12,
        atol=0.0,
    ):
        mismatch_count = int(
            (
                ~np.isclose(
                    flux[valid],
                    latest_frame_flux[valid],
                    rtol=1e-12,
                    atol=0.0,
                )
            ).sum()
        )
        raise ValueError(
            f"{mismatch_count} valid rows have endpoint flux different from "
            f"{schema.latest_frame_flux_column}."
        )

    if rows_dropped:
        features = np.ascontiguousarray(features[valid], dtype=np.float32)
        flux = flux[valid]
        datetimes = datetimes[valid]
        harp_numeric = harp_numeric[valid]
        source_row_indices = source_row_indices[valid]

    years = datetimes.dt.year.to_numpy(dtype=np.int64, copy=True)
    harp_values = harp_numeric.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.equal(harp_values, np.floor(harp_values))):
        raise ValueError("HARPNUM must contain integer-valued identifiers.")
    harp_numbers = harp_values.astype(np.int64)

    return RawTemporalData(
        features=features,
        flux=flux,
        years=years,
        harp_numbers=harp_numbers,
        source_row_indices=source_row_indices,
        rows_read=rows_read,
        rows_dropped=rows_dropped,
    )


def write_raw_split_csvs(
    csv_path: str | Path,
    *,
    source_row_count: int,
    train_source_indices: np.ndarray,
    test_source_indices: np.ndarray,
    train_output_path: str | Path,
    test_output_path: str | Path,
    skip_existing: bool = False,
) -> tuple[Path, Path]:
    """Route original one-line CSV records into atomic train/test files."""
    source_path = Path(csv_path).resolve()
    train_path = Path(train_output_path).resolve()
    test_path = Path(test_output_path).resolve()
    if source_row_count < 2:
        raise ValueError("source_row_count must be at least 2.")
    if train_path == test_path:
        raise ValueError("Train and test CSV paths must be different.")
    if source_path in {train_path, test_path}:
        raise ValueError("Split CSV output cannot overwrite the source CSV.")

    train_indices = np.asarray(train_source_indices, dtype=np.int64)
    test_indices = np.asarray(test_source_indices, dtype=np.int64)
    for name, indices in (
        ("train_source_indices", train_indices),
        ("test_source_indices", test_indices),
    ):
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if indices.min() < 0 or indices.max() >= source_row_count:
            raise IndexError(f"{name} contains a row outside the source CSV.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate row indices.")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Train and test source indices overlap.")

    assignments = np.full(source_row_count, -1, dtype=np.int8)
    assignments[train_indices] = 0
    assignments[test_indices] = 1
    write_train = not (skip_existing and train_path.exists())
    write_test = not (skip_existing and test_path.exists())
    if not write_train and not write_test:
        return train_path, test_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_temporary = train_path.with_name(
        f".{train_path.name}.{os.getpid()}.tmp"
    )
    test_temporary = test_path.with_name(
        f".{test_path.name}.{os.getpid()}.tmp"
    )

    train_written = 0
    test_written = 0
    try:
        with ExitStack() as stack:
            source_handle = stack.enter_context(source_path.open("rb"))
            train_handle = (
                stack.enter_context(train_temporary.open("wb")) if write_train else None
            )
            test_handle = (
                stack.enter_context(test_temporary.open("wb")) if write_test else None
            )
            header = source_handle.readline()
            if not header:
                raise ValueError(f"CSV has no header: {source_path.name}")
            if train_handle is not None:
                train_handle.write(header)
            if test_handle is not None:
                test_handle.write(header)
            for source_row_index, raw_line in enumerate(source_handle):
                if source_row_index >= source_row_count:
                    if skip_existing:
                        raise ValueError("Source CSV must contain one physical line per record.")
                    break
                assignment = assignments[source_row_index]
                if assignment == 0:
                    if train_handle is not None:
                        train_handle.write(raw_line)
                    train_written += 1
                elif assignment == 1:
                    if test_handle is not None:
                        test_handle.write(raw_line)
                    test_written += 1

        if train_written != train_indices.size:
            raise ValueError(
                f"Expected {train_indices.size} train rows but wrote {train_written}."
            )
        if test_written != test_indices.size:
            raise ValueError(
                f"Expected {test_indices.size} test rows but wrote {test_written}."
            )
        for needed, temporary, destination in (
            (write_train, train_temporary, train_path),
            (write_test, test_temporary, test_path),
        ):
            if not needed:
                continue
            if skip_existing:
                # Publish without overwriting a split created by another process.
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
                temporary.unlink()
            else:
                os.replace(temporary, destination)
    except BaseException:
        train_temporary.unlink(missing_ok=True)
        test_temporary.unlink(missing_ok=True)
        raise
    return train_path, test_path


def ensure_chronological_split_csvs(
    csv_path: str | Path,
    *,
    train_output_path: str | Path,
    test_output_path: str | Path,
    train_end_year: int = 2021,
    datetime_column: str = "history_window_end",
) -> tuple[Path, Path]:
    """Create missing splits using only endpoint dates and unchanged raw records."""
    train_path = Path(train_output_path).resolve()
    test_path = Path(test_output_path).resolve()
    missing = [path for path in (train_path, test_path) if not path.exists()]
    if not missing:
        return train_path, test_path
    source_path = Path(csv_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing {source_path.name}; provide the full temporal CSV "
            "to prepare the chronological split files."
        )
    timestamps = pd.read_csv(source_path, usecols=[datetime_column])
    dates = pd.to_datetime(timestamps[datetime_column], errors="coerce", utc=True)
    if dates.empty or dates.isna().any():
        raise ValueError("Full temporal CSV must contain valid endpoint timestamps.")
    years = dates.dt.year.to_numpy(dtype=np.int64)
    result = write_raw_split_csvs(
        source_path,
        source_row_count=years.size,
        train_source_indices=np.flatnonzero(years <= train_end_year),
        test_source_indices=np.flatnonzero(years > train_end_year),
        train_output_path=train_path,
        test_output_path=test_path,
        skip_existing=True,
    )
    for path in missing:
        print(f"prepared_split={path.name}", flush=True)
    return result


def _validated_train_count(total: int, train_fraction: float) -> int:
    if total < 2:
        raise ValueError("At least two rows or groups are required for splitting.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    return max(1, min(total - 1, int(math.floor(total * train_fraction))))


def _harp_modal_years(
    harp_numbers: np.ndarray,
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    order = np.argsort(harp_numbers, kind="stable")
    sorted_harps = harp_numbers[order]
    sorted_years = years[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_harps[1:] != sorted_harps[:-1], True]
    )
    unique_harps = sorted_harps[boundaries[:-1]]
    modal_years = np.empty(unique_harps.size, dtype=np.int64)
    cross_year_harps: list[int] = []

    for group_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        group_years, counts = np.unique(
            sorted_years[start:end],
            return_counts=True,
        )
        modal_years[group_index] = group_years[np.argmax(counts)]
        if group_years.size > 1:
            cross_year_harps.append(int(unique_harps[group_index]))
    return unique_harps, modal_years, cross_year_harps


def _make_harp_by_year_split(
    *,
    harp_numbers: np.ndarray,
    years: np.ndarray,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    unique_harps, modal_years, cross_year_harps = _harp_modal_years(
        harp_numbers,
        years,
    )
    endpoint_years = np.unique(years)
    stratum_years = np.unique(modal_years)
    if not np.array_equal(endpoint_years, stratum_years):
        missing_strata = np.setdiff1d(endpoint_years, stratum_years)
        raise ValueError(
            "Cannot stratify every endpoint year because no HARP has modal "
            f"membership in years: {missing_strata.tolist()}."
        )

    rng = np.random.default_rng(seed)
    train_harp_parts: list[np.ndarray] = []
    test_harp_parts: list[np.ndarray] = []
    stratum_counts: dict[int, dict[str, int]] = {}
    for year in stratum_years:
        year_harps = unique_harps[modal_years == year]
        if year_harps.size < 2:
            raise ValueError(
                "harp_by_year needs at least two modal-year HARP groups in "
                f"each year; year {int(year)} has {year_harps.size}."
            )
        shuffled_harps = rng.permutation(year_harps)
        train_count = _validated_train_count(
            int(year_harps.size),
            train_fraction,
        )
        train_harp_parts.append(shuffled_harps[:train_count])
        test_harp_parts.append(shuffled_harps[train_count:])
        stratum_counts[int(year)] = {
            "total_harps": int(year_harps.size),
            "train_harps": int(train_count),
            "test_harps": int(year_harps.size - train_count),
        }

    train_harps = np.concatenate(train_harp_parts)
    test_harps = np.concatenate(test_harp_parts)
    train_mask = np.isin(harp_numbers, train_harps)
    test_mask = np.isin(harp_numbers, test_harps)
    if np.any(train_mask & test_mask):
        raise RuntimeError("Internal error: a HARP was assigned to both splits.")
    if not np.all(train_mask | test_mask):
        raise RuntimeError("Internal error: at least one HARP was not assigned.")

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    train_years = np.unique(years[train_indices])
    test_years = np.unique(years[test_indices])
    missing_train_years = np.setdiff1d(endpoint_years, train_years)
    missing_test_years = np.setdiff1d(endpoint_years, test_years)
    if missing_train_years.size or missing_test_years.size:
        raise ValueError(
            "harp_by_year could not retain every endpoint year in both splits; "
            f"missing_train_years={missing_train_years.tolist()}, "
            f"missing_test_years={missing_test_years.tolist()}."
        )

    year_counts: dict[str, dict[str, int]] = {}
    for year in endpoint_years:
        year_mask = years == year
        year_counts[str(int(year))] = {
            **stratum_counts[int(year)],
            "train_rows": int(np.count_nonzero(year_mask & train_mask)),
            "test_rows": int(np.count_nonzero(year_mask & test_mask)),
            "train_row_harps": int(
                np.unique(harp_numbers[year_mask & train_mask]).size
            ),
            "test_row_harps": int(
                np.unique(harp_numbers[year_mask & test_mask]).size
            ),
        }

    details: dict[str, Any] = {
        "strategy": "harp_by_year",
        "train_fraction_requested": float(train_fraction),
        "seed": int(seed),
        "rule": "seeded_whole_harp_groups_stratified_by_modal_endpoint_year",
        "harp_year_rule": (
            "modal_endpoint_year_with_earliest_year_breaking_count_ties"
        ),
        "all_endpoint_years": endpoint_years.tolist(),
        "train_years": train_years.tolist(),
        "validation_years": test_years.tolist(),
        "train_unique_harps": int(train_harps.size),
        "validation_unique_harps": int(test_harps.size),
        "cross_year_harp_count": len(cross_year_harps),
        "cross_year_harps": cross_year_harps,
        "per_year": year_counts,
    }
    return train_indices, test_indices, details


def make_split(
    *,
    strategy: str,
    row_count: int,
    train_fraction: float = 0.8,
    seed: int = 42,
    years: np.ndarray | None = None,
    harp_numbers: np.ndarray | None = None,
) -> SplitResult:
    strategy_normalized = str(strategy).strip().lower()
    if strategy_normalized not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split strategy {strategy!r}; choose from {SPLIT_STRATEGIES}."
        )
    if row_count < 2:
        raise ValueError("At least two rows are required for splitting.")

    if strategy_normalized == "random":
        train_count = _validated_train_count(row_count, train_fraction)
        shuffled = np.random.default_rng(seed).permutation(row_count)
        train_indices = shuffled[:train_count]
        validation_indices = shuffled[train_count:]
        details: dict[str, Any] = {
            "strategy": strategy_normalized,
            "train_fraction_requested": float(train_fraction),
            "seed": int(seed),
            "rule": "seeded_row_permutation",
        }
    elif strategy_normalized == "chronological":
        if years is None:
            raise ValueError("years are required for a chronological split.")
        years_array = np.asarray(years, dtype=np.int64)
        if years_array.shape != (row_count,):
            raise ValueError("years must align one-to-one with the data rows.")
        unique_years = np.unique(years_array)
        train_year_count = _validated_train_count(
            len(unique_years),
            train_fraction,
        )
        train_years = unique_years[:train_year_count]
        validation_years = unique_years[train_year_count:]
        train_mask = np.isin(years_array, train_years)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(~train_mask)
        details = {
            "strategy": strategy_normalized,
            "train_fraction_requested": float(train_fraction),
            "rule": "first_fraction_of_sorted_distinct_endpoint_years",
            "cutoff_year_inclusive": int(train_years[-1]),
            "train_years": train_years.tolist(),
            "validation_years": validation_years.tolist(),
        }
    elif strategy_normalized == "harp":
        if harp_numbers is None:
            raise ValueError("harp_numbers are required for a HARP split.")
        harp_array = np.asarray(harp_numbers, dtype=np.int64)
        if harp_array.shape != (row_count,):
            raise ValueError("harp_numbers must align one-to-one with the data rows.")
        unique_harps = np.unique(harp_array)
        train_harp_count = _validated_train_count(
            len(unique_harps),
            train_fraction,
        )
        train_harps = unique_harps[:train_harp_count]
        validation_harps = unique_harps[train_harp_count:]
        train_mask = np.isin(harp_array, train_harps)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(~train_mask)
        details = {
            "strategy": strategy_normalized,
            "train_fraction_requested": float(train_fraction),
            "rule": "first_fraction_of_sorted_distinct_harp_numbers",
            "cutoff_harp_inclusive": int(train_harps[-1]),
            "train_unique_harps": int(train_harps.size),
            "validation_unique_harps": int(validation_harps.size),
            "train_harp_range": [
                int(train_harps[0]),
                int(train_harps[-1]),
            ],
            "validation_harp_range": [
                int(validation_harps[0]),
                int(validation_harps[-1]),
            ],
        }
    else:
        if years is None or harp_numbers is None:
            raise ValueError(
                "years and harp_numbers are required for a harp_by_year split."
            )
        years_array = np.asarray(years, dtype=np.int64)
        harp_array = np.asarray(harp_numbers, dtype=np.int64)
        if years_array.shape != (row_count,):
            raise ValueError("years must align one-to-one with the data rows.")
        if harp_array.shape != (row_count,):
            raise ValueError("harp_numbers must align one-to-one with the data rows.")
        train_indices, validation_indices, details = (
            _make_harp_by_year_split(
                harp_numbers=harp_array,
                years=years_array,
                train_fraction=train_fraction,
                seed=seed,
            )
        )

    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError(
            f"{strategy_normalized} split produced an empty train or validation set."
        )
    details["train_rows"] = int(train_indices.size)
    details["validation_rows"] = int(validation_indices.size)
    details["train_row_fraction_actual"] = float(train_indices.size / row_count)
    return SplitResult(
        train_indices=train_indices.astype(np.int64, copy=False),
        validation_indices=validation_indices.astype(np.int64, copy=False),
        details=details,
    )


def _signed_log1p_in_place(values: np.ndarray) -> None:
    negative = values < 0.0
    np.abs(values, out=values)
    np.log1p(values, out=values)
    values[negative] *= -1.0


def prepare_temporal_data(
    raw: RawTemporalData,
    schema: TemporalSchema,
    train_indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> PreparedTemporalData:
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if raw.features.shape[1] != schema.feature_count:
        raise ValueError("Feature matrix width does not match the temporal schema.")

    train_idx = np.asarray(train_indices, dtype=np.int64)
    if train_idx.ndim != 1 or train_idx.size == 0:
        raise ValueError("train_indices must be a non-empty one-dimensional array.")

    features = np.ascontiguousarray(raw.features, dtype=np.float32)
    temporal = features.reshape(
        features.shape[0],
        schema.frame_count,
        len(schema.base_features),
    )
    feature_stats: dict[str, dict[str, float]] = {}
    for feature_index, feature_name in enumerate(schema.base_features):
        feature_values = temporal[:, :, feature_index]
        _signed_log1p_in_place(feature_values)
        train_values = feature_values[train_idx]
        mean = float(np.mean(train_values, dtype=np.float64))
        std = float(np.std(train_values, dtype=np.float64))
        scale = max(std, eps)
        feature_values -= mean
        feature_values /= scale
        feature_stats[feature_name] = {
            "mean": mean,
            "std": std,
            "scale": scale,
        }

    log10_flux = np.log10(raw.flux)
    condition_mean = float(np.mean(log10_flux[train_idx], dtype=np.float64))
    condition_std = float(np.std(log10_flux[train_idx], dtype=np.float64))
    condition_scale = max(condition_std, eps)
    conditions = (
        (log10_flux - condition_mean) / condition_scale
    ).astype(np.float32)[:, None]

    normalization: dict[str, Any] = {
        "fit_scope": "training_split_only",
        "feature_transform": "sign(x) * log1p(abs(x))",
        "feature_statistics_scope": "pooled_across_rows_and_frames_per_base_feature",
        "feature_statistics": feature_stats,
        "condition": {
            "source_column": schema.flux_column,
            "transform": "log10",
            "mean": condition_mean,
            "std": condition_std,
            "scale": condition_scale,
        },
        "eps": float(eps),
    }
    return PreparedTemporalData(
        features=features,
        conditions=conditions,
        normalization=normalization,
    )


def invert_prepared_features(
    normalized_features: np.ndarray,
    schema: TemporalSchema,
    normalization: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(normalized_features, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != schema.feature_count:
        raise ValueError(
            f"Expected normalized features with shape [rows, {schema.feature_count}]."
        )
    temporal = values.reshape(
        values.shape[0],
        schema.frame_count,
        len(schema.base_features),
    )
    statistics = normalization["feature_statistics"]
    for feature_index, feature_name in enumerate(schema.base_features):
        stats = statistics[feature_name]
        transformed = (
            temporal[:, :, feature_index] * float(stats["scale"])
            + float(stats["mean"])
        )
        temporal[:, :, feature_index] = (
            np.sign(transformed) * np.expm1(np.abs(transformed))
        )
    return values


__all__ = [
    "IndexedTemporalDataset",
    "PreparedTemporalData",
    "RawTemporalData",
    "SPLIT_STRATEGIES",
    "SplitResult",
    "TemporalSchema",
    "inspect_temporal_schema",
    "invert_prepared_features",
    "load_temporal_csv",
    "make_split",
    "prepare_temporal_data",
    "write_raw_split_csvs",
]
