#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .data import (
        IndexedTemporalDataset, ensure_chronological_split_csvs, inspect_temporal_schema,
        load_temporal_csv, prepare_temporal_data,
    )
    from .model import TemporalConditionalOTFlow
    from .training import TrainingConfig, train_model
    from .flux_reference import (
        TRAINING_PROTOCOL, TRAINING_SCOPE,
        assert_training_dates, fit_condition_reference,
    )
else:
    from data import (
        IndexedTemporalDataset, ensure_chronological_split_csvs, inspect_temporal_schema,
        load_temporal_csv, prepare_temporal_data,
    )
    from model import TemporalConditionalOTFlow
    from training import TrainingConfig, train_model
    from flux_reference import (
        TRAINING_PROTOCOL, TRAINING_SCOPE,
        assert_training_dates, fit_condition_reference,
    )

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train chronological temporal OT-CFM on training data only, with "
            "one final-epoch checkpoint and no holdout evaluation."
        )
    )
    parser.add_argument(
        "--data-csv",
        default=str(
            SCRIPT_DIR / "data" / "train_chronological.csv"
        ),
        help=(
            "Training-only 60-frame CSV containing endpoint years through 2021. "
            "Missing default train/test CSVs are prepared from the downloaded full "
            "temporal CSV in data/. An explicit training CSV can be used directly."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "artifacts" / "chronological_training_only_seed3097"),
        help="New output directory; existing training artifacts are never overwritten.",
    )
    parser.add_argument("--split", choices=("chronological",), default="chronological")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--flux-quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=3097)
    parser.add_argument(
        "--epochs", type=int, default=75,
        help="Number of training epochs; save only the final epoch (default: 75).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
        help="Gradient norm cap; use 0 to disable.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--condition-embedding-dim", type=int, default=128)
    parser.add_argument("--max-time-period", type=float, default=10_000.0)
    parser.add_argument(
        "--eps-t",
        type=float,
        default=1e-3,
        help="Draw flow times from [eps_t, 1-eps_t].",
    )
    parser.add_argument(
        "--normalization-eps",
        type=float,
        default=1e-6,
        help="Minimum scale used while standardizing training data.",
    )
    parser.add_argument(
        "--datetime-column",
        default="history_window_end",
        help="Endpoint datetime column used for chronological splitting.",
    )
    parser.add_argument("--harp-column", default="HARPNUM")
    parser.add_argument(
        "--flux-column",
        default="flux",
        help="Positive endpoint flux column transformed using log10.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device: cpu (default), auto, cuda, cuda:N, or mps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; 0 is the memory-safe default.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional input-row cap for debugging.",
    )
    parser.add_argument(
        "--rk4-steps",
        type=int,
        default=64,
        help="Default integration step count recorded for sampling.",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but CUDA is unavailable."
        )
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_numerics(device: torch.device) -> None:
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def _write_history(
    history: dict[str, list[float]],
    path: Path,
) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse"])
        for epoch, train_loss in enumerate(history["train_mse"], start=1):
            writer.writerow([epoch, train_loss])
    os.replace(temporary_path, path)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if args.rk4_steps <= 0:
        raise ValueError("rk4-steps must be positive.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    checkpoint_epochs = [args.epochs]

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_numerics(device)
    data_path = Path(args.data_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifact_names = (
        "training_metadata.json", "history.csv", "condition_reference_summary.json",
        *(f"model_epoch_{epoch:03d}.pt" for epoch in checkpoint_epochs),
    )
    if any((output_dir / name).exists() for name in artifact_names):
        raise FileExistsError(
            f"Training artifacts already exist in {output_dir.name}. Choose a new --output-dir."
        )
    data_dir = SCRIPT_DIR / "data"
    if data_path == (data_dir / "train_chronological.csv").resolve():
        ensure_chronological_split_csvs(
            data_dir / "SHARPs_flux_temporal_u12m_w60_s10_wide.csv",
            train_output_path=data_path,
            test_output_path=data_dir / "test_chronological.csv",
            train_end_year=args.train_end_year,
            datetime_column=args.datetime_column,
        )
    # Only the prepared training split is parsed for features and flux.
    assert_training_dates(
        data_path, datetime_column=args.datetime_column,
        train_end_year=args.train_end_year, max_rows=args.max_rows,
    )
    print(f"reading={data_path.name}", flush=True)
    schema = inspect_temporal_schema(
        data_path,
        flux_column=args.flux_column,
        datetime_column=args.datetime_column,
        harp_column=args.harp_column,
    )
    if schema.frame_count != 60:
        raise ValueError(
            "This generator requires 60 frames (1,200 SHARPs coordinates); "
            f"the input has {schema.frame_count} frames."
        )
    raw = load_temporal_csv(
        data_path,
        schema,
        max_rows=args.max_rows,
    )
    print(
        f"rows_read={raw.rows_read} rows_kept={raw.features.shape[0]} "
        f"rows_dropped={raw.rows_dropped} frames={schema.frame_count} "
        f"x_dim={schema.feature_count}",
        flush=True,
    )

    train_indices = np.arange(raw.features.shape[0], dtype=np.int64)
    condition_reference = fit_condition_reference(
        raw.flux,
        quantile=args.flux_quantile,
        source={
            "scope": TRAINING_SCOPE,
            "files": [data_path.name],
            "flux_column": args.flux_column,
            "endpoint_year_min": int(raw.years.min()),
            "endpoint_year_max": int(raw.years.max()),
            "cutoff_year_inclusive": args.train_end_year,
        },
    )
    prepared = prepare_temporal_data(
        raw, schema, train_indices, eps=args.normalization_eps,
    )
    print(f"split=chronological train={train_indices.size}", flush=True)
    train_dataset = IndexedTemporalDataset(
        prepared.features, prepared.conditions, train_indices,
    )
    common_loader_arguments: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        common_loader_arguments["persistent_workers"] = True
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common_loader_arguments,
    )
    model = TemporalConditionalOTFlow(
        x_dim=train_dataset.x_dim,
        condition_dim=train_dataset.condition_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        time_embedding_dim=args.time_embedding_dim,
        condition_embedding_dim=args.condition_embedding_dim,
        max_time_period=args.max_time_period,
    ).to(device)
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=(
            args.gradient_clip if args.gradient_clip > 0.0 else None
        ),
        eps_t=args.eps_t,
        path_sigma=0.0,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model_parameters={parameter_count}", flush=True)
    metadata: dict[str, Any] = {
        "format_version": 2,
        "model_type": "temporal_conditional_ot_flow_matching",
        "flow_matching": {
            "objective": "optimal_transport",
            "coupling": "minibatch_linear_sum_assignment",
            "path": "linear_displacement",
            "path_sigma": 0.0,
            "loss": "unweighted_mean_squared_velocity_error",
            "tail_weight_lambda": 0.0,
            "tail_batch_fraction": None,
        },
        "data_csv": data_path.name,
        "schema": schema.to_dict(),
        "rows": {
            "read": raw.rows_read,
            "dropped_incomplete": raw.rows_dropped,
            "usable": int(raw.features.shape[0]),
        },
        "training_protocol": TRAINING_PROTOCOL,
        "train_end_year": args.train_end_year,
        "split": {
            "strategy": "chronological",
            "rule": "training_only_input_fixed_endpoint_year_cutoff",
            "cutoff_year_inclusive": args.train_end_year,
            "train_years": np.unique(raw.years).tolist(),
            "train_rows": int(train_indices.size),
        },
        "condition_reference": condition_reference,
        "checkpoint_epochs": checkpoint_epochs,
        "checkpoint_selection": "final_epoch",
        "completed_epochs": 0,
        "normalization": prepared.normalization,
        "model_config": model.config(),
        "training_config": training_config.to_dict(),
        "batch_size": args.batch_size,
        "sampling_defaults": {"rk4_steps": args.rk4_steps},
        "history_csv": "history.csv",
        "seed": args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "training_metadata.json"
    history_path = output_dir / "history.csv"
    _atomic_json_dump(condition_reference, output_dir / "condition_reference_summary.json")
    _atomic_json_dump(metadata, metadata_path)

    def save_epoch(epoch: int, history: dict[str, list[float]]) -> None:
        if epoch == args.epochs:
            checkpoint = {
                "format_version": 1,
                "epoch": epoch,
                "training_protocol": TRAINING_PROTOCOL,
                "train_end_year": args.train_end_year,
                "condition_reference": condition_reference,
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "model_config": model.config(),
                "schema": schema.to_dict(),
                "normalization": prepared.normalization,
                "sampling_defaults": metadata["sampling_defaults"],
                "model_type": metadata["model_type"],
            }
            checkpoint_path = output_dir / f"model_epoch_{epoch:03d}.pt"
            _atomic_torch_save(checkpoint, checkpoint_path)
            print(f"checkpoint={checkpoint_path.name}", flush=True)
        metadata["completed_epochs"] = epoch
        _atomic_json_dump(metadata, metadata_path)
        _write_history(history, history_path)

    train_model(
        model, train_loader, device=device, config=training_config,
        on_epoch_end=save_epoch,
    )
    print(f"metadata={metadata_path.name}", flush=True)
    print(f"history={history_path.name}", flush=True)
    return metadata


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
