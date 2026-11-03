from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader

if __package__:
    from .model import TemporalConditionalOTFlow
else:
    from model import TemporalConditionalOTFlow


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 75
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float | None = 5.0
    eps_t: float = 1e-3
    path_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.gradient_clip is not None and self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive or None.")
        if not 0.0 <= self.eps_t < 0.5:
            raise ValueError("eps_t must satisfy 0 <= eps_t < 0.5.")
        if self.path_sigma < 0.0:
            raise ValueError("path_sigma cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "eps_t": self.eps_t,
            "path_sigma": self.path_sigma,
        }


def pairwise_squared_distance(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must both be rank-2 tensors.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match.")
    source_sq = source.square().sum(dim=1, keepdim=True)
    target_sq = target.square().sum(dim=1)[None, :]
    return (
        source_sq + target_sq - 2.0 * source @ target.T
    ).clamp_min(0.0)


def minibatch_ot_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("OT source and target minibatches must have equal shapes.")
    if source.shape[0] == 0:
        raise ValueError("Cannot OT-match an empty minibatch.")
    if source.shape[0] == 1:
        return torch.zeros(1, device=source.device, dtype=torch.long)

    # The assignment is solved on CPU anyway. Compute its high-dimensional
    # distance matrix there in float64 so CUDA TF32/float32 cancellation cannot
    # corrupt the Hungarian cost matrix.
    source_for_cost = source.detach().to(device="cpu", dtype=torch.float64)
    target_for_cost = target.detach().to(device="cpu", dtype=torch.float64)
    cost = pairwise_squared_distance(
        source_for_cost,
        target_for_cost,
    ).numpy()
    if not np.isfinite(cost).all():
        raise FloatingPointError(
            "The minibatch OT cost matrix contains non-finite values. "
            "Check normalized feature values before training."
        )
    row_indices, target_indices = linear_sum_assignment(cost)
    permutation = np.empty_like(target_indices)
    permutation[row_indices] = target_indices
    return torch.from_numpy(
        permutation.astype(np.int64, copy=False)
    ).to(device=source.device)


def build_ot_flow_matching_batch(
    target: torch.Tensor,
    condition: torch.Tensor,
    *,
    eps_t: float = 1e-3,
    path_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim != 2:
        raise ValueError("target must have shape [batch, x_dim].")
    if condition.ndim == 1:
        condition = condition[:, None]
    if condition.ndim != 2 or condition.shape[0] != target.shape[0]:
        raise ValueError("condition must align with target rows.")
    if not 0.0 <= eps_t < 0.5:
        raise ValueError("eps_t must satisfy 0 <= eps_t < 0.5.")
    if path_sigma < 0.0:
        raise ValueError("path_sigma cannot be negative.")

    source = torch.randn_like(target)
    permutation = minibatch_ot_permutation(source, target)
    matched_target = target.index_select(0, permutation)
    matched_condition = condition.index_select(0, permutation)

    time = torch.rand(
        (target.shape[0], 1),
        device=target.device,
        dtype=target.dtype,
    )
    if eps_t:
        time = time * (1.0 - 2.0 * eps_t) + eps_t

    if path_sigma:
        path_noise = torch.randn_like(target)
    else:
        path_noise = torch.zeros_like(target)
    x_t = (
        (1.0 - time) * source
        + time * matched_target
        + path_sigma * path_noise
    )
    target_velocity = matched_target - source
    return x_t, time, matched_condition, target_velocity


def _run_epoch(
    model: TemporalConditionalOTFlow,
    data_loader: DataLoader,
    *,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_squared_error = 0.0
    total_values = 0
    with torch.enable_grad():
        for batch_index, (features, condition) in enumerate(data_loader):
            features = features.to(device=device, non_blocking=True)
            condition = condition.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            x_t, time, matched_condition, target_velocity = (
                build_ot_flow_matching_batch(
                    features,
                    condition,
                    eps_t=config.eps_t,
                    path_sigma=config.path_sigma,
                )
            )
            predicted_velocity = model(x_t, time, matched_condition)
            squared_error = (
                predicted_velocity - target_velocity
            ).square()
            loss = squared_error.mean()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Non-finite velocity loss at "
                    f"batch_index={batch_index}; "
                    f"features_finite={bool(torch.isfinite(features).all().item())}, "
                    f"conditions_finite={bool(torch.isfinite(condition).all().item())}, "
                    f"x_t_finite={bool(torch.isfinite(x_t).all().item())}, "
                    "target_velocity_finite="
                    f"{bool(torch.isfinite(target_velocity).all().item())}, "
                    "predicted_velocity_finite="
                    f"{bool(torch.isfinite(predicted_velocity).all().item())}."
                )

            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip,
                    error_if_nonfinite=True,
                    foreach=False,
                )
            optimizer.step()

            total_squared_error += float(squared_error.sum().item())
            total_values += int(squared_error.numel())

    if total_values == 0:
        raise ValueError("DataLoader produced no values.")
    return total_squared_error / total_values


def train_model(
    model: TemporalConditionalOTFlow,
    train_loader: DataLoader,
    *,
    device: torch.device,
    config: TrainingConfig,
    on_epoch_end: Callable[[int, dict[str, list[float]]], None],
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, foreach=False, fused=False,
    )
    history: dict[str, list[float]] = {"train_mse": []}
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, device=device, config=config, optimizer=optimizer,
        )
        history["train_mse"].append(train_loss)
        print(
            f"epoch={epoch:03d}/{config.epochs:03d} train_mse={train_loss:.8f}",
            flush=True,
        )
        on_epoch_end(epoch, history)
    return history


__all__ = [
    "TrainingConfig",
    "build_ot_flow_matching_batch",
    "minibatch_ot_permutation",
    "pairwise_squared_distance",
    "train_model",
]
