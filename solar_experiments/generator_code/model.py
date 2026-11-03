from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if max_period <= 1.0:
            raise ValueError("max_period must be greater than 1.")
        self.embedding_dim = int(embedding_dim)
        self.max_period = float(max_period)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 0:
            time = time.reshape(1, 1)
        elif time.ndim == 1:
            time = time[:, None]
        elif time.ndim != 2 or time.shape[1] != 1:
            raise ValueError(
                "time must have shape [batch] or [batch, 1], "
                f"got {tuple(time.shape)}."
            )

        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return time
        exponent = torch.arange(
            half_dim,
            device=time.device,
            dtype=time.dtype,
        )
        exponent /= max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(self.max_period) * exponent)
        phase = 2.0 * math.pi * time * frequencies[None, :]
        embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat([embedding, time], dim=-1)
        return embedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.network = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(self.normalization(values))


class TemporalConditionalOTFlow(nn.Module):
    """Conditional velocity field for an OT flow-matching CNF."""

    def __init__(
        self,
        *,
        x_dim: int,
        condition_dim: int = 1,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        time_embedding_dim: int = 64,
        condition_embedding_dim: int = 128,
        max_time_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("x_dim", x_dim),
            ("condition_dim", condition_dim),
            ("hidden_dim", hidden_dim),
            ("num_blocks", num_blocks),
            ("time_embedding_dim", time_embedding_dim),
            ("condition_embedding_dim", condition_embedding_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        self.x_dim = int(x_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.time_embedding_dim = int(time_embedding_dim)
        self.condition_embedding_dim = int(condition_embedding_dim)
        self.max_time_period = float(max_time_period)

        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=self.time_embedding_dim,
            max_period=self.max_time_period,
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.condition_embedding_dim,
                self.condition_embedding_dim,
            ),
            nn.SiLU(),
        )
        self.input_projection = nn.Sequential(
            nn.Linear(
                self.x_dim
                + self.time_embedding_dim
                + self.condition_embedding_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(self.hidden_dim)
            for _ in range(self.num_blocks)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.x_dim),
        )

    def config(self) -> dict[str, int | float]:
        return {
            "x_dim": self.x_dim,
            "condition_dim": self.condition_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "time_embedding_dim": self.time_embedding_dim,
            "condition_embedding_dim": self.condition_embedding_dim,
            "max_time_period": self.max_time_period,
        }

    def _prepare_condition(
        self,
        condition: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition[:, None]
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                "condition must have shape "
                f"[batch, {self.condition_dim}], got {tuple(condition.shape)}."
            )
        if condition.shape[0] == 1 and batch_size > 1:
            condition = condition.expand(batch_size, -1)
        elif condition.shape[0] != batch_size:
            raise ValueError(
                f"Condition batch {condition.shape[0]} does not match {batch_size}."
            )
        return condition

    @staticmethod
    def _prepare_time(
        time: torch.Tensor | float,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(time):
            time = torch.tensor(time, device=device, dtype=dtype)
        time = time.to(device=device, dtype=dtype)
        if time.ndim == 0:
            time = time.expand(batch_size)[:, None]
        elif time.ndim == 1 and time.shape[0] == batch_size:
            time = time[:, None]
        elif time.ndim == 2 and time.shape == (batch_size, 1):
            pass
        else:
            raise ValueError(
                "time must be scalar or align with the input batch; "
                f"got {tuple(time.shape)}."
            )
        return time

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor | float,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[1] != self.x_dim:
            raise ValueError(
                f"x_t must have shape [batch, {self.x_dim}], "
                f"got {tuple(x_t.shape)}."
            )
        condition = self._prepare_condition(condition, x_t.shape[0]).to(
            device=x_t.device,
            dtype=x_t.dtype,
        )
        time_tensor = self._prepare_time(
            time,
            x_t.shape[0],
            device=x_t.device,
            dtype=x_t.dtype,
        )
        hidden = self.input_projection(
            torch.cat(
                [
                    x_t,
                    self.time_embedding(time_tensor),
                    self.condition_embedding(condition),
                ],
                dim=-1,
            )
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden)

    def integrate_rk4(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        *,
        num_steps: int = 64,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        condition = self._prepare_condition(
            condition,
            x_start.shape[0],
        ).to(device=x_start.device, dtype=x_start.dtype)
        state = x_start
        step_size = 1.0 / num_steps
        for step in range(num_steps):
            time = step * step_size
            k1 = self(state, time, condition)
            k2 = self(state + 0.5 * step_size * k1, time + 0.5 * step_size, condition)
            k3 = self(state + 0.5 * step_size * k2, time + 0.5 * step_size, condition)
            k4 = self(state + step_size * k3, time + step_size, condition)
            state = state + (step_size / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        return state

    @torch.inference_mode()
    def sample(
        self,
        *,
        condition: torch.Tensor,
        num_samples: int,
        num_steps: int = 64,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        parameter = next(self.parameters())
        condition = condition.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        condition = self._prepare_condition(condition, num_samples)
        noise = torch.randn(
            (num_samples, self.x_dim),
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        return self.integrate_rk4(
            noise,
            condition,
            num_steps=num_steps,
        )


__all__ = [
    "ResidualMLPBlock",
    "SinusoidalTimeEmbedding",
    "TemporalConditionalOTFlow",
]
