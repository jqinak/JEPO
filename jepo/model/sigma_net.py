from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t_scalar: torch.Tensor) -> torch.Tensor:
        device = t_scalar.device
        half = self.out_dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=t_scalar.dtype) / max(half - 1, 1))
        angle = t_scalar[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        if emb.shape[-1] < self.out_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class StarVLASigmaNet(nn.Module):
    """条件方差网络：输出对角高斯 std/log_std。"""

    def __init__(
        self,
        *,
        ctx_dim: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        min_std: float = 0.02,
        max_std: float = 0.30,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min_std <= 0 or max_std < min_std:
            raise ValueError(f"Invalid std range: min_std={min_std}, max_std={max_std}")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.log_std_min = float(math.log(min_std))
        self.log_std_max = float(math.log(max_std))

        self.time_embed = TimeEmbedding(out_dim=hidden_dim, hidden_dim=hidden_dim)
        self.ctx_proj = nn.Linear(ctx_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        *,
        pooled_ctx: torch.Tensor,
        noisy_actions: torch.Tensor,
        t_scalar: float | torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, horizon, _ = noisy_actions.shape
        dtype = noisy_actions.dtype
        device = noisy_actions.device
        if isinstance(t_scalar, (float, int)):
            t = torch.full((bsz,), float(t_scalar), device=device, dtype=dtype)
        else:
            t = t_scalar.to(device=device, dtype=dtype).reshape(bsz)

        ctx_h = self.ctx_proj(pooled_ctx).unsqueeze(1).expand(bsz, horizon, -1)
        act_h = self.action_proj(noisy_actions)
        if state is None:
            state_h = torch.zeros(bsz, horizon, ctx_h.shape[-1], device=device, dtype=dtype)
        else:
            state_h = self.state_proj(state.reshape(bsz, -1)).unsqueeze(1).expand(bsz, horizon, -1)
        time_h = self.time_embed(t).unsqueeze(1).expand(bsz, horizon, -1)
        x = self.encoder(ctx_h + act_h + state_h + time_h)
        log_std_raw = self.out(x)
        squashed = torch.tanh(log_std_raw)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (squashed + 1.0) * 0.5
        std = torch.exp(log_std)
        return std, log_std
