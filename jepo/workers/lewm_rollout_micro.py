"""LEWM autoregressive micro-step embedding rollout (initialized from first observation only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _pad_time_left(seq: torch.Tensor, target_len: int, pad_pattern: str = "repeat_first") -> torch.Tensor:
    """seq: [B, T, D] -> last target_len slices, left-padded if needed."""
    b, t, d = seq.shape
    if t >= target_len:
        return seq[:, -target_len:, :]
    need = target_len - t
    if pad_pattern == "repeat_first":
        left = seq[:, 0:1, :].expand(b, need, d)
        return torch.cat([left, seq], dim=1)
    raise ValueError(f"Unknown pad_pattern: {pad_pattern}")


def coerce_pixels_btc_hw(
    pixels: torch.Tensor,
    *,
    batch_b: int,
    time_t: int,
) -> torch.Tensor:
    """Return ``[B, T, C, H, W]`` for :meth:`JEPA.encode`.

    Defense-in-depth: some callers mistakenly materialize pixels as ``[B*T,C,H,W]`` or ``[1,B*T,...]``.
    """
    if pixels.ndim == 5:
        b, t, c, h, w = pixels.shape
        if b == batch_b and t == time_t:
            return pixels
        if b == 1 and t == batch_b * time_t:
            return pixels.view(batch_b, time_t, c, h, w)
        if b == batch_b * time_t and t == 1:
            return pixels.view(batch_b, time_t, c, h, w)
    elif pixels.ndim == 4:
        n, c, h, w = pixels.shape
        if n == batch_b * time_t:
            return pixels.view(batch_b, time_t, c, h, w)
    raise ValueError(
        f"Cannot reshape pixels to [B={batch_b},T={time_t},C,H,W]; got {tuple(pixels.shape)} "
        "(check expert_views_per_traj nesting: outer batch, inner time steps)."
    )


def encode_pixels_bt(model: Any, pixels_btc_hw: torch.Tensor) -> torch.Tensor:
    """Encode ``[B, T, C, H, W]`` pixels to ``[B, T, Din]`` via ``model.encode``.

    ``lewm.jepa.JEPA.encode`` expects a 5D tensor and flattens ``(B,T)`` internally; do **not**
    pre-flatten here or ViT receives a wrong rank (e.g. 3D) and crashes.
    """
    if pixels_btc_hw.ndim != 5:
        raise ValueError(f"encode_pixels_bt expects [B,T,C,H,W], got {pixels_btc_hw.shape}")
    out = model.encode({"pixels": pixels_btc_hw.float()})
    return out["emb"]


def predict_micro_emb_sequence_open_loop(
    model: Any,
    first_frame_chw: torch.Tensor,
    micro_actions_bnad: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Roll LEWM predictor for n micro-steps.

    Uses only encoder(first_frame); subsequent latents come from predictor. Action history padded on the left.

    Args:
        model: JEPA-like with encode({"pixels"}), predict(emb_hist, action_emb_hist), action_encoder.
        first_frame_chw: [B, 3, H, W].
        micro_actions_bnad: [B, n, A] matched to Embedder expected dim.

    Returns:
        pred_embs [B, n, D_emb] predictor outputs at steps 0..n-1 (alignment with milestone index k).
    """
    device = first_frame_chw.device
    b, n_micro, act_dim = micro_actions_bnad.shape
    if n_micro < 1:
        raise ValueError("n_micro must be >= 1")

    inp = first_frame_chw.unsqueeze(1).contiguous()
    emb0_seq = encode_pixels_bt(model, inp)
    emb0 = emb0_seq[:, 0, :]
    emb_seq = emb0.unsqueeze(1)
    preds: list[torch.Tensor] = []

    for tau in range(n_micro):
        act_slice = micro_actions_bnad[:, : tau + 1, :].contiguous()
        act_hist = _pad_time_left(act_slice, history_size)
        act_emb_hist = model.action_encoder(act_hist)
        emb_hist = _pad_time_left(emb_seq, history_size)
        pred_step = model.predict(emb_hist, act_emb_hist)[:, -1:, :]
        preds.append(pred_step.squeeze(1))
        emb_seq = torch.cat([emb_seq, pred_step], dim=1)

    return torch.stack(preds, dim=1)


def _torch_or_numpy_to_float_chw_rgb(im_3hwc_or_chw: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize a single-frame array/tensor to CHW RGB float."""
    if isinstance(im_3hwc_or_chw, torch.Tensor):
        t = im_3hwc_or_chw.detach().float().cpu()
        if t.ndim != 3:
            raise ValueError(f"Tensor image must be 3D HWC or CHW, got shape {tuple(t.shape)}")
        a, bdim, cdim = t.shape
        if a in (1, 3) and a <= min(bdim, cdim):
            chw = t
            if a == 1:
                chw = chw.expand(3, bdim, cdim).clone()
        elif cdim in (1, 3):
            chw = t.permute(2, 0, 1).contiguous()
            if chw.shape[0] == 1:
                chw = chw.expand(3, chw.shape[1], chw.shape[2]).clone()
        else:
            raise ValueError(
                f"Cannot infer CHW vs HWC for tensor shape {tuple(t.shape)}; "
                "expected leading or trailing dim in {{1,3}} for RGB/gray."
            )
    else:
        arr = im_3hwc_or_chw
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image, shape={arr.shape}")
        chw = torch.from_numpy(arr).permute(2, 0, 1).float()
    return chw


def _prep_frame_chw01(im: Any, image_size: int) -> torch.Tensor:
    """One frame → ``[3, image_size, image_size]`` float in ``[0, 1]``."""
    if isinstance(im, Image.Image):
        arr = np.asarray(im.convert("RGB"))
        chw = torch.from_numpy(arr).permute(2, 0, 1).float()
    else:
        chw = _torch_or_numpy_to_float_chw_rgb(im)
    if float(chw.max()) > 1.0:
        chw = chw / 255.0
    return F.interpolate(chw.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(
        0
    )


def pil_batch_to_pixels_btc(
    batches_of_lists: list[list[Any]],
    image_size: int,
    device: torch.device,
    dtype=torch.float32,
    *,
    expected_batch: int | None = None,
    expected_time: int | None = None,
) -> torch.Tensor:
    """``batches_of_lists[b][t]`` frame → tensor ``[B, T, 3, H, W]`` in ``[0, 1]``.

    **Contract:** outer = batch trajectory index (same order as ``predicted_micro_actions``),
    inner = micro-step time, length ``num_micro_steps`` plus one when using next-frame GT observations.
    """
    if not batches_of_lists:
        raise ValueError("pil_batch_to_pixels_btc: batches_of_lists is empty")
    out_rows: list[torch.Tensor] = []
    for views in batches_of_lists:
        if isinstance(views, Image.Image):
            views_it: Any = [views]
        elif isinstance(views, np.ndarray) and views.ndim == 3:
            views_it = [views]
        elif isinstance(views, torch.Tensor) and views.ndim == 3:
            views_it = [views]
        elif isinstance(views, (list, tuple)):
            views_it = views
        else:
            raise TypeError(
                "Each trajectory must be a list/tuple of frames (or one H×W×C image/tensor). "
                f"Got {type(views)}."
            )
        chw_rows = [_prep_frame_chw01(im, image_size) for im in views_it]
        out_rows.append(torch.stack(chw_rows, dim=0))
    stacked = torch.stack(out_rows, dim=0).to(device=device, dtype=dtype)
    if stacked.ndim != 5:
        raise ValueError(f"pil_batch_to_pixels_btc: expected 5D [B,T,C,H,W], got {tuple(stacked.shape)}")
    if expected_batch is not None and stacked.shape[0] != expected_batch:
        raise ValueError(
            f"pil_batch_to_pixels_btc: batch dim {stacked.shape[0]} != expected_batch={expected_batch}"
        )
    if expected_time is not None and stacked.shape[1] != expected_time:
        raise ValueError(
            f"pil_batch_to_pixels_btc: time dim {stacked.shape[1]} != expected_time={expected_time}. "
            "Check ``trajectory_rollout.gt_use_next_observation`` vs ``len(expert_views)`` per trajectory."
        )
    return stacked
