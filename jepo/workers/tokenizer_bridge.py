from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TokenizerBridge:
    @staticmethod
    def normalize_examples(batch: Any) -> list[dict]:
        if isinstance(batch, list) and batch and isinstance(batch[0], dict):
            return batch
        raise TypeError("Expected dataloader batch as list[dict]. " f"Got type={type(batch)}.")

    @staticmethod
    def extract_actions(examples: list[dict], device: torch.device, horizon: int) -> torch.Tensor:
        actions = []
        for ex in examples:
            if "action" not in ex:
                raise KeyError("Example missing `action` key.")
            ac = np.asarray(ex["action"], dtype=np.float32)
            if ac.ndim != 2:
                raise ValueError(f"`action` must be [T, A], got shape={ac.shape}")
            actions.append(ac[-horizon:])
        return torch.from_numpy(np.stack(actions, axis=0)).to(device)

    @staticmethod
    def extract_states(examples: list[dict], device: torch.device) -> torch.Tensor | None:
        if not examples or "state" not in examples[0]:
            return None
        states = []
        for ex in examples:
            st = np.asarray(ex["state"], dtype=np.float32)
            if st.ndim == 1:
                st = st[None, :]
            states.append(st)
        return torch.from_numpy(np.stack(states, axis=0)).to(device)
