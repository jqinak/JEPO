from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    value = OmegaConf.select(config, key)
    return default if value is None else value


def _validate_reward_inputs(
    pred_embs: torch.Tensor,
    gt_embs: torch.Tensor,
    response_mask: torch.Tensor,
    action_dim: int,
    *,
    gt_offset: int,
) -> tuple[int, int]:
    if pred_embs.ndim != 3:
        raise ValueError(f"pred_embs must be 3D (B,T,D), got shape={tuple(pred_embs.shape)}")
    if gt_embs.ndim != 3:
        raise ValueError(f"gt_embs must be 3D (B,T,D), got shape={tuple(gt_embs.shape)}")
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be 2D (B,L), got shape={tuple(response_mask.shape)}")
    if gt_embs.shape[0] != pred_embs.shape[0]:
        raise ValueError(f"gt batch {gt_embs.shape[0]} != pred batch {pred_embs.shape[0]}")
    if gt_embs.shape[-1] != pred_embs.shape[-1]:
        raise ValueError(f"gt dim {gt_embs.shape[-1]} != pred dim {pred_embs.shape[-1]}")
    bsz, t_max, _ = pred_embs.shape
    if response_mask.shape[0] != bsz:
        raise ValueError(f"response_mask batch {response_mask.shape[0]} != pred batch {bsz}")
    expected_l = t_max * int(action_dim)
    if response_mask.shape[1] != expected_l:
        raise ValueError(f"response_mask length {response_mask.shape[1]} != pred_embs T*action_dim {expected_l}")
    if gt_embs.shape[1] < t_max + int(gt_offset):
        raise ValueError(
            f"gt_embs time {gt_embs.shape[1]} is too short for T_max={t_max} and gt_offset={gt_offset}"
        )
    return bsz, t_max


def _valid_step_mask(response_mask: torch.Tensor, action_dim: int, dtype: torch.dtype) -> torch.Tensor:
    bsz = response_mask.shape[0]
    return response_mask.reshape(bsz, -1, int(action_dim)).any(dim=-1).to(dtype=dtype)


def _infer_n_micro(response_mask: torch.Tensor, action_dim: int) -> torch.Tensor:
    return response_mask.sum(dim=-1).to(dtype=torch.long) // int(action_dim)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Args:
        a, b: (B, D) or (B, T, D)
    Returns:
        cosine similarity of same leading shape, clamped to [-1, 1]
    """
    return F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)


def _distribute_to_tokens(
    reward_per_step: torch.Tensor,
    response_mask: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """
    Repeat each per-step scalar reward ``action_dim`` times, then apply ``response_mask``.
    """
    bsz, t_max = reward_per_step.shape
    token_rewards = reward_per_step.unsqueeze(-1).expand(bsz, t_max, int(action_dim))
    token_rewards = token_rewards.reshape(bsz, t_max * int(action_dim))
    return token_rewards * response_mask.to(device=reward_per_step.device, dtype=reward_per_step.dtype)


def _normalize_per_sample(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize each sample independently over valid token positions. Padding remains zero.
    """
    mask = response_mask.to(device=token_rewards.device).bool()
    out = torch.zeros_like(token_rewards)
    for b in range(token_rewards.shape[0]):
        valid_idx = mask[b]
        vals = token_rewards[b, valid_idx]
        if vals.numel() == 0:
            continue
        std = vals.std()
        if std < 1e-8:
            continue
        out[b, valid_idx] = (vals - vals.mean()) / (std + 1e-8)
    return out


def compute_terminal_reward(
    pred_embs: torch.Tensor,
    gt_embs: torch.Tensor,
    response_mask: torch.Tensor,
    action_horizon: int,
    action_dim: int,
    gt_use_next_observation: bool,
    normalize_rewards: bool,
) -> torch.Tensor:
    del action_horizon
    gt_offset = 1 if bool(gt_use_next_observation) else 0
    bsz, t_max = _validate_reward_inputs(pred_embs, gt_embs, response_mask, action_dim, gt_offset=gt_offset)
    n_micro = _infer_n_micro(response_mask, action_dim).to(device=pred_embs.device)
    terminal_idx = n_micro - 1
    gather_idx = terminal_idx.view(bsz, 1, 1).expand(bsz, 1, pred_embs.shape[-1])
    terminal_pred = pred_embs.gather(dim=1, index=gather_idx).squeeze(1)
    terminal_gt = gt_embs.gather(dim=1, index=(gather_idx + gt_offset)).squeeze(1)
    reward = _cosine_sim(terminal_pred, terminal_gt).to(dtype=pred_embs.dtype)

    reward_per_step = torch.zeros(bsz, t_max, device=pred_embs.device, dtype=pred_embs.dtype)
    reward_per_step = reward_per_step.scatter(dim=1, index=terminal_idx.view(bsz, 1), src=reward.view(bsz, 1))
    token_rewards = _distribute_to_tokens(reward_per_step, response_mask, action_dim)
    if normalize_rewards:
        token_rewards = _normalize_per_sample(token_rewards, response_mask)
    return token_rewards


def compute_sparse_milestone_reward(
    pred_embs: torch.Tensor,
    gt_embs: torch.Tensor,
    response_mask: torch.Tensor,
    action_horizon: int,
    action_dim: int,
    gt_use_next_observation: bool,
    normalize_rewards: bool,
) -> torch.Tensor:
    gt_offset = 1 if bool(gt_use_next_observation) else 0
    bsz, t_max = _validate_reward_inputs(pred_embs, gt_embs, response_mask, action_dim, gt_offset=gt_offset)
    n_micro = _infer_n_micro(response_mask, action_dim).to(device=pred_embs.device)
    gt_slice = gt_embs[:, gt_offset : gt_offset + t_max, :]
    cos_all = _cosine_sim(pred_embs, gt_slice).to(dtype=pred_embs.dtype)

    step_ids = torch.arange(t_max, device=pred_embs.device).view(1, t_max)
    valid_step = step_ids < n_micro.view(bsz, 1)
    terminal_step = step_ids == (n_micro.view(bsz, 1) - 1)
    chunk_boundary = ((step_ids + 1) % int(action_horizon)) == 0
    milestone_mask = valid_step & (chunk_boundary | terminal_step)
    reward_per_step = cos_all * milestone_mask.to(dtype=pred_embs.dtype)

    token_rewards = _distribute_to_tokens(reward_per_step, response_mask, action_dim)
    if normalize_rewards:
        token_rewards = _normalize_per_sample(token_rewards, response_mask)
    return token_rewards


def compute_dense_milestone_reward(
    pred_embs: torch.Tensor,
    gt_embs: torch.Tensor,
    response_mask: torch.Tensor,
    action_horizon: int,
    action_dim: int,
    gt_use_next_observation: bool,
    normalize_rewards: bool,
) -> torch.Tensor:
    del action_horizon
    gt_offset = 1 if bool(gt_use_next_observation) else 0
    _, t_max = _validate_reward_inputs(pred_embs, gt_embs, response_mask, action_dim, gt_offset=gt_offset)
    gt_slice = gt_embs[:, gt_offset : gt_offset + t_max, :]
    cos_all = _cosine_sim(pred_embs, gt_slice).to(dtype=pred_embs.dtype)
    step_mask = _valid_step_mask(response_mask, action_dim, pred_embs.dtype).to(device=pred_embs.device)
    reward_per_step = cos_all * step_mask

    token_rewards = _distribute_to_tokens(reward_per_step, response_mask, action_dim)
    if normalize_rewards:
        token_rewards = _normalize_per_sample(token_rewards, response_mask)
    return token_rewards


def compute_jepo_reward(
    pred_embs: torch.Tensor,
    gt_embs: torch.Tensor,
    response_mask: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    reward_type = str(_cfg_get(config, "reward_type"))
    common = dict(
        action_horizon=int(_cfg_get(config, "action_horizon")),
        action_dim=int(_cfg_get(config, "action_dim")),
        gt_use_next_observation=bool(_cfg_get(config, "gt_use_next_observation", True)),
        normalize_rewards=bool(_cfg_get(config, "normalize_rewards", False)),
    )
    if reward_type == "terminal":
        return compute_terminal_reward(pred_embs, gt_embs, response_mask, **common)
    if reward_type == "sparse_milestone":
        return compute_sparse_milestone_reward(pred_embs, gt_embs, response_mask, **common)
    if reward_type == "dense_milestone":
        return compute_dense_milestone_reward(pred_embs, gt_embs, response_mask, **common)
    raise ValueError(f"Unknown reward_type: {reward_type!r}")


class JEPOLewmRewardWorker:
    """Vendored LEWM worker with JEPO trajectory-level reward dispatch."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
        self.smoke_random_init = bool(config.runtime.get("smoke_random_init", False))
        self.model = None
        self.history_size = 3
        self.num_preds = 1
        self.expected_action_dim = None
        self.image_size = 224
        self.use_fallback = bool(config.reward.fallback_to_action_embedding)
        self.strict_load = bool(config.reward.get("strict_lewm_load", True))
        self.min_load_ratio = float(config.reward.get("min_param_load_ratio", 0.90))
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self._try_load_lewm()
        self._try_load_action_normalizer()

    def _try_load_lewm(self):
        if self.smoke_random_init:
            try:
                self.model = self._build_jepa_from_config()
                self.model.to(self.device).eval()
                print("[smoke] lewm uses random initialization (no lewm_ckpt load).")
            except Exception as e:
                self.model = None
                self.use_fallback = True
                print(f"[smoke] lewm random init failed, fallback reward enabled: {e}")
            return

        ckpt_path = Path(self.config.paths.lewm_ckpt)
        if not ckpt_path.exists():
            if self.use_fallback:
                print(f"[LEWM] checkpoint missing, using fallback: {ckpt_path}")
                return
            raise FileNotFoundError(f"LE-WM checkpoint not found: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[LEWM] torch.load failed ({ckpt_path}): {e}")
            ckpt = None

        if isinstance(ckpt, torch.nn.Module):
            self.model = ckpt.to(self.device).eval()
            print(f"[LEWM] loaded full nn.Module from {ckpt_path}")
            return
        if ckpt is not None and hasattr(ckpt, "encode") and hasattr(ckpt, "predict"):
            self.model = ckpt.to(self.device).eval()
            print(f"[LEWM] loaded object with encode/predict from {ckpt_path}")
            return

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            try:
                self.model = self._build_jepa_from_config()
                filtered = {}
                for k, v in ckpt["state_dict"].items():
                    if k.startswith("model."):
                        filtered[k[len("model.") :]] = v
                self._load_state_dict_checked(self.model, filtered)
                self.model.to(self.device).eval()
                print(f"[LEWM] loaded state_dict (model.* keys) from {ckpt_path}, params matched checked.")
                return
            except Exception as e:
                self.model = None
                print(f"[LEWM] state_dict branch failed: {e}")

        if isinstance(ckpt, dict) and any(str(k).startswith("encoder.") for k in ckpt.keys()):
            try:
                self.model = self._build_jepa_from_config()
                self._load_state_dict_checked(self.model, ckpt)
                self.model.to(self.device).eval()
                print(f"[LEWM] loaded flat encoder.* state_dict from {ckpt_path}")
                return
            except Exception as e:
                self.model = None
                print(f"[LEWM] encoder.* branch failed: {e}")

        if not self.use_fallback:
            raise RuntimeError(
                "Failed to load LE-WM checkpoint in supported formats. "
                "Set reward.fallback_to_action_embedding=true to allow fallback."
            )
        print("[LEWM] all load branches failed; using fallback reward (cosine actions).")

    def _load_state_dict_checked(self, model: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        total = len(model.state_dict().keys())
        loaded = total - len(missing)
        ratio = 0.0 if total == 0 else loaded / total
        if self.strict_load and ratio < self.min_load_ratio:
            raise RuntimeError(
                f"LE-WM checkpoint load ratio too low: loaded={loaded}/{total} ({ratio:.2%}), "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    def _try_load_action_normalizer(self) -> None:
        jepo_cfg = OmegaConf.select(self.config, "reward.jepo") or OmegaConf.create({})
        norm_cfg = jepo_cfg.get("action_normalizer", None)
        if norm_cfg is None:
            norm_cfg = OmegaConf.create({"enabled": False})
        if not bool(norm_cfg.get("enabled", False)):
            return

        strict = bool(norm_cfg.get("strict", True))
        try:
            if norm_cfg.get("mean", None) is not None and norm_cfg.get("std", None) is not None:
                mean = torch.as_tensor(OmegaConf.to_container(norm_cfg.mean, resolve=True), dtype=torch.float32)
                std = torch.as_tensor(OmegaConf.to_container(norm_cfg.std, resolve=True), dtype=torch.float32)
            else:
                mean, std = self._build_action_stats_from_lewm_eval_cfg(norm_cfg)
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
            std = torch.where(std == 0, torch.ones_like(std), std)
            self.action_mean = mean
            self.action_std = std
            print(f"[LEWM] action normalizer enabled: dim={mean.shape[-1]}")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to initialize LEWM action normalizer: {e}") from e
            print(f"[LEWM] action normalizer disabled after init failure: {e}")

    def _build_action_stats_from_lewm_eval_cfg(self, norm_cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
        eval_cfg_path = norm_cfg.get("eval_cfg_path", None) or self.config.paths.get("lewm_eval_cfg_path", None)
        if eval_cfg_path is None:
            raise ValueError("reward.jepo.action_normalizer.eval_cfg_path or paths.lewm_eval_cfg_path is required")
        eval_cfg_path = Path(str(eval_cfg_path)).expanduser().resolve()
        eval_cfg = OmegaConf.load(str(eval_cfg_path))

        lewm_repo_raw = (
            norm_cfg.get("lewm_repo", None)
            or self.config.paths.get("lewm_stats_repo", None)
            or self.config.paths.lewm_repo
        )
        lewm_repo = Path(str(lewm_repo_raw)).expanduser().resolve()
        libero_dataset_py = lewm_repo / "libero_dataset.py"
        if not libero_dataset_py.exists():
            raise FileNotFoundError(f"Cannot find LEWM libero_dataset.py for action stats: {libero_dataset_py}")
        if str(lewm_repo) not in sys.path:
            sys.path.insert(0, str(lewm_repo))
        spec = importlib.util.spec_from_file_location("_jepo_lewm_libero_dataset_for_stats", libero_dataset_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import LEWM libero_dataset.py from {libero_dataset_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset_cfg = OmegaConf.to_container(eval_cfg.data.dataset, resolve=True)
        dataset_type = dataset_cfg.pop("type", "libero")
        if dataset_type != "libero":
            raise ValueError(f"Action normalizer expects a libero eval dataset, got {dataset_type!r}")
        dataset = module.LiberoParquetDataset(**dataset_cfg, transform=None)

        col_data = dataset.get_col_data("action")
        data = torch.from_numpy(np.array(col_data)).float()
        data = data[~torch.isnan(data).any(dim=1)]
        if data.numel() == 0:
            raise ValueError(f"No finite action rows found from eval cfg {eval_cfg_path}")
        mean = data.mean(0).clone()
        std = data.std(0).clone()
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    def _build_jepa_from_config(self):
        from omegaconf import OmegaConf
        from transformers import ViTConfig, ViTModel
        from lewm import ARPredictor, Embedder, JEPA, MLP

        override_cfg = self.config.paths.get("lewm_cfg_path", None)
        cfg_path = Path(override_cfg) if override_cfg else Path(self.config.paths.lewm_repo) / "config/train/lewm.yaml"
        cfg = OmegaConf.load(str(cfg_path))
        self.history_size = int(cfg.wm.history_size)
        self.num_preds = int(cfg.wm.num_preds)
        self.image_size = int(cfg.img_size)

        embed_dim = int(cfg.wm.get("embed_dim", 192))
        patch_size = int(cfg.patch_size)
        img_size = int(cfg.img_size)
        # The shipped LEWM checkpoint uses ViT-Tiny style dimensions: hidden=192, depth=12, heads=3.
        vit_cfg = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=embed_dim,
            num_hidden_layers=12,
            num_attention_heads=max(1, embed_dim // 64),
            intermediate_size=embed_dim * 4,
            qkv_bias=True,
        )
        encoder = ViTModel(vit_cfg)
        hidden_dim = int(encoder.config.hidden_size)
        action_dim = int(cfg.wm.get("action_dim", 7))
        frameskip = int(cfg.get("data", {}).get("dataset", {}).get("frameskip", 1)) if hasattr(cfg, "get") else 1
        effective_act_dim = frameskip * action_dim
        self.expected_action_dim = int(effective_act_dim)

        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
        predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
        return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder, projector=projector, pred_proj=predictor_proj)

    def _match_action_dim(self, actions: torch.Tensor) -> torch.Tensor:
        if self.expected_action_dim is None:
            return actions
        cur_dim = actions.shape[-1]
        exp_dim = int(self.expected_action_dim)
        if cur_dim == exp_dim:
            return actions
        if exp_dim % cur_dim == 0:
            factor = exp_dim // cur_dim
            return actions.repeat_interleave(factor, dim=-1)
        if cur_dim < exp_dim:
            return F.pad(actions, (0, exp_dim - cur_dim))
        return actions[..., :exp_dim]

    def _prepare_lewm_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = self._match_action_dim(actions)
        if self.action_mean is None or self.action_std is None:
            return actions
        mean = self.action_mean.to(device=actions.device, dtype=actions.dtype)
        std = self.action_std.to(device=actions.device, dtype=actions.dtype)
        if actions.shape[-1] != mean.shape[-1]:
            raise ValueError(
                f"LEWM action normalizer dim {mean.shape[-1]} != action dim {actions.shape[-1]} "
                "(check eval dataset frameskip/action_dim vs JEPO action_take_dim)."
            )
        return (actions - mean) / std

    @staticmethod
    def _pad_views_to_time(expert_views_per_traj: list[list[Any]], expected_time: int) -> list[list[Any]]:
        padded: list[list[Any]] = []
        for views in expert_views_per_traj:
            if not views:
                raise ValueError("expert_views_per_traj contains an empty trajectory")
            row = list(views[:expected_time])
            if len(row) < expected_time:
                row.extend([row[-1]] * (expected_time - len(row)))
            padded.append(row)
        return padded

    def compute_jepo_trajectory_rewards(
        self,
        expert_views_per_traj: list[list[Any]],
        predicted_micro_actions: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        gt_micro_actions: torch.Tensor | None,
        chunk_actions: int,
        rollout_n: int,
    ) -> dict[str, Any]:
        del chunk_actions
        jepo_cfg = OmegaConf.select(self.config, "reward.jepo") or OmegaConf.create({})
        action_dim = int(jepo_cfg.get("action_dim", predicted_micro_actions.shape[-1]))
        gt_use_next = bool(jepo_cfg.get("gt_use_next_observation", True))
        b_base = len(expert_views_per_traj)
        b_total, t_max, _ = predicted_micro_actions.shape
        if b_base < 1:
            raise ValueError("expert_views_per_traj must be non-empty")
        if b_total != b_base * int(rollout_n):
            raise ValueError(f"predicted rows {b_total} != base trajectories {b_base} * rollout_n {rollout_n}")

        reward_field = predicted_micro_actions.to(self.device)
        response_mask_dev = response_mask.to(device=self.device, dtype=reward_field.dtype)
        gt_action_pred_embs = None

        if self.model is not None:
            from jepo.workers.lewm_rollout_micro import (
                coerce_pixels_btc_hw,
                encode_pixels_bt,
                pil_batch_to_pixels_btc,
                predict_micro_emb_sequence_from_gt_history,
            )

            expect_t = t_max + (1 if gt_use_next else 0)
            padded_views = self._pad_views_to_time(expert_views_per_traj, expect_t)
            gt_pixels_base = pil_batch_to_pixels_btc(
                padded_views,
                self.image_size,
                self.device,
                expected_batch=b_base,
                expected_time=expect_t,
                imagenet_normalize=bool(jepo_cfg.get("imagenet_normalize", True)),
            )
            gt_pixels_base = coerce_pixels_btc_hw(gt_pixels_base, batch_b=b_base, time_t=expect_t)
            with torch.no_grad():
                gt_embs_base = encode_pixels_bt(self.model, gt_pixels_base.float())
                gt_embs = gt_embs_base.repeat_interleave(int(rollout_n), dim=0)
                pred_act_micro = self._prepare_lewm_actions(reward_field.float())
                pred_embs = predict_micro_emb_sequence_from_gt_history(
                    self.model,
                    gt_embs,
                    pred_act_micro,
                    self.history_size,
                    gt_offset=1 if gt_use_next else 0,
                )
                if gt_micro_actions is not None and bool(jepo_cfg.get("diagnostic_gt_action_cos", True)):
                    gt_act_base = gt_micro_actions.to(self.device, dtype=reward_field.dtype)
                    gt_act_rep = gt_act_base.repeat_interleave(int(rollout_n), dim=0)
                    gt_act_micro = self._prepare_lewm_actions(gt_act_rep.float())
                    gt_action_pred_embs = predict_micro_emb_sequence_from_gt_history(
                        self.model,
                        gt_embs,
                        gt_act_micro,
                        self.history_size,
                        gt_offset=1 if gt_use_next else 0,
                    )
        elif self.use_fallback:
            pred_embs = reward_field.float()
            if gt_micro_actions is None:
                gt_base = torch.zeros(b_base, t_max, pred_embs.shape[-1], device=self.device, dtype=pred_embs.dtype)
            else:
                gt_base = gt_micro_actions.to(self.device, dtype=pred_embs.dtype)
            gt_embs = gt_base.repeat_interleave(int(rollout_n), dim=0)
        else:
            raise RuntimeError("LEWM model unavailable and JEPO rewards require it.")

        token_rewards = compute_jepo_reward(pred_embs, gt_embs, response_mask_dev, jepo_cfg)

        n_micro = _infer_n_micro(response_mask_dev, action_dim).to(device=self.device)
        gt_offset = 1 if gt_use_next else 0
        warmup_steps = max(0, int(self.history_size) - int(gt_offset))
        if bool((n_micro <= warmup_steps).any()):
            raise ValueError(
                f"n_micro values must exceed LEWM warmup_steps={warmup_steps}; got {n_micro.detach().cpu().tolist()}"
            )
        if warmup_steps > 0:
            warmup_step_mask = torch.ones(
                response_mask_dev.shape[0],
                pred_embs.shape[1],
                device=self.device,
                dtype=token_rewards.dtype,
            )
            warmup_step_mask[:, :warmup_steps] = 0.0
            warmup_token_mask = warmup_step_mask.unsqueeze(-1).expand(-1, -1, action_dim).reshape_as(token_rewards)
            token_rewards = token_rewards * warmup_token_mask
        terminal_idx = n_micro - 1
        gather_idx = terminal_idx.view(b_total, 1, 1).expand(b_total, 1, pred_embs.shape[-1])
        term_cos = _cosine_sim(
            pred_embs.gather(1, gather_idx).squeeze(1),
            gt_embs.gather(1, gather_idx + gt_offset).squeeze(1),
        )
        gt_action_term_cos = None
        if gt_action_pred_embs is not None:
            gt_action_term_cos = _cosine_sim(
                gt_action_pred_embs.gather(1, gather_idx).squeeze(1),
                gt_embs.gather(1, gather_idx + gt_offset).squeeze(1),
            )
        step_mask = _valid_step_mask(response_mask_dev, action_dim, pred_embs.dtype).to(device=self.device)
        terminal_threshold = float(jepo_cfg.get("terminal_cos_threshold", 0.85))
        terminal_success = (term_cos >= terminal_threshold).to(dtype=term_cos.dtype)

        token_flat = token_rewards.detach().cpu()
        valid_mass = response_mask_dev.sum(dim=-1).clamp_min(1.0)
        row_reward = (token_rewards * response_mask_dev).sum(dim=-1) / valid_mass
        out = {
            "token_level_rewards": token_flat,
            "reward_mean": float(row_reward.mean().detach().cpu()),
            "reward_std": float(row_reward.std(unbiased=False).detach().cpu()),
            "step_reward_mean": float((token_rewards * response_mask_dev).sum().detach().cpu() / valid_mass.sum().detach().cpu()),
            "step_reward_std": float(token_rewards[response_mask_dev.bool()].std(unbiased=False).detach().cpu())
            if bool(response_mask_dev.bool().any())
            else 0.0,
            "reward_mean_raw": float(term_cos.mean().detach().cpu()),
            "reward_std_raw": float(term_cos.std(unbiased=False).detach().cpu()),
            "step_reward_mean_raw": float(term_cos.mean().detach().cpu()),
            "step_reward_std_raw": float(term_cos.std(unbiased=False).detach().cpu()),
            "reward/type_terminal": 1.0 if str(jepo_cfg.get("reward_type", "terminal")) == "terminal" else 0.0,
            "reward/type_sparse_milestone": 1.0
            if str(jepo_cfg.get("reward_type", "terminal")) == "sparse_milestone"
            else 0.0,
            "reward/type_dense_milestone": 1.0
            if str(jepo_cfg.get("reward_type", "terminal")) == "dense_milestone"
            else 0.0,
            "reward/terminal_cos_mean": float(term_cos.mean().detach().cpu()),
            "reward/terminal_cos_std": float(term_cos.std(unbiased=False).detach().cpu()),
            "reward/terminal_cos_pos_mean": float(((term_cos + 1.0) * 0.5).mean().detach().cpu()),
            "reward/terminal_success_rate": float(terminal_success.mean().detach().cpu()),
            "reward/terminal_cos_threshold": terminal_threshold,
            "reward/n_micro_steps_mean": float(n_micro.float().mean().detach().cpu()),
            "reward/n_micro_steps_min": float(n_micro.float().min().detach().cpu()),
            "reward/n_micro_steps_max": float(n_micro.float().max().detach().cpu()),
            "reward/lewm_history_size": float(self.history_size),
            "reward/lewm_warmup_steps_masked": float(warmup_steps),
            "reward/lewm_action_normalizer_enabled": 1.0 if self.action_mean is not None else 0.0,
            "reward/lewm_imagenet_normalize_enabled": 1.0 if bool(jepo_cfg.get("imagenet_normalize", True)) else 0.0,
            "reward/valid_step_fraction": float(step_mask.mean().detach().cpu()),
            "reward/normalize_token_rewards_applied": 1.0 if bool(jepo_cfg.get("normalize_rewards", False)) else 0.0,
        }
        if gt_action_term_cos is not None:
            out.update(
                {
                    "reward/terminal_cos_gt_action_mean": float(gt_action_term_cos.mean().detach().cpu()),
                    "reward/terminal_cos_gt_action_std": float(gt_action_term_cos.std(unbiased=False).detach().cpu()),
                    "reward/terminal_cos_gap_gt_minus_policy": float(
                        (gt_action_term_cos - term_cos).mean().detach().cpu()
                    ),
                }
            )
        return out
