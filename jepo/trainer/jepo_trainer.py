from __future__ import annotations

import copy
import glob
import importlib
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_CORE_ALGOS = None

def _get_core_algos():
    global _CORE_ALGOS
    if _CORE_ALGOS is not None:
        return _CORE_ALGOS
    try:
        from verl.trainer.ppo import core_algos as _ca
    except ModuleNotFoundError:
        _jepo_root = Path(__file__).resolve().parents[2]
        candidates = [_jepo_root / "verl", Path("/project/peilab/qjl/2026/wmrl/verl")]
        for candidate in candidates:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        if "verl" in sys.modules:
            del sys.modules["verl"]
        from verl.trainer.ppo import core_algos as _ca
    _CORE_ALGOS = _ca
    return _CORE_ALGOS

from jepo.data.full_trajectory_rollout_iterable import JEPOFullExpertTrajectoryIterable
from jepo.workers.actor_rollout_worker import ActorRolloutWorker
from jepo.workers.lewm_reward_worker import JEPOLewmRewardWorker
from jepo.workers.tokenizer_bridge import TokenizerBridge


def _chunks_to_micro_tensor(
    pred_c: torch.Tensor,
    *,
    s_chunks: int,
    chunk_actions: int,
    n_micro: int,
) -> torch.Tensor:
    """``pred_c`` [S, a, d] → first ``n_micro`` rows [n_micro, d] (drop flow padding tail)."""
    rows: list[torch.Tensor] = []
    a = int(chunk_actions)
    for j in range(int(s_chunks)):
        gs = j * a
        take = min(a, int(n_micro) - gs)
        if take > 0:
            rows.append(pred_c[j, :take].float())
    if not rows:
        raise RuntimeError("_chunks_to_micro_tensor: empty micro sequence")
    return torch.cat(rows, dim=0)


def _gt_chunks_to_micro_tensor(traj: dict, *, s_chunks: int, chunk_actions: int, n_micro: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    a = int(chunk_actions)
    for j in range(int(s_chunks)):
        gs = j * a
        take = min(a, int(n_micro) - gs)
        if take > 0:
            act = torch.as_tensor(traj["chunk_examples"][j]["action"], dtype=torch.float32)
            rows.append(act[:take])
    return torch.cat(rows, dim=0)


def _flatten_config_for_wandb(cfg) -> dict:
    """OmegaConf/DictConfig → 可 JSON 化的 dict，供 wandb.init(config=...)."""
    try:
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        return {}


def _tensor_scalar_stats(t: torch.Tensor, prefix: str) -> dict[str, float]:
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return {}
    x = t.detach().float().reshape(-1)
    return {
        f"{prefix}/mean": float(x.mean().cpu()),
        f"{prefix}/std": float(x.std(unbiased=False).cpu()),
        f"{prefix}/min": float(x.min().cpu()),
        f"{prefix}/max": float(x.max().cpu()),
        f"{prefix}/absmax": float(x.abs().max().cpu()),
    }


def _tensor_advanced_stats_rowwise(t: torch.Tensor, prefix: str) -> dict[str, float]:
    """2D tensor ``[batch, tokens]``: 跨行的 batch 粒度与沿 token 的变化。"""
    if not isinstance(t, torch.Tensor) or t.ndim != 2:
        return {}
    x = t.detach().float()
    row_mean = x.mean(dim=1)
    token_mean = x.mean(dim=0)
    return {
        f"{prefix}/row_mean/std_across_batch": float(row_mean.std(unbiased=False).cpu()),
        f"{prefix}/row_sum/mean_across_batch": float(x.sum(dim=1).mean().cpu()),
        f"{prefix}/row_sum/std_across_batch": float(x.sum(dim=1).std(unbiased=False).cpu()),
        f"{prefix}/token_mean/std_across_positions": float(token_mean.std(unbiased=False).cpu()),
    }


def _grpo_repeat_dispersion(tlr: torch.Tensor, advantages: torch.Tensor, returns_t: torch.Tensor, b_sz: int, repeat_n: int) -> dict[str, float]:
    """同一条 trajectory 的不同 rollout_n 副本之间的离散度。"""
    if repeat_n <= 1 or not isinstance(tlr, torch.Tensor) or tlr.ndim != 2:
        return {}
    bt = b_sz * repeat_n
    if tlr.shape[0] != bt:
        return {}
    row_r = tlr.detach().float().mean(dim=1)
    row_a = advantages.detach().float().mean(dim=1)
    row_ret = returns_t.detach().float().mean(dim=1)
    stds_r: list[float] = []
    stds_a: list[float] = []
    stds_ret: list[float] = []
    for bi in range(b_sz):
        sl = slice(bi * repeat_n, (bi + 1) * repeat_n)
        stds_r.append(float(row_r[sl].std(unbiased=False).cpu()))
        stds_a.append(float(row_a[sl].std(unbiased=False).cpu()))
        stds_ret.append(float(row_ret[sl].std(unbiased=False).cpu()))
    return {
        "monitor/grpo_across_repeat/tlr_rowmean_std_mean": float(np.mean(stds_r)),
        "monitor/grpo_across_repeat/tlr_rowmean_std_max": float(np.max(stds_r)),
        "monitor/grpo_across_repeat/adv_rowmean_std_mean": float(np.mean(stds_a)),
        "monitor/grpo_across_repeat/ret_rowmean_std_mean": float(np.mean(stds_ret)),
    }


def _histogram_payload(key: str, tensor: torch.Tensor, max_samples: int = 8192) -> dict:
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        return {}
    x = tensor.detach().float().reshape(-1).cpu().numpy()
    if x.size == 0:
        return {}
    if x.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(x.size, size=max_samples, replace=False)
        x = x[idx]
    return {key: wandb.Histogram(x)}


def _build_metrics_glossary_lines(use_trajectory: bool) -> list[str]:
    lines = [
        "",
        "—— WandB / 日志指标说明（细粒度）——",
        "【公共】rollout/old_log_prob_*：旧策略对采样链的对数概率（flow 各步 Normal 累加），按 token 展平后的统计。",
        "【公共】advantage/*、returns/*：优势与回报张量（与 token 维对齐）的 mean/std/min/max/absmax。",
        "【公共】actor/loss：PPO total = sum_chunk(pg − entropy_coef·entropy_agg)；轨迹模式对每 traj 多条 chunk 再对 batch 平均。",
        "【公共】actor/loss_avg_per_chunk：平均每 chunk 的上述 total（可与 loss 一起看是否多数 chunk）。",
        "【公共】actor/pg_loss（mean_chunks）：policy surrogate 均值；actor/entropy 为 entropy 聚合；actor/entropy_coeff_times_entropy 为被减项。",
        "【公共】actor/pg_clipfrac*、ratio_*：重要性采样比及 clip 比例；actor/ppo_kl 近似 KL。",
    ]
    if use_trajectory:
        lines += [
            "【Terminal 奖励】terminal/cos_mean、terminal/cos_pos_mean、terminal/success_rate：末步预测嵌入与 GT 嵌入的相似度、映射到 [0,1] 的分数、超过阈值比例。",
            "【奖励/优势】reward/token_*、advantage/*：token 级奖励和优势的均值、标准差与极值；terminal 模式若 normalize_rewards=true，reward/token_mean 接近 0 是预期现象。",
            "【策略稳定性】policy/kl、policy/clipfrac_*、policy/ratio_*、policy/entropy、policy/grad_norm：PPO 更新幅度、裁剪比例、探索度与梯度规模。",
            "【生成行为】generation/*、rollout/old_log_prob_*：每步轨迹长度、rollout 副本数与采样动作在旧策略下的概率统计。",
        ]
    return lines


def _format_static_config_report(config, use_trajectory: bool) -> list[str]:
    algo = OmegaConf.to_container(OmegaConf.select(config, "algorithm") or OmegaConf.create({}), resolve=True)
    tr = OmegaConf.to_container(OmegaConf.select(config, "trainer") or OmegaConf.create({}), resolve=True)
    rw = OmegaConf.to_container(OmegaConf.select(config, "reward") or OmegaConf.create({}), resolve=True)
    data = OmegaConf.to_container(OmegaConf.select(config, "data") or OmegaConf.create({}), resolve=True)
    runtime = OmegaConf.to_container(OmegaConf.select(config, "runtime") or OmegaConf.create({}), resolve=True)
    lines = [
        "=" * 88,
        "JEPO 训练监控 — 静态参数（本 run 仅报告一次；WandB config 中亦有完整 OmegaConf）",
        "=" * 88,
        f"trajectory_rollout.enabled = {use_trajectory}",
        "--- runtime (摘录) ---",
        OmegaConf.to_yaml(OmegaConf.create(runtime)).strip(),
        "--- data (摘录) ---",
        OmegaConf.to_yaml(OmegaConf.create(data)).strip(),
        "--- algorithm ---",
        OmegaConf.to_yaml(OmegaConf.create(algo)).strip(),
        "--- reward (terminal 监控关注末步相似度 / 成功率 / 优势) ---",
        OmegaConf.to_yaml(OmegaConf.create(rw)).strip(),
        "--- trainer (摘录，含 log/save/wandb) ---",
        OmegaConf.to_yaml(OmegaConf.create(tr)).strip(),
    ]
    if use_trajectory:
        tro = OmegaConf.to_container(OmegaConf.select(config, "trajectory_rollout") or OmegaConf.create({}), resolve=True)
        lines += ["--- trajectory_rollout ---", OmegaConf.to_yaml(OmegaConf.create(tro)).strip()]
    lines += _build_metrics_glossary_lines(use_trajectory)
    lines.append("=" * 88)
    return lines


def _log(msg: str) -> None:
    """写到 stdout 并立即刷新；重定向到文件时避免整块缓冲、指标滞后于 stderr 的 Warning。"""
    print(msg, flush=True)


def _log_trajectory_dynamic_step(
    global_step: int,
    *,
    b_sz: int,
    repeat_n: int,
    reward_out: dict,
    update_metrics: dict[str, float],
    rollout_aux: dict,
    token_stats: dict[str, float],
    advantage_extra: dict[str, float],
    advantage_stats: dict[str, float],
    old_log_prob_stats: dict[str, float],
) -> None:
    """Terminal reward 专用每步控制台监控。"""
    def _gf(d: dict, k: str, default: float = float("nan")) -> float:
        v = d.get(k, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    r = reward_out
    line1 = (
        f"[terminal reward step {global_step}] "
        f"cos={_gf(r, 'reward/terminal_cos_mean'):.4f}±{_gf(r, 'reward/terminal_cos_std'):.4f} "
        f"pos={_gf(r, 'reward/terminal_cos_pos_mean'):.4f} "
        f"succ={_gf(r, 'reward/terminal_success_rate'):.3f}@{_gf(r, 'reward/terminal_cos_threshold'):.2f} | "
        f"reward_tok μ/σ={_gf(token_stats, 'reward/token_level_tensor/mean'):.4f}/{_gf(token_stats, 'reward/token_level_tensor/std'):.4f} "
        f"adv μ/σ={_gf(advantage_stats, 'advantage/mean'):.4f}/{_gf(advantage_stats, 'advantage/std'):.4f} "
        f"adv_bt_std={_gf(advantage_extra, 'advantage/row_mean/std_across_batch'):.4f}"
    )
    m = update_metrics
    line2 = (
        f"[policy stability] "
        f"loss={_gf(m, 'actor/loss'):.4f} pg={_gf(m, 'actor/pg_loss'):.4f} "
        f"kl={_gf(m, 'actor/ppo_kl'):.5f} "
        f"clip={_gf(m, 'actor/pg_clipfrac'):.3f}/{_gf(m, 'actor/pg_clipfrac_lower'):.3f} "
        f"ratio={_gf(m, 'actor/ratio_mean'):.3f}/{_gf(m, 'actor/ratio_max'):.3f} "
        f"entropy={_gf(m, 'actor/entropy'):.4f} grad={_gf(m, 'actor/grad_norm'):.3f}"
    )
    line3 = (
        f"[generation behavior] "
        f"B={b_sz} rollout_n={repeat_n} "
        f"chunks={_gf(rollout_aux, 's_chunks'):.1f} micro_tokens={_gf(rollout_aux, 'micro_tokens'):.1f} "
        f"old_logp μ/σ={_gf(old_log_prob_stats, 'rollout/old_log_prob/mean'):.4f}/{_gf(old_log_prob_stats, 'rollout/old_log_prob/std'):.4f} "
        f"ret_std={_gf(r, 'reward_std'):.4f}"
    )
    line4 = (
        f"[terminal diagnostics] "
        f"gt_cos={_gf(r, 'diag/gt_terminal_cos_mean'):.4f} "
        f"gt_succ={_gf(r, 'diag/gt_terminal_success_rate'):.3f} | "
        f"actor_sigma={_gf(r, 'diag/actor_small_sigma_scale'):.4f} "
        f"actor_cos={_gf(r, 'diag/actor_terminal_cos_mean'):.4f} "
        f"actor_succ={_gf(r, 'diag/actor_terminal_success_rate'):.3f}"
    )
    _log(line1)
    _log(line2)
    _log(line3)
    _log(line4)


def _format_short_stats(d: dict[str, float], limit: int = 6) -> str:
    items = list(d.items())[:limit]
    return " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in items)


class JEPORayTrainer:
    """JEPO 训练器：rollout -> reward -> advantage -> actor update。"""

    def __init__(self, config):
        self.config = config  # 保存配置对象
        self._set_seed(int(config.runtime.seed))  # 固定随机种子
        self.bridge = TokenizerBridge()  # 批次桥接器
        self.actor_worker = ActorRolloutWorker(config)  # 策略 worker
        self.reward_worker = JEPOLewmRewardWorker(config)  # JEPO reward worker
        self.use_trajectory_rollout = bool(OmegaConf.select(config, "trajectory_rollout.enabled") or False)
        if self.use_trajectory_rollout:
            _log("[trajectory_rollout] enabled: full-episode iterable (batch=1) + LEWM open-loop rewards")
        self.rollout_cycle = itertools.cycle(self._build_rollout_stream())
        self.total_steps = int(config.trainer.total_training_steps)  # 总步数
        self.log_interval = int(config.trainer.log_interval)  # 日志间隔
        self.save_interval = int(config.trainer.save_interval)  # 保存间隔
        self.output_dir = Path(config.trainer.output_dir)  # 输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_step = 1
        if bool(config.trainer.get("auto_resume", False)):
            self.start_step = self._try_resume()
        self._wandb_initialized = False
        self._monitor_preamble_logged = False
        self.metrics_jsonl_enabled = bool(config.trainer.get("metrics_jsonl", True))
        self._metrics_jsonl_fh = None
        self._metrics_jsonl_path_announced = False
        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        """读取 starvla_cfg（如 qwen35vlPI_libero.yaml）里的 trackers / wandb_*，以及 trainer.wandb 覆盖项。"""
        tw = OmegaConf.select(self.config, "trainer.wandb") or OmegaConf.create({})
        starvla_path = Path(str(self.config.data.starvla_cfg))
        starvla_root = OmegaConf.load(starvla_path) if starvla_path.is_file() else OmegaConf.create({})

        enabled = OmegaConf.select(tw, "enabled")
        if enabled is None:
            trackers = OmegaConf.select(starvla_root, "trackers") or []
            trackers_list = OmegaConf.to_container(trackers, resolve=True)
            if not isinstance(trackers_list, list):
                trackers_list = []
            enabled = any(str(x).lower() == "wandb" for x in trackers_list)
        else:
            enabled = bool(enabled)

        if not enabled:
            return

        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError:
            _log("[wandb] 未安装 wandb，跳过远程记录。安装: pip install wandb")
            return

        entity = OmegaConf.select(tw, "entity") or OmegaConf.select(starvla_root, "wandb_entity")
        project = OmegaConf.select(tw, "project") or OmegaConf.select(starvla_root, "wandb_project")
        if not project:
            project = "jepo"
            _log("[wandb] 未配置 project，使用默认值 'jepo'")
        run_name = OmegaConf.select(tw, "run_name") or OmegaConf.select(tw, "name")
        if not run_name:
            run_name = f"jepo_{self.output_dir.name}_seed{int(self.config.runtime.seed)}"

        tags = OmegaConf.select(tw, "tags")
        tags_list = OmegaConf.to_container(tags, resolve=True) if tags is not None else None
        if not isinstance(tags_list, list):
            tags_list = []
        tags_list = [str(t) for t in tags_list]

        notes = OmegaConf.select(tw, "notes")
        notes_str = str(notes) if notes else None

        cfg_dict = _flatten_config_for_wandb(self.config)
        init_kwargs: dict = {
            "project": str(project),
            "name": str(run_name),
            "config": cfg_dict,
            "dir": str(self.output_dir),
        }
        if entity:
            init_kwargs["entity"] = str(entity)
        if tags_list:
            init_kwargs["tags"] = tags_list
        if notes_str:
            init_kwargs["notes"] = notes_str

        wandb.init(**init_kwargs)  # type: ignore[arg-type]
        self._wandb_initialized = True
        _log(f"[wandb] 已初始化: project={project} name={run_name}" + (f" entity={entity}" if entity else ""))

    def _wandb_log(self, global_step: int, metrics: dict) -> None:
        if not self._wandb_initialized:
            return
        import wandb  # type: ignore[import-untyped]

        out: dict = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                continue
            if isinstance(v, wandb.Histogram) or type(v).__name__ == "Histogram":
                out[str(k)] = v
                continue
            try:
                x = float(v)
                if np.isfinite(x):
                    out[str(k)] = x
            except (TypeError, ValueError):
                pass
        if out:
            wandb.log(out, step=int(global_step))

    def _append_metrics_jsonl(self, global_step: int, wb: dict) -> None:
        if not self.metrics_jsonl_enabled:
            return
        path = self.output_dir / "train_metrics.jsonl"
        if self._metrics_jsonl_fh is None:
            self._metrics_jsonl_fh = open(path, "a", encoding="utf-8")
            if not self._metrics_jsonl_path_announced:
                _log(f"[metrics] 本地逐标量记录: {path}（ trainer.metrics_jsonl=false 可关闭）")
                self._metrics_jsonl_path_announced = True
        row: dict = {"global_step": int(global_step), "trajectory_mode": bool(self.use_trajectory_rollout)}
        hist_prefixes = ("hist/", "hist__")
        for k, v in wb.items():
            sk = str(k)
            if sk.startswith(hist_prefixes):
                continue
            if isinstance(v, torch.Tensor):
                continue
            try:
                import wandb  # type: ignore[import-untyped]

                if isinstance(v, wandb.Histogram) or type(v).__name__ == "Histogram":
                    continue
            except Exception:
                if type(v).__name__ == "Histogram":
                    continue
            try:
                fv = float(v)
                if np.isfinite(fv):
                    row[sk.replace("/", "__")] = fv
            except (TypeError, ValueError):
                pass
        assert self._metrics_jsonl_fh is not None
        self._metrics_jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._metrics_jsonl_fh.flush()

    def _close_metrics_jsonl(self) -> None:
        if self._metrics_jsonl_fh is not None:
            try:
                self._metrics_jsonl_fh.flush()
                self._metrics_jsonl_fh.close()
            except Exception:
                pass
            self._metrics_jsonl_fh = None

    def _wandb_log_static_summary_once(self) -> None:
        """静态训练/奖励参数写入 WandB Summary，便于对比多 run。"""
        if not self._wandb_initialized:
            return
        import wandb  # type: ignore[import-untyped]

        if wandb.run is None:
            return
        run = wandb.run
        traj = OmegaConf.select(self.config, "reward.trajectory") or OmegaConf.create({})
        traj_roll = OmegaConf.select(self.config, "trajectory_rollout") or OmegaConf.create({})
        es = traj.get("enable_trajectory_sparse_milestone", None)
        ed = traj.get("enable_trajectory_dense_milestone", None)
        eterm = traj.get("enable_trajectory_terminal_bonus", None)
        summary: dict[str, float | str] = {
            "jepo_monitor/schema_version": 1.0,
            "static/trajectory_rollout_enabled": 1.0 if self.use_trajectory_rollout else 0.0,
            "static/rollout_n": float(self.config.algorithm.rollout_n),
            "static/train_batch_size": (
                float(v)
                if (v := OmegaConf.select(self.config, "trajectory_rollout.train_batch_size")) is not None
                else float(self.config.data.train_batch_size)
            ),
            "static/action_horizon": float(self.actor_worker.action_horizon),
            "static/action_dim": float(self.actor_worker.action_dim),
            "reward_static/sparse_milestone_scale": float(traj.get("sparse_milestone_scale", traj.get("milestone_scale", 1.0))),
            "reward_static/dense_milestone_scale": float(traj.get("dense_milestone_scale", traj.get("milestone_scale", 1.0))),
            "reward_static/terminal_bonus": float(traj.get("terminal_bonus", 0.5)),
            "reward_static/terminal_cos_threshold": float(traj.get("terminal_cos_threshold", 0.85)),
            "reward_static/flag_sparse_explicit": -1.0 if es is None else (1.0 if bool(es) else 0.0),
            "reward_static/flag_dense_explicit": -1.0 if ed is None else (1.0 if bool(ed) else 0.0),
            "reward_static/flag_terminal_explicit": -1.0 if eterm is None else (1.0 if bool(eterm) else 0.0),
            "reward_static/chunk_end_milestone_only": 1.0 if bool(traj.get("chunk_end_milestone_only", True)) else 0.0,
            "reward_static/credit_denom_mode": str(traj.get("credit_denom_mode", "chunk_tokens")),
            "reward_static/normalize_token_rewards": 1.0 if bool(traj.get("normalize_token_rewards", False)) else 0.0,
            "reward_static/gt_use_next_observation": 1.0 if bool(traj_roll.get("gt_use_next_observation", True)) else 0.0,
        }
        for k, v in summary.items():
            run.summary[k] = v

    def _wandb_finish(self) -> None:
        if not self._wandb_initialized:
            return
        import wandb  # type: ignore[import-untyped]

        if wandb.run is not None:
            wandb.finish()
        self._wandb_initialized = False

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_rollout_stream(self):
        if self.use_trajectory_rollout:
            return self._build_trajectory_dataloader()
        return self._build_dataloader()

    def _build_trajectory_dataloader(self):
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset

        starvla_cfg = OmegaConf.load(self.config.data.starvla_cfg)
        subset = OmegaConf.select(self.config, "data.libero_subset") or OmegaConf.select(self.config, "data.data_mix")
        if subset:
            starvla_cfg.datasets.vla_data.data_mix = str(subset)
        _log("[jepo] 构建 JEPOFullExpertTrajectoryIterable (batched full episodes) ...")
        base = get_vla_dataset(data_cfg=starvla_cfg.datasets.vla_data)
        traj_cfg = self.config.trajectory_rollout
        a = int(self.actor_worker.action_horizon)
        tb = int(traj_cfg.get("train_batch_size", self.config.data.train_batch_size))
        sampling_mode = str(traj_cfg.get("sampling_mode", "random_with_replacement"))
        shuffle_each_epoch = bool(traj_cfg.get("shuffle_each_epoch", True))
        dataset_summaries = []
        total_trajectories = 0
        total_transitions = 0
        for idx, ds in enumerate(base.datasets):
            n_traj = len(getattr(ds, "trajectory_ids", []))
            n_steps = int(len(ds))
            total_trajectories += n_traj
            total_transitions += n_steps
            ds_name = getattr(ds, "dataset_name", None) or getattr(ds, "tag", None) or f"dataset_{idx}"
            dataset_summaries.append(f"{ds_name}: trajectories={n_traj}, transitions={n_steps}")
        steps_per_full_pass = (total_trajectories + max(1, tb) - 1) // max(1, tb)
        total_steps_cfg = int(self.config.trainer.total_training_steps)
        pass_ratio = total_steps_cfg / max(1, steps_per_full_pass)
        _log(
            "[jepo data] "
            f"subset={starvla_cfg.datasets.vla_data.data_mix} | "
            f"sampling_mode={sampling_mode} shuffle_each_epoch={shuffle_each_epoch} | "
            f"trajectory_batch_size={tb} rollout_n={int(self.config.algorithm.rollout_n)} | "
            f"trajectories={total_trajectories} transitions={total_transitions} | "
            f"steps_per_full_pass≈{steps_per_full_pass} current_total_steps={total_steps_cfg} "
            f"({pass_ratio:.2f} theoretical passes)"
        )
        for summary in dataset_summaries:
            _log(f"[jepo data] {summary}")
        if sampling_mode == "sequential_epoch":
            _log("[jepo data] 采样说明：sequential_epoch 会按轨迹列表扫完一遍；shuffle_each_epoch=true 时每轮开始先打乱。")
        else:
            _log("[jepo data] 采样说明：random_with_replacement 为随机带放回采样；steps_per_full_pass 仅表示按 batch 顺序扫完全部轨迹所需的理论步数。")
        it_ds = JEPOFullExpertTrajectoryIterable(
            base,
            chunk_actions=a,
            seed=int(self.config.runtime.seed),
            max_sample_tries=int(traj_cfg.get("max_sample_tries", 512)),
            action_take_dim=int(traj_cfg.get("action_take_dim", self.actor_worker.action_dim)),
            gt_use_next_observation=bool(traj_cfg.get("gt_use_next_observation", True)),
            train_batch_size=tb,
            sampling_mode=sampling_mode,
            shuffle_each_epoch=shuffle_each_epoch,
        )
        return DataLoader(it_ds, batch_size=None, num_workers=0, pin_memory=False)

    def _build_dataloader(self):
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        starvla_cfg = OmegaConf.load(self.config.data.starvla_cfg)
        _log("正在构建数据加载器...")
        dataset = get_vla_dataset(data_cfg=starvla_cfg.datasets.vla_data)
        _log("数据加载器构建完成")
        _log(f"数据集长度: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Empty dataset from starVLA dataloader.")
        return DataLoader(
            dataset,
            batch_size=int(self.config.data.train_batch_size),
            num_workers=int(self.config.data.num_workers),
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
        )

    def _try_resume(self) -> int:
        pattern = str(self.output_dir / "global_step_*" / "wmrl_actor.pt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            return 1
        latest = ckpts[-1]
        state = torch.load(latest, map_location="cpu")
        self.actor_worker.action_model.load_state_dict(state["action_model"], strict=True)
        self.actor_worker.sigma_net.load_state_dict(state["sigma_net"], strict=True)
        self.actor_worker.optimizer.load_state_dict(state["optimizer"])
        step = int(Path(latest).parent.name.replace("global_step_", ""))
        _log(f"[resume] loaded {latest}, start from step {step + 1}")
        return step + 1

    @staticmethod
    def _repeat_examples(examples: list[dict], repeat_n: int) -> list[dict]:
        return [copy.deepcopy(ex) for ex in examples for _ in range(repeat_n)]

    @staticmethod
    def _build_group_index(base_batch_size: int, repeat_n: int) -> np.ndarray:
        ids = [f"uid-{i}" for i in range(base_batch_size) for _ in range(repeat_n)]
        return np.asarray(ids, dtype=object)

    @staticmethod
    def _assert_finite(name: str, value: torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN/Inf.")

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        group_index: np.ndarray,
        response_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not bool(self.config.algorithm.get("normalize_advantage", True)):
            return advantages
        mode = str(self.config.algorithm.get("adv_norm_mode", "batch")).lower()
        eps = float(self.config.algorithm.get("adv_norm_eps", 1e-6))

        if (
            response_mask is not None
            and response_mask.shape == advantages.shape
            and bool((response_mask < 0.5).any())
        ):
            mc = response_mask.sum(dim=-1).clamp(min=1.0)
            row_agg = (advantages * response_mask).sum(dim=-1) / mc
            out_s = torch.zeros_like(row_agg)
            if mode == "batch":
                mean_b = row_agg.mean()
                std_b = row_agg.std(unbiased=False)
                out_s = (row_agg - mean_b) / (std_b + eps)
            elif mode == "group":
                for gid in np.unique(group_index):
                    idx = np.where(group_index == gid)[0]
                    g = row_agg[idx]
                    mean = g.mean()
                    std = g.std(unbiased=False)
                    out_s[idx] = (g - mean) / (std + eps)
            else:
                raise ValueError(f"Unsupported algorithm.adv_norm_mode: {mode}")
            return out_s.unsqueeze(-1) * response_mask

        out = advantages.clone()
        if mode == "batch":
            mean = out.mean()
            std = out.std(unbiased=False)
            return (out - mean) / (std + eps)
        if mode == "group":
            for gid in np.unique(group_index):
                idx = np.where(group_index == gid)[0]
                group_adv = out[idx]
                mean = group_adv.mean()
                std = group_adv.std(unbiased=False)
                out[idx] = (group_adv - mean) / (std + eps)
            return out
        raise ValueError(f"Unsupported algorithm.adv_norm_mode: {mode}")

    def _compute_advantage(
        self,
        token_level_rewards: torch.Tensor,
        group_index: np.ndarray,
        response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_level_rewards.ndim != 2:
            raise ValueError(f"token_level_rewards should be 2D, got {token_level_rewards.shape}")
        if token_level_rewards.shape[0] != group_index.shape[0]:
            raise ValueError(
                f"Reward batch size {token_level_rewards.shape[0]} != group index size {group_index.shape[0]}"
            )
        if response_mask is None:
            response_mask = torch.ones_like(token_level_rewards)
        if response_mask.shape != token_level_rewards.shape:
            raise ValueError("response_mask must match token_level_rewards shape")
        adv_estimator = str(self.config.algorithm.adv_estimator).lower()
        if adv_estimator == "grpo":
            advantages, returns = _get_core_algos().compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=group_index,
            )
            advantages = self._normalize_advantages(advantages, group_index, response_mask=response_mask)
            self._assert_finite("advantages", advantages)
            self._assert_finite("returns", returns)
            return advantages, returns
        if adv_estimator == "gae":
            values = torch.zeros_like(token_level_rewards)
            advantages, returns = _get_core_algos().compute_gae_advantage_return(
                token_level_rewards=token_level_rewards,
                values=values,
                response_mask=response_mask,
                gamma=float(self.config.algorithm.gamma),
                lam=float(self.config.algorithm.lam),
            )
            advantages = self._normalize_advantages(advantages, group_index, response_mask=response_mask)
            self._assert_finite("advantages", advantages)
            self._assert_finite("returns", returns)
            return advantages, returns
        raise NotImplementedError(f"Unsupported adv estimator: {adv_estimator}")

    @staticmethod
    def _build_jepo_group_index(base_batch_size: int, repeat_n: int) -> np.ndarray:
        ids = [f"traj-{i}" for i in range(int(base_batch_size)) for _ in range(int(repeat_n))]
        return np.asarray(ids, dtype=object)

    def _run_trajectory_training_step(self, traj_batch: list):
        repeat_n = int(self.config.algorithm.rollout_n)
        base_b = len(traj_batch)
        expected_b = int(self.config.trajectory_rollout.get("train_batch_size", self.config.data.train_batch_size))
        if base_b != expected_b:
            raise ValueError(f"JEPO expects {expected_b} base trajectories per batch, got {base_b}")
        if repeat_n < 1:
            raise ValueError(f"Invalid rollout_n={repeat_n}")

        a = int(self.actor_worker.action_horizon)
        d = int(self.actor_worker.action_dim)
        per = a * d
        metas = [item.get("meta") or {} for item in traj_batch]
        n_micro = [int(meta.get("n_micro", meta.get("micro_steps", -1))) for meta in metas]
        if any(n < 1 for n in n_micro):
            raise ValueError(f"Invalid JEPO n_micro values: {n_micro}")
        s_chunks = [len(item["chunk_examples"]) for item in traj_batch]
        s_max = max(s_chunks)
        t_max = s_max * a
        padded_tokens = t_max * d

        traj_roll_cfg = OmegaConf.select(self.config, "trajectory_rollout") or OmegaConf.create({})
        use_next_gt_obs = bool(traj_roll_cfg.get("gt_use_next_observation", True))
        expert_views_base: list[list] = []
        gt_micro_base = torch.zeros(base_b, t_max, d, dtype=torch.float32)
        base_masks = torch.zeros(base_b, padded_tokens, dtype=torch.float32)

        for i, item in enumerate(traj_batch):
            expect_views = n_micro[i] + (1 if use_next_gt_obs else 0)
            views = item.get("expert_views")
            if views is None:
                raise ValueError(f"traj_batch[{i}] missing expert_views")
            if len(views) != expect_views:
                raise ValueError(
                    f"traj_batch[{i}] expert_views length {len(views)} != expected {expect_views} "
                    f"(n_micro={n_micro[i]}, gt_use_next_observation={use_next_gt_obs})"
                )
            expert_views_base.append(list(views))
            mic_full = _gt_chunks_to_micro_tensor(item, s_chunks=s_chunks[i], chunk_actions=a, n_micro=n_micro[i])
            gt_micro_base[i, : n_micro[i]] = mic_full
            base_masks[i, : n_micro[i] * d] = 1.0

        pred_micro_rows: list[torch.Tensor] = []
        logprob_rows: list[torch.Tensor] = []
        flat_chunk_examples_ordered: list[dict] = []
        chain_slices_ordered: list[torch.Tensor] = []
        compact_old_log_probs: list[torch.Tensor] = []
        row_chunk_counts: list[int] = []

        for bi, item in enumerate(traj_batch):
            chunk_flat = list(item["chunk_examples"])
            for _rn in range(repeat_n):
                noise = self.actor_worker.sample_noise_for_chunks(chunk_flat)
                roll = self.actor_worker.generate_actions_chunk_flat(chunk_flat, noise)
                pred_c = roll["predicted_actions"].float()
                x_chain = roll["x_chain"]
                micro_1d = _chunks_to_micro_tensor(pred_c, s_chunks=s_chunks[bi], chunk_actions=a, n_micro=n_micro[bi])
                pred_pad = torch.zeros(t_max, d, dtype=torch.float32)
                pred_pad[: n_micro[bi]] = micro_1d
                pred_micro_rows.append(pred_pad)

                old_lp_flat = self.actor_worker.compute_log_prob(chunk_flat, x_chain).float()
                logp_pad = torch.zeros(padded_tokens, dtype=torch.float32)
                for j in range(s_chunks[bi]):
                    logp_pad[j * per : (j + 1) * per] = old_lp_flat[j]
                    flat_chunk_examples_ordered.append(chunk_flat[j])
                    chain_slices_ordered.append(x_chain[j : j + 1])
                    compact_old_log_probs.append(old_lp_flat[j : j + 1])
                logprob_rows.append(logp_pad)
                row_chunk_counts.append(int(s_chunks[bi]))

        predicted_micro = torch.stack(pred_micro_rows, dim=0)
        logprob_ordered = torch.stack(logprob_rows, dim=0)
        response_mask_bt = base_masks.repeat_interleave(repeat_n, dim=0).contiguous()
        token_rows = response_mask_bt.shape[0]
        if predicted_micro.shape != (base_b * repeat_n, t_max, d):
            raise RuntimeError(f"predicted_micro shape {tuple(predicted_micro.shape)} != ({base_b * repeat_n}, {t_max}, {d})")
        if logprob_ordered.shape != (token_rows, padded_tokens):
            raise RuntimeError(f"logprob shape {tuple(logprob_ordered.shape)} != ({token_rows}, {padded_tokens})")

        reward_out = self.reward_worker.compute_jepo_trajectory_rewards(
            expert_views_base,
            predicted_micro,
            response_mask_bt,
            gt_micro_actions=gt_micro_base,
            chunk_actions=a,
            rollout_n=repeat_n,
        )
        if bool(traj_roll_cfg.get("diagnostic_terminal_eval", True)):
            gt_diag = self.reward_worker.compute_jepo_trajectory_rewards(
                expert_views_base,
                gt_micro_base,
                base_masks,
                gt_micro_actions=gt_micro_base,
                chunk_actions=a,
                rollout_n=1,
            )
            reward_out.update(
                {
                    "diag/gt_terminal_cos_mean": gt_diag.get("reward/terminal_cos_mean"),
                    "diag/gt_terminal_cos_std": gt_diag.get("reward/terminal_cos_std"),
                    "diag/gt_terminal_cos_pos_mean": gt_diag.get("reward/terminal_cos_pos_mean"),
                    "diag/gt_terminal_success_rate": gt_diag.get("reward/terminal_success_rate"),
                }
            )

            diag_sigma_scale = float(traj_roll_cfg.get("diagnostic_actor_sigma_scale", 0.0))
            diag_pred_rows: list[torch.Tensor] = []
            for bi, item in enumerate(traj_batch):
                chunk_flat = list(item["chunk_examples"])
                diag_noise = self.actor_worker.sample_noise_for_chunks(chunk_flat)
                diag_roll = self.actor_worker.generate_actions_chunk_flat(
                    chunk_flat,
                    diag_noise,
                    sigma_scale=diag_sigma_scale,
                )
                diag_micro_1d = _chunks_to_micro_tensor(
                    diag_roll["predicted_actions"].float(),
                    s_chunks=s_chunks[bi],
                    chunk_actions=a,
                    n_micro=n_micro[bi],
                )
                diag_pred_pad = torch.zeros(t_max, d, dtype=torch.float32)
                diag_pred_pad[: n_micro[bi]] = diag_micro_1d
                diag_pred_rows.append(diag_pred_pad)

            diag_pred_micro = torch.stack(diag_pred_rows, dim=0)
            actor_diag = self.reward_worker.compute_jepo_trajectory_rewards(
                expert_views_base,
                diag_pred_micro,
                base_masks,
                gt_micro_actions=gt_micro_base,
                chunk_actions=a,
                rollout_n=1,
            )
            reward_out.update(
                {
                    "diag/actor_small_sigma_scale": diag_sigma_scale,
                    "diag/actor_terminal_cos_mean": actor_diag.get("reward/terminal_cos_mean"),
                    "diag/actor_terminal_cos_std": actor_diag.get("reward/terminal_cos_std"),
                    "diag/actor_terminal_cos_pos_mean": actor_diag.get("reward/terminal_cos_pos_mean"),
                    "diag/actor_terminal_success_rate": actor_diag.get("reward/terminal_success_rate"),
                }
            )
        token_ordered = reward_out["token_level_rewards"].detach().cpu().float()
        if token_ordered.shape != response_mask_bt.shape:
            raise RuntimeError(f"reward shape {tuple(token_ordered.shape)} != response_mask {tuple(response_mask_bt.shape)}")

        gid = JEPORayTrainer._build_jepo_group_index(base_b, repeat_n)
        self._assert_finite("token_level_rewards", token_ordered)
        advantages, returns = self._compute_advantage(token_ordered, gid, response_mask=response_mask_bt)
        if advantages.shape != logprob_ordered.shape:
            raise RuntimeError(f"advantages vs log_probs shape mismatch: {advantages.shape}, {logprob_ordered.shape}")

        min_rs_traj = float(self.config.trainer.get("min_reward_std_trajectory", 0.0))
        if min_rs_traj > 0.0 and float(token_ordered.std(unbiased=False)) <= min_rs_traj:
            _log(
                f"[jepo] warning: token reward std low {float(token_ordered.std(unbiased=False)):.6e} "
                f"<= {min_rs_traj:.6e}"
            )

        chained_chains = torch.cat(chain_slices_ordered, dim=0)
        old_log_probs_chunkwise = torch.cat(compact_old_log_probs, dim=0)
        update_metrics = self.actor_worker.update_actor_variable_trajectory_chunks(
            row_chunk_counts=row_chunk_counts,
            advantages=advantages,
            flat_chunk_examples=flat_chunk_examples_ordered,
            chains=chained_chains,
            old_log_probs=old_log_probs_chunkwise,
            response_mask=response_mask_bt,
        )

        rollout_meta = {
            "predicted_micro_reference": predicted_micro.detach().cpu(),
            "micro_tokens": float(sum(n_micro) * d / max(1, base_b)),
            "padded_response_tokens": float(padded_tokens),
            "s_chunks": float(sum(s_chunks) / max(1, base_b)),
            "s_chunks_max": float(s_max),
            "chunks_total": float(sum(row_chunk_counts)),
            "n_micro": float(sum(n_micro) / max(1, base_b)),
            "n_micro_min": float(min(n_micro)),
            "n_micro_max": float(max(n_micro)),
            "pad_tail": float(sum(int(meta.get("pad_tail", 0)) for meta in metas) / max(1, base_b)),
            "base_trajectories": float(base_b),
            "rollout_rows": float(base_b * repeat_n),
        }
        return (
            token_ordered,
            advantages,
            returns,
            chained_chains,
            logprob_ordered,
            update_metrics,
            reward_out,
            rollout_meta,
        )


    def fit(self):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(line_buffering=True)
            except Exception:
                pass
        data_iter = itertools.cycle(self.rollout_cycle)
        try:
            for global_step in range(self.start_step, self.total_steps + 1):
                raw_batch = next(data_iter)

                if global_step == self.start_step and not self._monitor_preamble_logged:
                    for line in _format_static_config_report(self.config, self.use_trajectory_rollout):
                        _log(line)
                    self._wandb_log_static_summary_once()
                    self._monitor_preamble_logged = True

                if self.use_trajectory_rollout:
                    traj_batch = raw_batch
                    if not isinstance(traj_batch, list):
                        raise TypeError("trajectory dataloader must yield list[dict]")
                    (
                        token_level_rewards,
                        advantages,
                        returns,
                        chained_chains,
                        old_log_probs,
                        update_metrics,
                        reward_out,
                        rollout_aux,
                    ) = self._run_trajectory_training_step(traj_batch)
                    repeat_n = int(self.config.algorithm.rollout_n)
                    b_mon = len(traj_batch)
                    tok_st = _tensor_scalar_stats(token_level_rewards.float(), "reward/token_level_tensor")
                    adv_st = _tensor_scalar_stats(advantages.float(), "advantage")
                    adv_ex = _tensor_advanced_stats_rowwise(advantages.float(), "advantage")
                    old_lp_st = _tensor_scalar_stats(old_log_probs.float(), "rollout/old_log_prob")
                    _log_trajectory_dynamic_step(
                        global_step,
                        b_sz=b_mon,
                        repeat_n=repeat_n,
                        reward_out=reward_out,
                        update_metrics=update_metrics,
                        rollout_aux=rollout_aux,
                        token_stats=tok_st,
                        advantage_extra=adv_ex,
                        advantage_stats=adv_st,
                        old_log_prob_stats=old_lp_st,
                    )
                    if global_step % self.log_interval == 0:
                        _log(
                            f"[step {global_step} trajectory recap] chunks≈{float(rollout_aux.get('s_chunks', 0)):.1f} "
                            f"n_micro≈{float(rollout_aux.get('micro_tokens', 0)):.1f} "
                            f"(见上方 [monitor step …] 每步完整曲线)"
                        )
                else:
                    examples = self.bridge.normalize_examples(raw_batch)
                    repeat_n = int(self.config.algorithm.rollout_n)
                    repeated_examples = self._repeat_examples(examples, repeat_n)

                    noisy_dict = self.actor_worker.sample_noisy_actions(repeated_examples)
                    rollout = self.actor_worker.generate_actions(repeated_examples, noise=noisy_dict["noise"])
                    old_log_probs = self.actor_worker.compute_log_prob(repeated_examples, rollout["x_chain"])
                    if old_log_probs.shape[0] != len(repeated_examples):
                        raise RuntimeError(
                            f"old_log_probs batch mismatch: {old_log_probs.shape[0]} vs {len(repeated_examples)}"
                        )

                    reward_out = self.reward_worker.compute_rewards(repeated_examples, rollout["predicted_actions"])
                    token_level_rewards = reward_out["token_level_rewards"]
                    self._assert_finite("token_level_rewards", token_level_rewards)
                    group_index = self._build_group_index(len(examples), repeat_n)
                    advantages, returns = self._compute_advantage(token_level_rewards, group_index)
                    if advantages.shape != old_log_probs.shape:
                        raise RuntimeError(
                            f"advantages shape {advantages.shape} != old_log_probs shape {old_log_probs.shape}"
                        )

                    min_reward_std = float(self.config.trainer.get("min_reward_std", 1e-6))
                    if reward_out["reward_std"] <= min_reward_std:
                        raise RuntimeError(
                            f"Reward std too small ({reward_out['reward_std']:.6e}) <= min_reward_std ({min_reward_std:.6e})."
                        )

                    update_metrics = self.actor_worker.update_actor(
                        {
                            "examples": repeated_examples,
                            "x_chain": rollout["x_chain"],
                            "old_log_probs": old_log_probs,
                            "advantages": advantages,
                        }
                    )

                for key in ("actor/loss", "actor/ppo_kl", "actor/grad_norm"):
                    self._assert_finite(key, torch.tensor(update_metrics[key]))

                if global_step % self.save_interval == 0 or global_step == self.total_steps:
                    ckpt = self.actor_worker.save_checkpoint(str(self.output_dir), global_step)
                    _log(f"[step {global_step}] saved checkpoint: {ckpt}")

                wb: dict = {}
                if self.use_trajectory_rollout:
                    reward_stats = _tensor_scalar_stats(token_level_rewards.float(), "reward/token_level_tensor")
                    reward_row_stats = _tensor_advanced_stats_rowwise(token_level_rewards.float(), "reward/token_level_tensor")
                    adv_stats = _tensor_scalar_stats(advantages.float(), "advantage")
                    adv_row_stats = _tensor_advanced_stats_rowwise(advantages.float(), "advantage")
                    ret_stats = _tensor_scalar_stats(returns.float(), "returns")
                    old_lp_stats = _tensor_scalar_stats(old_log_probs.float(), "rollout/old_log_prob")
                    repeat_stats = _grpo_repeat_dispersion(
                        token_level_rewards,
                        advantages,
                        returns,
                        len(traj_batch),
                        int(self.config.algorithm.rollout_n),
                    )

                    focused = {
                        "terminal/cos_mean": reward_out.get("reward/terminal_cos_mean"),
                        "terminal/cos_std": reward_out.get("reward/terminal_cos_std"),
                        "terminal/cos_pos_mean": reward_out.get("reward/terminal_cos_pos_mean"),
                        "terminal/success_rate": reward_out.get("reward/terminal_success_rate"),
                        "terminal/cos_threshold": reward_out.get("reward/terminal_cos_threshold"),
                        "diagnostic/gt_terminal_cos_mean": reward_out.get("diag/gt_terminal_cos_mean"),
                        "diagnostic/gt_terminal_cos_std": reward_out.get("diag/gt_terminal_cos_std"),
                        "diagnostic/gt_terminal_cos_pos_mean": reward_out.get("diag/gt_terminal_cos_pos_mean"),
                        "diagnostic/gt_terminal_success_rate": reward_out.get("diag/gt_terminal_success_rate"),
                        "diagnostic/actor_small_sigma_scale": reward_out.get("diag/actor_small_sigma_scale"),
                        "diagnostic/actor_terminal_cos_mean": reward_out.get("diag/actor_terminal_cos_mean"),
                        "diagnostic/actor_terminal_cos_std": reward_out.get("diag/actor_terminal_cos_std"),
                        "diagnostic/actor_terminal_cos_pos_mean": reward_out.get("diag/actor_terminal_cos_pos_mean"),
                        "diagnostic/actor_terminal_success_rate": reward_out.get("diag/actor_terminal_success_rate"),
                        "reward/token_mean": reward_stats.get("reward/token_level_tensor/mean"),
                        "reward/token_std": reward_stats.get("reward/token_level_tensor/std"),
                        "reward/token_absmax": reward_stats.get("reward/token_level_tensor/absmax"),
                        "reward/row_mean_std_across_batch": reward_row_stats.get("reward/token_level_tensor/row_mean/std_across_batch"),
                        "advantage/mean": adv_stats.get("advantage/mean"),
                        "advantage/std": adv_stats.get("advantage/std"),
                        "advantage/absmax": adv_stats.get("advantage/absmax"),
                        "advantage/row_mean_std_across_batch": adv_row_stats.get("advantage/row_mean/std_across_batch"),
                        "advantage/repeat_rowmean_std": repeat_stats.get("monitor/grpo_across_repeat/adv_rowmean_std_mean"),
                        "returns/mean": ret_stats.get("returns/mean"),
                        "returns/std": ret_stats.get("returns/std"),
                        "policy/loss": update_metrics.get("actor/loss"),
                        "policy/pg_loss": update_metrics.get("actor/pg_loss"),
                        "policy/kl": update_metrics.get("actor/ppo_kl"),
                        "policy/clipfrac_hi": update_metrics.get("actor/pg_clipfrac"),
                        "policy/clipfrac_lo": update_metrics.get("actor/pg_clipfrac_lower"),
                        "policy/ratio_mean": update_metrics.get("actor/ratio_mean"),
                        "policy/ratio_max": update_metrics.get("actor/ratio_max"),
                        "policy/entropy": update_metrics.get("actor/entropy"),
                        "policy/grad_norm": update_metrics.get("actor/grad_norm"),
                        "generation/batch": float(len(traj_batch)),
                        "generation/rollout_n": float(int(self.config.algorithm.rollout_n)),
                        "generation/s_chunks": rollout_aux.get("s_chunks"),
                        "generation/micro_tokens": rollout_aux.get("micro_tokens"),
                        "rollout/old_log_prob_mean": old_lp_stats.get("rollout/old_log_prob/mean"),
                        "rollout/old_log_prob_std": old_lp_stats.get("rollout/old_log_prob/std"),
                        "meta/trajectory_mode": 1.0,
                    }
                    for key, value in focused.items():
                        try:
                            wb[key] = float(value)
                        except (TypeError, ValueError):
                            pass
                    wb["meta/trajectory_mode"] = 1.0
                    wb["meta/repeated_rollout_trajectories"] = float(repeat_n * len(traj_batch))
                    if bool(self.config.trainer.get("wandb_histograms", True)):
                        wb.update(_histogram_payload("hist/token_level_reward", token_level_rewards))
                        wb.update(_histogram_payload("hist/advantage", advantages))
                        wb.update(_histogram_payload("hist/returns", returns))
                else:
                    for rk, rv in reward_out.items():
                        if rk == "token_level_rewards" or isinstance(rv, torch.Tensor):
                            continue
                        try:
                            wb[str(rk).replace("/", "__")] = float(rv)
                        except (TypeError, ValueError):
                            pass
                    wb.update(update_metrics)
                    wb.update(_tensor_scalar_stats(token_level_rewards.float(), "reward/token_level_tensor"))
                    wb.update(_tensor_advanced_stats_rowwise(token_level_rewards.float(), "monitor/tlr"))
                    wb.update(_tensor_scalar_stats(advantages.float(), "advantage"))
                    wb.update(_tensor_advanced_stats_rowwise(advantages.float(), "monitor/adv"))
                    wb.update(_tensor_scalar_stats(returns.float(), "returns"))
                    wb.update(_tensor_advanced_stats_rowwise(returns.float(), "monitor/ret"))
                    wb.update(_tensor_scalar_stats(old_log_probs.float(), "rollout/old_log_prob"))
                    pa = rollout["predicted_actions"]
                    wb["meta/base_batch"] = float(len(examples))
                    wb["meta/repeated_batch"] = float(len(repeated_examples))
                    wb["meta/rollout_n"] = float(repeat_n)
                    wb["meta/action_horizon"] = float(pa.shape[1])
                    wb["meta/action_dim"] = float(pa.shape[-1])
                    wb.update(_tensor_scalar_stats(pa.float(), "rollout/predicted_actions"))
                    if bool(self.config.trainer.get("wandb_histograms", True)):
                        wb.update(_histogram_payload("hist/token_level_reward", token_level_rewards))
                        wb.update(_histogram_payload("hist/advantage", advantages))
                    _log(
                        f"[monitor step {global_step} non-trajectory] B={len(examples)} "
                        f"rew_mean={reward_out.get('reward_mean', float('nan')):.6f} rew_std={reward_out.get('reward_std', float('nan')):.6f}"
                    )
                    m = update_metrics
                    _log(
                        f"[monitor loss] total={m.get('actor/loss', float('nan')):.5f} "
                        f"avg/chunk={m.get('actor/loss_avg_per_chunk', float('nan')):.5f} "
                        f"pg={m.get('actor/pg_loss', float('nan')):.5f} "
                        f"entropy={m.get('actor/entropy', float('nan')):.5f} "
                        f"ent_coef×H={m.get('actor/entropy_coeff_times_entropy', float('nan')):.5f}"
                    )
                    _log(
                        f"[monitor policy] kl={m.get('actor/ppo_kl', float('nan')):.5f} "
                        f"clip_hi={m.get('actor/pg_clipfrac', float('nan')):.4f} clip_lo={m.get('actor/pg_clipfrac_lower', float('nan')):.4f} "
                        f"ratio_m={m.get('actor/ratio_mean', float('nan')):.4f} ratio_max={m.get('actor/ratio_max', float('nan')):.4f} "
                        f"ratio_raw_max={m.get('actor/ratio_raw_max', float('nan')):.4f} |grad|={m.get('actor/grad_norm', float('nan')):.5f}"
                    )
                wb["trainer/global_step"] = float(global_step)
                self._wandb_log(global_step, wb)
                self._append_metrics_jsonl(global_step, wb)
        finally:
            self._close_metrics_jsonl()
            self._wandb_finish()
