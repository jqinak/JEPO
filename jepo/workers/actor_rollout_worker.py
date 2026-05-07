from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image as PILImage
from torch.distributions import Normal

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

from starVLA.deployment.model_server.tools.image_tools import to_pil_preserve
from jepo.model.sigma_net import StarVLASigmaNet
from jepo.workers.tokenizer_bridge import TokenizerBridge


@dataclass
class EncodedContext:
    vl_embs_list: list[torch.Tensor]
    pooled_ctx: torch.Tensor
    state: torch.Tensor | None


class ActorRolloutWorker:
    """策略侧 worker：rollout、log_prob 计算、PPO 更新。"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")

        from jepo.model.base_framework import baseframework
        # self.model = baseframework.from_pretrained(config.paths.starvla_ckpt).to(self.device)
        
        if config.runtime.smoke_random_init:
            print("[smoke] actor uses random initialization (no ckpt load).")
            from jepo.model.base_framework import build_framework
            def _random_init_model_(model, std=0.02):
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if not p.is_floating_point():
                            continue
                        if p.ndim >= 2:
                            torch.nn.init.normal_(p, mean=0.0, std=std)  # 线性层/注意力权重
                        else:
                            torch.nn.init.zeros_(p)  # bias / norm bias
                    for n, b in model.named_buffers():
                        if torch.is_floating_point(b):
                            b.zero_()  # 例如 running stats
            framework_cfg = OmegaConf.load(str(config.data.starvla_cfg))
            self.model = build_framework(framework_cfg).to(self.device)
            _random_init_model_(self.model)
            print("[smoke] _random_init_model_ finished!")
        else:
            self.model = baseframework.from_pretrained(config.paths.starvla_ckpt).to(self.device)
        
        # self.model.train()
        self.bridge = TokenizerBridge()

        self.action_model = self.model.action_model
        self.action_dim = int(self.action_model.action_dim)
        self.action_horizon = int(self.action_model.action_horizon)
        self.num_timestep_buckets = int(self.action_model.num_timestep_buckets)
        self.num_flow_steps = int(config.algorithm.num_flow_steps)
        self.ppo_epochs = int(config.algorithm.get("ppo_epochs", 1))
        self.ppo_mini_batch_size = int(config.algorithm.get("ppo_mini_batch_size", 0))

        state_dim = int(getattr(self.action_model.config, "state_dim", self.action_dim))
        ctx_dim = int(self.action_model.input_embedding_dim)
        sigma_cfg = config.algorithm.get("sigma_net", {})
        self.sigma_net = StarVLASigmaNet(
            ctx_dim=ctx_dim,
            action_dim=self.action_dim,
            state_dim=state_dim,
            hidden_dim=int(sigma_cfg.get("hidden_dim", 1024)),
            num_layers=int(sigma_cfg.get("num_layers", 4)),
            num_heads=int(sigma_cfg.get("num_heads", 8)),
            min_std=float(sigma_cfg.get("min_std", 0.02)),
            max_std=float(sigma_cfg.get("max_std", 0.30)),
            dropout=float(sigma_cfg.get("dropout", 0.0)),
        ).to(self.device)

        for p in self.model.qwen_vl_interface.parameters():
            p.requires_grad = False

        trainable_groups = [
            {"params": [p for p in self.action_model.parameters() if p.requires_grad], "lr": float(config.algorithm.actor_lr)},
            {"params": [p for p in self.sigma_net.parameters() if p.requires_grad], "lr": float(config.algorithm.sigma_lr)},
        ]
        self.optimizer = torch.optim.AdamW(trainable_groups, betas=(0.9, 0.999), weight_decay=0.0)

    def _encode_examples(self, examples: list[dict], require_grad: bool) -> EncodedContext:
        batch_images = []
        for ex in examples:
            imgs = to_pil_preserve(ex["image"])
            # Qwen VLM build_qwenvl_inputs 要求每样本为「视角列表」：for img in imgs
            if isinstance(imgs, PILImage.Image):
                imgs = [imgs]
            batch_images.append(imgs)
        instructions = [ex["lang"] for ex in examples]
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)
        self.model.qwen_vl_interface.eval()
        if require_grad:
            with autocast_ctx:
                qwen_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
                outputs = self.model.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        else:
            with torch.no_grad():
                with autocast_ctx:
                    qwen_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
                    outputs = self.model.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)

        expected_layers = len(self.action_model.model.transformer_blocks)
        vl_embs_list = list(outputs.hidden_states[-expected_layers:])
        # VLM 在 autocast 下常为 bf16；action DiT / diffusers Linear 多为 fp32，需对齐否则 matmul 报错
        vl_embs_list = [h.float() for h in vl_embs_list]
        pooled_ctx = vl_embs_list[-1].mean(dim=1)
        state = self.bridge.extract_states(examples, self.device)
        return EncodedContext(vl_embs_list=vl_embs_list, pooled_ctx=pooled_ctx, state=state)

    def _predict_velocity(self, ctx: EncodedContext, noisy_actions: torch.Tensor, t_scalar: float) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        t_scalar = max(0.0, min(1.0, float(t_scalar)))
        t_discrete = int(t_scalar * (self.num_timestep_buckets - 1))
        timesteps = torch.full((batch_size,), t_discrete, device=self.device, dtype=torch.long)

        action_features = self.action_model.action_encoder(noisy_actions, timesteps)
        if self.action_model.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device)
            pos_embs = self.action_model.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        state_features = self.action_model.state_encoder(ctx.state) if ctx.state is not None else None
        future_tokens = self.action_model.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        temb = self.action_model.model.timestep_encoder(timesteps)
        model_output = sa_embs
        for layer_idx, layer in enumerate(self.action_model.model.transformer_blocks):
            model_output = layer(hidden_states=model_output, encoder_hidden_states=ctx.vl_embs_list[layer_idx], temb=temb)
        pred = self.action_model.action_decoder(model_output)
        return pred[:, -self.action_horizon :]

    def sample_noisy_actions(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        gt_actions = self.bridge.extract_actions(examples, self.device, self.action_horizon)
        batch_size = gt_actions.shape[0]
        noise = torch.randn_like(gt_actions)
        t = torch.rand(batch_size, device=self.device, dtype=gt_actions.dtype).view(batch_size, 1, 1)
        noisy_actions = (1 - t) * noise + t * gt_actions
        flow = gt_actions - noise
        return {"noise": noise.detach().cpu(), "flow": flow.detach().cpu(), "gt_noisy_actions": noisy_actions.detach().cpu(), "gt_timestep_embeddings": t.detach().cpu()}

    def sample_noise_for_chunks(self, chunk_examples_flat: list[dict]) -> torch.Tensor:
        """Independent Gaussian noises per chunk-row (training-time exploration), shape [N, horizon, dim]."""
        return torch.randn(
            len(chunk_examples_flat), self.action_horizon, self.action_dim, dtype=torch.float32
        ).cpu()

    def generate_actions(
        self,
        examples: list[dict],
        noise: torch.Tensor | None = None,
        *,
        sigma_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        examples = self.bridge.normalize_examples(examples)
        ctx = self._encode_examples(examples, require_grad=False)
        batch_size = len(examples)
        if noise is None:
            noise = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device, dtype=ctx.vl_embs_list[-1].dtype)
        else:
            noise = noise.to(self.device)

        k_steps = self.num_flow_steps
        dt = -1.0 / k_steps
        time = 1.0
        x_chain = torch.empty(batch_size, k_steps + 1, self.action_horizon, self.action_dim, device=self.device, dtype=noise.dtype)
        x_chain[:, 0] = noise
        curr = noise
        with torch.no_grad():
            for k in range(k_steps):
                t_scalar = 1.0 - time
                flow = self._predict_velocity(ctx, curr, t_scalar)
                mean_next = curr + dt * flow
                std, _ = self.sigma_net(pooled_ctx=ctx.pooled_ctx, noisy_actions=curr, t_scalar=t_scalar, state=ctx.state)
                if sigma_scale <= 0.0:
                    curr = mean_next.to(curr.dtype)
                else:
                    dist = Normal(mean_next.float(), (std.float() * float(sigma_scale)).clamp_min(1e-6))
                    curr = dist.sample().to(curr.dtype)
                x_chain[:, k + 1] = curr
                time += dt
        return {"predicted_actions": curr.detach().cpu(), "x_chain": x_chain.detach().cpu(), "noise": noise.detach().cpu()}

    def _compute_log_prob(self, examples: list[dict], x_chain: torch.Tensor, return_entropy: bool, require_grad: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        examples = self.bridge.normalize_examples(examples)
        ctx = self._encode_examples(examples, require_grad=require_grad)
        chain = x_chain.to(self.device)
        bsz, kp1, horizon, action_dim = chain.shape
        k_steps = kp1 - 1
        dt = -1.0 / k_steps

        logp = torch.zeros(bsz, horizon, action_dim, device=self.device, dtype=torch.float32)
        entropy = torch.zeros_like(logp) if return_entropy else None
        const_term = 0.5 * (math.log(2.0 * math.pi) + 1.0)
        for k in range(k_steps):
            xk = chain[:, k]
            xk1 = chain[:, k + 1]
            t_scalar = k / k_steps
            flow = self._predict_velocity(ctx, xk, t_scalar)
            mean_next = xk + dt * flow
            std, log_std = self.sigma_net(pooled_ctx=ctx.pooled_ctx, noisy_actions=xk, t_scalar=t_scalar, state=ctx.state)
            dist = Normal(mean_next.float(), std.float().clamp_min(1e-6))
            logp += dist.log_prob(xk1.float())
            if return_entropy and entropy is not None:
                entropy += log_std.float() + const_term

        logp_vec = logp.reshape(bsz, horizon * action_dim)
        ent_vec = entropy.reshape(bsz, horizon * action_dim) if entropy is not None else None
        return logp_vec, ent_vec

    def compute_log_prob(self, examples: list[dict], x_chain: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logp, _ = self._compute_log_prob(examples, x_chain, return_entropy=False, require_grad=False)
        return logp.detach().cpu()

    def _update_one_minibatch(self, batch: dict[str, Any]) -> dict[str, float]:
        examples = batch["examples"]
        x_chain = batch["x_chain"]
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)

        self.action_model.train()
        self.sigma_net.train()
        self.optimizer.zero_grad(set_to_none=True)
        new_log_probs, entropy = self._compute_log_prob(examples, x_chain=x_chain, return_entropy=True, require_grad=True)
        assert entropy is not None
        log_ratio = (new_log_probs - old_log_probs).float()
        max_ratio_guard = float(self.config.algorithm.get("max_ratio_guard", 20.0))
        # 旧写法 exp(clamp(log_ratio, ±20)) 最大约 exp(20)≈5e8，与 max_ratio_guard=20（线性比值）语义冲突。
        # 诊断用 ratio 截断到与 guard 一致：|log_ratio| ≤ ln(max_ratio_guard)。
        log_cap = math.log(max_ratio_guard + 1e-12)
        ratio = torch.exp(torch.clamp(log_ratio, min=-log_cap, max=log_cap))
        response_mask = torch.ones_like(advantages, device=self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = _get_core_algos().compute_policy_loss(
                old_log_prob=old_log_probs,
                log_prob=new_log_probs,
                advantages=advantages,
                response_mask=response_mask,
                cliprange=float(self.config.algorithm.clip_ratio),
                cliprange_low=float(self.config.algorithm.clip_ratio),
                cliprange_high=float(self.config.algorithm.clip_ratio),
                clip_ratio_c=3.0,
            )
        entropy_loss = _get_core_algos().agg_loss(entropy, response_mask, loss_agg_mode="token-mean")
        ec = float(self.config.algorithm.get("entropy_coeff", 0.0))
        total_loss = pg_loss - ec * entropy_loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.action_model.parameters()) + list(self.sigma_net.parameters()),
            max_norm=float(self.config.algorithm.max_grad_norm),
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError("actor total_loss contains NaN/Inf.")
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("actor grad_norm contains NaN/Inf.")
        target_kl = float(self.config.algorithm.get("target_kl", 0.0))
         
        if bool(self.config.algorithm.get("kl_stop_on_exceed", False)) and target_kl > 0.0 and float(ppo_kl) > target_kl and not bool(getattr(self.config.runtime, "smoke_random_init", False)):
                raise RuntimeError(f"PPO KL divergence too large: {float(ppo_kl):.6f} > target_kl={target_kl:.6f}")
        ratio_raw_max = float(torch.exp(torch.clamp(log_ratio, min=-50.0, max=50.0)).max().detach().cpu())
        # if ratio_raw_max > max_ratio_guard and not bool(
        #     getattr(self.config.runtime, "smoke_random_init", False)
        # ):
        #     raise RuntimeError(
        #         f"PPO ratio max too large: {ratio_raw_max:.4f} > max_ratio_guard={max_ratio_guard:.4f}"
        #     )
        ratio_max = ratio.max()
        return {
            "actor/loss": float(total_loss.detach().cpu()),
            "actor/loss_avg_per_chunk": float(total_loss.detach().cpu()),
            "actor/pg_loss": float(pg_loss.detach().cpu()),
            "actor/pg_loss_sum_over_chunks": float(pg_loss.detach().cpu()),
            "actor/entropy": float(entropy_loss.detach().cpu()),
            "actor/entropy_coeff_times_entropy": float(ec * entropy_loss.detach().cpu()),
            "actor/ppo_kl": float(ppo_kl.detach().cpu()),
            "actor/ratio_mean": float(ratio.mean().detach().cpu()),
            "actor/ratio_max": float(ratio_max.detach().cpu()),
            "actor/ratio_raw_max": float(ratio_raw_max),
            "actor/pg_clipfrac": float(pg_clipfrac.detach().cpu()),
            "actor/pg_clipfrac_lower": float(pg_clipfrac_lower.detach().cpu()),
            "actor/grad_norm": float(grad_norm.detach().cpu()),
        }

    def update_actor(self, batch: dict[str, Any]) -> dict[str, float]:
        examples: list[dict] = batch["examples"]
        x_chain = batch["x_chain"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]

        total_bsz = len(examples)
        mini = self.ppo_mini_batch_size if self.ppo_mini_batch_size > 0 else total_bsz
        if total_bsz % mini != 0:
            raise ValueError(f"Batch size {total_bsz} must be divisible by ppo_mini_batch_size={mini}")

        metrics_accum = []
        for _ in range(self.ppo_epochs):
            for start in range(0, total_bsz, mini):
                end = start + mini
                self.optimizer.zero_grad(set_to_none=True)
                mini_batch = {
                    "examples": examples[start:end],
                    "x_chain": x_chain[start:end],
                    "old_log_probs": old_log_probs[start:end],
                    "advantages": advantages[start:end],
                }
                metric = self._update_one_minibatch(mini_batch)
                self.optimizer.step()
                metrics_accum.append(metric)

        out = {}
        for k in metrics_accum[0].keys():
            out[k] = float(sum(m[k] for m in metrics_accum) / len(metrics_accum))
        return out

    def generate_actions_chunk_flat(
        self,
        flat_chunk_examples: list[dict],
        noise: torch.Tensor,
        *,
        sigma_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        return self.generate_actions(flat_chunk_examples, noise=noise, sigma_scale=sigma_scale)

    def update_actor_trajectory_chunks(
        self,
        *,
        s_chunks: int,
        advantages: torch.Tensor,
        flat_chunk_examples: list[dict],
        chains: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Sum PPO surrogate over all chunks per trajectory, single backward/step per traj row."""
        bt = advantages.shape[0]
        total_chunks = chains.shape[0]
        if total_chunks != bt * int(s_chunks):
            raise RuntimeError(f"chunks {total_chunks} != batch {bt} * s_chunks {s_chunks}")
        if old_log_probs.shape[0] != total_chunks:
            raise RuntimeError("old_log_probs chunk dim mismatch.")
        per = int(self.action_horizon * self.action_dim)
        row_metrics: list[dict[str, float]] = []

        mask_bt = response_mask
        if mask_bt is not None and mask_bt.shape != advantages.shape:
            raise ValueError("response_mask must match advantages shape in trajectory PPO update.")

        for bi in range(bt):
            self.optimizer.zero_grad(set_to_none=True)
            self.action_model.train()
            self.sigma_net.train()
            total_loss: torch.Tensor | None = None
            ec = float(self.config.algorithm.get("entropy_coeff", 0.0))
            chunk_pg: list[float] = []
            chunk_entropy_loss: list[float] = []
            chunk_kl: list[float] = []
            chunk_clip: list[float] = []
            chunk_clip_lo: list[float] = []
            chunk_ratio_mean: list[float] = []
            chunk_ratio_max: list[float] = []
            chunk_ratio_raw_max: list[float] = []

            for j in range(int(s_chunks)):
                ci = bi * int(s_chunks) + j
                single_ex = [flat_chunk_examples[ci]]
                x_ch = chains[ci : ci + 1].to(self.device)
                ol = old_log_probs[ci : ci + 1].to(self.device)
                adv_slice = advantages[bi : bi + 1, j * per : (j + 1) * per].to(self.device)

                if mask_bt is not None:
                    resp_mask_slice = mask_bt[bi : bi + 1, j * per : (j + 1) * per].to(self.device)
                else:
                    resp_mask_slice = torch.ones_like(adv_slice)

                new_lp, entropy = self._compute_log_prob(single_ex, x_chain=x_ch, return_entropy=True, require_grad=True)
                assert entropy is not None
                log_ratio = (new_lp - ol).float()
                max_ratio_guard = float(self.config.algorithm.get("max_ratio_guard", 20.0))
                log_cap = math.log(max_ratio_guard + 1e-12)
                ratio = torch.exp(torch.clamp(log_ratio, min=-log_cap, max=log_cap))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = _get_core_algos().compute_policy_loss(
                        old_log_prob=ol,
                        log_prob=new_lp,
                        advantages=adv_slice,
                        response_mask=resp_mask_slice,
                        cliprange=float(self.config.algorithm.clip_ratio),
                        cliprange_low=float(self.config.algorithm.clip_ratio),
                        cliprange_high=float(self.config.algorithm.clip_ratio),
                        clip_ratio_c=3.0,
                    )
                entropy_loss = _get_core_algos().agg_loss(entropy, resp_mask_slice, loss_agg_mode="token-mean")
                chunk_loss = pg_loss - ec * entropy_loss
                chunk_pg.append(float(pg_loss.detach().cpu()))
                chunk_entropy_loss.append(float(entropy_loss.detach().cpu()))
                chunk_kl.append(float(ppo_kl.detach().cpu()))
                chunk_clip.append(float(pg_clipfrac.detach().cpu()))
                chunk_clip_lo.append(float(pg_clipfrac_lower.detach().cpu()))
                chunk_ratio_mean.append(float(ratio.mean().detach().cpu()))
                chunk_ratio_max.append(float(ratio.max().detach().cpu()))
                chunk_ratio_raw_max.append(
                    float(torch.exp(torch.clamp(log_ratio, min=-50.0, max=50.0)).max().detach().cpu())
                )

                if total_loss is None:
                    total_loss = chunk_loss
                else:
                    total_loss = total_loss + chunk_loss

            assert total_loss is not None
            n_c = len(chunk_pg)
            eff_ent = sum(chunk_entropy_loss) / max(1, n_c)
            eff_pg = sum(chunk_pg) / max(1, n_c)

            total_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                list(self.action_model.parameters()) + list(self.sigma_net.parameters()),
                max_norm=float(self.config.algorithm.max_grad_norm),
            )
            if not torch.isfinite(total_loss) or not torch.isfinite(gn):
                raise FloatingPointError("trajectory actor update non-finite loss/grad.")

            target_kl = float(self.config.algorithm.get("target_kl", 0.0))
            mean_kl = float(sum(chunk_kl) / max(1, n_c))
            if (
                bool(self.config.algorithm.get("kl_stop_on_exceed", False))
                and target_kl > 0.0
                and mean_kl > target_kl
                and not bool(getattr(self.config.runtime, "smoke_random_init", False))
            ):
                raise RuntimeError(f"PPO KL too large: {mean_kl:.6f} > {target_kl}")

            self.optimizer.step()
            row_metrics.append(
                {
                    # total_loss = sum_j ( pg_j - ec * entropy_j ) ，与单次 graph 反向一致
                    "actor/loss": float(total_loss.detach().cpu()),
                    "actor/loss_avg_per_chunk": float(total_loss.detach().cpu()) / max(1, n_c),
                    "actor/pg_loss": eff_pg,
                    "actor/pg_loss_sum_over_chunks": float(sum(chunk_pg)),
                    "actor/entropy": eff_ent,
                    "actor/entropy_coeff_times_entropy": ec * eff_ent,
                    "actor/ppo_kl": float(sum(chunk_kl) / max(1, n_c)),
                    "actor/pg_clipfrac": float(sum(chunk_clip) / max(1, n_c)),
                    "actor/pg_clipfrac_lower": float(sum(chunk_clip_lo) / max(1, n_c)),
                    "actor/ratio_mean": float(sum(chunk_ratio_mean) / max(1, n_c)),
                    "actor/ratio_max": float(sum(chunk_ratio_max) / max(1, n_c)),
                    "actor/ratio_raw_max": max(chunk_ratio_raw_max) if chunk_ratio_raw_max else 0.0,
                    "actor/grad_norm": float(gn.detach().cpu()),
                }
            )

        out_metrics: dict[str, float] = {}
        if row_metrics:
            for k in row_metrics[0].keys():
                out_metrics[k] = float(sum(row[k] for row in row_metrics) / len(row_metrics))
        return out_metrics

    def update_actor_variable_trajectory_chunks(
        self,
        *,
        row_chunk_counts: list[int],
        advantages: torch.Tensor,
        flat_chunk_examples: list[dict[str, Any]],
        chains: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, float]:
        bt = advantages.shape[0]
        if len(row_chunk_counts) != bt:
            raise RuntimeError(f"row_chunk_counts length {len(row_chunk_counts)} != batch rows {bt}")
        if response_mask.shape != advantages.shape:
            raise ValueError("response_mask must match advantages shape in JEPO trajectory PPO update.")
        expected_chunks = int(sum(int(x) for x in row_chunk_counts))
        if chains.shape[0] != expected_chunks:
            raise RuntimeError(f"chains rows {chains.shape[0]} != expected chunks {expected_chunks}")
        if old_log_probs.shape[0] != expected_chunks:
            raise RuntimeError(f"old_log_probs rows {old_log_probs.shape[0]} != expected chunks {expected_chunks}")
        if len(flat_chunk_examples) != expected_chunks:
            raise RuntimeError(f"flat examples {len(flat_chunk_examples)} != expected chunks {expected_chunks}")

        per = int(self.action_horizon * self.action_dim)
        row_metrics: list[dict[str, float]] = []
        cursor = 0

        for bi, s_chunks in enumerate(row_chunk_counts):
            self.optimizer.zero_grad(set_to_none=True)
            self.action_model.train()
            self.sigma_net.train()
            total_loss: torch.Tensor | None = None
            ec = float(self.config.algorithm.get("entropy_coeff", 0.0))
            chunk_pg: list[float] = []
            chunk_entropy_loss: list[float] = []
            chunk_kl: list[float] = []
            chunk_clip: list[float] = []
            chunk_clip_lo: list[float] = []
            chunk_ratio_mean: list[float] = []
            chunk_ratio_max: list[float] = []
            chunk_ratio_raw_max: list[float] = []

            for j in range(int(s_chunks)):
                ci = cursor + j
                single_ex = [flat_chunk_examples[ci]]
                x_ch = chains[ci : ci + 1].to(self.device)
                ol = old_log_probs[ci : ci + 1].to(self.device)
                adv_slice = advantages[bi : bi + 1, j * per : (j + 1) * per].to(self.device)
                resp_mask_slice = response_mask[bi : bi + 1, j * per : (j + 1) * per].to(self.device)

                new_lp, entropy = self._compute_log_prob(single_ex, x_chain=x_ch, return_entropy=True, require_grad=True)
                assert entropy is not None
                log_ratio = (new_lp - ol).float()
                max_ratio_guard = float(self.config.algorithm.get("max_ratio_guard", 20.0))
                log_cap = math.log(max_ratio_guard + 1e-12)
                ratio = torch.exp(torch.clamp(log_ratio, min=-log_cap, max=log_cap))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = _get_core_algos().compute_policy_loss(
                        old_log_prob=ol,
                        log_prob=new_lp,
                        advantages=adv_slice,
                        response_mask=resp_mask_slice,
                        cliprange=float(self.config.algorithm.clip_ratio),
                        cliprange_low=float(self.config.algorithm.clip_ratio),
                        cliprange_high=float(self.config.algorithm.clip_ratio),
                        clip_ratio_c=3.0,
                    )
                entropy_loss = _get_core_algos().agg_loss(entropy, resp_mask_slice, loss_agg_mode="token-mean")
                chunk_loss = pg_loss - ec * entropy_loss
                chunk_pg.append(float(pg_loss.detach().cpu()))
                chunk_entropy_loss.append(float(entropy_loss.detach().cpu()))
                chunk_kl.append(float(ppo_kl.detach().cpu()))
                chunk_clip.append(float(pg_clipfrac.detach().cpu()))
                chunk_clip_lo.append(float(pg_clipfrac_lower.detach().cpu()))
                chunk_ratio_mean.append(float(ratio.mean().detach().cpu()))
                chunk_ratio_max.append(float(ratio.max().detach().cpu()))
                chunk_ratio_raw_max.append(
                    float(torch.exp(torch.clamp(log_ratio, min=-50.0, max=50.0)).max().detach().cpu())
                )
                total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss

            cursor += int(s_chunks)
            assert total_loss is not None
            n_c = max(1, len(chunk_pg))
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.action_model.parameters()) + list(self.sigma_net.parameters()),
                max_norm=float(self.config.algorithm.max_grad_norm),
            )
            if not torch.isfinite(total_loss) or not torch.isfinite(grad_norm):
                raise FloatingPointError("JEPO trajectory actor update non-finite loss/grad.")

            target_kl = float(self.config.algorithm.get("target_kl", 0.0))
            mean_kl = float(sum(chunk_kl) / n_c)
            if (
                bool(self.config.algorithm.get("kl_stop_on_exceed", False))
                and target_kl > 0.0
                and mean_kl > target_kl
                and not bool(getattr(self.config.runtime, "smoke_random_init", False))
            ):
                raise RuntimeError(f"PPO KL too large: {mean_kl:.6f} > {target_kl}")

            self.optimizer.step()
            row_metrics.append(
                {
                    "actor/loss": float(total_loss.detach().cpu()),
                    "actor/loss_avg_per_chunk": float(total_loss.detach().cpu()) / n_c,
                    "actor/pg_loss": float(sum(chunk_pg) / n_c),
                    "actor/pg_loss_sum_over_chunks": float(sum(chunk_pg)),
                    "actor/entropy": float(sum(chunk_entropy_loss) / n_c),
                    "actor/entropy_coeff_times_entropy": ec * float(sum(chunk_entropy_loss) / n_c),
                    "actor/ppo_kl": mean_kl,
                    "actor/pg_clipfrac": float(sum(chunk_clip) / n_c),
                    "actor/pg_clipfrac_lower": float(sum(chunk_clip_lo) / n_c),
                    "actor/ratio_mean": float(sum(chunk_ratio_mean) / n_c),
                    "actor/ratio_max": float(sum(chunk_ratio_max) / n_c),
                    "actor/ratio_raw_max": max(chunk_ratio_raw_max) if chunk_ratio_raw_max else 0.0,
                    "actor/grad_norm": float(grad_norm.detach().cpu()),
                }
            )

        out_metrics: dict[str, float] = {}
        if row_metrics:
            for key in row_metrics[0].keys():
                out_metrics[key] = float(sum(row[key] for row in row_metrics) / len(row_metrics))
        return out_metrics


    def save_checkpoint(self, save_dir: str, step: int) -> str:
        out_dir = Path(save_dir) / f"global_step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "jepo_actor.pt"
        torch.save(
            {
                "action_model": self.action_model.state_dict(),
                "sigma_net": self.sigma_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        return str(ckpt_path)
