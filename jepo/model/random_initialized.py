# Copyright 2025 wmrl — utility helpers for randomly initialized policies.
"""
随机初始化 QwenPI：骨干拓扑与本地 Qwen3.5-0.8B 的 ``config.json`` 一致（权重非预训练）。

维度摘要（来源 ``starVLA/playground/Pretrained_models/Qwen3.5-0.8B/config.json``）::

    text_config.hidden_size = 1024
    text_config.num_hidden_layers = 24
    text_config.intermediate_size = 3584
    vision_config.depth = 12, hidden_size = 768, out_hidden_size = 1024

Action head 默认与 LIBERO QwenPI YAML（ ``cross_attention_dim=1024`` 、DiT ``num_layers=24`` ）对齐。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModel, AutoProcessor

from starVLA.model.modules.vlm.QWen3_5 import (
    _ACTION_TOKEN_MAX,
    _ACTION_TOKEN_MIN,
    _QWen3_5_VL_Interface,
)
from jepo.model.QwenPI import Qwen_PI

import starVLA.model.modules.vlm as vlm_mod

# 本地 Qwen3.5-0.8B 目录（用于读取架构 config + tokenizer/processor）
DEFAULT_BASE_VLM = "/project/peilab/qjl/2026/starVLA/playground/Pretrained_models/Qwen3.5-0.8B"

# 与训练过的 LIBERO QwenPI 同一套 framework/action 形状（可按需替换为你的 YAML）
DEFAULT_FRAMEWORK_CFG_YAML = (
    "/project/peilab/qjl/2026/starVLA/playground/Checkpoints/libero_qwen35_08b_PI_50/config.yaml"
)


class RandomQwen35VLInterface(_QWen3_5_VL_Interface):
    """
    与 ``_QWen3_5_VL_Interface`` 相同接口；底层 ``Qwen3_5`` 从 ``AutoConfig`` 仅架构实例化，
    ``AutoProcessor`` 仍从目录加载（词表与预处理与官方一致，非随机）。
    """

    def __init__(
        self,
        config,
        *,
        base_vlm_path: str = DEFAULT_BASE_VLM,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        torch.nn.Module.__init__(self)

        qwenvl_config = config.framework.get("qwenvl", {})
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                attn_implementation = "sdpa"

        hf_cfg = AutoConfig.from_pretrained(base_vlm_path, trust_remote_code=True)

        # 与 starVLA QwenPI + transformers 对齐：可用时可指定注意力实现
        for cfg_obj in (hf_cfg, getattr(hf_cfg, "text_config", None)):
            if cfg_obj is not None and hasattr(cfg_obj, "attn_implementation"):
                setattr(cfg_obj, "attn_implementation", attn_implementation)

        # 架构一致、权重随机初始化（不加载 ``model.safetensors``）
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            self.model = Qwen3_5ForConditionalGeneration(hf_cfg)
        except Exception:
            self.model = AutoModel.from_config(hf_cfg, trust_remote_code=True)

        self.model = self.model.to(dtype=dtype)

        self.processor = AutoProcessor.from_pretrained(base_vlm_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

        self.config = config

        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        mid = qwenvl_config.get("base_vlm", "")
        if isinstance(mid, str) and "-Action" in mid:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX


@contextlib.contextmanager
def _patch_get_vlm(factory):
    prev = vlm_mod.get_vlm_model
    vlm_mod.get_vlm_model = factory
    try:
        yield
    finally:
        vlm_mod.get_vlm_model = prev


def build_random_qwen_pi(
    *,
    cfg: Optional[OmegaConf] = None,
    framework_cfg_yaml: Optional[str | Path] = None,
    base_vlm_path: str = DEFAULT_BASE_VLM,
    seed: Optional[int] = None,
) -> Qwen_PI:
    """
    构造与 ``Qwen_PI`` 相同结构、骨干与 Qwen3.5-0.8B 目录中 ``config.json`` 一致、权重随机的模型。

    Args:
        cfg: 完整 OmegaConf；若省略则 ``framework_cfg_yaml`` 必须可 ``OmegaConf.load``。
        framework_cfg_yaml: 含 ``framework`` / ``datasets`` 的 YAML（默认同 LIBERO checkpoint 配置）。
        base_vlm_path: 指向含 ``config.json`` 与 processor 的 Qwen3.5-0.8B 目录。
        seed: 固定随机种子（可选）。
    """
    if cfg is None:
        path = Path(framework_cfg_yaml or DEFAULT_FRAMEWORK_CFG_YAML)
        cfg = OmegaConf.load(str(path))

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _factory(_merged_cfg):
        return RandomQwen35VLInterface(_merged_cfg, base_vlm_path=base_vlm_path)

    with _patch_get_vlm(_factory):
        model = Qwen_PI(cfg)

    return model


__all__ = [
    "DEFAULT_BASE_VLM",
    "DEFAULT_FRAMEWORK_CFG_YAML",
    "RandomQwen35VLInterface",
    "build_random_qwen_pi",
]
