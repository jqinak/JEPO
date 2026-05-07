from __future__ import annotations

import torch

from jepo.workers.lewm_reward_worker import (
    compute_dense_milestone_reward,
    compute_jepo_reward,
    compute_sparse_milestone_reward,
    compute_terminal_reward,
)


B = 8
S = 5
ACTION_HORIZON = 8
ACTION_DIM = 7
D = 256
T_MAX = S * ACTION_HORIZON
L = T_MAX * ACTION_DIM


def _mask(lengths: list[int]) -> torch.Tensor:
    response_mask = torch.zeros(B, L, dtype=torch.float32)
    for b, n_micro in enumerate(lengths):
        response_mask[b, : n_micro * ACTION_DIM] = 1.0
    return response_mask


def _embs(lengths: list[int], terminal_signs: list[float] | None = None, *, gt_extra: int = 1):
    pred = torch.zeros(B, T_MAX, D, dtype=torch.float32)
    gt = torch.zeros(B, T_MAX + gt_extra, D, dtype=torch.float32)
    pred[..., 0] = 1.0
    gt[..., 0] = 1.0
    if terminal_signs is not None:
        for b, sign in enumerate(terminal_signs):
            pred[b, lengths[b] - 1, 0] = float(sign)
    return pred, gt


def test_terminal_continuous_positive_reward_and_padding_zero():
    lengths = [T_MAX] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths, [1.0] * B)
    token_rewards = compute_terminal_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=False,
    )
    for b in range(B):
        valid_idx = response_mask[b].bool()
        assert token_rewards[b, valid_idx].sum() > 0
        assert token_rewards[b, ~valid_idx].sum() == 0


def test_terminal_continuous_negative_reward_is_kept():
    lengths = [T_MAX] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths, [-1.0] * B)
    token_rewards = compute_terminal_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=False,
    )
    assert token_rewards.sum() < 0
    assert torch.count_nonzero(token_rewards).item() == B * ACTION_DIM


def test_terminal_continuous_mixed_signs():
    lengths = [T_MAX] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths, [1.0] * 4 + [-1.0] * 4)
    token_rewards = compute_terminal_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=False,
    )
    row_sums = token_rewards.sum(dim=-1)
    assert (row_sums[:4] > 0).all()
    assert (row_sums[4:] < 0).all()


def test_variable_valid_length_terminal_positions():
    lengths = [40] * 4 + [37] * 4
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths, [1.0] * B)
    token_rewards = compute_terminal_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=False,
    )
    for b in range(4):
        start = 39 * ACTION_DIM
        assert torch.count_nonzero(token_rewards[b, start : start + ACTION_DIM]).item() == ACTION_DIM
    for b in range(4, B):
        start = 36 * ACTION_DIM
        assert torch.count_nonzero(token_rewards[b, start : start + ACTION_DIM]).item() == ACTION_DIM
        assert token_rewards[b, 37 * ACTION_DIM :].abs().sum() == 0
    assert token_rewards[response_mask == 0].abs().sum() == 0


def test_sparse_milestone_reward_count_equals_chunk_count():
    lengths = [40] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths)
    token_rewards = compute_sparse_milestone_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=False,
    )
    assert torch.count_nonzero(token_rewards[0]).item() == 5 * ACTION_DIM


def test_dense_milestone_reward_count_equals_valid_step_count():
    lengths = [37] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths)
    token_rewards = compute_dense_milestone_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=False,
    )
    assert torch.count_nonzero(token_rewards[0]).item() == 37 * ACTION_DIM


def test_dispatcher_routing():
    lengths = [40] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths)
    for reward_type in ("terminal", "sparse_milestone", "dense_milestone"):
        out = compute_jepo_reward(
            pred,
            gt,
            response_mask,
            {
                "reward_type": reward_type,
                "action_horizon": ACTION_HORIZON,
                "action_dim": ACTION_DIM,
                "gt_use_next_observation": True,
                "normalize_rewards": False,
            },
        )
        assert out.shape == (B, L)
        assert out.dtype == pred.dtype


def test_normalization_keeps_padding_zero():
    lengths = [40] * B
    response_mask = _mask(lengths)
    pred, gt = _embs(lengths, [1.0] * 4 + [-1.0] * 4)
    token_rewards = compute_terminal_reward(
        pred,
        gt,
        response_mask,
        ACTION_HORIZON,
        ACTION_DIM,
        gt_use_next_observation=True,
        normalize_rewards=True,
    )
    for b in range(B):
        vals = token_rewards[b, response_mask[b].bool()]
        assert torch.isclose(vals.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(vals.std(), torch.tensor(1.0), atol=1e-6)
    assert token_rewards[response_mask == 0].abs().sum() == 0
