"""Attention pooling contracts for masked trajectories."""

import pytest
import torch

from src.agents.rl.models.nn_modules import AttentionTrajectoryEncoder


def test_padding_is_ignored_even_when_nonfinite() -> None:
    encoder = AttentionTrajectoryEncoder(num_layers=2)
    samples = torch.randn(2, 3, 4)
    times = torch.tensor([[0.0, 0.1, 0.2]]).expand(2, -1).clone()
    expected = encoder(samples, times)
    padded = torch.cat((samples, torch.full((2, 2, 4), float("nan"))), dim=1)
    padded_times = torch.cat((times, torch.full((2, 2), float("nan"))), dim=1)
    mask = torch.tensor([[False, False, False, True, True]]).expand(2, -1)
    actual = encoder(padded, padded_times, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual[:, 0].sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters()
    )


def test_encoder_rejects_empty_or_invalid_samples() -> None:
    encoder = AttentionTrajectoryEncoder()
    samples = torch.zeros(2, 3, 4)
    times = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="valid point"):
        encoder(samples, times, torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        encoder(samples, times, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="timestamps"):
        encoder(samples, times[:, :2])
    samples[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        encoder(samples, times)


def test_encoder_requires_transformer_layers() -> None:
    with pytest.raises(ValueError, match="positive"):
        AttentionTrajectoryEncoder(num_layers=0)
