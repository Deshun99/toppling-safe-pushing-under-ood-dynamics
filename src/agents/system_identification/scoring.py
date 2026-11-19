"""Shared observation window for trajectory-based system identification."""

import numpy as np
from numpy.typing import NDArray


def get_scoring_indices(
    n_samples: int,
    *,
    warmup_steps: int = 0,
    sample_stride: int = 1,
) -> NDArray[np.intp]:
    """Select aligned samples, masking contact warm-up after downsampling.

    Keep the original recording's sampling phase: stride 12 with 8 warm-up
    steps selects 12, 24, ... rather than 8, 20, .... Warm-up remains simulated
    and recorded, but its contact transients are never scored.
    Empty windows are rejected; shorten the stride to retain evidence.
    """
    if (
        not isinstance(sample_stride, int)
        or isinstance(sample_stride, bool)
        or sample_stride < 1
    ):
        raise ValueError("sample_stride must be a positive integer")
    if (
        not isinstance(warmup_steps, int)
        or warmup_steps < 0
        or warmup_steps >= n_samples
    ):
        raise ValueError("warmup_steps must leave at least one commanded-action sample")
    indices = np.arange(0, n_samples, sample_stride, dtype=np.intp)
    indices = indices[indices >= warmup_steps]
    if not len(indices):
        raise ValueError("No samples in scoring window; reduce sample_stride")
    return indices
