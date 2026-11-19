"""Learned world model for the 'learned' ablation condition.

Replaces InternalSim + SysID: predicts the next observed trajectory from the
current observation, action reference, candidate delta, and most-recent real
trajectory without running any PyBullet simulation.
"""

import numpy as np
import torch
from torch import nn

from src.agents.rl.models.nn_modules import MLP

_OBS_DIM = 7
_ACTION_DIM = 3
_TRAJ_FEATURES = 5  # Must match DenseTrajectory.NUM_FEATURES.


def interpolate_model_trajectory(
    trajectory: np.ndarray, points: int = 16
) -> np.ndarray:
    """Resample scaled policy features for the fixed-size learned world model.

    Attention retains every recorded sample; the model learns on evenly spaced
    timestamps. Interpolation commutes with the channel-wise physical scaling.
    """
    samples = np.asarray(trajectory, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != _TRAJ_FEATURES or not len(samples):
        raise ValueError("world model trajectory must have shape (N, 5), N > 0")
    if not np.all(np.isfinite(samples)) or np.any(np.diff(samples[:, 0]) < 0):
        raise ValueError("world model trajectory must be finite and time ordered")
    grid = np.linspace(samples[0, 0], samples[-1, 0], points)
    return np.column_stack(
        [np.interp(grid, samples[:, 0], channel) for channel in samples.T]
    ).astype(np.float32)


class LatentWorldModel(nn.Module):
    """MLP that predicts the next observed trajectory for a candidate action.

    Input: concatenation of scaled observation (7), action reference (3),
    candidate delta (3), and flattened real trajectory (P * 5).
    Output: predicted next trajectory of shape (P, 5), in the same normalised
    feature space as TrajectoryEncoder.extract_trajectory_features.
    """

    def __init__(self, trajectory_points: int = 16, hidden_dim: int = 256) -> None:
        super().__init__()
        traj_flat = trajectory_points * _TRAJ_FEATURES
        in_dim = _OBS_DIM + _ACTION_DIM + _ACTION_DIM + traj_flat
        self.net = MLP(in_dim, traj_flat, hidden_dim)
        self.trajectory_points = trajectory_points

    def forward(
        self,
        obs: torch.Tensor,        # (batch, 7)
        action_ref: torch.Tensor,  # (batch, 3)
        delta: torch.Tensor,       # (batch, 3)
        real_traj: torch.Tensor,   # (batch, P, 5)
    ) -> torch.Tensor:             # (batch, P, 5)
        x = torch.cat([obs, action_ref, delta, real_traj.flatten(1)], dim=-1)
        return self.net(x).reshape(-1, self.trajectory_points, _TRAJ_FEATURES)

    def predict_numpy(
        self,
        obs: np.ndarray,        # (7,)
        action_ref: np.ndarray,  # (3,)
        delta: np.ndarray,       # (3,)
        real_traj: np.ndarray,   # (P, 5)
    ) -> np.ndarray:             # (1, P, 5)
        """Run a single no-grad inference; returns shape (1, P, 5) for the critic."""
        device = next(self.parameters()).device
        real_traj = interpolate_model_trajectory(real_traj, self.trajectory_points)
        with torch.no_grad():
            inputs = (
                torch.as_tensor(np.array(value), dtype=torch.float32, device=device)[None]
                for value in (obs, action_ref, delta, real_traj)
            )
            out = self(*inputs)
        return out.cpu().numpy()
