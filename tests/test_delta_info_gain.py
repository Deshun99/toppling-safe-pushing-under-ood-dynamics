"""Particle-cloud spread and information-gain shaping for delta training."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.system_identification.sysid import Z_LO, Z_HI
from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.rl.training.train_delta import (
    normalized_posterior_spread,
    posterior_spread,
    posterior_mean_physics,
    sysid_relative_error,
    tip_stability_margin,
    sysid_tip_margin_error,
    sysid_accuracy_record,
    run_episode,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.records import ActionRecord


def test_sysid_relative_error_is_zero_when_estimate_matches_truth() -> None:
    err = sysid_relative_error((1.0, 0.5, 0.1), (1.0, 0.5, 0.1))
    assert err == {
        "sysid_rel_err_m": 0.0,
        "sysid_rel_err_mu": 0.0,
        "sysid_rel_err_h_c": 0.0,
    }


def test_sysid_relative_error_reports_per_parameter_fraction() -> None:
    err = sysid_relative_error((1.1, 0.4, 0.2), (1.0, 0.5, 0.1))
    assert err["sysid_rel_err_m"] == pytest.approx(0.1)
    assert err["sysid_rel_err_mu"] == pytest.approx(0.2)
    assert err["sysid_rel_err_h_c"] == pytest.approx(1.0)


def test_posterior_mean_physics_is_none_without_an_estimator() -> None:
    assert posterior_mean_physics(SimpleNamespace(estimator=None)) is None


def test_posterior_mean_physics_weights_the_particle_cloud() -> None:
    particles = np.array([[1.0, 0.3, 0.1], [2.0, 0.5, 0.2]])
    weights = np.array([0.25, 0.75])
    controller = SimpleNamespace(
        estimator=object(), _get_particles=lambda: (particles, weights)
    )
    expected = tuple(np.average(particles, axis=0, weights=weights))
    assert posterior_mean_physics(controller) == pytest.approx(expected)


def test_tip_stability_margin_is_mass_independent_width_minus_mu_hc() -> None:
    # w/2 - mu*h_c; mass is not an argument and cannot change the margin.
    assert tip_stability_margin(0.10, 0.5, 0.06) == pytest.approx(0.05 - 0.03)


def test_tip_margin_error_is_zero_when_hidden_tip_params_match() -> None:
    # Identical mu and h_c give an identical margin even when mass differs 2x.
    record = sysid_tip_margin_error((2.0, 0.5, 0.06), (1.0, 0.5, 0.06), width=0.10)
    assert record["tip_margin_est"] == pytest.approx(record["tip_margin_true"])
    assert record["tip_margin_abs_err"] == pytest.approx(0.0)


def test_tip_margin_error_tracks_mu_and_hc_divergence() -> None:
    record = sysid_tip_margin_error((1.0, 0.6, 0.08), (1.0, 0.4, 0.05), width=0.12)
    assert record["tip_margin_est"] == pytest.approx(0.06 - 0.6 * 0.08)
    assert record["tip_margin_true"] == pytest.approx(0.06 - 0.4 * 0.05)
    assert record["tip_margin_abs_err"] == pytest.approx(
        abs((0.06 - 0.6 * 0.08) - (0.06 - 0.4 * 0.05))
    )


def test_sysid_accuracy_record_bundles_estimate_rel_error_and_tip_margin() -> None:
    record = sysid_accuracy_record((1.1, 0.4, 0.2), (1.0, 0.5, 0.1), width=0.3)
    assert (record["sysid_est_m"], record["sysid_est_mu"], record["sysid_est_h_c"]) == (
        pytest.approx(1.1),
        pytest.approx(0.4),
        pytest.approx(0.2),
    )
    assert record["sysid_rel_err_h_c"] == pytest.approx(1.0)
    assert record["tip_margin_true"] == pytest.approx(0.15 - 0.5 * 0.1)
    assert record["tip_margin_est"] == pytest.approx(0.15 - 0.4 * 0.2)


def _record() -> ActionRecord:
    trajectory = DenseTrajectory(
        t=[0.0, 0.05], x=[0.0, 0.0], psi=[0.0, 0.0], xdot=[0.0, 0.0], psidot=[0.0, 0.0]
    )
    return ActionRecord((0.0, 0.0, 0.0, 0.0), PushAction(0.02, 1.0, 0.05), trajectory)


_WIDE = np.column_stack((Z_LO, (Z_LO + Z_HI) / 2, Z_HI)).T  # 3 particles spanning prior
_WIDE_SPREAD = normalized_posterior_spread(_WIDE, np.full(3, 1 / 3), Z_LO, Z_HI)


class _ShrinkingEstimator:
    """Posterior that collapses to a point the first time the cloud is shrunk."""

    def __init__(self) -> None:
        self.particles = _WIDE.copy()
        self.weights = np.full(3, 1 / 3)

    def shrink(self) -> None:
        midpoint = (Z_LO + Z_HI) / 2
        self.particles = np.tile(midpoint, (3, 1))


class _ShrinkingController:
    def __init__(self, shrink_on_observe: int) -> None:
        self.estimator = _ShrinkingEstimator()
        self._observe_count = 0
        self._shrink_on = shrink_on_observe
        self.remaining = 2
        self.stop_reason = "proposal_budget_exhausted"

    def reset_episode(self) -> None:
        self.reference = np.full(3, -0.5)

    def probe(self, observation):
        return self.reference.copy()

    def observe(self, record, next_observation, toppled=False) -> None:
        self._observe_count += 1
        if self._observe_count == self._shrink_on:
            self.estimator.shrink()

    def build_context(self, observation):
        return SimpleNamespace(real_trajectory=np.ones((4, 5), np.float32))

    def propose(self, observation, *, randomize=False):
        if not self.remaining:
            return None
        self.remaining -= 1
        return SimpleNamespace(
            delta=np.full(3, 0.1),
            action=self.reference + 0.1,
            forecasts=np.full((2, 4, 5), 9.0, dtype=np.float32),
            weights=np.array([0.3, 0.7]),
            context=SimpleNamespace(
                real_trajectory=np.full((4, 5), 7.0, np.float32),
                reference_action=self.reference.copy(),
            ),
        )


class _ConstRewardEnv:
    def __init__(self, task_rewards: list[float]) -> None:
        self.rewards = task_rewards
        self.steps = 0

    def reset(self, seed):
        self.steps = 0
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        reward = self.rewards[self.steps]
        self.steps += 1
        end = self.steps == len(self.rewards)
        return (
            np.full(7, self.steps, dtype=np.float32),
            reward,
            end,
            False,
            dict(action_record=_record(), toppled=False, success=False),
        )


def test_info_gain_shapes_learning_reward_but_not_logged_task_reward() -> None:
    beta = 2.0
    # Probe is push 1 (observe #1); cloud shrinks on push 2's observation (observe #2).
    controller = _ShrinkingController(shrink_on_observe=2)
    episode = run_episode(
        _ConstRewardEnv([-1.0, -1.0, -1.0]),
        controller,
        SimpleNamespace(num_timesteps=0),
        seed=0,
        info_gain_beta=beta,
    )
    assert episode.record["episode_reward"] == pytest.approx(-3.0)
    learning_rewards = [example.reward for example in episode.examples]
    assert learning_rewards[0] == pytest.approx(-1.0 + beta * _WIDE_SPREAD)
    assert learning_rewards[1] == pytest.approx(-1.0)


def test_info_gain_beta_zero_leaves_learning_reward_equal_to_task_reward() -> None:
    controller = _ShrinkingController(shrink_on_observe=2)
    episode = run_episode(
        _ConstRewardEnv([-1.0, -1.0, -1.0]),
        controller,
        SimpleNamespace(num_timesteps=0),
        seed=0,
        info_gain_beta=0.0,
    )
    assert [example.reward for example in episode.examples] == pytest.approx(
        [-1.0, -1.0]
    )


def test_posterior_spread_is_none_without_an_estimator() -> None:
    controller = SimpleNamespace(estimator=None)
    assert posterior_spread(controller) is None


def test_posterior_spread_reads_z_particles_from_a_smc2_estimator() -> None:
    estimator = SMC2SysID(
        (0.08, 0.08, 0.2),
        K=16,
        n_theta=16,
        n_state=1,
        start_std=(0.0, 0.0, 0.0, 0.0),
        seed=0,
    )
    controller = SimpleNamespace(estimator=estimator)
    expected = normalized_posterior_spread(
        estimator.particles, estimator.weights, Z_LO, Z_HI
    )
    assert posterior_spread(controller) == pytest.approx(expected)


def test_spread_is_zero_for_identical_particles() -> None:
    particles = np.tile(np.array([0.0, 0.5, 0.5]), (5, 1))
    weights = np.full(5, 0.2)
    assert normalized_posterior_spread(particles, weights, Z_LO, Z_HI) == pytest.approx(
        0.0, abs=1e-12
    )


def test_spread_is_larger_for_a_wider_cloud() -> None:
    weights = np.full(4, 0.25)
    tight = np.tile(np.array([0.0, 0.5, 0.5]), (4, 1)) + np.array(
        [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]
    )
    wide = np.column_stack(
        (
            np.linspace(Z_LO[0], Z_HI[0], 4),
            np.linspace(Z_LO[1], Z_HI[1], 4),
            np.linspace(Z_LO[2], Z_HI[2], 4),
        )
    )
    tight_spread = normalized_posterior_spread(tight, weights, Z_LO, Z_HI)
    wide_spread = normalized_posterior_spread(wide, weights, Z_LO, Z_HI)
    assert wide_spread > tight_spread > 0.0


def test_spread_shrinks_when_weight_concentrates_on_a_subset() -> None:
    particles = np.column_stack(
        (
            np.linspace(Z_LO[0], Z_HI[0], 6),
            np.linspace(Z_LO[1], Z_HI[1], 6),
            np.linspace(Z_LO[2], Z_HI[2], 6),
        )
    )
    uniform = np.full(6, 1 / 6)
    concentrated = np.array([0.0, 0.0, 0.48, 0.48, 0.02, 0.02])
    spread_uniform = normalized_posterior_spread(particles, uniform, Z_LO, Z_HI)
    spread_concentrated = normalized_posterior_spread(
        particles, concentrated, Z_LO, Z_HI
    )
    assert spread_concentrated < spread_uniform
