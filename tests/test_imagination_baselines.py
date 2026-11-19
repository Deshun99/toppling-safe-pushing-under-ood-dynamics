"""Tests for the imagination baselines: providers, imagination, and the B5 filter."""
import numpy as np
import pytest

from src.agents.rl.filter import PessimisticFilter
from src.agents.rl.imagination import Imagination
from src.agents.rl.models.sac import SAC, SACConfig
from src.agents.rl.physics_providers import (
    EnsemblePhysics,
    EstimatedPhysics,
    NominalPhysics,
    OraclePhysics,
    make_provider,
)
from src.agents.simulator.internal_sim import InternalSim
from src.evaluation.rl.evaluate_dyna import parse_parameter_errors, perturb_theta
from src.simulation.environment.env import ACTION_RANGES, PushEnv
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.object import CuboidSpec

SPEC = CuboidSpec(w=.08, L=.08, H=.20, m=1.0, mu=.5, h_c=.08)
GEOMETRY = (SPEC.w, SPEC.L, SPEC.H)


def test_oracle_provider_returns_true_parameters():
    provider = OraclePhysics(SPEC.theta)
    assert provider.theta(1, 2, 3) == pytest.approx(SPEC.theta)
    provider.observe(object())  # no-op, must not raise


def test_nominal_provider_uses_training_means():
    provider = NominalPhysics()
    m, mu, h_c = provider.theta(*GEOMETRY)
    assert (m, mu, h_c) == pytest.approx((1.0, 0.5, 0.475 * SPEC.H))


def test_estimated_provider_falls_back_until_first_refit(force_env):
    provider = EstimatedPhysics(GEOMETRY)
    assert provider.theta(*GEOMETRY) == pytest.approx(
        NominalPhysics().theta(*GEOMETRY)
    )
    _, _, _, _, info = force_env.step(np.array([0.0, 0.5, 0.5]))
    provider.observe(info["action_record"])
    assert provider.point is not None
    assert provider.theta(*GEOMETRY) == pytest.approx(provider.point)


def test_ensemble_provider_retains_hypotheses(force_env):
    provider = EnsemblePhysics(GEOMETRY, replay_budget=64)
    _, _, _, _, info = force_env.step(np.array([0.0, 0.5, 0.5]))
    provider.observe(info["action_record"])
    hypotheses, weights = provider.hypotheses_with_weights(*GEOMETRY)
    assert len(hypotheses) == len(weights) >= 1
    assert sum(weights) == pytest.approx(1.0)
    assert provider.theta(*GEOMETRY) in hypotheses


def test_make_provider_mapping():
    assert isinstance(make_provider("b2", GEOMETRY), NominalPhysics)
    assert isinstance(make_provider("b3", GEOMETRY), EstimatedPhysics)
    assert isinstance(make_provider("b4", GEOMETRY), EnsemblePhysics)
    assert isinstance(make_provider("b6", GEOMETRY, SPEC.theta), OraclePhysics)
    with pytest.raises(ValueError):
        make_provider("b6", GEOMETRY)
    with pytest.raises(ValueError):
        make_provider("b9", GEOMETRY)


def test_parameter_error_parsing_and_perturbation():
    assert parse_parameter_errors(["mass:1.25", "h_c:0.75"]) == {
        "mass": 1.25,
        "h_c": 0.75,
    }
    assert parse_parameter_errors(None) == {}
    for bad in ("width:1.1", "mass:", "mass:0", "mass:-1", "mass:x"):
        with pytest.raises(ValueError):
            parse_parameter_errors([bad])
    theta = (1.0, 0.5, 0.08)
    assert perturb_theta(theta, {"mass": 1.25}) == pytest.approx(
        (1.25, 0.5, 0.08)
    )
    assert perturb_theta(theta, {}) == pytest.approx(theta)


def test_imagine_step_scores_like_the_environment(force_env):
    sim = InternalSim()
    try:
        imagination = Imagination(
            sim, OraclePhysics(SPEC.theta), force_env.reward,
            force_env.action_ranges, force_env.target_distance,
            force_env.epsilon, force_env.psi_succ_tol, force_env.v_succ_tol,
        )
        observation, _ = force_env.reset(seed=0)
        next_obs, reward, toppled, success = imagination.imagine_step(
            observation, np.array([0.0, 0.5, 0.5], dtype=np.float32)
        )
        assert next_obs.shape == (7,)
        assert next_obs[4:] == pytest.approx(observation[4:])
        assert np.isfinite(reward)
        assert isinstance(toppled, bool) and isinstance(success, bool)
    finally:
        sim.close()


def test_generate_adds_imagined_transitions(force_env):
    sim = InternalSim()
    try:
        model = SAC(SACConfig(obs_dim=7, action_dim=3, buffer_size=64))
        observation, _ = force_env.reset(seed=0)
        model.replay.add(observation, np.zeros(3), 0.0, observation, False)
        imagination = Imagination(
            sim, OraclePhysics(SPEC.theta), force_env.reward,
            force_env.action_ranges, force_env.target_distance,
            force_env.epsilon, force_env.psi_succ_tol, force_env.v_succ_tol,
            horizon=1,
        )
        imagination.record_state(observation)
        stats = imagination.generate(model, n_rollouts=3)
        assert stats.transitions == 3
        assert model.replay.size == 4
    finally:
        sim.close()


def test_rollout_starts_come_from_real_states_only(force_env):
    sim = InternalSim()
    try:
        model = SAC(SACConfig(obs_dim=7, action_dim=3, buffer_size=64))
        real_obs, _ = force_env.reset(seed=0)
        # Only imagined junk in the replay: a plausible but distinct state.
        fake_obs = np.array([0.9, 0.0, 0.0, 0.0, *GEOMETRY], dtype=np.float32)
        model.replay.add(fake_obs, np.zeros(3), 0.0, fake_obs, False)
        imagination = Imagination(
            sim, OraclePhysics(SPEC.theta), force_env.reward,
            force_env.action_ranges, force_env.target_distance,
            force_env.epsilon, force_env.psi_succ_tol, force_env.v_succ_tol,
            horizon=1, seed=0,
        )
        imagination.record_state(real_obs)
        seen = []

        def spy_imagine_step(obs, action):
            seen.append(np.asarray(obs).copy())
            return Imagination.imagine_step(imagination, obs, action)

        imagination.imagine_step = spy_imagine_step
        imagination.generate(model, n_rollouts=2)
        assert len(seen) == 2
        for start in seen:
            assert start == pytest.approx(real_obs)
            assert start != pytest.approx(fake_obs)
    finally:
        sim.close()


class FakeSim:
    """Predict a canned trajectory with a fixed peak tilt."""

    def __init__(self, peak_tilt: float) -> None:
        self.peak_tilt = peak_tilt

    def predict(self, spec, start, action, warmup_steps=0, rest=None):
        traj = DenseTrajectory()
        traj.psi.extend([0.0, self.peak_tilt])
        traj.x.extend([start[0], start[0] + 0.01])
        traj.xdot.extend([0.0, 0.0])
        traj.psidot.extend([0.0, 0.0])
        traj.t.extend([0.0, 0.01])
        return traj

    def close(self) -> None:
        pass


class SafePolicy:
    def predict(self, observations, deterministic=True):
        return np.array([0.0, 0.5, 0.5], dtype=np.float32)


def _ensemble_with(theta: tuple[float, float, float]):
    class FixedEnsemble:
        def hypotheses_with_weights(self, w, L, H):
            return [theta], [1.0]
    return FixedEnsemble()


def _observation() -> np.ndarray:
    """A plausible observation: (d, psi, xdot, psidot, w, L, H)."""
    return np.array([0.5, 0.0, 0.0, 0.0, *GEOMETRY], dtype=np.float32)


def test_filter_rejects_all_toppling_candidates(force_env):
    provider = _ensemble_with((1.0, 0.5, 0.01))  # crit_tilt = atan(0.04/0.01) ~ 1.33
    safety_filter = PessimisticFilter(
        FakeSim(1.5), provider, ACTION_RANGES, 0.5, candidates=2
    )
    action = safety_filter.select_action(SafePolicy(), _observation())
    assert action[1] == -1.0  # zero force fallback
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_filter_keeps_safe_candidates(force_env):
    provider = _ensemble_with((1.0, 0.5, 0.08))
    safety_filter = PessimisticFilter(
        FakeSim(0.01), provider, ACTION_RANGES, 0.5, candidates=2
    )
    action = safety_filter.select_action(SafePolicy(), _observation())
    assert action == pytest.approx(np.array([0.0, 0.5, 0.5], dtype=np.float32))


@pytest.fixture
def force_env():
    env = PushEnv(fixed_spec=SPEC, target_distance=0.5)
    env.reset(seed=0)
    yield env
    env.close()
