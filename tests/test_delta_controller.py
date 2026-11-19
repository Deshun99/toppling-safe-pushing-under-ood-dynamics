"""Behavioral contracts for delta search, historical evidence, and ownership."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.rl.models.delta_actor_critic import (
    DeltaActorCritic,
    DeltaActorCriticConfig,
)
from src.agents.rl.training.train_delta import _record_training_example

import numpy as np
import pytest

from src.agents.rl.models.delta_controller import (
    DeltaController,
    DeltaControllerConfig,
)
from src.agents.system_identification.sysid import Hypothesis, SysIDResult
from src.simulation.environment import ACTION_RANGES, PushEnv
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.action_mapping import normalized_to_physical
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord

GEOMETRY = (0.08, 0.08, 0.2)
OBSERVATION = np.array([0.7, 0.01, 0.1, 0.02, *GEOMETRY])
PARTICLES = np.array(
    [[np.log(0.5), 0.2, 0.3], [np.log(1.0), 0.5, 0.8], [np.log(2.0), 0.7, 0.5]]
)


def trajectory(start=(0.3, 0.01, 0.1, 0.02), *, peak=0.01, displacement=0.02):
    x, _, vx, wy = start
    return DenseTrajectory(
        [0.0, 0.1, 0.2],
        [x, x + displacement / 2, x + displacement],
        [0.0, peak, 0.0],
        [vx, vx, vx],
        [wy, wy, wy],
    )


def record(action=None):
    return ActionRecord(
        (0.3, 0.01, 0.1, 0.02), action or PushAction(0.1, 25.0, 0.275), trajectory()
    )


class Estimator:
    def __init__(self):
        self.particles = PARTICLES.copy()
        self.weights = np.array([0.7, 0.1, 0.2])
        self.updates = []
        self.closed = False

    def update(self, history):
        self.updates.append(list(history))
        return SysIDResult(
            Hypothesis((0.5, 0.2, 0.06), 0.0), [Hypothesis((0.5, 0.2, 0.06), 0.0, 1.0)]
        )

    def close(self):
        self.closed = True


class Simulator:
    def __init__(self, peak=lambda spec, action: 0.01):
        self.calls = []
        self.peak = peak
        self.closed = False

    def predict(self, spec, start, action):
        self.calls.append((spec, start, action))
        return trajectory(
            start, peak=self.peak(spec, action), displacement=action.F * spec.m * 0.001
        )

    def close(self):
        self.closed = True


class Distribution:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.samples = 0

    def sample(self, count):
        self.samples += 1
        result = np.array(next(self.batches), dtype=float)
        assert len(result) == count
        return result


class Model:
    config: DeltaActorCriticConfig = DeltaActorCriticConfig()

    def __init__(self, batches):
        self.cached = Distribution(batches)
        self.forwards = 0
        self.scored = []

    def distribution(self, observation, real, reference_action=None, *, uniform=False):
        assert observation.shape == (7,)
        assert real.shape == (16, 5)
        self.forwards += 1
        return self.cached

    def score(self, observation, real, deltas, forecasts, weights):
        self.scored.append((deltas.copy(), forecasts.copy(), weights.copy()))
        return deltas[:, 1]


def controller(batches, *, simulator=None, **config):
    model = Model(batches)
    estimator = Estimator()
    simulator = simulator or Simulator()
    control = DeltaController(
        model,
        DeltaControllerConfig(**config),
        estimator=estimator,
        simulator=simulator,
    )
    control.initialize_for_object(GEOMETRY)
    control.observe(record(), OBSERVATION)
    return control, model, estimator, simulator


def test_probe_and_moving_reference_use_actual_execution_and_reset_episode_only():
    control, _, estimator, _ = controller([])
    np.testing.assert_allclose(control.probe(OBSERVATION), [-1.0, -0.98, -1.0])
    control.observe(record(PushAction(0.12, 30.0, 0.32)), OBSERVATION)
    np.testing.assert_allclose(control.action_reference, [0.25, 0.2, 0.2])
    assert len(control.history) == len(estimator.updates) == 2
    control.reset_episode()
    assert control.action_reference is None and control.real_trajectory is None
    assert len(control.history) == 2
    with pytest.raises(RuntimeError, match="probe"):
        control.propose(OBSERVATION)


def test_resampling_reuses_one_forward_and_scores_full_ensemble_from_current_state():
    batches = [[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, -0.2, 0.0], [0.0, 0.2, 0.0]]]
    control, model, _, sim = controller(batches, candidates_per_batch=2, max_batches=2)
    selected = control.propose(OBSERVATION)
    assert selected is not None
    assert model.forwards == 1 and model.cached.samples == 2
    np.testing.assert_allclose(selected.action, [0.0, 0.2, 0.0], atol=1e-15)
    np.testing.assert_allclose(control.action_reference, [0.0, 0.0, 0.0], atol=1e-15)
    assert selected.forecasts.shape == (3, 16, 5)  # estimator result() retains only one
    np.testing.assert_allclose(selected.weights, [0.7, 0.1, 0.2])
    assert len(sim.calls) == 6
    np.testing.assert_allclose(sim.calls[0][1], [0.3, 0.01, 0.1, 0.02])
    assert sim.calls[0][0].m == pytest.approx(0.5)
    assert sim.calls[1][0].h_c == pytest.approx(0.16)


def test_predicted_toppling_does_not_reject_the_highest_scoring_action() -> None:
    sim = Simulator(peak=lambda spec, action: 1.2 if action.F > 25 else 0.01)
    control, model, _, _ = controller(
        [[[0.0, -0.2, 0.0], [0.0, 0.2, 0.0]]],
        simulator=sim,
        candidates_per_batch=2,
        max_batches=1,
    )
    selected = control.propose(OBSERVATION)
    assert selected is not None
    np.testing.assert_allclose(selected.delta, [0.0, 0.2, 0.0])
    assert control.stop_reason is None
    assert len(sim.calls) == 6
    assert len(model.scored[0][0]) == 2
    assert np.max(np.abs(selected.forecasts[:, :, 2])) > 5


def test_earlier_controller_settings_ignore_removed_filter_without_mutation() -> None:
    settings = dict(near_topple_fraction=0.8, n_particles=256)
    config = DeltaControllerConfig.from_settings(settings)
    assert config.n_particles == 256
    assert not hasattr(config, "near_topple_fraction")
    assert settings["near_topple_fraction"] == 0.8


def test_finite_budget_stops_without_simulating_invalid_actions_or_clipping():
    batches = [[[1.1, 0.0, 0.0]]] * 3
    control, model, _, sim = controller(batches, candidates_per_batch=1, max_batches=3)
    assert control.propose(OBSERVATION) is None
    assert model.forwards == 1 and model.cached.samples == 3
    assert len(sim.calls) == 0


def test_toppled_records_are_retained_but_never_mutate_assimilated_history():
    control, _, estimator, _ = controller([])
    old = control.history[0].traj.x.copy()
    fallen = record(PushAction(0.12, 30.0, 0.32))
    control.observe(fallen, OBSERVATION, toppled=True)
    fallen.traj.x[0] = 999.0
    assert len(control.records) == 2 and len(control.history) == 1
    assert len(estimator.updates) == 1
    assert control.history[0].traj.x == old
    assert control.records[-1].traj.x[0] != 999.0
    assert control.stop_reason == "observed_topple"
    assert control.propose(OBSERVATION) is None
    control.reset_episode()
    control.observe(record(), OBSERVATION)
    assert len(estimator.updates[-1]) == 2


def test_historical_forecasts_snapshot_particles_start_reference_and_bounds():
    control, _, estimator, sim = controller([[[0.0, 0.1, 0.0]]], candidates_per_batch=1)
    proposal = control.propose(OBSERVATION)
    assert proposal is not None
    context = proposal.context
    example = SimpleNamespace(metadata=context)
    baseline = control.forecast_for_training(example, proposal.delta)
    estimator.particles[:] = [np.log(3.0), 1.0, 0.2]
    estimator.weights[:] = [0.0, 0.0, 1.0]
    control.observe(record(PushAction(0.15, 45.0, 0.4)), OBSERVATION)
    regenerated = control.forecast_for_training(example, proposal.delta)
    np.testing.assert_allclose(regenerated, baseline)
    # A fresh action delta must get matching fresh simulations, not old forecasts.
    changed = control.forecast_for_training(example, np.array([0.0, -0.1, 0.0]))
    assert not np.allclose(changed, baseline)
    assert sim.calls[-1][2].F == pytest.approx(22.5)
    assert sim.calls[-1][0].m == pytest.approx(2.0)
    with pytest.raises(ValueError):
        context.particles[0, 0] = 100.0
    before = len(sim.calls)
    with pytest.raises(ValueError, match="within"):
        control.forecast_for_training(example, np.array([2.0, 0.0, 0.0]))
    assert len(sim.calls) == before


def test_injected_resources_remain_owned_by_caller_and_cannot_leak_between_objects():
    control, _, estimator, sim = controller([])
    with pytest.raises(RuntimeError, match="fresh injected"):
        control.initialize_for_object(GEOMETRY)
    control.close()
    control.close()
    assert not estimator.closed and not sim.closed
    with pytest.raises(RuntimeError, match="closed"):
        control.propose(OBSERVATION)


@pytest.mark.parametrize("backend", ["smc2", "cmaes"])
def test_real_probe_and_one_candidate_smoke(backend):
    config = DeltaControllerConfig(
        backend=backend,
        n_particles=3,
        candidates_per_batch=1,
        max_batches=1,
        cma_replay_budget=6,
        estimator_options={"mcmc_steps": 0} if backend == "smc2" else {"restarts": 0},
    )
    model = Model([[[0.0, 0.0, 0.0]]])
    control = DeltaController(model, config)
    env = PushEnv(fixed_spec=CuboidSpec(*GEOMETRY, 1.0, 0.5, 0.1))
    try:
        observation, _ = env.reset(seed=2)
        control.initialize_for_object(observation[4:])
        action = control.probe(observation)
        next_observation, _, terminated, truncated, info = env.step(action)
        control.observe(
            info["action_record"], next_observation, toppled=info["toppled"]
        )
        proposal = control.propose(next_observation)
        assert proposal is not None
        assert proposal.forecasts.shape[1:] == (16, 5)
        assert proposal.weights.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(proposal.action, action, atol=1e-6)
        physical = normalized_to_physical(proposal.action, GEOMETRY, ACTION_RANGES)
        assert physical.F == pytest.approx(config.probe_force)
    finally:
        control.close()
        env.close()


@pytest.mark.parametrize(
    "setting",
    [
        dict(max_batches=0),
        dict(n_particles=0),
        dict(backend="unknown"),
        dict(probe_force=float("nan")),
        dict(seed=-1),
    ],
)
def test_invalid_settings_fail_before_allocating_clients(
    setting: dict[str, object],
) -> None:
    with pytest.raises((ValueError, TypeError)):
        DeltaControllerConfig(**setting)


def test_duplicate_forecasts_preserve_particle_order_and_weights() -> None:
    control, _, _, sim = controller([])
    context = control.build_context(OBSERVATION)
    context = replace(
        context,
        particles=context.particles[[2, 0, 2, 1, 0]],
        weights=np.array([0.1, 0.2, 0.3, 0.15, 0.25]),
    )
    forecasts = control._forecast(context, np.zeros(3))
    assert len(sim.calls) == 3
    np.testing.assert_array_equal(forecasts[0], forecasts[2])
    np.testing.assert_array_equal(forecasts[1], forecasts[4])
    expected = []
    for theta in context.particles:
        single = replace(context, particles=theta[None], weights=np.ones(1))
        expected.append(control._forecast(single, np.zeros(3))[0])
    np.testing.assert_array_equal(forecasts, expected)
    np.testing.assert_array_equal(context.weights, [0.1, 0.2, 0.3, 0.15, 0.25])


def test_nearby_physics_particles_are_not_merged() -> None:
    control, _, _, sim = controller([])
    context = control.build_context(OBSERVATION)
    particles = np.repeat(context.particles[:1], 2, axis=0)
    particles[1, 0] = np.nextafter(particles[0, 0], np.inf)
    context = replace(context, particles=particles, weights=np.full(2, 0.5))
    control._forecast(context, np.zeros(3))
    assert len(sim.calls) == 2


def test_known_physics_skips_inference_and_resets_for_each_object() -> None:
    model, sim = Model([[[0.0, 0.1, 0.0]]]), Simulator()
    control = DeltaController(
        model,
        DeltaControllerConfig(backend="known", candidates_per_batch=1),
        simulator=sim,
    )
    control.initialize_for_object(GEOMETRY, known_parameters=[1.0, 0.4, 0.08])
    control.observe(record(), OBSERVATION)
    proposal = control.propose(OBSERVATION)
    assert proposal is not None and proposal.forecasts.shape == (1, 16, 5)
    np.testing.assert_array_equal(proposal.context.particles, [[1.0, 0.4, 0.08]])
    np.testing.assert_array_equal(proposal.context.observation, OBSERVATION)
    assert control.estimator is None and control.history == []
    control.reset_episode()
    np.testing.assert_array_equal(control._get_particles()[0], [[1.0, 0.4, 0.08]])
    control.initialize_for_object(GEOMETRY, known_parameters=[1.5, 0.6, 0.10])
    np.testing.assert_array_equal(control._get_particles()[0], [[1.5, 0.6, 0.10]])
    control.close()


@pytest.mark.parametrize(
    "parameters", [None, [0, 0.4, 0.1], [1, -0.1, 0.1], [1, 0.4, 0.3]]
)
def test_known_physics_requires_valid_parameters(
    parameters: list[float] | None,
) -> None:
    control = DeltaController(
        Model([]), DeltaControllerConfig(backend="known"), simulator=Simulator()
    )
    with pytest.raises(ValueError):
        control.initialize_for_object(GEOMETRY, known_parameters=parameters)
    control.close()


def test_reused_pybullet_forecasts_match_independent_replays() -> None:
    from src.agents.simulator.internal_sim import InternalSim

    sim = InternalSim()
    control, _, _, _ = controller([], simulator=sim)
    try:
        context = control.build_context(OBSERVATION)
        context = replace(
            context,
            particles=context.particles[[2, 0, 2]],
            weights=np.array([0.2, 0.5, 0.3]),
        )
        delta = np.array([0.0, -0.5, -0.5])
        with patch.object(sim, "predict", wraps=sim.predict) as predict:
            reused = control._forecast(context, delta)
        assert predict.call_count == 2
        independent = []
        for theta in context.particles:
            single = replace(context, particles=theta[None], weights=np.ones(1))
            independent.append(control._forecast(single, delta)[0])
        np.testing.assert_array_equal(reused, independent)
    finally:
        control.close()
        sim.close()


class VariableLengthSimulator(Simulator):
    def predict(
        self,
        spec: CuboidSpec,
        start: tuple[float, float, float, float],
        action: PushAction,
    ) -> DenseTrajectory:
        predicted = super().predict(spec, start, action)
        count = 1 if spec.m < 1 else 3
        channels = predicted.to_numpy()[:count].T.tolist()
        return DenseTrajectory(*channels)


def test_attention_controller_preserves_raw_samples_through_proposal() -> None:
    config = DeltaActorCriticConfig(
        hidden_dim=16, trajectory_feature_strategy="self-attention"
    )
    model = DeltaActorCritic(config)
    control = DeltaController(
        model, estimator=Estimator(), simulator=VariableLengthSimulator()
    )
    control.initialize_for_object(GEOMETRY)
    with patch(
        "numpy.interp", side_effect=AssertionError("attention must not interpolate")
    ):
        control.observe(record(), OBSERVATION)
        assert control.real_trajectory.shape == (3, 5)
        proposal = control.propose(OBSERVATION)
    assert proposal is not None
    assert isinstance(proposal.forecasts, tuple)
    assert {len(item) for item in proposal.forecasts} == {1, 3}
    assert all(not item.flags.writeable for item in proposal.forecasts)
    example = _record_training_example(
        {
            "observation": OBSERVATION,
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
        },
        proposal,
    )
    model.train_step([example], control.forecast_for_training)
    control.close()


def test_uniform_warmup_bypasses_actor_and_critic_ranking() -> None:
    model = DeltaActorCritic(DeltaActorCriticConfig(hidden_dim=16))
    control = DeltaController(model, estimator=Estimator(), simulator=Simulator())
    control.initialize_for_object(GEOMETRY)
    control.observe(record(), OBSERVATION)
    with (
        patch.object(
            model.actor, "forward", side_effect=AssertionError("warmup actor")
        ),
        patch.object(model, "score", side_effect=AssertionError("warmup ranking")),
    ):
        proposal = control.propose(OBSERVATION, randomize=True)
    assert proposal is not None and np.all(np.abs(proposal.action) <= 1)
    control.close()


def test_forecast_pool_matches_serial_forecast() -> None:
    from src.agents.simulator.internal_sim import InternalSim
    from src.agents.simulator.forecast_pool import ForecastPool

    def build(pool):
        control = DeltaController(
            Model([]),
            DeltaControllerConfig(),
            estimator=Estimator(),
            simulator=InternalSim(),
            forecast_pool=pool,
        )
        control.initialize_for_object(GEOMETRY)
        control.observe(record(), OBSERVATION)
        return control

    serial_ctrl = build(None)
    pool = ForecastPool(workers=2)
    pooled_ctrl = build(pool)
    delta = np.zeros(3)
    serial = serial_ctrl._forecast(serial_ctrl.build_context(OBSERVATION), delta)
    parallel = pooled_ctrl._forecast(pooled_ctrl.build_context(OBSERVATION), delta)
    pool.close()
    assert len(parallel) == len(serial) > 0
    for a, b in zip(parallel, serial):
        np.testing.assert_array_equal(a, b)
