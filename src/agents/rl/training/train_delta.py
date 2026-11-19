"""Train delta policies with scheduled replay updates and one-step soft TD targets.

Target critics bootstrap actor-sampled next deltas with fresh forecasts from the
saved next physics context. Simulator-ranked proposals collect experience.
"""

import argparse
import json
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import torch

from src.agents.rl.models.delta_controller import (
    CANDIDATE_SELECTION,
    DeltaController,
    DeltaControllerConfig,
    Proposal,
)
from src.agents.rl.models.delta_actor_critic import (
    DeltaActorCritic,
    DeltaActorCriticConfig,
    TrainingExample,
)
from src.agents.rl.training.constants import (
    TRAIN_MODE, ADAPT_MODE, EVALUATE_MODE,
    CURRICULUM_MIN_PUSHES, CURRICULUM_SUCCESS_WINDOW, CURRICULUM_SUCCESS_THRESHOLD,
)
from src.agents.rl.training.delta_utils import parse_args
from src.agents.simulator.forecast_pool import ForecastPool
from src.agents.system_identification.sysid import Z_LO, Z_HI
from src.agents.rl.training.io_utils import (
    create_output_directory,
    log_console_output,
    to_json_value,
)
from src.evaluation.rl.constants import (
    OBJECT_SEED_STRIDE,
    EvaluationCase,
)
from src.evaluation.rl.evaluation_sets import EvaluationSet
from src.simulation.environment import ACTION_RANGES, ActionRanges, PushEnv
from src.simulation.environment.config import load_action_ranges
from src.simulation.environment.reward import RewardConfig, RewardModel
from src.simulation.modelling.action_mapping import normalized_to_physical


def normalized_posterior_spread(
    particles: np.ndarray, weights: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """Weighted total variance of z-particles, each coordinate scaled to its prior span.

    Measures how spread out the physics posterior is in a resampling-robust way:
    it reflects where particles sit, not how the weights are distributed. Summing
    per-coordinate variances after scaling to [0, 1] keeps log-mass, friction, and
    relative COM height commensurate so no single parameter dominates.
    """
    scaled = (np.asarray(particles, dtype=float) - lo) / (hi - lo)
    weights = np.asarray(weights, dtype=float)
    mean = weights @ scaled
    variance = weights @ ((scaled - mean) ** 2)
    return float(variance.sum())


def posterior_spread(controller: "DeltaController") -> float | None:
    """Current physics-posterior spread, or None when the backend keeps no posterior.

    The 'known' backend has no estimator and the CMA-ES backend exposes no
    z-particle cloud, so both return None and information-gain shaping is skipped.
    """
    estimator = getattr(controller, "estimator", None)
    if estimator is None or not hasattr(estimator, "particles"):
        return None
    return normalized_posterior_spread(
        estimator.particles, estimator.weights, Z_LO, Z_HI
    )


def posterior_mean_physics(
    controller: "DeltaController",
) -> tuple[float, float, float] | None:
    """Weighted posterior mean of (m, mu, h_c), or None when no inference runs.

    The 'known' backend keeps no estimator, so there is no estimate to score against
    the truth and this returns None.
    """
    if getattr(controller, "estimator", None) is None:
        return None
    particles, weights = controller._get_particles()
    mean = np.average(particles, axis=0, weights=weights)
    return float(mean[0]), float(mean[1]), float(mean[2])


def sysid_relative_error(
    estimate: tuple[float, float, float], truth: tuple[float, float, float]
) -> dict[str, float]:
    """Per-parameter relative error |estimate - truth| / |truth| for (m, mu, h_c)."""
    return {
        f"sysid_rel_err_{name}": abs(float(e) - float(t)) / abs(float(t))
        for name, e, t in zip(("m", "mu", "h_c"), estimate, truth)
    }


def tip_stability_margin(width: float, mu: float, h_c: float) -> float:
    """Mass-independent core of the tipping condition, w/2 - mu*h_c (metres).

    The quasi-static topple boundary is (F/mg)(h - h_c) <= w/2 - mu*h_c; its
    right side holds every hidden tip parameter and no mass. Larger is more
    stable. Comparing the estimated and true margins tests whether the agent
    knows the tipping boundary even when its mass estimate is wrong.
    """
    return float(width) / 2.0 - float(mu) * float(h_c)


def sysid_tip_margin_error(
    estimate: tuple[float, float, float],
    truth: tuple[float, float, float],
    width: float,
) -> dict[str, float]:
    """Estimated vs true tip-stability margin and their absolute error (metres)."""
    est = tip_stability_margin(width, estimate[1], estimate[2])
    true = tip_stability_margin(width, truth[1], truth[2])
    return {
        "tip_margin_est": est,
        "tip_margin_true": true,
        "tip_margin_abs_err": abs(est - true),
    }


def sysid_accuracy_record(
    estimate: tuple[float, float, float],
    truth: tuple[float, float, float],
    width: float,
) -> dict[str, float]:
    """Raw estimate, per-parameter relative error, and tip-margin accuracy.

    The raw estimate is logged so the mass-dependent critical tip force can be
    reconstructed later; the tip margin isolates the mass-free topple boundary.
    """
    record = {
        "sysid_est_m": float(estimate[0]),
        "sysid_est_mu": float(estimate[1]),
        "sysid_est_h_c": float(estimate[2]),
    }
    record.update(sysid_relative_error(estimate, truth))
    record.update(sysid_tip_margin_error(estimate, truth, width))
    return record


# Interpolated trajectory feature layout is [time, displacement, tilt, forward
# velocity, angular velocity]; column 2 is tilt (psi). Must match the order in
# TrajectoryEncoder._extract_interpolated_trajectory_features.
_TILT_INDEX = 2
_TRAJ_FEATURES = 5


def world_model_tilt_error(examples: list["TrainingExample"]) -> dict[str, float]:
    """Mean peak-tilt prediction error of the learned world model over an episode.

    For each executed action, compare the world model's predicted peak tilt
    against the actual peak tilt of the resulting trajectory, in the normalized
    feature space the model operates in (column 2 = psi). Meaningful only in
    learned mode, where each forecast is a single predicted trajectory. Returns
    no keys when no executed example carries both a forecast and its outcome.
    """
    errors = []
    for example in examples:
        context = example.next_context
        if context is None or example.forecasts is None:
            continue
        predicted = np.asarray(example.forecasts, dtype=float).reshape(-1, _TRAJ_FEATURES)
        actual = np.asarray(context.real_trajectory, dtype=float).reshape(-1, _TRAJ_FEATURES)
        predicted_peak = float(np.max(np.abs(predicted[:, _TILT_INDEX])))
        actual_peak = float(np.max(np.abs(actual[:, _TILT_INDEX])))
        errors.append(abs(predicted_peak - actual_peak))
    if not errors:
        return {}
    return {"wm_tilt_peak_abs_err": float(np.mean(errors))}


@dataclass
class EpisodeRun:
    """Episode outcomes and recorded one-step transitions."""

    record: dict[str, Any]
    examples: list[TrainingExample]
    traces: list[dict[str, Any]] = field(default_factory=list)


def _execute_action(
    env: PushEnv,
    observation: np.ndarray,
    action: np.ndarray,
    controller: DeltaController,
    info_gain_beta: float = 0.0,
) -> dict[str, Any]:
    previous_observation = np.asarray(observation).copy()
    next_observation, reward, terminated, truncated, info = env.step(action)
    action_record = info.get("action_record")
    spread_before = posterior_spread(controller) if info_gain_beta else None
    if action_record is not None:
        controller.observe(
            action_record, next_observation, toppled=bool(info.get("toppled", False))
        )
    return dict(
        observation=previous_observation,
        reward=float(reward),
        info_gain=_information_gain(controller, spread_before, info_gain_beta),
        next_observation=next_observation,
        terminated=bool(terminated),
        truncated=bool(truncated),
        success=bool(info.get("success", False)),
        toppled=bool(info.get("toppled", False)),
        action_record=action_record,
        execution_status=info.get("execution_status", "completed"),
        execution_failed=bool(info.get("execution_failed", False)),
        normalized_action=np.asarray(action).copy(),
    )


def _information_gain(
    controller: DeltaController, spread_before: float | None, info_gain_beta: float
) -> float:
    """Posterior-spread reduction from the just-observed push, clamped nonnegative.

    A momentary widening of the cloud should not penalize a push, so only
    shrinkage is credited. Returns 0.0 whenever the backend keeps no posterior.
    """
    if not info_gain_beta or spread_before is None:
        return 0.0
    spread_after = posterior_spread(controller)
    if spread_after is None:
        return 0.0
    return max(0.0, spread_before - spread_after)


def _record_training_example(
    trace: dict[str, Any],
    proposal: Proposal,
    next_context: Any = None,
    info_gain_beta: float = 0.0,
) -> TrainingExample:
    # All these fields were available BEFORE execution. Never substitute
    # action_record.traj for the candidate's stored simulated forecasts. The
    # critic learns the shaped reward; the task reward stays in trace for logs.
    return TrainingExample(
        observation=np.asarray(trace["observation"]).copy(),
        real_trajectory=np.asarray(proposal.context.real_trajectory).copy(),
        delta=np.asarray(proposal.delta).copy(),
        forecasts=deepcopy(proposal.forecasts),
        weights=np.asarray(proposal.weights).copy(),
        reward=trace["reward"] + info_gain_beta * trace.get("info_gain", 0.0),
        terminated=trace["terminated"],
        truncated=trace["truncated"],
        next_context=next_context,
        reference_action=np.asarray(proposal.context.reference_action).copy(),
        metadata=proposal.context,
    )


def collect_actions(
    env: PushEnv,
    controller: DeltaController,
    model: DeltaActorCritic,
    seed: int,
    *,
    learning_start_step: int = 0,
    max_steps: int | None = None,
    info_gain_beta: float = 0.0,
    uniform_warmup: bool = True,
) -> Iterator[tuple[dict[str, Any], TrainingExample | None]]:
    """Yield each executed push so the caller can learn before the next proposal."""
    observation, _ = env.reset(seed=seed)
    controller.reset_episode()
    action, proposal = controller.probe(observation), None
    for push_count in count(1):
        trace = _execute_action(env, observation, action, controller, info_gain_beta)
        transition = _build_transition(trace, proposal, controller, info_gain_beta)
        model.num_timesteps += 1
        yield trace, transition
        observation = trace["next_observation"]
        if trace["terminated"] or trace["truncated"]:
            return
        if max_steps is not None and push_count >= max_steps:
            controller.stop_reason = "interaction_budget"
            return
        proposal = controller.propose(
            observation,
            randomize=uniform_warmup and model.num_timesteps < learning_start_step,
        )
        if proposal is None:
            return
        action = proposal.action


def _build_transition(
    trace: dict[str, Any],
    proposal: Proposal | None,
    controller: DeltaController,
    info_gain_beta: float = 0.0,
) -> TrainingExample | None:
    if proposal is None or (trace["action_record"] is None and not trace["terminated"]):
        return None
    next_context = None
    if not trace["terminated"]:
        next_context = controller.build_context(trace["next_observation"])
    return _record_training_example(trace, proposal, next_context, info_gain_beta)


def _summarize_episode(
    traces: list[dict[str, Any]],
    controller: DeltaController,
    transition_count: int,
) -> dict[str, Any]:
    last = traces[-1]
    stop_reason = controller.stop_reason
    if last["terminated"] or last["truncated"]:
        stop_reason = None
    return dict(
        real_interactions=len(traces),
        probe_interactions=1,
        actor_interactions=len(traces) - 1,
        episode_reward=sum(trace["reward"] for trace in traces),
        info_gain_total=sum(trace.get("info_gain", 0.0) for trace in traces),
        terminated=last["terminated"],
        truncated=last["truncated"],
        success=last["success"],
        toppled=last["toppled"],
        execution_status=last.get("execution_status", "completed"),
        execution_failed=last.get("execution_failed", False),
        timeout=last["truncated"],
        stopped=stop_reason is not None,
        stop_reason=stop_reason,
        transition_count=transition_count,
        final_observation=last["next_observation"],
    )


def run_episode(
    env: PushEnv,
    controller: DeltaController,
    model: DeltaActorCritic,
    *,
    seed: int,
    learning_start_step: int = 0,
    on_transition: Callable[[TrainingExample | None, DeltaController], None]
    | None = None,
    max_steps: int | None = None,
    info_gain_beta: float = 0.0,
    uniform_warmup: bool = True,
) -> EpisodeRun:
    """Learn from each yielded transition before collection resumes."""
    traces, transitions = [], []
    pushes = collect_actions(
        env,
        controller,
        model,
        seed,
        learning_start_step=learning_start_step,
        max_steps=max_steps,
        info_gain_beta=info_gain_beta,
        uniform_warmup=uniform_warmup,
    )
    for trace, transition in pushes:
        traces.append(trace)
        if transition is not None:
            transitions.append(transition)
        if on_transition is not None:
            on_transition(transition, controller)
    record = _summarize_episode(traces, controller, len(transitions))
    record["seed"] = seed
    return EpisodeRun(record, transitions, traces)


def resolve_cases(
    args: argparse.Namespace,
) -> tuple[list[EvaluationCase], dict[str, Any], str]:
    """Reuse evaluation-set sampling and global seed indices for comparisons."""
    path = _get_case_path(args)
    evaluation_set = EvaluationSet.load(
        path,
        args.split,
        sampled=not bool(args.benchmark_config),
        count=args.num_objects,
        seed=args.seed,
    )
    if args.benchmark_config and args.smoke:
        evaluation_set = evaluation_set.limit(1)
    snapshot = evaluation_set.config
    return list(evaluation_set.cases), snapshot, snapshot.get("name", path.stem)


def environment_settings(
    args: argparse.Namespace, source_metadata: dict[str, Any] | None
) -> dict[str, Any]:
    """Restore the finite-horizon task unless a new horizon is explicit."""
    saved = (source_metadata or {}).get("environment", {})
    settings = dict(
        T_max=args.max_decisions
        if args.max_decisions is not None
        else saved.get("T_max", 10),
        target_distance=args.target_distance,
        epsilon=args.epsilon,
        psi_succ_tol=0.02,
        v_succ_tol=0.05,
        robot=saved.get("robot", False) if args.robot is None else args.robot,
    )
    for name in ("target_distance", "epsilon", "psi_succ_tol", "v_succ_tol"):
        if name in saved:
            settings[name] = saved[name]
    for name, default in (("wait_for_rest", False), ("rest_timeout", 2.0)):
        value = getattr(args, name, None)
        previous = saved.get(name, default)
        if args.checkpoint is not None and value is not None and value != previous:
            raise ValueError(
                "Checkpoint settling settings cannot be changed during adaptation/evaluation"
            )
        settings[name] = previous if value is None else value
    return settings


def _build_fresh_model(args: argparse.Namespace) -> DeltaActorCritic:
    return DeltaActorCritic(
        DeltaActorCriticConfig(
            hidden_dim=args.hidden_dim,
            trajectory_points=args.trajectory_points,
            trajectory_feature_strategy=args.trajectory_feature_strategy,
            delta_scale=tuple(args.delta_scale),
            learning_rate=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            batch_size=args.batch_size,
            gradient_steps=args.gradient_steps,
            seed=args.seed,
            world_model=args.world_model,
        ),
        device="cpu",
    )


def _warm_start_from(
    args: argparse.Namespace, source: DeltaActorCritic
) -> DeltaActorCritic:
    """New config, weights from source. Optimizers start fresh; latent model copies only when both present."""
    model = _build_fresh_model(args)
    model.actor.load_state_dict(source.actor.state_dict())
    model.critic.load_state_dict(source.critic.state_dict())
    model.target_critic.load_state_dict(source.target_critic.state_dict())
    with torch.no_grad():
        model.log_alpha.copy_(source.log_alpha)
    model.num_timesteps = source.num_timesteps
    model.num_updates = source.num_updates
    if source.latent_world_model is not None and model.latent_world_model is not None:
        model.latent_world_model.load_state_dict(source.latent_world_model.state_dict())
    return model


def _load_model_and_ranges(
    args: argparse.Namespace,
) -> tuple[DeltaActorCritic, ActionRanges]:
    if args.checkpoint is not None:
        source_model = DeltaActorCritic.load(args.checkpoint, device="cpu")
        action_ranges = load_action_ranges(args.checkpoint, source_model.training_metadata)
        if args.f_max is not None and args.f_max != action_ranges.force[1]:
            raise ValueError(
                "The loaded checkpoint fixes the action bounds; --f-max cannot change them"
            )
        return source_model, action_ranges
    if args.init_from is not None:
        source_model = DeltaActorCritic.load(args.init_from, device="cpu")
        action_ranges = load_action_ranges(args.init_from, source_model.training_metadata)
        if args.f_max is not None and args.f_max != action_ranges.force[1]:
            raise ValueError(
                "The loaded checkpoint fixes the action bounds; --f-max cannot change them"
            )
        return _warm_start_from(args, source_model), action_ranges
    action_ranges = ActionRanges(force=(0.0, args.f_max or ACTION_RANGES.force[1]))
    return _build_fresh_model(args), action_ranges


def _settings_metadata(
    args: argparse.Namespace, model: DeltaActorCritic
) -> dict[str, Any] | None:
    """Source for reward and environment settings.

    Train always takes these from the command line; adapt/evaluate restore them
    from the checkpoint. Warm-start (train with --init-from) loads weights but not
    settings, so training can continue under different reward and task settings.
    """
    return None if args.mode == TRAIN_MODE else model.training_metadata


def _build_controller_config(
    args: argparse.Namespace,
    model: DeltaActorCritic,
) -> DeltaControllerConfig:
    settings = environment_settings(args, _settings_metadata(args, model))
    if "--world-model" in args.explicit_options and args.world_model != model.config.world_model:
        raise ValueError("A checkpoint fixes the world-model variant")
    current = DeltaControllerConfig(
        wait_for_rest=settings["wait_for_rest"],
        rest_timeout=settings["rest_timeout"],
        backend=args.backend,
        n_particles=args.particles,
        seed=args.seed,
        trajectory_points=model.config.trajectory_points,
        candidates_per_batch=args.candidates,
        max_batches=args.max_batches,
        probe_force=args.probe_force,
        probe_duration=args.probe_duration,
        probe_height_fraction=args.probe_height_fraction,
        cma_replay_budget=args.cma_replay_budget,
        estimator_options={"mcmc_steps": args.mcmc_steps}
        if args.backend == "smc2"
        else {},
        world_model=model.config.world_model,
    )
    return _restore_controller_config(args, model, current)


def _restore_controller_config(
    args: argparse.Namespace, model: DeltaActorCritic, current: DeltaControllerConfig,
) -> DeltaControllerConfig:
    """Frozen runs retain inference/proposal settings unless explicitly overridden."""
    saved = (_settings_metadata(args, model) or {}).get("controller", {})
    values = dict(asdict(current), **saved)
    flags = {
        "backend": "--backend", "n_particles": "--particles",
        "candidates_per_batch": "--candidates", "max_batches": "--max-batches",
        "probe_force": "--probe-force", "probe_duration": "--probe-duration",
        "probe_height_fraction": "--probe-height-fraction",
        "cma_replay_budget": "--cma-replay-budget",
    }
    for name, flag in flags.items():
        if flag in args.explicit_options:
            values[name] = getattr(current, name)
    if values["backend"] != saved.get("backend", values["backend"]):
        values["estimator_options"] = current.estimator_options
    if "--mcmc-steps" in args.explicit_options and values["backend"] == "smc2":
        values["estimator_options"] = dict(
            values["estimator_options"], mcmc_steps=args.mcmc_steps
        )
    for name in ("seed", "world_model", "trajectory_points", "wait_for_rest", "rest_timeout"):
        values[name] = getattr(current, name)
    return DeltaControllerConfig.from_settings(values)


def _get_case_path(args: argparse.Namespace) -> Path:
    return Path(
        args.benchmark_config or args.sampler_config or args.distribution_config
    )


def _resolve_reward_config(
    args: argparse.Namespace, source_metadata: dict[str, Any] | None
) -> RewardConfig:
    saved = RewardConfig(**(source_metadata or {}).get("reward", {}))
    overrides = {
        name: value
        for name, value in (
            ("strategy", args.reward_strategy),
            ("proximity_bonus", args.proximity_bonus),
            ("proximity_scale", args.proximity_scale),
            ("no_progress_penalty", args.no_progress_penalty),
            ("no_progress_threshold", args.no_progress_threshold),
            ("R_topple", args.topple_penalty),
        )
        if value is not None
    }
    config = replace(saved, **overrides)
    if args.checkpoint is not None and config != saved:
        raise ValueError(
            "Checkpoint reward settings cannot be changed during adaptation/evaluation"
        )
    # R_topple is a shared terminal penalty, valid under either strategy; only the
    # proximity-specific overrides require goal_proximity.
    proximity_overrides = {
        name for name in overrides if name not in ("strategy", "R_topple")
    }
    if config.strategy == "linear" and proximity_overrides:
        raise ValueError("Proximity settings require --reward-strategy goal_proximity")
    return config


def _build_run_metadata(
    args: argparse.Namespace,
    run_id: str,
    distribution_snapshot: dict[str, Any],
    model: DeltaActorCritic,
    action_ranges: ActionRanges,
    controller_config: DeltaControllerConfig,
) -> dict[str, Any]:
    source_metadata = _settings_metadata(args, model)
    return dict(
        run_id=run_id,
        mode=args.mode,
        arguments=to_json_value(vars(args)),
        distribution=distribution_snapshot,
        distribution_config_path=str(_get_case_path(args).resolve()),
        training_distribution=(source_metadata or {}).get(
            "training_distribution",
            distribution_snapshot if args.mode == TRAIN_MODE else {},
        ),
        action_ranges=asdict(action_ranges),
        environment=environment_settings(args, source_metadata),
        reward=asdict(_resolve_reward_config(args, source_metadata)),
        controller=asdict(controller_config),
        model_config=asdict(model.config),
        source_checkpoint=str(args.checkpoint or args.init_from)
        if (args.checkpoint or args.init_from)
        else None,
        learning="one-step soft TD; actor-sampled bootstrap; scheduled replay updates; detached simulator",
        candidate_selection=CANDIDATE_SELECTION,
        search_stops="stop collection; retain all executed TD transitions",
    )


def _configure_gui(env: PushEnv) -> None:
    """Frame the pushing workspace and hide panels that obscure small objects."""
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client)
    p.resetDebugVisualizerCamera(
        cameraDistance=max(1.2, 1.5 * env.target_distance),
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[env.target_distance / 2, 0, 0.1],
        physicsClientId=env.client,
    )
    p.addUserDebugLine(
        [env.target_distance, -0.2, 0.005],
        [env.target_distance, 0.2, 0.005],
        lineColorRGB=[0, 0.8, 0],
        lineWidth=3,
        physicsClientId=env.client,
    )


def _create_target_env(case: EvaluationCase, metadata: dict[str, Any]) -> PushEnv:
    gui = metadata["arguments"]["gui"]
    return PushEnv(
        gui=gui,
        realtime_scale=metadata["arguments"]["speed"] if gui else 0.0,
        fixed_spec=case.spec,
        action_ranges=ActionRanges(**metadata["action_ranges"]),
        reward=RewardModel(RewardConfig(**metadata["reward"])),
        **metadata["environment"],
    )


@contextmanager
def _open_object(
    model: DeltaActorCritic,
    case: EvaluationCase,
    object_seed: int,
    metadata: dict[str, Any],
    forecast_pool: ForecastPool | None = None,
) -> Iterator[tuple[DeltaController, PushEnv]]:
    config = replace(
        DeltaControllerConfig.from_settings(metadata["controller"]), seed=object_seed
    )
    with ExitStack() as resources:
        # Open the visible world before creating any headless physics clients.
        env = _create_target_env(case, metadata)
        resources.callback(env.close)
        if metadata["arguments"]["gui"]:
            _configure_gui(env)
        video_path = metadata["arguments"].get("video")
        if video_path is not None:
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, str(video_path), physicsClientId=env.client
            )
            resources.callback(p.stopStateLogging, log_id)
        controller = DeltaController(
            model,
            config,
            env.action_ranges,
            target_distance=env.target_distance,
            forecast_pool=forecast_pool,
        )
        resources.callback(controller.close)
        controller.initialize_for_object(
            (case.spec.w, case.spec.L, case.spec.H),
            known_parameters=case.spec.theta if config.backend == "known" else None,
        )
        yield controller, env


class DeltaRun:
    """Own the model, replay, and saved results for one delta experiment."""

    def __init__(
        self,
        args: argparse.Namespace,
        model: DeltaActorCritic,
        metadata: dict[str, Any],
        directory: Path,
        forecast_pool: ForecastPool | None = None,
    ) -> None:
        self.args = args
        self.model = model
        self.metadata = metadata
        self.directory = directory
        self.forecast_pool = forecast_pool
        self.replay: deque[TrainingExample] = deque(maxlen=args.replay_capacity)
        self.rng = np.random.default_rng(args.seed + 1)
        self.episodes: list[dict[str, Any]] = []
        self.losses: list[dict[str, float]] = []
        self.objects: list[dict[str, Any]] = []
        self.collected_steps = 0
        self.learning_start_step = model.num_timesteps + args.learning_starts
        # A warm-started policy drives from the first push; only fresh runs need
        # uniform-random warmup to seed a blank policy.
        self.uniform_warmup = args.init_from is None
        self.learning_started = False
        self.stage_index: int | None = None
        self.stage_start_step = 0

    def run_objects(self, cases: list[EvaluationCase]) -> None:
        """Train the flat object stream, or a success-gated distance curriculum."""
        if self.args.curriculum_distances:
            self._run_curriculum(cases)
        else:
            self._run_flat(cases)

    def _run_curriculum(self, cases: list[EvaluationCase]) -> None:
        """Train each distance until its push cap or early success gate is met."""
        distances = self.args.curriculum_distances
        print(
            f"Curriculum: {len(distances)} stages at distances {distances}, "
            f"advance at {self.args.curriculum_steps} pushes, or after "
            f">{CURRICULUM_MIN_PUSHES} pushes and "
            f">{CURRICULUM_SUCCESS_THRESHOLD:.0%} success in the latest "
            f"{CURRICULUM_SUCCESS_WINDOW} episodes of each stage.",
            flush=True,
        )
        for stage, distance in enumerate(distances):
            print(
                f"=== Curriculum stage {stage + 1}/{len(distances)}: "
                f"target_distance={distance} ===",
                flush=True,
            )
            self.metadata["environment"]["target_distance"] = distance
            self.stage_index = stage
            self.stage_start_step = self.collected_steps
            self._train_stage(cases)
            self.save_stage_checkpoint(stage, distance)

    def _has_completed_stage(self) -> bool:
        """Stop at the push cap, or earlier after enough pushes and recent success."""
        if self.stage_index is None:
            return False
        stage_pushes = self.collected_steps - self.stage_start_step
        if stage_pushes >= self.args.curriculum_steps:
            return True
        if stage_pushes <= CURRICULUM_MIN_PUSHES:
            return False
        recent = self.episodes[-CURRICULUM_SUCCESS_WINDOW:]
        if len(recent) < CURRICULUM_SUCCESS_WINDOW:
            return False
        if any(episode["curriculum_stage"] != self.stage_index for episode in recent):
            return False
        return (
            sum(episode["success"] for episode in recent) / CURRICULUM_SUCCESS_WINDOW
            > CURRICULUM_SUCCESS_THRESHOLD
        )

    def _train_stage(self, cases: list[EvaluationCase]) -> None:
        round_index = 0
        push_limit = self.stage_start_step + self.args.curriculum_steps
        while not self._has_completed_stage():
            for case in cases:
                self.run_object(case, len(self.objects), push_limit=push_limit)
                if self._has_completed_stage():
                    return
            round_index += 1
            sampling_args = argparse.Namespace(
                **(vars(self.args) | {"seed": self.args.seed + round_index})
            )
            cases, _, _ = resolve_cases(sampling_args)

    def save_stage_checkpoint(self, stage: int, distance: float) -> None:
        self.model.training_metadata = self.metadata
        path = self.directory / f"delta_stage{stage}_d{distance:.2f}.pt"
        self.model.save(path)
        print(f"Stage {stage} (d={distance}) checkpoint saved: {path}", flush=True)

    def _run_flat(self, cases: list[EvaluationCase]) -> None:
        """
        Run each object in the evaluation set, collecting episodes and saving results.
        """
        round_index = 0
        while True:
            for case in cases:
                if (
                    self.args.mode == TRAIN_MODE
                    and self.collected_steps >= self.args.total_steps
                ):
                    return
                push_limit = self.args.total_steps if self.args.mode == TRAIN_MODE else None
                self.run_object(case, len(self.objects), push_limit=push_limit)
            if self.args.mode != TRAIN_MODE:
                return
            round_index += 1
            sampling_args = argparse.Namespace(
                **(vars(self.args) | {"seed": self.args.seed + round_index})
            )
            cases, _, _ = resolve_cases(sampling_args)

    def run_object(
        self, case: EvaluationCase, object_index: int, *, push_limit: int | None = None
    ) -> None:
        seed_index = object_index if self.args.mode == TRAIN_MODE else case.seed_index
        object_seed = self.args.seed + OBJECT_SEED_STRIDE * seed_index
        if self.args.mode != TRAIN_MODE:
            self._reset_learning_for_object(object_seed)
        details = dict(
            split=case.split, seed_index=case.seed_index, **case.classify(self.metadata)
        )
        self.objects.append(dict(spec=asdict(case.spec), **details))
        print(
            f"object={object_index} hidden parameters: m={case.spec.m:.6g} kg, "
            f"mu={case.spec.mu:.6g}, h_c={case.spec.h_c:.6g} m",
            flush=True,
        )
        with _open_object(
            self.model, case, object_seed, self.metadata, self.forecast_pool
        ) as (controller, env):
            self.run_episodes(
                controller, env, object_index, object_seed,
                details, case.spec.theta, case.spec.w, push_limit=push_limit,
            )
            if self.args.mode == ADAPT_MODE:
                self.model.training_metadata = self.metadata
                self.model.save(self.directory / f"object_{object_index:04d}.pt")

    def _reset_learning_for_object(self, object_seed: int) -> None:
        """Start each held-out object from independent checkpoint weights and replay."""
        self.model = DeltaActorCritic.load(self.args.checkpoint, device="cpu")
        self.model.rng.manual_seed(object_seed)
        self.replay.clear()
        self.learning_started = False
        self.learning_start_step = self.model.num_timesteps + self.args.learning_starts

    def run_episodes(
        self, controller: DeltaController, env: PushEnv,
        object_index: int, object_seed: int, object_details: dict[str, Any],
        true_physics: tuple[float, float, float], object_width: float,
        *, push_limit: int | None = None,
    ) -> None:
        for episode_index in range(self.args.episodes_per_object):
            remaining = None if push_limit is None else push_limit - self.collected_steps
            if remaining is not None and remaining <= 0:
                break
            updates_before = self.model.num_updates
            episode = run_episode(
                env, controller, self.model,
                seed=object_seed + episode_index,
                learning_start_step=self.learning_start_step,
                on_transition=self.learn_from_transition,
                max_steps=remaining,
                info_gain_beta=self.args.info_gain_beta,
                uniform_warmup=self.uniform_warmup,
            )
            self._annotate_episode(
                episode, controller, object_details, object_index, episode_index,
                updates_before, true_physics, object_width,
            )
            self.save_episode(episode)
            if self._has_completed_stage():
                break

    def _annotate_episode(
        self, episode: EpisodeRun, controller: DeltaController,
        object_details: dict[str, Any], object_index: int, episode_index: int,
        updates_before: int, true_physics: tuple[float, float, float], object_width: float,
    ) -> None:
        """Attach object identity, task settings and learning/model accuracy metrics."""
        episode.record.update(object_details)
        episode.record.update(object_index=object_index, episode_index=episode_index)
        episode.record["curriculum_stage"] = self.stage_index
        episode.record["target_distance"] = self.metadata["environment"]["target_distance"]
        episode.record["gradient_steps"] = self.model.num_updates - updates_before
        estimate = posterior_mean_physics(controller)
        if estimate is not None:
            episode.record.update(sysid_accuracy_record(estimate, true_physics, object_width))
        if getattr(self.model, "latent_world_model", None) is not None:
            episode.record.update(world_model_tilt_error(episode.examples))

    def learn_from_transition(
        self,
        transition: TrainingExample | None,
        controller: DeltaController,
    ) -> None:
        self.collected_steps += 1
        if self.args.mode == EVALUATE_MODE:
            return
        if transition is not None:
            self.replay.append(transition)
        if self.model.num_timesteps <= self.learning_start_step or not self.replay:
            return
        if not self.learning_started:
            print(
                f"Learning warmup complete at push={self.model.num_timesteps}; "
                f"replay={len(self.replay)}; updates every {self.args.train_freq} pushes.",
                flush=True,
            )
            self.learning_started = True
        if (self.model.num_timesteps - self.learning_start_step) % self.args.train_freq:
            return
        for _ in range(self.args.gradient_steps):
            self.update_from_replay(controller)

    def update_from_replay(self, controller: DeltaController) -> None:
        indices = self.rng.integers(len(self.replay), size=self.args.batch_size)
        batch = [self.replay[int(index)] for index in indices]
        start = time.perf_counter()
        losses = self.model.train_step(batch, controller.forecast_for_training)
        losses["update_seconds"] = time.perf_counter() - start
        self.losses.append(to_json_value(losses))
        self.record_update(losses)

    def record_update(self, metrics: dict[str, float]) -> None:
        record = dict(
            push_count=self.collected_steps,
            object_index=len(self.objects) - 1,
            replay_size=len(self.replay),
            **metrics,
        )
        with (self.directory / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if (
            self.model.num_updates == 1
            or self.model.num_updates % self.args.log_every == 0
        ):
            values = " ".join(f"{name}={value:.5g}" for name, value in record.items())
            print(f"learning {values}", flush=True)

    def save_checkpoint(self) -> None:
        self.model.training_metadata = self.metadata
        path = self.directory / "delta_latest.pt"
        temporary = path.with_suffix(".pt.tmp")
        self.model.save(temporary)
        temporary.replace(path)
        print(f"Checkpoint saved at push={self.collected_steps}: {path}", flush=True)

    def save_episode(self, episode: EpisodeRun) -> None:
        self.save_actions(episode)
        record = to_json_value(episode.record)
        self.episodes.append(record)
        with (self.directory / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"object={record['object_index']} episode={record['episode_index']} "
            f"pushes={record['real_interactions']} reward={record['episode_reward']:.3f} "
            f"success={record['success']} toppled={record['toppled']} "
            f"timeout={record['timeout']} stopped={record['stopped']} "
            f"transitions={len(episode.examples)}",
            flush=True,
        )
        if self.args.gui and self.args.linger:
            time.sleep(self.args.linger)
        if self.args.mode != EVALUATE_MODE and (
            self.collected_steps // self.args.checkpoint_every
            > (self.collected_steps - record["real_interactions"])
            // self.args.checkpoint_every
        ):
            self.save_checkpoint()

    def save_actions(self, episode: EpisodeRun) -> None:
        record = episode.record
        parameters = self.objects[record["object_index"]]["spec"]
        first_push = self.collected_steps - len(episode.traces)
        with (self.directory / "actions.jsonl").open("a") as stream:
            for episode_push, trace in enumerate(episode.traces, start=1):
                row = self._action_log_row(
                    trace, record, parameters, episode_push, first_push
                )
                stream.write(json.dumps(to_json_value(row), allow_nan=False) + "\n")

    def _action_log_row(
        self, trace: dict[str, Any], record: dict[str, Any],
        parameters: dict[str, float], episode_push: int, first_push: int,
    ) -> dict[str, Any]:
        geometry = tuple(parameters[name] for name in ("w", "L", "H"))
        action = normalized_to_physical(
            trace["normalized_action"], geometry,
            ActionRanges(**self.metadata["action_ranges"]), clip=True,
        )
        row = dict(
            push_count=first_push + episode_push, episode_push=episode_push,
            is_probe=episode_push == 1, normalized_action=trace["normalized_action"],
            action=dict(height_m=action.h, force_n=action.F, duration_s=action.tau),
            object_parameters=parameters,
            remaining_distance_before_m=float(abs(trace["observation"][0])),
            remaining_distance_after_m=float(abs(trace["next_observation"][0])),
        )
        row.update({key: record[key] for key in (
            "object_index", "episode_index", "curriculum_stage", "target_distance", "seed"
        )})
        row.update({key: trace[key] for key in (
            "reward", "success", "toppled", "terminated", "truncated",
            "execution_status", "execution_failed",
        )})
        return row

    def summarize(self) -> dict[str, Any]:
        return dict(
            episode_count=len(self.episodes),
            real_interactions=sum(
                episode["real_interactions"] for episode in self.episodes
            ),
            probe_interactions=sum(
                episode["probe_interactions"] for episode in self.episodes
            ),
            actor_interactions=sum(
                episode["actor_interactions"] for episode in self.episodes
            ),
            transitions=sum(episode["transition_count"] for episode in self.episodes),
            gradient_steps=len(self.losses),
            successes=sum(episode["success"] for episode in self.episodes),
            topples=sum(episode["toppled"] for episode in self.episodes),
            timeouts=sum(episode["timeout"] for episode in self.episodes),
            search_stops=sum(episode["stopped"] for episode in self.episodes),
        )

    def save_results(self) -> dict[str, Any]:
        if self.args.mode == TRAIN_MODE:
            self.model.training_metadata = self.metadata
            self.model.save(self.directory / "delta_final.pt")
        results = dict(
            schema_version=1,
            metadata=self.metadata,
            objects=self.objects,
            summary=self.summarize(),
            losses=self.losses,
        )
        (self.directory / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False)
        )
        print(f"Delta run saved to {self.directory}", flush=True)
        return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train fresh, adapt independent checkpoint copies, or evaluate frozen weights.

    SysID persists across an object's episodes. Adaptation resets model and replay
    for each object; evaluation freezes the networks while still updating SysID.
    """
    cases, snapshot, distribution_name = resolve_cases(args)
    model, action_ranges = _load_model_and_ranges(args)
    controller_config = _build_controller_config(args, model)
    directory, run_id = create_output_directory(
        args.out_dir, Path("artifacts/delta") / args.mode / distribution_name
    )
    metadata = _build_run_metadata(
        args,
        run_id,
        snapshot,
        model,
        action_ranges,
        controller_config,
    )
    (directory / "config.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    with log_console_output(directory / "training.log"), ExitStack() as run_resources:
        pool = None
        if args.forecast_workers > 1:
            pool = ForecastPool(args.forecast_workers)
            run_resources.callback(pool.close)
        experiment = DeltaRun(args, model, metadata, directory, forecast_pool=pool)
        experiment.run_objects(cases)
        return experiment.save_results()


def main(argv: list[str] | None = None, *, default_mode: str = TRAIN_MODE) -> None:
    torch.set_num_threads(1)
    run(parse_args(argv, default_mode=default_mode))


if __name__ == "__main__":
    main()
