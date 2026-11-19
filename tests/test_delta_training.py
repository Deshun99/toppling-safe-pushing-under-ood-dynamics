"""One-step transitions, per-push TD updates, and runnable delta modes."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import pytest

from src.agents.rl.models.delta_actor_critic import (
    DeltaActorCritic,
    TrainingExample,
    ForecastFunction,
)
from src.agents.rl.training.constants import DEFAULT_DISTRIBUTION_PATH
from src.agents.rl.training.io_utils import create_output_directory
from src.agents.rl.training.train_delta import (
    environment_settings,
    parse_args,
    resolve_cases,
    run,
    run_episode,
    DeltaRun,
    world_model_tilt_error,
    _build_controller_config,
    _build_fresh_model,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.records import ActionRecord


def _record(value):
    trajectory = DenseTrajectory(
        t=[0.0, 0.05],
        x=[0.0, value],
        psi=[0.0, 0.0],
        xdot=[0.0, 0.0],
        psidot=[0.0, 0.0],
    )
    return ActionRecord((0.0, 0.0, 0.0, 0.0), PushAction(0.02, 1.0, 0.05), trajectory)


class EpisodeEnvironment:
    """Final reward is intentionally unlike every predicted trajectory."""

    def __init__(self, rewards: list[float], *, terminal: bool = True) -> None:
        self.T_max = len(rewards)
        self.rewards = rewards
        self.terminal = terminal
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
            end and self.terminal,
            end and not self.terminal,
            dict(
                action_record=_record(self.steps),
                toppled=end and self.terminal,
                success=False,
                max_tilt=0.0,
            ),
        )


class EpisodeController:
    def __init__(self, proposals=2):
        self.remaining = proposals
        self.observed = []
        self.stop_reason = "proposal_budget_exhausted"

    def reset_episode(self):
        self.reference = np.full(3, -0.5)

    def probe(self, observation):
        return self.reference.copy()

    def observe(self, record, next_observation, toppled=False):
        self.observed.append((record, toppled))

    def build_context(self, observation: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(
            observation=observation.copy(),
            real_trajectory=np.ones((4, 5), np.float32),
            reference_action=self.reference.copy(),
            weights=np.ones(1),
        )

    def propose(self, observation, *, randomize=False):
        if not self.remaining:
            return None
        self.remaining -= 1
        context = SimpleNamespace(
            real_trajectory=np.full((4, 5), 7.0, dtype=np.float32),
            reference_action=self.reference.copy(),
        )
        return SimpleNamespace(
            delta=np.full(3, 0.1),
            action=self.reference + 0.1,
            forecasts=np.full((2, 4, 5), 99.0, dtype=np.float32),
            weights=np.array([0.3, 0.7]),
            context=context,
        )


def test_td_transitions_preserve_preexecution_context():
    env = EpisodeEnvironment([-1.0, 2.0, -10.0])
    controller = EpisodeController()
    model = SimpleNamespace(num_timesteps=0)
    episode = run_episode(env, controller, model, seed=7)
    assert [example.reward for example in episode.examples] == [2.0, -10.0]
    assert episode.record["real_interactions"] == 3
    assert episode.record["probe_interactions"] == 1
    assert episode.record["actor_interactions"] == 2
    assert episode.record["toppled"]
    assert controller.observed[-1][1]
    assert episode.examples[0].metadata is not None
    np.testing.assert_array_equal(episode.examples[0].forecasts, 99.0)
    np.testing.assert_array_equal(episode.examples[0].real_trajectory, 7.0)
    np.testing.assert_array_equal(episode.examples[0].observation, np.ones(7))


def test_search_stop_retains_executed_td_transitions() -> None:
    env = EpisodeEnvironment([-1.0, 2.0, -10.0])
    episode = run_episode(
        env,
        EpisodeController(proposals=1),
        SimpleNamespace(num_timesteps=0),
        seed=0,
    )
    assert env.steps == 2
    assert episode.record["episode_reward"] == 1.0
    assert episode.record["stopped"]
    assert not episode.record["terminated"]
    assert not episode.record["truncated"]
    assert not episode.record["timeout"]
    assert episode.record["actor_interactions"] == 1
    assert len(episode.examples) == 1
    assert episode.examples[0].reward == 2.0


def test_probe_terminal_and_explicit_horizon_have_distinct_labels():
    model = SimpleNamespace(num_timesteps=0)
    probe = run_episode(
        EpisodeEnvironment([-100.0]), EpisodeController(), model, seed=1
    )
    assert probe.examples == []
    assert probe.record["real_interactions"] == 1
    assert probe.record["terminated"]
    horizon = run_episode(
        EpisodeEnvironment([-0.1, -0.2], terminal=False),
        EpisodeController(),
        model,
        seed=1,
    )
    assert horizon.record["timeout"]
    assert horizon.examples[0].reward == -0.2


def test_cli_requires_explicit_checkpoint_mode_and_preserves_outputs(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--mode", "evaluate"])
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "vanilla.pt"])
    with pytest.raises(SystemExit):
        parse_args(["--num-objects", "0"])
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "result.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        create_output_directory(directory, tmp_path / "default")
    assert marker.read_text() == "preserve"


def test_checkpoint_horizon_restored_and_explicit_override_recorded():
    fresh = parse_args([])
    assert fresh.max_decisions is None
    assert environment_settings(fresh, None)["T_max"] == 10
    evaluation = parse_args(["--mode", "evaluate", "--checkpoint", "unused.pt"])
    saved = {"environment": {"T_max": 4, "target_distance": 1.2, "epsilon": 0.03}}
    restored = environment_settings(evaluation, saved)
    assert restored["T_max"] == 4
    assert restored["target_distance"] == 1.2
    assert restored["epsilon"] == 0.03
    override = parse_args(
        ["--mode", "evaluate", "--checkpoint", "unused.pt", "--max-decisions", "2"]
    )
    assert environment_settings(override, saved)["T_max"] == 2
    smoke = parse_args(["--mode", "evaluate", "--checkpoint", "unused.pt", "--smoke"])
    assert environment_settings(smoke, saved)["T_max"] == 3


def test_paper_preset_and_explicit_overrides() -> None:
    args = parse_args(["--paper"])
    settings = environment_settings(args, None)
    assert settings["robot"] and settings["T_max"] == 40
    assert settings["epsilon"] == .08 and args.f_max == 15
    assert args.particles == 32 and args.mcmc_steps == 2
    assert args.trajectory_feature_strategy == "self-attention"
    assert len(args.curriculum_distances) == 13 and args.curriculum_steps == 10_000
    assert args.curriculum_distances[0] == .10 and args.curriculum_distances[-1] == 1.
    assert args.info_gain_beta == 50
    override = parse_args([
        "--paper", "--particles", "7", "--info-gain-beta", "0",
        "--curriculum-steps", "10000",
    ])
    assert override.particles == 7 and override.info_gain_beta == 0
    assert override.curriculum_steps == 10_000
    for mode in ("none", "learned"):
        assert parse_args(["--paper", "--world-model", mode]).info_gain_beta == 0


def test_paper_evaluation_defaults_preserve_checkpoint_task_and_embodiment() -> None:
    args = parse_args(["--paper", "--mode", "evaluate", "--checkpoint", "unused.pt"])
    assert args.episodes_per_object == 5
    assert args.benchmark_config.name == "fixed_cases.json"
    saved = {"environment": {"robot": True, "T_max": 40, "epsilon": .08}}
    assert environment_settings(args, saved)["robot"]
    assert environment_settings(args, saved)["T_max"] == 40
    assert not environment_settings(args, None)["robot"]


def test_checkpoint_restores_controller_and_explicit_override() -> None:
    source_args = parse_args(["--paper", "--world-model", "learned"])
    model = _build_fresh_model(source_args)
    model.training_metadata = {"controller": {
        "n_particles": 32, "candidates_per_batch": 4, "max_batches": 2,
        "probe_force": .8, "world_model": "learned",
        "backend": "smc2", "estimator_options": {"mcmc_steps": 1},
    }}
    args = parse_args(["--mode", "evaluate", "--checkpoint", "unused.pt"])
    restored = _build_controller_config(args, model)
    assert restored.n_particles == 32 and restored.probe_force == .8
    assert restored.world_model == "learned"
    overridden = parse_args([
        "--mode", "evaluate", "--checkpoint", "unused.pt", "--particles", "7"
    ])
    assert _build_controller_config(overridden, model).n_particles == 7
    switched = parse_args([
        "--mode", "evaluate", "--checkpoint", "unused.pt", "--backend", "cmaes"
    ])
    assert not _build_controller_config(switched, model).estimator_options
    invalid = parse_args([
        "--mode", "evaluate", "--checkpoint", "unused.pt", "--world-model", "physics"
    ])
    with pytest.raises(ValueError, match="world-model variant"):
        _build_controller_config(invalid, model)


@pytest.mark.parametrize("world_model", ["none", "learned", "physics"])
def test_paper_robot_train_and_frozen_evaluation(tmp_path: Path, world_model: str) -> None:
    source = tmp_path / "train"
    train = parse_args([
        "--paper", "--smoke", "--world-model", world_model, "--out-dir", str(source)
    ])
    trained = run(train)
    assert trained["metadata"]["environment"]["robot"]
    assert trained["summary"]["gradient_steps"] > 0
    checkpoint = source / "delta_final.pt"
    evaluation = parse_args([
        "--paper", "--smoke", "--mode", "evaluate", "--checkpoint", str(checkpoint),
        "--out-dir", str(tmp_path / "evaluate"),
    ])
    result = run(evaluation)
    assert result["metadata"]["controller"]["world_model"] == world_model
    assert result["metadata"]["environment"]["robot"]
    assert result["summary"]["gradient_steps"] == 0


@pytest.mark.parametrize("failed_step", [1, 2])
def test_execution_failure_learns_terminal_reward_without_fabricated_record(failed_step: int) -> None:
    class FailedEnvironment(EpisodeEnvironment):
        def step(self, action: np.ndarray) -> tuple:
            result = super().step(action)
            if self.steps != failed_step:
                return result
            obs, reward, _, _, _ = result
            return obs, reward, True, False, {
                "execution_status": "travel_limit", "execution_failed": True,
            }

    controller = EpisodeController()
    episode = run_episode(
        FailedEnvironment([0., -1.]), controller, SimpleNamespace(num_timesteps=0), seed=0
    )
    assert episode.record["execution_failed"]
    assert episode.record["execution_status"] == "travel_limit"
    assert episode.record["real_interactions"] == failed_step
    assert len(controller.observed) == failed_step - 1
    assert len(episode.examples) == failed_step - 1
    assert episode.traces[-1]["action_record"] is None
    if episode.examples:
        example = episode.examples[0]
        assert example.reward == -1. and example.terminated
        assert example.next_context is None
        np.testing.assert_array_equal(example.forecasts, 99.)
        np.testing.assert_array_equal(example.real_trajectory, 7.)


def test_fixed_cases_preserve_membership_and_seed_indices() -> None:
    arguments = ["--mode", "evaluate", "--checkpoint", "unused.pt"]
    fixed, _, _ = resolve_cases(
        parse_args(
            arguments
            + [
                "--benchmark-config",
                "configs/rl/evaluation_sets/fixed_cases.json",
                "--split",
                "mass_ood",
            ]
        )
    )
    assert len(fixed) == 4
    assert {case.split for case in fixed} == {"mass_ood"}
    assert [case.seed_index for case in fixed] == [8, 9, 10, 11]
    training = json.loads(DEFAULT_DISTRIBUTION_PATH.read_text())
    for case in fixed:
        assert (
            case.classify({"training_distribution": training})["membership"]
            == "outside_ranges"
        )


def test_sampled_cases_preserve_membership_and_seed_indices() -> None:
    arguments = ["--mode", "evaluate", "--checkpoint", "unused.pt"]
    training = json.loads(DEFAULT_DISTRIBUTION_PATH.read_text())
    grouped = [
        "--sampler-config",
        "configs/rl/evaluation_sets/sampled_reserved_regions.json",
        "--num-objects",
        "2",
    ]
    all_cases, _, _ = resolve_cases(parse_args(arguments + grouped))
    subset, _, _ = resolve_cases(
        parse_args(arguments + grouped + ["--split", "reserved_region_3"])
    )
    selected_from_all = [
        case for case in all_cases if case.split == "reserved_region_3"
    ]
    assert subset == selected_from_all
    assert [case.seed_index for case in subset] == [4, 5]
    for case in subset:
        assert (
            case.classify({"training_distribution": training})["membership"]
            == "excluded_region"
        )


def train_smoke_checkpoint(
    directory: Path, backend: str, strategy: str
) -> tuple[Path, dict[str, Any]]:
    args = parse_args(
        [
            "--smoke",
            "--backend",
            backend,
            "--out-dir",
            str(directory),
            "--trajectory-feature-strategy",
            strategy,
        ]
    )
    results = run(args)
    return directory / "delta_final.pt", results


@pytest.fixture(
    params=[
        ("interpolate", "smc2"),
        ("interpolate", "cmaes"),
        ("self-attention", "smc2"),
        ("self-attention", "cmaes"),
    ]
)
def trained_delta(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    strategy, backend = request.param
    checkpoint, results = train_smoke_checkpoint(tmp_path / "source", backend, strategy)
    assert checkpoint.is_file()
    return checkpoint, results


def test_td_training_smoke(trained_delta: tuple[Path, dict[str, Any]]) -> None:
    checkpoint, trained = trained_delta
    assert trained["summary"]["probe_interactions"] >= 2
    assert trained["summary"]["real_interactions"] == 6
    assert trained["summary"]["gradient_steps"] >= 1
    assert trained["summary"]["transitions"] >= 1
    model = DeltaActorCritic.load(checkpoint)
    assert (
        model.config.trajectory_feature_strategy
        == trained["metadata"]["model_config"]["trajectory_feature_strategy"]
    )


def forbid_learning(*args: object, **kwargs: object) -> None:
    raise AssertionError("Frozen evaluation must never train")


def test_frozen_td_evaluation(
    trained_delta: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, trained = trained_delta
    before = deepcopy(DeltaActorCritic.load(checkpoint).actor.state_dict())
    monkeypatch.setattr(DeltaActorCritic, "train_step", forbid_learning)
    directory = tmp_path / "evaluation"
    argv = ["--mode", "evaluate", "--checkpoint", str(checkpoint), "--smoke"]
    argv += ["--backend", trained["metadata"]["controller"]["backend"]]
    argv += ["--out-dir", str(directory)]
    evaluated = run(parse_args(argv))
    assert evaluated["summary"]["gradient_steps"] == 0
    assert not list(directory.glob("*.pt"))
    for name, value in DeltaActorCritic.load(checkpoint).actor.state_dict().items():
        np.testing.assert_array_equal(value, before[name])
    assert (
        json.loads((directory / "results.json").read_text())["summary"]
        == evaluated["summary"]
    )
    episode = json.loads((directory / "episodes.jsonl").read_text().splitlines()[0])
    assert episode["probe_interactions"] == 1 and episode["real_interactions"] >= 1


def test_td_evaluation_rejects_changed_action_bounds(
    trained_delta: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    checkpoint, _ = trained_delta
    args = parse_args(
        [
            "--mode",
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--f-max",
            "15",
            "--out-dir",
            str(tmp_path / "wrong_bounds"),
        ]
    )
    with pytest.raises(ValueError, match="bounds"):
        run(args)


def test_known_physics_training_skips_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.agents.rl.models import delta_controller

    def forbid_estimation(*args: object, **kwargs: object) -> None:
        raise AssertionError("Known-physics training must not instantiate SysID")

    monkeypatch.setattr(delta_controller, "SMC2SysID", forbid_estimation)
    monkeypatch.setattr(delta_controller, "SysID", forbid_estimation)
    result = run(
        parse_args(
            ["--smoke", "--backend", "known", "--out-dir", str(tmp_path / "known")]
        )
    )
    assert result["summary"]["gradient_steps"] > 0
    assert result["metadata"]["controller"]["backend"] == "known"
    output = capsys.readouterr().out
    assert "toppled=" in output and "reward=" in output


def test_transition_updates_happen_before_next_action_selection() -> None:
    env = EpisodeEnvironment([-1.0, 2.0, -10.0])
    control = EpisodeController()
    model = SimpleNamespace(num_timesteps=0)
    updated_at = []

    def learn(transition: object, controller: object) -> None:
        updated_at.append(
            (env.steps, None if transition is None else transition.reward)
        )

    original_propose = control.propose

    def propose(observation: np.ndarray, *, randomize: bool = False) -> object:
        assert updated_at[-1][0] == env.steps
        return original_propose(observation, randomize=randomize)

    control.propose = propose
    run_episode(env, control, model, seed=7, on_transition=learn)
    assert updated_at == [(1, None), (2, 2.0), (3, -10.0)]
    assert model.num_timesteps == 3


def test_epsilon_flag_sets_environment_success_tolerance() -> None:
    settings = environment_settings(parse_args(["--epsilon", "0.08"]), None)
    assert settings["epsilon"] == 0.08


def test_epsilon_defaults_to_standard_tolerance_when_omitted() -> None:
    settings = environment_settings(parse_args([]), None)
    assert settings["epsilon"] == 0.05


def test_default_delta_td_budgets_match_sac() -> None:
    from src.agents.rl.models.sac import SACConfig
    from src.agents.rl.training.constants import (
        DEFAULT_TOTAL_STEPS,
        DEFAULT_LEARNING_STARTS,
    )

    args = parse_args([])
    assert args.total_steps == DEFAULT_TOTAL_STEPS
    assert args.learning_starts == DEFAULT_LEARNING_STARTS
    assert args.batch_size == SACConfig.batch_size
    assert args.replay_capacity == SACConfig.buffer_size
    assert args.hidden_dim == SACConfig.hidden_dim
    assert args.tau == SACConfig.tau
    assert args.gradient_steps == 1


def test_td_training_resamples_objects_to_exact_budget_and_respects_warmup(
    tmp_path: Path,
) -> None:
    args = parse_args(["--smoke", "--backend", "known", "--out-dir", str(tmp_path)])
    args.total_steps, args.learning_starts, args.episodes_per_object = 7, 3, 1
    result = run(args)
    assert result["summary"]["real_interactions"] == 7
    assert result["summary"]["gradient_steps"] == 4
    assert len(result["objects"]) >= 3
    assert DeltaActorCritic.load(tmp_path / "delta_final.pt").num_timesteps == 7


def adapt_two_objects(checkpoint: Path, directory: Path) -> dict[str, Any]:
    args = parse_args(
        [
            "--mode",
            "adapt",
            "--checkpoint",
            str(checkpoint),
            "--backend",
            "known",
            "--out-dir",
            str(directory),
        ]
    )
    args.num_objects = 2
    args.episodes_per_object = 1
    args.max_decisions = 3
    args.batch_size = 2
    args.candidates = 2
    return run(args)


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_adaptation_resets_td_weights_and_replay_per_object(
    tmp_path: Path,
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _ = train_smoke_checkpoint(tmp_path / "source", "known", strategy)
    initial_updates = DeltaActorCritic.load(checkpoint).num_updates
    starts = {}
    original = DeltaActorCritic.train_step

    def update(
        model: DeltaActorCritic,
        examples: Sequence[TrainingExample],
        forecast: ForecastFunction,
    ) -> dict[str, float]:
        geometries = {tuple(example.observation[4:]) for example in examples}
        assert len(geometries) == 1
        starts.setdefault(next(iter(geometries)), model.num_updates)
        return original(model, examples, forecast)

    monkeypatch.setattr(DeltaActorCritic, "train_step", update)
    result = adapt_two_objects(checkpoint, tmp_path / "adapt")
    assert len(starts) == 2 and set(starts.values()) == {initial_updates}
    assert result["summary"]["gradient_steps"] >= 2
    assert len(list((tmp_path / "adapt").glob("object_*.pt"))) == 2


def test_forecast_workers_defaults_to_one() -> None:
    assert parse_args([]).forecast_workers == 1


def test_forecast_workers_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--forecast-workers", "0"])


def test_single_worker_builds_no_pool(tmp_path, monkeypatch) -> None:
    import src.agents.rl.training.train_delta as td

    constructed = []
    monkeypatch.setattr(td, "ForecastPool", lambda *a, **k: constructed.append(1))
    args = parse_args("--smoke --backend known".split() + ["--out-dir", str(tmp_path)])
    td.run(args)
    assert constructed == []


def test_init_from_allowed_in_train_mode_sets_path() -> None:
    args = parse_args(["--init-from", "prior.pt"])
    assert str(args.init_from) == "prior.pt"
    assert args.mode == "train" and args.checkpoint is None


def test_init_from_rejected_outside_train_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--mode", "adapt", "--checkpoint", "c.pt", "--init-from", "p.pt"])


def test_init_from_and_checkpoint_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--init-from", "p.pt", "--checkpoint", "c.pt"])


def test_warm_start_disables_uniform_warmup(tmp_path) -> None:
    model = SimpleNamespace(num_timesteps=5)
    fresh = DeltaRun(parse_args([]), model, {}, tmp_path)
    warm = DeltaRun(parse_args(["--init-from", "p.pt"]), model, {}, tmp_path)
    assert fresh.uniform_warmup is True
    assert warm.uniform_warmup is False


def test_world_model_tilt_error_measures_peak_tilt_gap() -> None:
    def traj(peak_tilt: float) -> np.ndarray:
        a = np.zeros((3, 5))
        a[1, 2] = peak_tilt  # column 2 is tilt
        return a

    ex = [
        SimpleNamespace(
            forecasts=traj(0.1)[None],  # (1, 3, 5) predicted
            next_context=SimpleNamespace(real_trajectory=traj(0.3)),  # actual
        )
    ]
    assert world_model_tilt_error(ex)["wm_tilt_peak_abs_err"] == pytest.approx(0.2)


def test_world_model_tilt_error_skips_examples_without_next_trajectory() -> None:
    ex = [SimpleNamespace(forecasts=np.zeros((1, 3, 5)), next_context=None)]
    assert world_model_tilt_error(ex) == {}


def test_learned_world_model_logs_tilt_error(tmp_path) -> None:
    run(parse_args(["--smoke", "--world-model", "learned", "--out-dir", str(tmp_path)]))
    eps = [json.loads(line) for line in open(tmp_path / "episodes.jsonl")]
    assert any("wm_tilt_peak_abs_err" in e for e in eps)


def test_curriculum_parses_distance_list_and_steps() -> None:
    args = parse_args(
        ["--curriculum-distances", "0.1", "0.2", "--curriculum-steps", "5"]
    )
    assert args.curriculum_distances == [0.1, 0.2]
    assert args.curriculum_steps == 5


def test_curriculum_requires_train_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "adapt",
                "--checkpoint",
                "c.pt",
                "--curriculum-distances",
                "0.1",
                "--curriculum-steps",
                "5",
            ]
        )


def test_curriculum_distances_and_steps_required_together() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--curriculum-distances", "0.1"])
    with pytest.raises(SystemExit):
        parse_args(["--curriculum-steps", "5"])


def test_curriculum_distances_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--curriculum-distances", "0.1", "-0.2", "--curriculum-steps", "5"])


def test_curriculum_runs_all_stages_and_tags_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parse_args([
        "--num-objects", "1", "--episodes-per-object", "7",
        "--curriculum-distances", "0.08", "0.12", "--curriculum-steps", "4",
    ])
    experiment, checkpoints = _run_budget_only_collection(
        args, monkeypatch, tmp_path, successes=[True] * 60,
    )
    assert checkpoints == [4, 8]
    assert {e["curriculum_stage"] for e in experiment.episodes} == {0, 1}
    assert {e["target_distance"] for e in experiment.episodes} == {0.08, 0.12}
    out = capsys.readouterr().out
    assert "Curriculum: 2 stages" in out
    assert "stage 1/2" in out and "stage 2/2" in out


def _run_budget_only_collection(
    args: Any, monkeypatch: pytest.MonkeyPatch, directory: Path,
    *, successes: Sequence[bool] | None = None,
) -> tuple[DeltaRun, list[int]]:
    """Exercise collection/budgeting without simulation, networks or training."""
    model = SimpleNamespace(num_timesteps=0, num_updates=0)
    metadata = {"environment": {"target_distance": args.target_distance}}
    experiment = DeltaRun(args, model, metadata, directory)
    checkpoints: list[int] = []
    outcomes = iter(successes) if successes is not None else None

    def save_episode(episode: Any) -> None:
        if outcomes is not None:
            episode.record["success"] = next(outcomes)
        experiment.episodes.append(episode.record)

    def count_push(transition: Any, controller: Any) -> None:
        experiment.collected_steps += 1

    monkeypatch.setattr(experiment, "learn_from_transition", count_push)
    monkeypatch.setattr(experiment, "save_episode", save_episode)
    monkeypatch.setattr(experiment, "save_stage_checkpoint", lambda *_: checkpoints.append(experiment.collected_steps))
    monkeypatch.setattr(
        "src.agents.rl.training.train_delta._open_object",
        lambda *_: nullcontext((EpisodeController(proposals=100), EpisodeEnvironment([0.] * 3, terminal=False))),
    )
    cases, _, _ = resolve_cases(args)
    experiment.run_objects(cases)
    return experiment, checkpoints


@pytest.mark.parametrize("stage_budget", [1, 5, 7])
def test_curriculum_stops_at_hard_cap_without_success(
    stage_budget: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    args = parse_args([
        "--num-objects", "1", "--episodes-per-object", "7",
        "--curriculum-distances", ".1", ".2", ".3",
        "--curriculum-steps", str(stage_budget),
    ])
    experiment, checkpoints = _run_budget_only_collection(args, monkeypatch, tmp_path)
    assert checkpoints == [stage_budget, 2 * stage_budget, 3 * stage_budget]
    assert experiment.model.num_timesteps == 3 * stage_budget
    for stage in range(3):
        episodes = [e for e in experiment.episodes if e["curriculum_stage"] == stage]
        assert sum(e["real_interactions"] for e in episodes) == stage_budget
        assert not any(e["success"] for e in episodes)
        assert episodes[-1]["stop_reason"] == "interaction_budget"


def test_curriculum_advances_early_and_resets_pushes_and_success_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    args = parse_args([
        "--num-objects", "1", "--episodes-per-object", "7",
        "--curriculum-distances", ".1", ".2", ".3",
        "--curriculum-steps", "10000",
    ])
    experiment, checkpoints = _run_budget_only_collection(
        args, monkeypatch, tmp_path, successes=[True] * 2001,
    )
    assert checkpoints == [2001, 4002, 6003]
    for stage in range(3):
        episodes = [e for e in experiment.episodes if e["curriculum_stage"] == stage]
        assert len(episodes) == 667
        assert all(e["real_interactions"] == 3 and not e["stopped"] for e in episodes)


@pytest.mark.parametrize(
    "pushes, successes, count, stage, expected",
    [(2000, 30, 30, 0, False), (2001, 27, 30, 0, False),
     (2001, 28, 30, 0, True), (2001, 29, 29, 0, False),
     (2001, 30, 30, 1, False), (9999, 0, 0, 0, False),
     (10000, 0, 0, 0, True)],
)
def test_curriculum_requires_strict_minimum_and_stage_local_success_window(
    pushes: int, successes: int, count: int, stage: int, expected: bool, tmp_path: Path,
) -> None:
    args = parse_args(["--paper"])
    model = SimpleNamespace(num_timesteps=0, num_updates=0)
    experiment = DeltaRun(args, model, {}, tmp_path)
    experiment.stage_index = stage
    experiment.stage_start_step = 500
    experiment.collected_steps = 500 + pushes
    experiment.episodes = [
        dict(curriculum_stage=0, success=index < successes) for index in range(count)
    ]
    assert experiment._has_completed_stage() is expected


def test_curriculum_keeps_training_until_recent_success_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    args = parse_args([
        "--num-objects", "1", "--episodes-per-object", "7",
        "--curriculum-distances", ".1", "--curriculum-steps", "10000",
    ])
    experiment, checkpoints = _run_budget_only_collection(
        args, monkeypatch, tmp_path, successes=[False] * 700 + [True] * 28,
    )
    assert len(experiment.episodes) == 728
    assert checkpoints == [728 * 3]


def test_flat_budget_keeps_the_same_collection_stop_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    args = parse_args(["--num-objects", "1", "--episodes-per-object", "2", "--total-steps", "7"])
    experiment, checkpoints = _run_budget_only_collection(args, monkeypatch, tmp_path)
    assert experiment.collected_steps == experiment.model.num_timesteps == 7
    assert not checkpoints
    last = experiment.episodes[-1]
    assert last["stopped"] and last["stop_reason"] == "interaction_budget"
    assert not last["terminated"] and not last["truncated"] and not last["timeout"]


def test_warm_start_continues_steps_and_takes_new_reward(tmp_path) -> None:
    source, _ = train_smoke_checkpoint(tmp_path / "src", "known", "interpolate")
    source_steps = DeltaActorCritic.load(source).num_timesteps
    args = parse_args(
        [
            "--smoke",
            "--backend",
            "known",
            "--out-dir",
            str(tmp_path / "warm"),
            "--init-from",
            str(source),
            "--topple-penalty",
            "7",
        ]
    )
    result = run(args)
    # New reward comes from args, not restored from the loaded checkpoint's metadata.
    assert result["metadata"]["reward"]["R_topple"] == 7
    # The step counter continues past the loaded checkpoint rather than resetting.
    resumed = DeltaActorCritic.load(tmp_path / "warm" / "delta_final.pt")
    assert resumed.num_timesteps > source_steps
