"""Delta action bounds and command-line settings for training and evaluation."""

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import torch

from src.agents.rl.training.constants import (
    TRAIN_MODE,
    ADAPT_MODE,
    EVALUATE_MODE,
    DEFAULT_DISTRIBUTION_PATH,
    DEFAULT_TOTAL_STEPS,
    DEFAULT_LEARNING_STARTS,
    DEFAULT_CURRICULUM_MAX_PUSHES,
)
from src.agents.rl.models.sac import SACConfig
from src.simulation.execution.settling import RestConfig
from src.evaluation.rl.constants import DEFAULT_EPISODE_LINGER_S, DEFAULT_PLAYBACK_SPEED


def compute_delta_bounds(
    reference_action: torch.Tensor,
    delta_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect configured delta limits with the final action's [-1, 1] bounds."""
    reference = reference_action.to(torch.float64)
    lower = torch.maximum(-delta_scale.double(), -1 - reference)
    upper = torch.minimum(delta_scale.double(), 1 - reference)
    # Round inward so float32 samples remain valid against float64 references.
    lower_float, upper_float = lower.float(), upper.float()
    lower_float = torch.where(
        lower_float.double() < lower,
        torch.nextafter(lower_float, torch.full_like(lower_float, float("inf"))),
        lower_float,
    )
    upper_float = torch.where(
        upper_float.double() > upper,
        torch.nextafter(upper_float, torch.full_like(upper_float, -float("inf"))),
        upper_float,
    )
    return lower_float, upper_float


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--paper", action="store_true", help="use the gantry paper experiment defaults"
    )
    parser.add_argument(
        "--robot", action=argparse.BooleanOptionalAction, default=None,
        help="execute with the gantry; otherwise restore the checkpoint embodiment",
    )
    parser.add_argument(
        "--mode",
        choices=(TRAIN_MODE, ADAPT_MODE, EVALUATE_MODE),
        default=parser.get_default("mode"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="warm-start training from a saved delta checkpoint; weights and "
        "architecture load from it while all other settings come from these args",
    )
    _add_case_arguments(parser)
    _add_collection_arguments(parser)


def _add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--distribution-config", type=Path, default=DEFAULT_DISTRIBUTION_PATH
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--benchmark-config",
        type=Path,
        help="fixed benchmark groups; adapt/evaluate only",
    )
    sources.add_argument(
        "--sampler-config",
        type=Path,
        help="single or grouped benchmark samplers; adapt/evaluate only",
    )
    parser.add_argument("--split", nargs="+", help="selected benchmark groups, or all")


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--num-objects", type=int, default=8)
    parser.add_argument("--episodes-per-object", type=int, default=20)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="explicit horizon override; defaults to checkpoint T_max or 10 fresh",
    )
    parser.add_argument("--target-distance", type=float, default=1.0)
    parser.add_argument(
        "--curriculum-distances",
        type=float,
        nargs="+",
        help="ordered target distances for curriculum training; replaces "
        "--target-distance; advance at the push cap or early on success (train only)",
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        help="maximum pushes per distance; advance earlier after >2000 pushes "
        "and over 90%% success in its latest 30 episodes (train only)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="goal distance tolerance in metres for success; restored from checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gui", action="store_true", help="Watch frozen evaluation in PyBullet"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_PLAYBACK_SPEED,
        help="GUI pacing: 1 = real time, 0.5 = half speed, 0 = unpaced",
    )
    parser.add_argument(
        "--linger",
        type=float,
        default=DEFAULT_EPISODE_LINGER_S,
        help="Pause after each GUI episode (seconds)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Record the GUI session to an MP4 file (requires --gui and ffmpeg on PATH)",
    )


def _add_controller_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("smc2", "cmaes", "known"),
        default="smc2",
        help="known uses true object physics without inference",
    )
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument(
        "--forecast-workers",
        type=int,
        default=1,
        help="parallel CPU worker processes for internal-sim forecasts; 1 = serial (no pool)",
    )
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--cma-replay-budget", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--probe-force", type=float, default=0.5)
    parser.add_argument("--probe-duration", type=float, default=0.05)
    parser.add_argument("--probe-height-fraction", type=float, default=0.1)
    parser.add_argument("--f-max", type=float, default=None)
    parser.add_argument(
        "--world-model",
        choices=("none", "physics", "learned"),
        default="physics",
        help=(
            "none: model-free, no InternalSim or SysID; "
            "physics: InternalSim+SysID ensemble (default); "
            "learned: neural latent model replaces both"
        ),
    )


def _add_learning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="target-environment interaction budget for training, including probes",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="uniform delta warmup; defaults to 10000 pushes for train, 0 for adapt",
    )
    parser.add_argument("--tau", type=float, default=SACConfig.tau)
    parser.add_argument("--hidden-dim", type=int, default=SACConfig.hidden_dim)
    parser.add_argument("--trajectory-points", type=int, default=16)
    parser.add_argument(
        "--trajectory-feature-strategy",
        choices=("interpolate", "self-attention"),
        default="interpolate",
        help="interpolate samples for an MLP, or attend to all recorded samples",
    )
    parser.add_argument("--delta-scale", type=float, nargs=3, default=(0.3, 0.4, 0.4))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)


def _add_update_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=SACConfig.batch_size)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="run gradient-steps updates every N real pushes after warmup (default 1)",
    )
    parser.add_argument("--replay-capacity", type=int, default=SACConfig.buffer_size)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="print learning diagnostics every N gradient updates",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="replace delta_latest.pt every N training pushes",
    )
    parser.add_argument("--smoke", action="store_true")


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if (args.mode != TRAIN_MODE) != (args.checkpoint is not None):
        parser.error(
            "train creates fresh delta weights; adapt/evaluate require --checkpoint"
        )
    if args.init_from is not None:
        if args.checkpoint is not None:
            parser.error(
                "use --checkpoint for adapt/evaluate or --init-from to warm-start "
                "training, not both"
            )
        if args.mode != TRAIN_MODE:
            parser.error(
                "--init-from warm-starts training; adapt/evaluate use --checkpoint"
            )
    if (args.curriculum_distances is None) != (args.curriculum_steps is None):
        parser.error(
            "--curriculum-distances and --curriculum-steps must be used together"
        )
    if args.curriculum_distances is not None:
        if args.mode != TRAIN_MODE:
            parser.error("--curriculum-distances is train-mode only")
        if args.curriculum_steps < 1:
            parser.error("--curriculum-steps must be positive")
        if any(not np.isfinite(d) or d <= 0 for d in args.curriculum_distances):
            parser.error("--curriculum-distances must all be positive and finite")
    if args.gui and args.mode != EVALUATE_MODE:
        parser.error("--gui requires --mode evaluate")
    if args.video is not None and not args.gui:
        parser.error("--video requires --gui")
    if args.mode == TRAIN_MODE and (
        args.benchmark_config or args.sampler_config or args.split
    ):
        parser.error(
            "train uses --distribution-config; benchmark sources/splits are for adapt/evaluate"
        )
    _validate_counts(parser, args)
    _validate_ranges(parser, args)


def _validate_counts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "total_steps",
        "num_objects",
        "episodes_per_object",
        "particles",
        "cma_replay_budget",
        "candidates",
        "max_batches",
        "hidden_dim",
        "trajectory_points",
        "batch_size",
        "gradient_steps",
        "train_freq",
        "replay_capacity",
        "log_every",
        "checkpoint_every",
        "forecast_workers",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def _validate_ranges(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if any(not np.isfinite(value) or value < 0 for value in (args.speed, args.linger)):
        parser.error("--speed and --linger must be finite and nonnegative")
    if args.learning_starts is not None and args.learning_starts < 0:
        parser.error("--learning-starts must be nonnegative")
    if not np.isfinite(args.tau) or not 0 <= args.tau <= 1:
        parser.error("--tau must be within [0, 1]")
    if args.seed < 0 or args.mcmc_steps < 0:
        parser.error("seed and mcmc-steps must be nonnegative")
    if args.max_decisions is not None and args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    for name in ("target_distance", "epsilon", "lr", "probe_force", "probe_duration"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.f_max is not None and (not np.isfinite(args.f_max) or args.f_max <= 0):
        parser.error("--f-max must be positive and finite")
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        parser.error("--gamma must be within [0, 1]")


def _apply_smoke_settings(args: argparse.Namespace) -> None:
    args.total_steps = 6
    args.learning_starts = 1
    args.num_objects = 1
    args.episodes_per_object = 2
    args.max_decisions = 3
    args.particles = 4
    args.mcmc_steps = 0
    args.cma_replay_budget = 16
    args.candidates = 16
    args.max_batches = 2
    args.hidden_dim = 32
    args.batch_size = 4


def _add_reward_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reward-strategy",
        choices=("linear", "goal_proximity"),
        help="fresh runs default to linear; checkpoints restore their saved strategy",
    )
    parser.add_argument(
        "--proximity-bonus", type=float, help="goal_proximity score bound (default 100)"
    )
    parser.add_argument(
        "--proximity-scale",
        type=float,
        help="distance in metres at half the proximity score (default 0.25)",
    )
    parser.add_argument(
        "--no-progress-penalty",
        type=float,
        help="extra goal_proximity penalty for insufficient progress (default 0, disabled)",
    )
    parser.add_argument(
        "--no-progress-threshold",
        type=float,
        help="net goal-distance reduction in metres at or below which to penalize (default 0.001)",
    )
    parser.add_argument(
        "--topple-penalty",
        type=float,
        help="reward subtracted on toppling; restore checkpoint setting when omitted (default 100)",
    )
    parser.add_argument(
        "--info-gain-beta",
        type=float,
        default=0.0,
        help="weight on particle-cloud information-gain reward shaping; 0 disables (needs smc2/cmaes)",
    )


def _validate_reward_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for name in ("no_progress_penalty", "no_progress_threshold"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.topple_penalty is not None and (
        not np.isfinite(args.topple_penalty) or args.topple_penalty < 0
    ):
        parser.error("--topple-penalty must be finite and nonnegative")
    if not np.isfinite(args.info_gain_beta) or args.info_gain_beta < 0:
        parser.error("--info-gain-beta must be finite and nonnegative")
    if args.proximity_bonus is not None and (
        not np.isfinite(args.proximity_bonus) or args.proximity_bonus < 0
    ):
        parser.error("--proximity-bonus must be finite and nonnegative")
    if args.proximity_scale is not None and (
        not np.isfinite(args.proximity_scale) or args.proximity_scale <= 0
    ):
        parser.error("--proximity-scale must be finite and positive")


def _add_rest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait-for-rest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="wait for passive settling after pushes; restore checkpoint setting when omitted",
    )
    parser.add_argument(
        "--rest-timeout",
        type=float,
        help="maximum passive wait in simulated seconds; 0 waits until rest (default 2)",
    )


def _set_paper_defaults(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    """Use supplied-paper settings and supplementary run budgets; flags override."""
    preliminary, _ = parser.parse_known_args(argv)
    if not preliminary.paper:
        return
    parser.set_defaults(
        particles=32, candidates=4, max_batches=2,
        trajectory_feature_strategy="self-attention", trajectory_points=16,
    )
    if preliminary.mode != TRAIN_MODE:
        parser.set_defaults(episodes_per_object=5)
        if not preliminary.sampler_config:
            parser.set_defaults(benchmark_config=Path(__file__).resolve().parents[4]
                                / "configs/rl/evaluation_sets/fixed_cases.json")
        return
    distances = [.10, .15, .20, .25, .30, .35, .40, .50, .60, .70, .80, .90, 1.0]
    parser.set_defaults(
        robot=True, max_decisions=40, epsilon=.08, f_max=15.,
        episodes_per_object=10,
        curriculum_distances=None if preliminary.smoke else distances,
        curriculum_steps=None if preliminary.smoke else DEFAULT_CURRICULUM_MAX_PUSHES,
        batch_size=64, train_freq=8, learning_starts=200,
        reward_strategy="goal_proximity", topple_penalty=10.,
        no_progress_penalty=1.5,
        info_gain_beta=50. if preliminary.world_model == "physics" else 0.,
    )


def _warn_unused_options(args: argparse.Namespace) -> None:
    if args.world_model != "physics" and args.forecast_workers > 1:
        warnings.warn(
            "--forecast-workers has no effect with --world-model %s: "
            "the forecast pool is only used in physics mode." % args.world_model,
            stacklevel=2,
        )
    if args.info_gain_beta and args.backend == "known":
        warnings.warn(
            "--info-gain-beta has no effect with --backend known: the true physics "
            "are given, so the posterior has no uncertainty to reduce.",
            stacklevel=2,
        )


def parse_args(
    argv: list[str] | None = None, *, default_mode: str = TRAIN_MODE
) -> argparse.Namespace:
    """Keep CPU smoke budgets small while retaining the actual online pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(mode=default_mode)
    _add_run_arguments(parser)
    _add_controller_arguments(parser)
    _add_learning_arguments(parser)
    _add_update_arguments(parser)
    _add_reward_arguments(parser)
    _add_rest_arguments(parser)
    options = sys.argv[1:] if argv is None else argv
    _set_paper_defaults(parser, options)
    args = parser.parse_args(options)
    args.explicit_options = [v.split("=", 1)[0] for v in options if v.startswith("--")]
    _validate_arguments(parser, args)
    _validate_reward_arguments(parser, args)
    try:
        RestConfig(bool(args.wait_for_rest), 2.0 if args.rest_timeout is None else args.rest_timeout)
    except ValueError as error:
        parser.error(str(error))
    if args.learning_starts is None:
        args.learning_starts = DEFAULT_LEARNING_STARTS if args.mode == TRAIN_MODE else 0
    _warn_unused_options(args)
    if args.smoke:
        _apply_smoke_settings(args)
        if args.paper:
            args.train_freq = 1
    return args
