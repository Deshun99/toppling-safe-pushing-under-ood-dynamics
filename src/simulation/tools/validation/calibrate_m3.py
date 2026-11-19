"""Inspect maximum-force pushes and reward scales before M3 training.

Tests maximum-duration pushes at low and high heights, and compares terminal
bonuses with per-action reward terms from random episodes. A failed maximum
push does not establish that the force limit is unsuitable: weaker or shorter
pushes within the same action range may succeed. These trials do not search
for safe target-reaching actions or measure learned-policy performance.

Prints measured outcomes and their limits, without a keep/tune decision for
F_max. No files or settings are changed. Freeze settings only after suitable
training-distribution evaluation (outline sections 5.3 and 11.1).

Run from the repo root:  python -m src.simulation.tools.validation.calibrate_m3 [--fmax-list 10,15,20,25]
                          [--episodes 200] [--seed 0]
"""

import argparse

import numpy as np

from src.simulation.environment import (
    ACTION_RANGES,
    ActionRanges,
    PushEnv,
    RewardConfig,
)
from src.simulation.modelling.object import CuboidSpec

# Specs that stress the corners of the default object bounds (outline section 9.1).
EXTREME_SPECS: dict[str, CuboidSpec] = {
    "heaviest_high_mu": CuboidSpec(
        w=0.10, L=0.10, H=0.25, m=1.5, mu=0.70, h_c=0.15
    ),  # hardest to move
    "light_tall_high_com": CuboidSpec(
        w=0.05, L=0.05, H=0.25, m=0.5, mu=0.30, h_c=0.15
    ),  # easiest to topple
    "mid_a": CuboidSpec(w=0.075, L=0.075, H=0.18, m=1.0, mu=0.50, h_c=0.09),
    "mid_b": CuboidSpec(w=0.09, L=0.06, H=0.15, m=1.25, mu=0.40, h_c=0.075),
}

# Max-push action: lowest height (0.1H), max force, max duration (0.5 s).
MAX_PUSH = (-1.0, 1.0, 1.0)
# Deliberately unsafe action: highest height (0.9H), max force, max duration.
UNSAFE_PUSH = (1.0, 1.0, 1.0)


def sweep_fmax(fmax_values: list[float], seed: int) -> list[dict]:
    """Push every extreme spec with the max action at each candidate F_max."""
    rows = []
    for fmax in fmax_values:
        for name, spec in EXTREME_SPECS.items():
            env = PushEnv(
                action_ranges=ActionRanges(force=(0.0, fmax)), fixed_spec=spec
            )
            try:
                row = _measure_max_pushes(env, seed)
                row.update(fmax=fmax, spec=name, m=spec.m, mu=spec.mu)
                rows.append(row)
                # One deliberately unsafe push: does the danger exist?
                _, unsafe_info = env.reset(seed=seed)
                _, _, _, _, unsafe_info = env.step(UNSAFE_PUSH)
                rows[-1]["unsafe_toppled"] = bool(unsafe_info["toppled"])
                rows[-1]["unsafe_max_tilt"] = float(unsafe_info["max_tilt"])
            finally:
                env.close()
    return rows


def _measure_max_pushes(env: PushEnv, seed: int) -> dict:
    env.reset(seed=seed)
    displacements = []
    for _ in range(env.T_max):
        observation, _, terminated, truncated, info = env.step(MAX_PUSH)
        displacements.append(env.target_distance - observation[0])
        if terminated or truncated:
            break
    return {
        "reached": bool(info["success"]),
        "toppled": bool(info["toppled"]),
        "actions": len(displacements),
        "dx_first": float(displacements[0]),
        "dx_total": float(displacements[-1]),
    }


def reward_diagnostics(episodes: int, seed: int) -> dict:
    """Random episodes over the default object bounds; decompose the reward into terms."""
    reward = RewardConfig()
    env = PushEnv(action_ranges=ACTION_RANGES)
    env.action_space.seed(seed)
    try:
        term_r = []  # per-action |r| with terminal bonuses removed
        ep_return = []
        ep_success = []
        ep_topple = []
        for episode in range(episodes):
            total, success, toppled, action_terms = _measure_random_episode(
                env, seed + episode, reward
            )
            term_r.extend(action_terms)
            ep_return.append(total)
            ep_success.append(success)
            ep_topple.append(toppled)
    finally:
        env.close()
    return {
        "mean_episode_return": float(np.mean(ep_return)),
        "success_rate": float(np.mean(ep_success)),
        "topple_rate": float(np.mean(ep_topple)),
        "mean_per_action_term": float(np.mean(term_r)),
    }


def _measure_random_episode(
    env: PushEnv,
    seed: int,
    reward: RewardConfig,
) -> tuple[float, bool, bool, list[float]]:
    env.reset(seed=seed)
    total_reward = 0.0
    action_terms = []
    while True:
        _, value, terminated, truncated, info = env.step(env.action_space.sample())
        terminal_bonus = (reward.R_succ if info["success"] else 0.0) - (
            reward.R_topple if info["toppled"] else 0.0
        )
        action_terms.append(abs(value - terminal_bonus))
        total_reward += value
        if terminated or truncated:
            return (
                total_reward,
                bool(info["success"]),
                bool(info["toppled"]),
                action_terms,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fmax-list",
        type=str,
        default="10,15,20,25",
        help="comma-separated F_max candidates in N (default: 10,15,20,25)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="random episodes for the reward diagnostics (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for all episodes (default: 0)"
    )
    return parser.parse_args()


def _print_force_sweep(rows: list[dict]) -> None:
    print("=== F_max sweep: max push (h=0.1H, F=F_max, tau=0.5 s) ===")
    print(
        f"{'F_max':>5} {'spec':<20} {'m':>5} {'mu':>4} "
        f"{'reached':>8} {'toppled':>8} {'actions':>8} "
        f"{'dx_first':>9} {'dx_total':>9} {'unsafe_toppled':>15}"
    )
    for r in rows:
        print(
            f"{r['fmax']:>5.0f} {r['spec']:<20} {r['m']:>5.2f} {r['mu']:>4.2f} "
            f"{str(r['reached']):>8} {str(r['toppled']):>8} "
            f"{r['actions']:>8} {r['dx_first']:>9.3f} "
            f"{r['dx_total']:>9.2f} {str(r['unsafe_toppled']):>15}"
        )


def _print_reward_metrics(diag: dict, episodes: int, reward: RewardConfig) -> None:
    print(
        "\n=== Reward diagnostics over "
        f"{episodes} random episodes (the default object bounds) ==="
    )
    print(f"success rate:            {diag['success_rate']:.2f}")
    print(f"topple rate:             {diag['topple_rate']:.2f}")
    print(f"mean episode return:     {diag['mean_episode_return']:+.2f}")
    print(
        f"mean per-action |term|:  {diag['mean_per_action_term']:.3f} "
        f"(R_succ={reward.R_succ}, R_topple={reward.R_topple})"
    )


def _print_force_findings(rows: list[dict]) -> None:
    current_fmax = ACTION_RANGES.force[1]
    heaviest = next(
        r for r in rows if r["spec"] == "heaviest_high_mu" and r["fmax"] == current_fmax
    )
    light_tall = next(
        r
        for r in rows
        if r["spec"] == "light_tall_high_com" and r["fmax"] == current_fmax
    )
    print(
        f"Heaviest/highest-friction object, low-height max-push trial at {current_fmax:g} N: "
        f"first-push displacement={heaviest['dx_first']:.3f} m; "
        f"trial ended toppled={heaviest['toppled']}."
    )
    print(
        f"Light/tall object, high-height maximum push: toppled={light_tall['unsafe_toppled']}."
    )
    print(
        "Toppling in a maximum-push trial does not show that F_max needs changing. "
        "These trials do not establish whether the action range permits safe "
        "target-reaching; test weaker and shorter pushes to assess that."
    )


def _print_calibration_summary(
    rows: list[dict], diagnostics: dict, reward: RewardConfig
) -> None:
    print("\n=== Findings ===")
    _print_force_findings(rows)
    bonuses_dominate = (
        min(reward.R_succ, reward.R_topple) >= 5.0 * diagnostics["mean_per_action_term"]
    )
    print(f"Terminal bonuses >= 5x mean per-action |term|: {bonuses_dominate}")
    if not bonuses_dominate:
        print(
            "Terminal bonuses fall below the reward-scale check in these random "
            "episodes. Review reward scales using representative rollouts."
        )


def main() -> None:
    args = _parse_args()
    fmax_values = [float(value) for value in args.fmax_list.split(",")]
    if ACTION_RANGES.force[1] not in fmax_values:
        fmax_values.append(ACTION_RANGES.force[1])
    reward = RewardConfig()
    rows = sweep_fmax(fmax_values, args.seed)
    _print_force_sweep(rows)
    diagnostics = reward_diagnostics(args.episodes, args.seed)
    _print_reward_metrics(diagnostics, args.episodes, reward)
    _print_calibration_summary(rows, diagnostics, reward)


if __name__ == "__main__":
    main()
