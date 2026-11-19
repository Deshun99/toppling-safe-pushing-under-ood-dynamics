"""Watch the gymnasium env run with a random agent (demo / debugging).

Opens the PyBullet GUI and samples the action space (h, F, tau) until the
cuboid topples or reaches the target (max T_max = 10 decisions per episode).
The hidden object parameters are printed to the console each episode; tilt
is printed in radians (1 rad ~ 57 deg). The physics is paced at real time by
default so the push is watchable (--speed); trajectories are identical at
any speed.

Run from the repo root:  python -m src.simulation.tools.demos.env_demo [--seed N] [--episodes K]
                          [--speed S] [--linger S] [--cam-dist D]
"""

import argparse
import time

import pybullet as p

from src.simulation.environment import PushEnv


def run_episode(env: PushEnv, episode: int) -> None:
    _, info = env.reset(seed=episode)
    print(f"--- episode {episode}: true theta (m, mu, h_c) = {info['true_theta']} ---")
    done = False
    t = 0
    while not done:
        _, r, terminated, truncated, info = env.step(env.action_space.sample())
        tilt = max(abs(p) for p in env.last_trajectory.psi)
        h, F, tau = info["executed"]
        print(
            f"  step {t}: (h={h:.3f}, F={F:.1f}, tau={tau:.2f})  "
            f"r={r:+.3f}  toppled={info['toppled']}  max_tilt={tilt:.2f} rad"
        )
        done = terminated or truncated
        t += 1
    if info["success"]:
        outcome = "success"
    elif info["toppled"]:
        outcome = "toppled"
    else:
        outcome = "truncated"
    print(f"  outcome: {outcome} after {t} decisions\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the first episode (default: 0)"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="number of episodes to run (default: 1)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="physics replay speed: 1.0 = real time, "
        "0 = as fast as possible (default: 1.0)",
    )
    parser.add_argument(
        "--linger",
        type=float,
        default=2.0,
        help="seconds to keep the final state on screen before closing (default: 2.0)",
    )
    parser.add_argument(
        "--cam-dist",
        type=float,
        default=None,
        help="debug-visualizer camera distance in m, i.e. the "
        "zoom level (default: leave the camera alone; "
        "~0.35 is a close-up of the cuboid)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = PushEnv(gui=True, realtime_scale=args.speed)
    if args.cam_dist is not None:
        p.resetDebugVisualizerCamera(
            cameraDistance=args.cam_dist,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.1],  # mid-height of the nominal cuboid
            physicsClientId=env.client,
        )
    try:
        for i in range(args.episodes):
            run_episode(env, args.seed + i)
        if args.linger > 0:
            print(f"keeping the final state on screen for {args.linger} s...")
            time.sleep(args.linger)
    finally:
        env.close()


if __name__ == "__main__":
    main()
