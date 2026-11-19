"""Watch the gantry push a block forward through physical pad contact.

Run from the repo root: python -m src.simulation.tools.demos.arm_demo
Use --heights-only to run the original positioning demonstration.
"""

import argparse
import math
import time
from typing import Final

import pybullet as p
import pybullet_data

from src.simulation.execution.robot import (
    PAD_HALF_DEPTH,
    PAD_HALF_HEIGHT,
    GantryPushExecutor,
)
from src.simulation.modelling.constants import DT, GRAVITY
from src.simulation.modelling.object import Cuboid, CuboidSpec

GOAL_TOLERANCE = 0.05  # matches the non-robot environment
STANDOFF = 0.01  # clearance between the pad face and the rear face (m)

PUSH_SPEED: Final = 0.08  # nominal horizontal slide speed (m/s) used to shorten pushes near the goal
HEIGHTS = (0.04, 0.12, 0.18)  # 0.2H, 0.6H, 0.9H of the demo block (H = 0.20)


def _spawn_world() -> tuple[int, Cuboid]:
    client = p.connect(p.GUI)
    p.setGravity(0, 0, GRAVITY, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    plane = p.loadURDF("plane.urdf", physicsClientId=client)
    p.changeDynamics(plane, -1, lateralFriction=1.0, physicsClientId=client)
    block = Cuboid(
        CuboidSpec(w=0.08, L=0.08, H=0.20, m=1.0, mu=0.5, h_c=0.08), client
    )
    block.spawn(position=(0.0, 0.0, 0.105))
    for _ in range(round(0.5 / DT)):
        p.stepSimulation(physicsClientId=client)
    p.resetDebugVisualizerCamera(1.5, 45, -20, (0.1, 0, 0.45), physicsClientId=client)
    return client, block


def mark_goal(client: int, goal_x: float, tolerance: float = GOAL_TOLERANCE) -> None:
    """Draw the block-centre goal and tolerance used by the non-robot env."""
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=(tolerance, 0.12, 0.001),
        rgbaColor=(0.1, 0.8, 0.2, 0.35), physicsClientId=client,
    )
    p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
        basePosition=(goal_x, 0, 0.001), physicsClientId=client,
    )
    p.addUserDebugLine(
        (goal_x, -0.14, 0.004), (goal_x, 0.14, 0.004),
        lineColorRGB=(0, 0.6, 0.1), lineWidth=3, physicsClientId=client,
    )
    p.addUserDebugText(
        f"GOAL: block centre x={goal_x:.2f} m (+/- {tolerance:.2f} m)",
        (goal_x - tolerance, 0.16, 0.03), textColorRGB=(0, 0.5, 0.1),
        textSize=1.2, physicsClientId=client,
    )


def _pad_target(block: Cuboid, h: float) -> tuple[float, float]:
    """Pad-centre standoff pose whose face covers the rear-face height h."""
    block_x, _, _ = block.pose()[0]
    x_pad = block_x - block.spec.w / 2 - PAD_HALF_DEPTH - STANDOFF
    return x_pad, max(PAD_HALF_HEIGHT, h)


def visit_heights(executor: GantryPushExecutor, block: Cuboid) -> None:
    for h in HEIGHTS:
        x_pad, z_pad = _pad_target(block, h)
        print(f"pad to rear-face height {h:.2f} m -> pad centre ({x_pad:.3f}, {z_pad:.3f})")
        executor.move_to_joints(executor.pad_ik(x_pad, z_pad))
    print("homing...")
    executor.home()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="physics replay speed: 1.0 = real time, "
        "0 = as fast as possible (default: 1.0)",
    )
    parser.add_argument("--linger", type=float, default=2.0,
                        help="seconds to keep the final state on screen (default: 2.0)")
    parser.add_argument("--heights-only", action="store_true")
    parser.add_argument("--pushes", "--max-pushes", dest="pushes", type=int, default=20,
                        help="maximum pushes before giving up (default: 20)")
    parser.add_argument("--height", type=float, default=0.04)
    parser.add_argument("--force", type=float, default=10.0, help="motor force ceiling in N")
    parser.add_argument("--duration", type=float, default=0.6, help="seconds per push")
    parser.add_argument("--goal-x", type=float, default=0.50,
                        help="block-centre goal in world X, metres (default: 0.50)")
    args = parser.parse_args()
    if args.pushes < 1 or args.speed < 0 or args.linger < 0:
        parser.error("pushes must be positive; speed and linger must be nonnegative")
    if not math.isfinite(args.goal_x) or not 0 < args.duration < math.inf:
        parser.error("goal-x must be finite and duration must be finite and positive")
    return args


def _push_outcome(block: Cuboid, goal_x: float) -> str | None:
    x, tilt, _, _ = block.state()
    if abs(tilt) >= block.spec.crit_tilt:
        return "block toppled"
    if abs(goal_x - x) <= GOAL_TOLERANCE:
        return f"goal reached: block centre x={x:.3f} m"
    if x > goal_x + GOAL_TOLERANCE:
        return "goal overshot; forward-only gantry cannot correct it"
    return None


def push_forward(executor: GantryPushExecutor, block: Cuboid,
                 args: argparse.Namespace) -> None:
    # Reject an unreachable centre goal before starting the scripted sequence.
    executor.pad_ik(args.goal_x - block.spec.w / 2 - PAD_HALF_DEPTH, args.height)
    for index in range(args.pushes):
        outcome = _push_outcome(block, args.goal_x)
        if outcome:
            print(outcome)
            return
        before = block.state()[0]
        duration = min(args.duration, (args.goal_x - before) / PUSH_SPEED)
        executor.execute(block, args.height, args.force, duration)
        after = block.state()[0]
        print(f"push {index + 1}: x={after:.3f} m, moved {after - before:.3f} m")
    print(_push_outcome(block, args.goal_x) or
          f"maximum {args.pushes} pushes reached before the goal")


def main() -> None:
    args = _parse_args()
    client, block = _spawn_world()
    try:
        mark_goal(client, args.goal_x)
        executor = GantryPushExecutor(client, realtime_scale=args.speed)
        if args.heights_only:
            visit_heights(executor, block)
        else:
            push_forward(executor, block, args)
        if args.linger > 0:
            print(f"keeping the final state on screen for {args.linger} s...")
            time.sleep(args.linger)
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()

