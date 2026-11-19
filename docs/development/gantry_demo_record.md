# PR record: sliding gantry robot and goal-directed contact-pushing demo

- **This repository:** PR #1
- **Status:** merged into `main` as a record of the gantry/demo contribution

## Code in this PR

- `src/simulation/tools/demos/arm_demo.py` — standalone GUI demo: the overhead
  X/Z sliding gantry approaches a block and pushes it forward through
  physical pad contact toward a marked goal. Ported to the current executor
  API (`GantryPushExecutor.execute(cuboid, h, F, tau)` with admittance
  control); the nominal 0.08 m/s slide speed used for near-goal push
  shortening is a local constant.
- `tests/test_arm_demo.py` — deterministic goal-termination checks for the
  demo's push loop, independent of physics.

## What it contributed

- **Robot model:** overhead-frame URDF with horizontal and vertical prismatic
  joints, an upright pushing pad, and a recessed support bracket so the pad
  contacts the block first.
- **Contact pushing:** slow approach to the rear face, vertical position
  holding, and force-limited horizontal velocity control. The push timer
  starts after contact is detected.
- **Goal-directed demo:** labelled green goal line and floor tolerance band;
  continue pushing until the goal is reached, the block topples or
  overshoots, or the maximum push budget is exhausted.
- **Configurable controls:** motor force ceiling, contact height, push
  duration, maximum pushes, goal position, playback speed, and final-state
  display time.
- **Failure handling:** reject invalid commands and unreachable goal
  positions; stop the horizontal motor and raise an error on contact timeout
  or travel-limit failure.
- **Regression coverage:** URDF/contact geometry, slide positioning, upright
  pad orientation, consecutive pushes without teleporting or homing, and
  goal-termination logic.

## How to run the demo

```bash
.venv/bin/python -m src.simulation.tools.demos.arm_demo --speed 1 --linger 10
```

The standalone robot and demo foundation from this work is extended by the
environment integration and baseline studies recorded in PR #2. The full
original description lives in this repository's PR #1.
