"""Contact-force requests executed by the paper's two-axis Cartesian gantry.

The outer admittance loop integrates force error into a position target. The
motor effort ceiling belongs to the inner servo and is independent of requested
contact force. Forecasts deliberately omit this executor.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Final

import pybullet as p

from src.simulation.execution.force_push import AppliedForce, DenseTrajectory, PushExecutor
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import Cuboid, ObjectState

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
JointPositions = tuple[float, float]
DEFAULT_URDF_PATH: Final = Path(__file__).resolve().parents[3] / 'assets/two_joint_arm/two_joint_arm.urdf'
PAD_ZERO_HEIGHT: Final = 0.65
PAD_X_OFFSET: Final = 0.12
X_TRAVEL: Final = (0.0, 1.28)
Z_TRAVEL: Final = (-0.63, 0.0)
DEFAULT_BASE_POSITION: Final = (-0.35, 0.0, 0.0)
HOME_POSITIONS: Final = (0.03, 0.0)
PAD_LATERAL_FRICTION: Final = 0.1
PAD_HALF_DEPTH: Final = 0.015
PAD_HALF_HEIGHT: Final = 0.02
APPROACH_SPEED: Final = 0.02
POSITION_CONTROL_FORCE: Final = 100.0
K_ADM: Final = 1.0  # m/(N s), paper Eq. (2)
MAX_PUSH_SPEED: Final = 3.0
FORCE_FILTER_TIME_CONSTANT: Final = 0.02
FORCE_FILTER_ALPHA: Final = DT / (FORCE_FILTER_TIME_CONSTANT + DT)
# Lower inner gains avoid impulsive contact loss under the outer force-error loop.
PUSH_POSITION_GAIN: Final = 0.001
PUSH_VELOCITY_GAIN: Final = 0.01


class TravelLimitError(ValueError):
    """A requested pose or forward push exceeds gantry travel."""


class BlockToppledError(ValueError):
    """The block toppled while establishing contact, before the timed push."""


class OutsidePlaneError(ValueError):
    """The block left the fixed x-z pushing plane."""


@dataclass
class ExecutionReport:
    """Normal-force summaries refer only to contact within the timed push."""

    h_cmd: float
    F_cmd: float
    tau_cmd: float
    contact_made: bool = False
    approach_steps: int = 0
    h_real: float | None = None
    F_real: float | None = None
    tau_real: float = 0.0
    max_tilt_approach: float = 0.0
    max_tilt_push: float = 0.0
    stray_contact_steps: int = 0
    impulse_approach: float = 0.0
    impulse_real: float = 0.0
    execution_time: float = 0.0
    timed_steps: int = 0
    contact_steps: int = 0


class GantryPushExecutor(PushExecutor):
    """Reach contact, then track requested force with an admittance controller."""

    def __init__(
        self, client_id: int, urdf: str | Path = DEFAULT_URDF_PATH,
        base_position: Vector3 = DEFAULT_BASE_POSITION, realtime_scale: float = 0.0,
    ) -> None:
        self.client = client_id
        self.realtime_scale = realtime_scale
        self.base_position = base_position
        p.setTimeStep(DT, physicsClientId=self.client)
        self.arm = p.loadURDF(str(urdf), basePosition=base_position,
                              useFixedBase=True, physicsClientId=self.client)
        self.horizontal = self._find_joint('horizontal_joint')
        self.vertical = self._find_joint('vertical_joint')
        self.pad = self._find_joint('pad_joint')
        p.changeDynamics(self.arm, self.pad, lateralFriction=PAD_LATERAL_FRICTION,
                         physicsClientId=self.client)
        self.last_execution: ExecutionReport | None = None
        self.timed_start: ObjectState | None = None
        self.applied_forces: list[AppliedForce] = []
        self.place_joints(HOME_POSITIONS)

    def _find_joint(self, name: str) -> int:
        for index in range(p.getNumJoints(self.arm, physicsClientId=self.client)):
            info = p.getJointInfo(self.arm, index, physicsClientId=self.client)
            if info[1].decode() == name:
                return index
        raise RuntimeError(f'joint {name!r} not found in gantry URDF')

    def pad_ik(self, x_pad: float, z_pad: float) -> JointPositions:
        positions = (x_pad - self.base_position[0] - PAD_X_OFFSET,
                     z_pad - self.base_position[2] - PAD_ZERO_HEIGHT)
        self._validate_positions(positions)
        return positions

    @staticmethod
    def _validate_positions(positions: JointPositions) -> None:
        for value, (lower, upper) in zip(positions, (X_TRAVEL, Z_TRAVEL)):
            # Resting floor penetration can displace the nominal lower pad edge.
            if not math.isfinite(value) or not lower - 1e-4 <= value <= upper + 1e-4:
                raise TravelLimitError(f'slide target {positions} is outside gantry travel')

    def joint_positions(self) -> JointPositions:
        return tuple(p.getJointState(self.arm, joint, physicsClientId=self.client)[0]
                     for joint in (self.horizontal, self.vertical))

    def pad_pose(self) -> tuple[Vector3, Quaternion]:
        return p.getLinkState(self.arm, self.pad, physicsClientId=self.client)[:2]

    def _command_joints(self, positions: JointPositions) -> None:
        self._validate_positions(positions)
        for joint, position in zip((self.horizontal, self.vertical), positions):
            p.setJointMotorControl2(self.arm, joint, p.POSITION_CONTROL,
                                   targetPosition=position, force=POSITION_CONTROL_FORCE,
                                   maxVelocity=0.5, physicsClientId=self.client)

    def place_joints(self, positions: JointPositions) -> None:
        """Place joints only at construction or explicit diagnostic setup."""
        self._command_joints(positions)
        for joint, position in zip((self.horizontal, self.vertical), positions):
            p.resetJointState(self.arm, joint, position, targetVelocity=0.0,
                             physicsClientId=self.client)

    def _advance(self) -> None:
        p.stepSimulation(physicsClientId=self.client)
        if self.realtime_scale > 0:
            time.sleep(DT / self.realtime_scale)

    def move_to_joints(
        self, positions: JointPositions, tolerance: float = 1e-3, timeout_steps: int = 720,
    ) -> int:
        self._command_joints(positions)
        for step in range(1, timeout_steps + 1):
            self._advance()
            if all(abs(current - target) <= tolerance
                   for current, target in zip(self.joint_positions(), positions)):
                return step
        raise TimeoutError(f'gantry did not reach {positions} within {timeout_steps} steps')

    def home(self) -> int:
        """Raise before retracting so a fresh block cannot obstruct homing."""
        horizontal, vertical = self.joint_positions()
        horizontal = min(X_TRAVEL[1], max(X_TRAVEL[0], horizontal))
        steps = 0
        if vertical < -1e-3:
            steps += self.move_to_joints((horizontal, HOME_POSITIONS[1]))
        if abs(horizontal - HOME_POSITIONS[0]) > 1e-3:
            steps += self.move_to_joints(HOME_POSITIONS)
        return steps

    def _horizontal_velocity(self, speed: float, force: float) -> None:
        p.setJointMotorControl2(self.arm, self.horizontal, p.VELOCITY_CONTROL,
                               targetVelocity=speed, force=force, physicsClientId=self.client)

    def _check_forward_travel(self) -> None:
        if self.joint_positions()[0] >= X_TRAVEL[1] - 0.002:
            raise TravelLimitError('push reached the gantry horizontal travel limit')

    def _contacts(self, cuboid: Cuboid) -> tuple:
        return p.getContactPoints(self.arm, cuboid.body, self.pad, physicsClientId=self.client)

    def _pad_contact(self, cuboid: Cuboid) -> bool:
        return bool(self._contacts(cuboid))

    def _pad_contact_force(self, cuboid: Cuboid) -> float:
        return sum(point[9] for point in self._contacts(cuboid))

    def _stray_contact_step(self, cuboid: Cuboid) -> bool:
        points = p.getContactPoints(self.arm, cuboid.body, physicsClientId=self.client)
        return any(point[3] != self.pad for point in points)

    def _position_for_approach(self, cuboid: Cuboid, height: float) -> JointPositions:
        position, orientation = cuboid.pose()
        rear = cuboid.rear_face_point(position, orientation, height)
        target = self.pad_ik(rear[0] - PAD_HALF_DEPTH, rear[2])
        current, _ = self.pad_pose()
        if abs(rear[1] - current[1]) > 0.01:
            raise OutsidePlaneError('block is outside the gantry pushing plane')
        if abs(current[2] - rear[2]) > 0.002:
            x = min(current[0], rear[0] - PAD_HALF_DEPTH - 0.01)
            self._move_and_count(self.pad_ik(x, current[2]))
            self._move_and_count(self.pad_ik(x, rear[2]))
            self._move_and_count(self.pad_ik(rear[0] - PAD_HALF_DEPTH - 0.01, rear[2]))
        return target

    def _move_and_count(self, target: JointPositions) -> None:
        steps = self.move_to_joints(target)
        if self.last_execution is not None:
            self.last_execution.approach_steps += steps

    def _approach(self, cuboid: Cuboid, height: float) -> None:
        target = self._position_for_approach(cuboid, height)
        self._command_joints((self.joint_positions()[0], target[1]))
        self._horizontal_velocity(APPROACH_SPEED, 1.0)
        for _ in range(2400):
            self._check_forward_travel()
            self._advance()
            self._measure_approach(cuboid)
            if abs(cuboid.state()[1]) > cuboid.spec.crit_tilt:
                raise BlockToppledError('block toppled before the timed push')
            if self._pad_contact(cuboid):
                return
        raise TimeoutError('gantry could not establish pad contact with the block')

    def _measure_approach(self, cuboid: Cuboid) -> None:
        report = self.last_execution
        if report is None:
            return
        report.approach_steps += 1
        report.max_tilt_approach = max(report.max_tilt_approach, abs(cuboid.state()[1]))
        report.stray_contact_steps += int(self._stray_contact_step(cuboid))
        report.impulse_approach += self._pad_contact_force(cuboid) * DT
        if self._pad_contact(cuboid):
            report.contact_made = True
            report.h_real = self.pad_pose()[0][2]

    @staticmethod
    def _validate_push(h: float, F: float, tau: float, warmup_steps: int) -> None:
        if not all(math.isfinite(v) for v in (h, F, tau)):
            raise ValueError('push parameters must be finite')
        if F < 0 or F > POSITION_CONTROL_FORCE or tau <= 0:
            raise ValueError('force must be in [0, 100] N and duration must be positive')
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError('warmup_steps must be a nonnegative integer')

    @staticmethod
    def admittance_velocity(force: float, measured_force: float, block_velocity: float) -> float:
        """Paper Eq. (2): force error and ongoing object motion set pad speed."""
        target = max(0.0, block_velocity) + K_ADM * (force - measured_force)
        return min(MAX_PUSH_SPEED, max(0.0, target))

    def _command_push(self, x_target: float, velocity: float) -> None:
        self._check_forward_travel()
        p.setJointMotorControl2(
            self.arm, self.horizontal, p.POSITION_CONTROL,
            targetPosition=min(X_TRAVEL[1], x_target), targetVelocity=velocity,
            force=POSITION_CONTROL_FORCE, maxVelocity=MAX_PUSH_SPEED,
            positionGain=PUSH_POSITION_GAIN, velocityGain=PUSH_VELOCITY_GAIN,
            physicsClientId=self.client,
        )

    def execute(
        self, cuboid: Cuboid, h: float, F: float, tau: float, warmup_steps: int = 0,
    ) -> DenseTrajectory:
        """Exclude positioning/approach from the timed samples and replay start."""
        self._validate_push(h, F, tau, warmup_steps)
        if cuboid.spec.H < 2 * PAD_HALF_HEIGHT:
            raise ValueError('object is shorter than the pushing pad')
        self.last_execution = ExecutionReport(h, F, tau)
        self.applied_forces = []
        self.timed_start = None
        feasible_h = min(max(h, PAD_HALF_HEIGHT), cuboid.spec.H - PAD_HALF_HEIGHT)
        try:
            for _ in range(warmup_steps):
                self._advance()
            if F > 0:
                self._approach(cuboid, feasible_h)
            self.timed_start = cuboid.state()
            return self._timed_push(cuboid, F, tau)
        finally:
            report = self.last_execution
            report.execution_time = (report.approach_steps + report.timed_steps) * DT
            self._horizontal_velocity(0.0, POSITION_CONTROL_FORCE)

    def _timed_push(
        self, cuboid: Cuboid, force: float, tau: float,
    ) -> DenseTrajectory:
        trajectory = DenseTrajectory()
        x_target = self.joint_positions()[0]
        measured = self._pad_contact_force(cuboid)
        for step in range(max(1, round(tau / DT))):
            if force > 0:
                measured += FORCE_FILTER_ALPHA * (self._pad_contact_force(cuboid) - measured)
                velocity = self.admittance_velocity(force, measured, cuboid.state()[2])
                x_target += velocity * DT
                self._command_push(x_target, velocity)
            else:
                self._horizontal_velocity(0.0, POSITION_CONTROL_FORCE)
            self._advance()
            trajectory.append(step * DT, *cuboid.state())
            self._measure_push(cuboid)
        return trajectory

    def _measure_push(self, cuboid: Cuboid) -> None:
        report = self.last_execution
        contacts = self._contacts(cuboid)
        normal_force = sum(point[9] for point in contacts)
        report.timed_steps += 1
        report.max_tilt_push = max(report.max_tilt_push, abs(cuboid.state()[1]))
        report.stray_contact_steps += int(self._stray_contact_step(cuboid))
        if contacts:
            report.contact_steps += 1
            report.tau_real = report.contact_steps * DT
            report.impulse_real += normal_force * DT
            report.F_real = report.impulse_real / report.tau_real
        force, point = self._resultant(contacts, cuboid)
        position, orientation = cuboid.pose()
        inverse_position, inverse_orientation = p.invertTransform(position, orientation)
        local_point, _ = p.multiplyTransforms(inverse_position, inverse_orientation, point,
                                               (0, 0, 0, 1))
        self.applied_forces.append(AppliedForce(force[0], local_point[2] + cuboid.spec.h_c))

    @staticmethod
    def _resultant(contacts: tuple, cuboid: Cuboid) -> tuple[Vector3, Vector3]:
        """Return world force on the object and its normal-weighted contact point.

        PyBullet normal and lateral vectors describe force on body A (gantry),
        so invert them for body B (object). The force patch is approximated by
        one application point; its full distributed contact wrench is not saved.
        """
        force = [0.0, 0.0, 0.0]
        point = [0.0, 0.0, 0.0]
        normal = sum(contact[9] for contact in contacts)
        for contact in contacts:
            for axis in range(3):
                force[axis] -= (contact[9] * contact[7][axis]
                                + contact[10] * contact[11][axis]
                                + contact[12] * contact[13][axis])
                point[axis] += contact[9] * contact[6][axis]
        if normal:
            point = [coordinate / normal for coordinate in point]
        else:
            point = list(cuboid.pose()[0])
        return tuple(force), tuple(point)
