from collections.abc import Iterator

import pybullet as p
import pybullet_data
import pytest

from src.simulation.modelling.object import Cuboid

from src.simulation.execution.robot import (
    GantryPushExecutor,
    HOME_POSITIONS,
    OutsidePlaneError,
    PAD_HALF_HEIGHT,
    PAD_LATERAL_FRICTION,
)


@pytest.fixture
def client() -> Iterator[int]:
    client_id = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    plane = p.loadURDF('plane.urdf', physicsClientId=client_id)
    p.changeDynamics(plane, -1, lateralFriction=1.0, physicsClientId=client_id)
    yield client_id
    p.disconnect(client_id)


@pytest.fixture
def executor(client: int) -> GantryPushExecutor:
    return GantryPushExecutor(client)


def test_gantry_has_two_linear_axes_and_fixed_pad(executor: GantryPushExecutor) -> None:
    infos = [p.getJointInfo(executor.arm, i, physicsClientId=executor.client)
             for i in range(p.getNumJoints(executor.arm, physicsClientId=executor.client))]
    assert [i[1].decode() for i in infos] == ['horizontal_joint', 'vertical_joint', 'pad_joint']
    assert [i[2] for i in infos] == [p.JOINT_PRISMATIC, p.JOINT_PRISMATIC, p.JOINT_FIXED]
    assert p.getDynamicsInfo(executor.arm, executor.pad, physicsClientId=executor.client)[1] == PAD_LATERAL_FRICTION


@pytest.mark.parametrize('target', [(-0.23, .02), (1.05, .65), (-.065, .04), (.3, .18)])
def test_pad_reaches_targets_and_remains_upright(executor: GantryPushExecutor, target: tuple[float, float]) -> None:
    executor.move_to_joints(executor.pad_ik(*target))
    position, orientation = executor.pad_pose()
    assert position == pytest.approx((target[0], 0, target[1]), abs=.002)
    assert orientation == pytest.approx((0, 0, 0, 1), abs=1e-8)


@pytest.mark.parametrize('target', [(-.24, .1), (1.06, .1), (0, .01), (0, .66), (float('nan'), .1)])
def test_outside_travel_is_rejected(executor: GantryPushExecutor, target: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match='travel'):
        executor.pad_ik(*target)


def test_home_is_clear_of_block(executor: GantryPushExecutor) -> None:
    assert executor.joint_positions() == pytest.approx(HOME_POSITIONS)
    assert executor.pad_pose()[0][2] > .5


def test_translated_base_is_respected(client: int) -> None:
    executor = GantryPushExecutor(client, base_position=(1, 2, .1))
    executor.place_joints(executor.pad_ik(1.4, .3))
    assert executor.pad_pose()[0] == pytest.approx((1.4, 2, .3))


@pytest.fixture
def block(executor: GantryPushExecutor) -> Cuboid:
    from src.simulation.modelling.object import Cuboid, CuboidSpec
    block = Cuboid(CuboidSpec(w=.08, L=.08, H=.20, m=1, mu=.5, h_c=.08), executor.client)
    block.spawn(position=(0, 0, .105))
    for _ in range(120):
        p.stepSimulation(physicsClientId=executor.client)
    return block


def test_contact_push_moves_block_and_preserves_height(executor: GantryPushExecutor, block: Cuboid) -> None:
    from src.simulation.modelling.constants import DT
    initial = block.state()[0]
    record = executor.execute(block, h=.04, F=10, tau=.2, warmup_steps=8)
    assert block.state()[0] > initial + .01
    assert len(record.t) == round(.2 / DT)
    assert record.t[-1] == pytest.approx((len(record.t) - 1) * DT)
    assert executor.pad_pose()[0][2] == pytest.approx(.04, abs=.002)
    assert abs(block.state()[1]) < block.spec.crit_tilt


def test_successive_pushes_do_not_teleport_or_home(executor: GantryPushExecutor, block: Cuboid, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('push must not teleport or home')
    monkeypatch.setattr(executor, 'place_joints', forbidden)
    monkeypatch.setattr(executor, 'home', forbidden)
    for _ in range(3):
        before = block.state()[0]
        executor.execute(block, .04, 10, .15)
        assert block.state()[0] > before + .01
        assert executor.pad_pose()[0][2] == pytest.approx(.04, abs=.002)


def test_zero_force_does_not_approach_block(executor: GantryPushExecutor, block: Cuboid) -> None:
    before = block.state()[0]
    pad = executor.pad_pose()[0]
    executor.execute(block, .04, 0, .3)
    assert block.state()[0] == pytest.approx(before, abs=.001)
    assert executor.pad_pose()[0] == pytest.approx(pad, abs=.001)


@pytest.mark.parametrize('height,force,duration', [(.04, -1, .3),
                                                   (.04, 101, .3), (.04, 10, 0)])
def test_invalid_push_fails_before_motion(executor: GantryPushExecutor, block: Cuboid, height: float, force: float, duration: float) -> None:
    before = executor.pad_pose()[0]
    with pytest.raises(ValueError):
        executor.execute(block, height, force, duration)
    assert executor.pad_pose()[0] == before


def test_block_outside_pushing_plane_raises(executor: GantryPushExecutor, client: int) -> None:
    from src.simulation.modelling.object import Cuboid, CuboidSpec
    off_plane = Cuboid(CuboidSpec(w=.08, L=.08, H=.20, m=1, mu=.5, h_c=.08), client)
    off_plane.spawn(position=(0, .05, .105))
    for _ in range(120):
        p.stepSimulation(physicsClientId=client)
    with pytest.raises(OutsidePlaneError):
        executor._approach(off_plane, .04)
    off_plane.remove()


def test_out_of_window_heights_clamp_into_pad_window(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor.execute(block, h=0.0, F=10, tau=.3)  # below the pad window
    report = executor.last_execution
    assert report.h_cmd == 0.0
    assert report.h_real == pytest.approx(PAD_HALF_HEIGHT, abs=.01)


def test_travel_limit_stops_motor(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor.place_joints((1.28, -.61))
    with pytest.raises(ValueError, match='travel limit'):
        executor.execute(block, .04, 10, .3)
    assert p.getJointState(executor.arm, executor.horizontal,
                           physicsClientId=executor.client)[0] <= 1.28 + .001


def test_support_bracket_is_recessed_behind_pad() -> None:
    import xml.etree.ElementTree as ET
    from src.simulation.execution.robot import DEFAULT_URDF_PATH
    root = ET.parse(DEFAULT_URDF_PATH).getroot()
    pad_origin = root.find("joint[@name='pad_joint']/origin")
    px, py, pz = map(float, pad_origin.attrib['xyz'].split())
    pad_box = root.find("link[@name='pad']/collision/geometry/box")
    dx, dy, dz = map(float, pad_box.attrib['size'].split())
    for collision in root.findall("link[@name='vertical_slide']/collision"):
        x, y, z = map(float, collision.find('origin').attrib['xyz'].split())
        sx, sy, sz = map(float, collision.find('geometry/box').attrib['size'].split())
        if abs(y - py) < (sy + dy) / 2 and abs(z - pz) < (sz + dz) / 2:
            assert x + sx / 2 <= px + dx / 2 - .004


def test_approach_reaches_pad_contact_without_support_contact(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor._approach(block, .04)
    contacts = p.getContactPoints(executor.arm, block.body,
                                  physicsClientId=executor.client)
    assert any(c[3] == executor.pad for c in contacts)
    assert all(c[3] == executor.pad for c in contacts)


def test_execution_report_records_realized_push(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor.execute(block, h=.04, F=10, tau=.2, warmup_steps=8)
    report = executor.last_execution
    assert report.contact_made
    assert report.h_real == pytest.approx(.04, abs=.01)
    assert report.F_real > 0
    assert 0 < report.tau_real <= .2
    assert report.approach_steps > 0
    assert report.impulse_real > 0
    assert report.execution_time > 0


def test_clean_push_has_no_stray_contact(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor.execute(block, .04, 10, .15)
    assert executor.last_execution.stray_contact_steps == 0


def test_zero_force_report_has_no_contact(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor.execute(block, .04, 0, .3)
    report = executor.last_execution
    assert not report.contact_made
    assert report.F_real is None
    assert report.tau_real == 0
    assert report.approach_steps == 0


def test_report_survives_failed_push(executor: GantryPushExecutor, block: Cuboid) -> None:
    executor.place_joints((1.28, -.61))
    with pytest.raises(ValueError, match='travel limit'):
        executor.execute(block, .04, 10, .3)
    report = executor.last_execution
    assert report is not None
    assert report.F_cmd == 10
    assert not report.contact_made


@pytest.mark.parametrize('requested_force,feedback,speed,expected', [
    (5.0, 4.0, .2, 1.2), (5.0, 5.0, .2, .2),
    (5.0, 8.0, .2, 0.0), (15.0, 0.0, 0.0, 3.0),
    (1.0, .5, -.2, .5),
])
def test_admittance_uses_force_error_and_block_velocity(requested_force: float, feedback: float, speed: float, expected: float) -> None:
    assert GantryPushExecutor.admittance_velocity(requested_force, feedback, speed) == expected


def test_recorded_force_matches_timed_motion_samples(executor: GantryPushExecutor, block: Cuboid) -> None:
    from src.simulation.modelling.constants import DT
    trajectory = executor.execute(block, .04, 10, .2, warmup_steps=8)
    assert len(executor.applied_forces) == len(trajectory.t) == round(.2 / DT)
    assert executor.timed_start is not None
    assert trajectory.t[0] == 0.0
    assert executor.timed_start[0] < trajectory.x[-1]
    assert any(sample.force_x > 0 for sample in executor.applied_forces)
    assert all(0 <= sample.height <= block.spec.H for sample in executor.applied_forces)
    assert executor.last_execution.approach_steps > 0
    assert executor.last_execution.tau_cmd == .2


def test_height_change_retracts_before_vertical_motion(executor: GantryPushExecutor, block: Cuboid, monkeypatch: pytest.MonkeyPatch) -> None:
    executor.execute(block, .04, 5, .05)
    calls = []
    move = executor.move_to_joints

    def record_move(target: tuple[float, float], *args: object, **kwargs: object) -> int:
        calls.append(target)
        return move(target, *args, **kwargs)

    monkeypatch.setattr(executor, 'move_to_joints', record_move)
    old_x, old_z = executor.joint_positions()
    executor.execute(block, .10, 5, .05)
    assert len(calls) >= 2
    assert calls[0][0] < old_x
    assert calls[0][1] == pytest.approx(old_z)
    assert calls[1][0] == pytest.approx(calls[0][0])
    assert calls[1][1] > calls[0][1]


def test_geometry_matches_paper_workspace(executor: GantryPushExecutor) -> None:
    horizontal = p.getJointInfo(executor.arm, executor.horizontal, physicsClientId=executor.client)
    vertical = p.getJointInfo(executor.arm, executor.vertical, physicsClientId=executor.client)
    assert horizontal[8:10] == (0.0, 1.28)
    assert vertical[8:10] == (-.63, 0.0)
    executor.place_joints((1.28, -.63))
    assert executor.pad_pose()[0] == pytest.approx((1.05, 0, .02))


def test_home_recovers_small_dynamic_travel_overshoot(executor: GantryPushExecutor) -> None:
    p.resetJointState(executor.arm, executor.horizontal, 1.281,
                     physicsClientId=executor.client)
    p.resetJointState(executor.arm, executor.vertical, -.61,
                     physicsClientId=executor.client)
    executor.home()
    assert executor.joint_positions() == pytest.approx(HOME_POSITIONS, abs=.002)


@pytest.mark.parametrize('force', [5.0, 10.0])
def test_nominal_contact_force_tracks_request(
    executor: GantryPushExecutor, block: Cuboid, force: float,
) -> None:
    trajectory = executor.execute(block, .04, force, .3)
    report = executor.last_execution
    assert report.F_real == pytest.approx(force, rel=.25)
    assert report.tau_real >= .9 * .3
    assert max(map(abs, trajectory.psi)) < .05
