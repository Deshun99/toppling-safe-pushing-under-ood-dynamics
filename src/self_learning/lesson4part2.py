import time

import pybullet as p
import pybullet_data as pd

p.connect(p.GUI)

p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pd.getDataPath())

p.loadURDF("plane.urdf")
robot_arm = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.0, 0.0, 0.0],
    useFixedBase=True,
)

p.resetDebugVisualizerCamera(
    cameraTargetPosition=[0.0, 0.0, 0.0],
    cameraDistance=5,
    cameraPitch=-30,
    cameraYaw=-45,
)

number_of_joints = p.getNumJoints(robot_arm)

for joint_index in range(number_of_joints):
    joint_information = p.getJointInfo(robot_arm, joint_index)

    joint_idx = joint_information[0]
    joint_name = joint_information[1].decode("utf-8")
    joint_type = joint_information[2]

    joint_minimum_limit = joint_information[8]
    joint_maximum_limit = joint_information[9]

    print(
        f"{joint_idx} - Name: {joint_name} | Type: {joint_type} | "
        f"Minimum Rotation Limit: {joint_minimum_limit} | "
        f"Maximum Rotation Limit: {joint_maximum_limit}"
    )

for numSteps in range(240 * 30):
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()
