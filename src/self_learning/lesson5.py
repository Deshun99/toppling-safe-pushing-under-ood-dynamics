import math
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

for joint_index in range(number_of_joints):
    p.resetJointState(
        bodyUniqueId=robot_arm,
        jointIndex=joint_index,
        targetValue=1.5,  # Roughly 57 degrees in radians
        targetVelocity=6.0,
    )

for numSteps in range(240 * 120):
    p.stepSimulation()
    time.sleep(1 / 240)

    # p.setJointMotorControl2( # Control the first joint of the robot arm to reach a target position
    #     bodyUniqueId=robot_arm,
    #     jointIndex=0,
    #     controlMode=p.POSITION_CONTROL, # Turn until joint 0 reaches the target position
    #     targetPosition=1.1, # Roughly 63 degrees in radians
    #     force=70.0, # Maximum torque the robot can apply to reach the target velocity
    # )

    # p.setJointMotorControl2( # Control the first joint of the robot arm to reach a target velocity. Robot will keep turning until it reaches the target velocity or the joint limit is reached
    #     bodyUniqueId=robot_arm,
    #     jointIndex=0,
    #     controlMode=p.VELOCITY_CONTROL, # Control the velocity of the first joint
    #     targetVelocity=2.0,
    #     force=50.0, # Maximum torque the robot can apply to reach the target velocity
    # )

    p.setJointMotorControl2(  # in this mode, we aren't telling the robot where to go, nor how fast to go, we are just telling it how much torque to apply to the joint. The robot will keep turning until it reaches the joint limit or the torque is no longer sufficient to overcome friction and other forces acting on the joint.
        bodyUniqueId=robot_arm,
        jointIndex=0,
        controlMode=p.TORQUE_CONTROL,
        force=50.0,  # The actual torque applied to the joint.
    )  # Heavily used in reinforcement learning, where the agent learns to apply exact torque to the joints to achieve a goal, such as reaching a target position or following a trajectory.

    if numSteps % 240 != 0:
        continue
    for joint_index in range(number_of_joints):
        joint_position, joint_velocity, reaction_force, applied_torque = (
            p.getJointState(robot_arm, joint_index)
        )

        print(
            f"Joint Index: {joint_index} | "
            f"Position: {math.degrees(joint_position):.4f} degrees | "
            f"Velocity: {joint_velocity:.4f}"
        )

p.disconnect()
