import math
import time

import pybullet as p
import pybullet_data

p.connect(p.GUI)

p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
ballId = p.loadURDF(
    "soccerball.urdf",
    basePosition=[1.5, 1.5, 1.0],
)

p.resetDebugVisualizerCamera(
    cameraTargetPosition=[1.5, 1.5, 0.5],
    cameraDistance=4,
    cameraPitch=-30,
    cameraYaw=-45,
)

p.changeDynamics(ballId, -1, mass=1.0)

p.addUserDebugLine(
    lineFromXYZ=[1.0, 1.5, 0.1],
    lineToXYZ=[2.0, 1.5, 0.1],
    lineColorRGB=[0, 255, 0],
    lineWidth=4,
)

p.addUserDebugLine(
    lineFromXYZ=[1.5, 1.0, 0.1],
    lineToXYZ=[1.5, 2.0, 0.1],
    lineColorRGB=[0, 255, 0],
    lineWidth=4,
)

alpha = 6.0
distance_from_target_point = 0.0
target_coord = [1.5, 1.5, 0.0]
exponential_reward_score = 0.0

for numSteps in range(240 * 30):
    ball_pos, _ = p.getBasePositionAndOrientation(ballId)

    distance_from_target_point = math.sqrt(
        (ball_pos[0] - target_coord[0]) ** 2 + (ball_pos[1] - target_coord[1]) ** 2
    )

    exponential_reward_score = math.exp(-alpha * distance_from_target_point)

    p.stepSimulation()
    time.sleep(1 / 240)

    if numSteps % 240 == 0:
        print(
            f"\n[ STEP: {numSteps} ]\n"
            f"x = {ball_pos[0]:.4f}\n"
            f"y = {ball_pos[1]:.4f}\n"
            f"z = {ball_pos[2]:.4f}\n"
            f"Distance from Target Point: {distance_from_target_point:.4f}\n"
            f"Exponential Reward Score: {exponential_reward_score:.4f}"
        )
