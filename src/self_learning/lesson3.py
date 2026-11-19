import time

import pybullet as p
import pybullet_data

p.connect(p.GUI)

p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.loadURDF("plane.urdf")
ballId = p.loadURDF("soccerball.urdf", basePosition=[0, 0, 0.5])

p.resetDebugVisualizerCamera(
    cameraTargetPosition=[0, 0, 2],
    cameraDistance=10,
    cameraPitch=-30,
    cameraYaw=-45,
)

p.changeDynamics(ballId, -1, mass=1.0)

targetXPos = 5.0
currentDistanceFromTarget = 0.0
rewardScore = 0.0

for numSteps in range(5000):
    p.stepSimulation()
    time.sleep(1 / 240)

    # "ballPos" = [x, y, z] -> 0, 1, 2
    ballPos, ballOrient = p.getBasePositionAndOrientation(ballId)

    currentDistanceFromTarget = abs(targetXPos - ballPos[0])
    rewardScore = 1.0 - currentDistanceFromTarget

    p.applyExternalForce(
        objectUniqueId=ballId,
        linkIndex=-1,
        forceObj=[1.0, 0.0, 0.0],
        posObj=ballPos,
        flags=p.LINK_FRAME,
    )

    if numSteps % 200 == 0:
        print(
            f"\n[ STEP: {numSteps} ]\n"
            f"x = {ballPos[0]:.4f}\n"
            f"y = {ballPos[1]:.4f}\n"
            f"z = {ballPos[2]:.4f}\n"
            f"Reward Score: {rewardScore:.4f}"
        )

p.disconnect()
