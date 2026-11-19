import time

import pybullet as p
import pybullet_data

p.connect(p.GUI)

p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")

ball_id = p.loadURDF("soccerball.urdf", basePosition=[0, 0, 2])

p.resetDebugVisualizerCamera(
    cameraDistance=10, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0, 0, 10]
)  # Can still change camera angle in GUI by dragging with the mouse and holding control key

p.changeDynamics(
    ball_id,
    -1,  # link index (-1 for base link)
    mass=1.0,
)

for num_steps in range(5000):
    p.stepSimulation()
    time.sleep(1 / 240)

    ball_position, ball_orientation = p.getBasePositionAndOrientation(ball_id)

    p.applyExternalForce(
        objectUniqueId=ball_id,
        linkIndex=-1,  # Apply force to the base link
        forceObj=[10, 0, 0],  # Force vector (downward)
        posObj=ball_position,  # Position where the force is applied
        flags=p.WORLD_FRAME,  # Apply force in world frame
    )

    print(
        f"Step: {num_steps}, Ball Position: {ball_position}, Ball Orientation: {ball_orientation}"
    )

p.disconnect()
