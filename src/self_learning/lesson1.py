import time

import pybullet as p
import pybullet_data

p.connect(p.GUI)

p.setGravity(0, 0, -9.81)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")

p.loadURDF("soccerball.urdf")

for num_steps in range(5000):
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()
