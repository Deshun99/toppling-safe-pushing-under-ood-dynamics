"""Fixed physics timing shared by target simulation and imagination."""

from typing import Final

PHYSICS_HZ: Final = 240
DT: Final = 1.0 / PHYSICS_HZ
GRAVITY: Final = -9.81
