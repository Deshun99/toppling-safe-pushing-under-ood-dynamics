import pytest

from src.simulation.environment.env import PushEnv
from src.simulation.modelling.object import CuboidSpec


@pytest.fixture
def spec():
    return CuboidSpec(w=0.08, L=0.08, H=0.20, m=1.0, mu=0.3, h_c=0.08)


@pytest.fixture
def env(spec):
    instance = PushEnv(fixed_spec=spec)
    try:
        instance.reset(seed=42)
        yield instance
    finally:
        instance.close()
