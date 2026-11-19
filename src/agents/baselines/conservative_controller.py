"""B0: conservative analytic pushing from SysID point estimates (outline §10).

At each decision the controller combines the approximate tipping model
(outline §3.3) with the current point estimate (m, mu, h_c) to pick a push
that stays safely below the tipping bound while making bounded progress
toward the target. The internal simulator serves system identification only;
the controller never rolls it out.
"""

from dataclasses import dataclass

from src.simulation.environment.env import ActionRanges
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import GRAVITY
from src.simulation.modelling.object import Geometry, PhysicsParameters

DEFAULT_HEIGHT_FRACTION = 0.1  # lowest policy height; below h_c for study ranges
DEFAULT_VELOCITY_CAP = 0.12  # m/s, maximum target block speed per push
DEFAULT_TIP_SAFETY_FRACTION = 0.5  # of the analytic tipping bound
DEFAULT_MIN_VELOCITY = 0.01  # m/s floor so friction balance still makes progress
DEFAULT_STOP_EPSILON = 0.05  # m, stop inside the goal band or past it


@dataclass(frozen=True)
class ConservativeController:
    """Pick safe pushes from the tipping model and a physics point estimate."""

    h_fraction: float = DEFAULT_HEIGHT_FRACTION
    velocity_cap: float = DEFAULT_VELOCITY_CAP
    tip_safety_fraction: float = DEFAULT_TIP_SAFETY_FRACTION
    min_velocity: float = DEFAULT_MIN_VELOCITY
    stop_epsilon: float = DEFAULT_STOP_EPSILON

    def choose_action(
        self,
        geometry: Geometry,
        theta_hat: PhysicsParameters,
        distance: float,
        decisions_left: int,
        action_ranges: ActionRanges,
    ) -> PushAction:
        """Map the remaining distance and point estimate to a safe push.

        The tipping inequality (F/(m g))(h - h_c) <= w/2 - mu h_c caps the
        force when the estimate places the commanded height above h_c;
        otherwise the friction-balance force is the operative choice.
        """
        width, _, height = geometry
        mass_hat, mu_hat, com_hat = theta_hat
        tau_max = action_ranges.duration[1]
        if distance <= self.stop_epsilon:
            return PushAction(h=height * self.h_fraction, F=0.0, tau=0.05)
        h = height * self.h_fraction
        needed_speed = distance / (max(decisions_left, 1) * tau_max)
        target_speed = min(self.velocity_cap, max(self.min_velocity, needed_speed))
        force = mass_hat * (target_speed / tau_max + mu_hat * abs(GRAVITY))
        if h > com_hat:
            tipping_force = (
                mass_hat
                * abs(GRAVITY)
                * (width / 2 - mu_hat * com_hat)
                / (h - com_hat)
            )
            force = min(force, self.tip_safety_fraction * max(tipping_force, 0.0))
        return PushAction(
            h=h,
            F=max(0.0, min(force, action_ranges.force[1])),
            tau=tau_max,
        )
