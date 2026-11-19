"""M2/E4 identifiability study (outline §9.5, milestone M2).

Scripted push sequences on a target object with known hidden theta. For each
excitation regime, records the action history, runs CMA-ES system
identification from it, and reports per-parameter estimation error and the
spread of the retained top-K hypothesis set.

Run from the repo root:  .venv/bin/python -m src.evaluation.system_identification.diagnostics.identifiability
"""

import numpy as np

from src.agents.system_identification.sysid import SysID, SysIDResult
from src.evaluation.system_identification.shared.constants import SPEC, WARMUP
from src.evaluation.system_identification.shared.excitation import (
    compute_errors,
    get_regimes,
    settle_or_replace_object,
)
from src.simulation.environment.env import PushEnv
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.records import ActionRecord


def evaluate_regimes(target: PushEnv) -> dict[str, SysIDResult]:
    results = {}
    for name, actions in get_regimes().items():
        target.reset()
        history: list[ActionRecord] = []
        for action in actions:
            if action.h > SPEC.h_c:
                settle_or_replace_object(target)
            history.append(target.push(action, warmup_steps=WARMUP))
        estimator = SysID(geometry=(SPEC.w, SPEC.L, SPEC.H))
        try:
            result = estimator.update(history)
        finally:
            estimator.close()
        results[name] = result
        print_estimate(name, result)
    return results


def print_estimate(sequence_name: str, result: SysIDResult) -> None:
    mass, friction, com_height = result.point.theta
    errors = compute_errors(result)
    mass_std, friction_std, com_std = errors["spread"]
    print(
        f"{sequence_name:20s} | m_hat={mass:5.2f} (err {errors['m_err'] * 100:4.0f}%) | "
        f"mu_hat={friction:4.2f} (err {errors['mu_err']:4.2f}) | "
        f"hc_hat={com_height:4.3f} (err {errors['hc_err'] * 100:4.0f}%) | "
        f"top-K std: (m {mass_std:.2f}, mu {friction_std:.2f}, hc {com_std:.3f})"
    )


def check_bistability(target: PushEnv, repeats: int = 10) -> None:
    print("\nBistable-band check (outline §13 h_c risk): repeat the boundary")
    print("push (0.60H, F=5.0) from rest — the outcome flips run to run:")
    topples = 0
    for _ in range(repeats):
        settle_or_replace_object(target)
        record = target.push(
            PushAction(h=0.6 * SPEC.H, F=5.0, tau=0.4), warmup_steps=WARMUP
        )
        peak_tilt = max(abs(tilt) for tilt in record.traj.psi[WARMUP:])
        topples += np.degrees(peak_tilt) > np.degrees(SPEC.crit_tilt)
    print(f"  {topples}/{repeats} runs toppled at identical commanded push")


def main() -> None:
    target = PushEnv(fixed_spec=SPEC)
    try:
        print(
            f"true theta: m={SPEC.m:.2f} kg, mu={SPEC.mu:.2f}, h_c={SPEC.h_c:.3f} m ({SPEC.h_c / SPEC.H:.2f}H)\n"
        )
        results = evaluate_regimes(target)
        print("\nTop-K hypotheses for 'full mixed':")
        for hypothesis in results["full mixed"].hypotheses:
            mass, friction, com_height = hypothesis.theta
            print(
                f"  m={mass:5.2f}  mu={friction:4.2f}  h_c={com_height:4.3f}  loss={hypothesis.loss:9.1f}  w={hypothesis.weight:.2f}"
            )
        check_bistability(target)
        print("\nM2/E4 identifiability study complete.")
    finally:
        target.close()


if __name__ == "__main__":
    main()
