"""Offline calibration.

Three quantities are fixed before serving and never during it:

    tau   - the detector threshold,
    beta  - the false-cut rate it induces (fraction of firings that land before
            the true knee, where stopping would change the answer),
    gamma - the post-knee tail bound (how much quality a past-knee cut can give
            up), taken as a high quantile of the per-request past-knee loss.

The tool runs the target model once over a labeled calibration set, records for
each candidate tau the false-cut rate and the tail bound, and keeps the smallest
tau whose false-cut rate stays under the operator's target. A smaller tau fires
sooner (more reclamation, more throughput) but risks cutting before the knee;
the sweep finds the most aggressive tau that still respects the false-cut target.

This module is deliberately model-agnostic. It consumes per-request *traces*
already collected from the target model (the readiness signal at each step and
the true knee and quality found by running the request to completion). The
trace-collection script that produces these is part of the artifact; see the
README. The output is a small JSON config the server reads at startup.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Sequence


@dataclass
class RequestTrace:
    """One labeled calibration request.

    signal:        readiness signal s_i(t) at each thinking step.
    knee_step:     index of the true knee (found offline by running to the end
                   and locating where added reasoning stops changing the answer).
    full_quality:  task quality of the uninterrupted answer (the baseline).
    quality_if_cut_at: quality if reasoning were ended at each step, same length
                   as ``signal``. Used to measure realized loss for any tau.
    """

    signal: Sequence[float]
    knee_step: int
    full_quality: float
    quality_if_cut_at: Sequence[float]


@dataclass
class CalibrationResult:
    tau: float
    beta: float          # realized false-cut rate at this tau
    gamma: float         # post-knee tail bound (quantile of past-knee loss)


def _first_crossing(signal: Sequence[float], tau: float, debounce: int) -> int | None:
    """Index of the first step where signal clears tau on ``debounce`` steps."""
    streak = 0
    for t, s in enumerate(signal):
        streak = streak + 1 if s >= tau else 0
        if streak >= debounce:
            return t
    return None


def evaluate_tau(
    traces: Sequence[RequestTrace],
    tau: float,
    debounce: int = 2,
    gamma_quantile: float = 0.99,
) -> CalibrationResult:
    """Measure beta and gamma for one candidate threshold tau."""
    fires = 0
    premature = 0
    past_knee_losses: List[float] = []

    for tr in traces:
        cut = _first_crossing(tr.signal, tau, debounce)
        if cut is None:
            continue  # detector never fires on this request
        fires += 1
        if cut < tr.knee_step:
            premature += 1
        else:
            loss = tr.full_quality - tr.quality_if_cut_at[cut]
            past_knee_losses.append(max(0.0, loss))

    beta = premature / fires if fires else 0.0
    gamma = _quantile(past_knee_losses, gamma_quantile) if past_knee_losses else 0.0
    return CalibrationResult(tau=tau, beta=beta, gamma=gamma)


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, int(round(q * (len(ys) - 1))))
    return ys[idx]


def calibrate(
    traces: Sequence[RequestTrace],
    beta_target: float = 0.02,
    tau_grid: Sequence[float] | None = None,
    debounce: int = 2,
    gamma_quantile: float = 0.99,
) -> CalibrationResult:
    """Pick the smallest tau whose false-cut rate stays under ``beta_target``.

    Smaller tau is more aggressive, so we scan from small to large and take the
    first tau that satisfies the target. Falls back to the most conservative tau
    on the grid if none qualifies.
    """
    grid = tau_grid if tau_grid is not None else [i / 100.0 for i in range(50, 100)]
    best: CalibrationResult | None = None
    for tau in sorted(grid):
        r = evaluate_tau(traces, tau, debounce, gamma_quantile)
        if r.beta <= beta_target:
            return r
        best = r
    return best  # type: ignore[return-value]


def write_config(result: CalibrationResult, end_of_thinking_id: int, path: str) -> None:
    """Serialize a calibration result to the JSON the server reads at startup."""
    payload = {
        "tau": result.tau,
        "beta": result.beta,
        "gamma": result.gamma,
        "end_of_thinking_id": end_of_thinking_id,
        "debounce_steps": 2,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
