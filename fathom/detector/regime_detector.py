"""Regime detector.

At each decode step the engine already produces the model's next-token
distribution. The detector reads a single scalar from it, the readiness signal

    s_i(t) = probability mass the model places on ending its reasoning,

i.e. the mass on the end-of-thinking marker (and, optionally, close variants of
it). As a request converges on an answer this mass climbs and holds, so a single
threshold tau separates "still reasoning" from "past the knee."

The threshold tau is fixed offline by the calibration tool (see
``fathom/calibration``) so that at most a fraction beta of firings are premature
(land before the true knee). The detector itself is intentionally trivial at
serving time: one comparison per running sequence, no extra forward pass.

A two-step debounce (the signal must clear tau on two consecutive steps before
the request is flagged) suppresses isolated spikes in s_i(t).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass
class DetectorConfig:
    """Calibrated detector parameters. Produced offline, read at startup."""

    tau: float                      # readiness threshold; flag when s_i(t) >= tau
    end_of_thinking_id: int         # token id of the end-of-thinking marker
    variant_ids: Sequence[int] = () # optional close variants whose mass is added
    debounce_steps: int = 2         # consecutive steps above tau before flagging

    def __post_init__(self) -> None:
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must lie in (0, 1)")
        if self.debounce_steps < 1:
            raise ValueError("debounce_steps must be >= 1")


class RegimeDetector:
    """Flags the step at which a request's reasoning has passed its knee.

    Stateless across requests except for a small per-request streak counter used
    by the debounce. ``observe`` is called once per running sequence per decode
    step with that sequence's next-token probability vector.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.cfg = config
        self._streak: Dict[int, int] = {}   # request_id -> consecutive steps >= tau
        self._flagged: Dict[int, bool] = {} # request_id -> already flagged

    def readiness_signal(self, probs: Sequence[float]) -> float:
        """s_i(t): probability mass on ending reasoning at this step.

        ``probs`` is the model's next-token distribution for one sequence. We sum
        the mass on the end-of-thinking marker and any configured close variants.
        Reading it costs a few lookups; no forward pass of our own.
        """
        s = float(probs[self.cfg.end_of_thinking_id])
        for vid in self.cfg.variant_ids:
            s += float(probs[vid])
        return s

    def observe(self, request_id: int, probs: Sequence[float]) -> bool:
        """Update the detector with one step of one request.

        Returns True the first time the request is confirmed past its knee
        (signal at or above tau on ``debounce_steps`` consecutive steps).
        Subsequent calls for an already-flagged request return False; the
        request is flagged exactly once.
        """
        if self._flagged.get(request_id, False):
            return False

        s = self.readiness_signal(probs)
        if s >= self.cfg.tau:
            self._streak[request_id] = self._streak.get(request_id, 0) + 1
        else:
            self._streak[request_id] = 0

        if self._streak[request_id] >= self.cfg.debounce_steps:
            self._flagged[request_id] = True
            return True
        return False

    def is_flagged(self, request_id: int) -> bool:
        return self._flagged.get(request_id, False)

    def release(self, request_id: int) -> None:
        """Drop per-request state once a request leaves the system."""
        self._streak.pop(request_id, None)
        self._flagged.pop(request_id, None)
