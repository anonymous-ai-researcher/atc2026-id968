"""Load-adaptive controller.

The detector says when a cut is *safe*; the controller says when a cut is
*worth making*. It reads three signals the engine already exposes and folds them
into a single aggressiveness knob:

    theta = min(1, max(p_q, p_m, p_h))

where each p is a normalized pressure, zero while its signal is comfortable and
rising to one as the signal saturates:

    p_q : queue-depth pressure   (admitted but not-yet-running requests)
    p_m : KV-memory pressure     (fraction of cache in use)
    p_h : latency-headroom pressure (shrinking slack to the SLO)

Taking the maximum makes whichever resource binds first drive the response, so
theta is zero only while all three are comfortable.

Two measures keep theta from chattering as load fluctuates:
  1. each raw signal is smoothed by an exponential moving average before its
     pressure is formed, so a single noisy step does not swing theta;
  2. the response is asymmetric: theta rises promptly when pressure climbs but
     relaxes slowly, so brief dips do not prematurely stop reclamation.
"""

from dataclasses import dataclass


def _pressure(value: float, comfort: float, saturate: float) -> float:
    """Shortfall of a signal past its comfort threshold, divided by its range.

    Returns 0 while value <= comfort and 1 once value >= saturate, linear in
    between. ``saturate`` must be strictly greater than ``comfort``.
    """
    if saturate <= comfort:
        raise ValueError("saturate must exceed comfort")
    if value <= comfort:
        return 0.0
    if value >= saturate:
        return 1.0
    return (value - comfort) / (saturate - comfort)


@dataclass
class ControllerConfig:
    # comfort / saturation points for each signal (units match the engine)
    queue_comfort: float = 4.0       # waiting requests
    queue_saturate: float = 64.0
    mem_comfort: float = 0.70        # fraction of KV cache in use
    mem_saturate: float = 0.95
    headroom_comfort: float = 0.50   # slack to SLO as a fraction (1.0 = idle)
    headroom_saturate: float = 0.05
    ema_decay: float = 0.9           # smoothing of raw signals
    relax_rate: float = 0.2          # max downward step of theta per decode step


class LoadAdaptiveController:
    """Maps live load to theta in [0, 1], smoothed and asymmetric."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.cfg = config or ControllerConfig()
        self._ema_q: float | None = None
        self._ema_m: float | None = None
        self._ema_h: float | None = None
        self._theta = 0.0

    def _smooth(self, prev: float | None, raw: float) -> float:
        if prev is None:
            return raw
        d = self.cfg.ema_decay
        return d * prev + (1.0 - d) * raw

    def update(self, queue_depth: float, mem_fraction: float, slo_headroom: float) -> float:
        """Recompute theta from one decode step's load signals.

        ``slo_headroom`` is the slack to the SLO as a fraction in [0, 1], where
        1.0 means fully idle and 0.0 means at the deadline. Returns the new theta.
        """
        self._ema_q = self._smooth(self._ema_q, queue_depth)
        self._ema_m = self._smooth(self._ema_m, mem_fraction)
        self._ema_h = self._smooth(self._ema_h, slo_headroom)

        c = self.cfg
        p_q = _pressure(self._ema_q, c.queue_comfort, c.queue_saturate)
        p_m = _pressure(self._ema_m, c.mem_comfort, c.mem_saturate)
        # headroom pressure rises as headroom *shrinks*, so invert the direction
        p_h = _pressure(-self._ema_h, -c.headroom_comfort, -c.headroom_saturate)

        target = min(1.0, max(p_q, p_m, p_h))

        # asymmetric response: rise immediately, relax at a bounded rate
        if target >= self._theta:
            self._theta = target
        else:
            self._theta = max(target, self._theta - c.relax_rate)
        return self._theta

    @property
    def theta(self) -> float:
        return self._theta
