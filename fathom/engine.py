"""Reference control loop.

This module wires the four components into the decode-step loop FATHOM adds
around a serving engine. It is engine-agnostic: it talks to the host engine
through a small adapter interface (see ``fathom/integration/vllm_adapter.py`` for
the vLLM binding) and contains no model or kernel code of its own.

The loop, once per decode step, is:

    1. detector.observe(...)   for each running sequence -> update flags
    2. controller.update(...)  from live load -> theta
    3. scheduler.select(...)   flagged requests, theta -> ids to end
    4. actuator.end_reasoning  inject end-of-thinking for each chosen id
    5. freed KV blocks return to the engine's block manager (host side)

The dual trigger lives in the gap between steps 1 and 3: a request is only ever
ended if the detector flagged it (safe) *and* theta called for reclamation
(useful).
"""

from dataclasses import dataclass, field
from typing import Dict, List

from fathom.detector.regime_detector import RegimeDetector, DetectorConfig
from fathom.controller.load_controller import LoadAdaptiveController, ControllerConfig
from fathom.actuator.depth_actuator import DepthActuator
from fathom.scheduler.elastic_scheduler import ElasticJobScheduler, FlaggedRequest


@dataclass
class RequestState:
    """Per-request bookkeeping the loop needs from the host engine each step."""

    request_id: int
    next_token_probs: list          # model's next-token distribution this step
    kv_blocks: int                  # current KV footprint (rho_i)
    thinking: bool = True           # still in the thinking phase?


@dataclass
class LoadSnapshot:
    queue_depth: float
    mem_fraction: float
    slo_headroom: float             # slack to SLO as a fraction in [0, 1]


@dataclass
class StepResult:
    theta: float
    newly_flagged: List[int] = field(default_factory=list)
    ended: List[int] = field(default_factory=list)


class FathomEngine:
    """Engine-agnostic control plane around a host serving engine."""

    def __init__(
        self,
        detector_cfg: DetectorConfig,
        controller_cfg: ControllerConfig | None = None,
    ) -> None:
        self.detector = RegimeDetector(detector_cfg)
        self.controller = LoadAdaptiveController(controller_cfg)
        self.actuator = DepthActuator(detector_cfg.end_of_thinking_id)
        self.scheduler = ElasticJobScheduler()

    def step(
        self,
        running: List[RequestState],
        load: LoadSnapshot,
        inject_fn,
    ) -> StepResult:
        """Run one decode-step of the control loop.

        ``inject_fn(request_id)`` is the host callback that appends the
        end-of-thinking token to a sequence (the actuator's effect, performed on
        the host's own sequence object). Returns what changed this step.
        """
        result = StepResult(theta=0.0)

        # 1. sensing: update detector flags from this step's distributions
        for r in running:
            if r.thinking and self.detector.observe(r.request_id, r.next_token_probs):
                result.newly_flagged.append(r.request_id)

        # 2. load -> theta
        theta = self.controller.update(load.queue_depth, load.mem_fraction, load.slo_headroom)
        result.theta = theta
        if theta <= 0.0:
            return result  # uncontended: act on nothing

        # 3. which flagged, still-thinking requests give up their tail
        flagged = [
            FlaggedRequest(r.request_id, r.kv_blocks)
            for r in running
            if r.thinking and self.detector.is_flagged(r.request_id)
            and not self.actuator.was_actuated(r.request_id)
        ]
        chosen = self.scheduler.select(flagged, theta)

        # 4. act: inject end-of-thinking on the host side
        for rid in chosen:
            inject_fn(rid)
            self.actuator._actuated.add(rid)  # mark; host performs the edit
            result.ended.append(rid)
        return result

    def release(self, request_id: int) -> None:
        """Drop all per-request state once a request finishes."""
        self.detector.release(request_id)
        self.actuator.release(request_id)
