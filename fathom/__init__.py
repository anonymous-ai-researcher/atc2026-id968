"""FATHOM: scheduling reasoning depth as a load-adaptive, quality-bounded resource.

FATHOM adds a thin control plane around an unmodified serving engine. Four
components cooperate on the running batch each decode step:

    RegimeDetector      - reads a per-request readiness signal and flags the
                          step at which reasoning has passed its knee.
    LoadAdaptiveController - maps live cluster load to a single aggressiveness
                          knob theta in [0, 1].
    DepthActuator       - ends a flagged request's reasoning by injecting the
                          end-of-thinking token (budget forcing).
    ElasticJobScheduler - picks which flagged requests give up their tail,
                          ordered by KV footprint, and recycles freed memory.

The components are deliberately decoupled: the detector decides *whether a cut
is safe*, the controller decides *whether a cut is worth making*, and only when
both agree (the dual trigger) does the actuator fire.
"""

from fathom.detector.regime_detector import RegimeDetector, DetectorConfig
from fathom.controller.load_controller import LoadAdaptiveController, ControllerConfig
from fathom.actuator.depth_actuator import DepthActuator
from fathom.scheduler.elastic_scheduler import ElasticJobScheduler
from fathom.engine import FathomEngine, RequestState

__all__ = [
    "RegimeDetector",
    "DetectorConfig",
    "LoadAdaptiveController",
    "ControllerConfig",
    "DepthActuator",
    "ElasticJobScheduler",
    "FathomEngine",
    "RequestState",
]

__version__ = "1.0.0"
