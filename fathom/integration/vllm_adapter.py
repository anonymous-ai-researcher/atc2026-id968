"""vLLM integration: where FATHOM attaches to the serving engine.

FATHOM lives entirely in the scheduling and sampling layer. It changes no model
code, no attention or other GPU kernels, and nothing in the batching machinery,
so it inherits the engine's existing optimizations. The same four hook points
exist in comparable engines such as SGLang.

This file documents the attachment points and provides a thin wrapper. It is a
reference sketch: the exact symbols differ across vLLM versions, so the artifact
pins a version in ``requirements.txt`` and the README notes the functions to
wrap. The four hooks, in the order they fire each decode step:

    1. sampler post-hook   -> RegimeDetector.observe(...)
       After the sampler computes next-token logits/probs, read s_i(t) off the
       distribution already in hand.

    2. scheduler pre-step  -> LoadAdaptiveController.update(...)
       Read queue depth, KV-cache utilization, and current tail latency, all of
       which the scheduler already tracks, and map them to theta.

    3. scheduler pre-step  -> ElasticJobScheduler.select(...)
       Choose flagged sequences by KV footprint to meet the theta target.

    4. sequence step        -> DepthActuator.end_reasoning(...)
       Append the end-of-thinking token to each chosen sequence at the step
       boundary. Freed blocks return to vLLM's BlockManager through its normal
       free path; admission of waiting requests is unchanged.
"""

from fathom.engine import FathomEngine, RequestState, LoadSnapshot
from fathom.detector.regime_detector import DetectorConfig
from fathom.controller.load_controller import ControllerConfig


class FathomVLLM:
    """Thin wrapper that drives FathomEngine from vLLM's scheduler loop.

    Usage (pseudocode, see README for the concrete wrapping):

        fathom = FathomVLLM.from_config("configs/r1_distill_32b.json")
        # inside vLLM's step(), after sampling:
        fathom.on_decode_step(running_seqs, llm_engine)
    """

    def __init__(self, detector_cfg: DetectorConfig, controller_cfg: ControllerConfig | None = None) -> None:
        self.core = FathomEngine(detector_cfg, controller_cfg)

    @classmethod
    def from_config(cls, path: str) -> "FathomVLLM":
        import json
        with open(path) as f:
            cfg = json.load(f)
        det = DetectorConfig(
            tau=cfg["tau"],
            end_of_thinking_id=cfg["end_of_thinking_id"],
            debounce_steps=cfg.get("debounce_steps", 2),
        )
        return cls(det)

    def on_decode_step(self, running_seqs, llm_engine) -> None:
        """Bridge one vLLM decode step into the FATHOM control loop.

        ``running_seqs`` are vLLM's running sequences; ``llm_engine`` exposes the
        block manager and load counters. This reads per-sequence distributions
        and load, runs the control loop, and injects end-of-thinking tokens for
        the sequences the loop selects.
        """
        states = [
            RequestState(
                request_id=s.request_id,
                next_token_probs=s.last_token_probs,   # populated by the sampler hook
                kv_blocks=s.num_kv_blocks,
                thinking=s.in_thinking_phase,
            )
            for s in running_seqs
        ]
        load = LoadSnapshot(
            queue_depth=llm_engine.num_waiting,
            mem_fraction=llm_engine.kv_cache_usage,
            slo_headroom=llm_engine.slo_headroom,
        )
        seq_by_id = {s.request_id: s for s in running_seqs}

        def inject(request_id: int) -> None:
            seq_by_id[request_id].append_token(self.core.actuator.end_of_thinking_id)

        self.core.step(states, load, inject)
