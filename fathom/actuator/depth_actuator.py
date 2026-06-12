"""Depth actuator.

Ending a request's reasoning is a one-token edit. When the scheduler decides to
reclaim a request, the actuator appends the end-of-thinking marker to that
request's sequence at the next step boundary. The model reads it as the cue to
stop reasoning and begin answering, the budget-forcing mechanism. Neither the
model weights nor the engine's batching change; the injection is a single token
written into one sequence where the engine already advances every running
request together.

Ending the thinking does not erase the reasoning produced so far, which the
answer still attends to; it removes only the tail that would have followed. The
request then reaches an answer in a handful of tokens rather than thousands, so
it stops extending its KV cache and frees that memory far sooner.
"""

from typing import Protocol


class SequenceHandle(Protocol):
    """The minimal interface the actuator needs from an engine sequence.

    A real integration backs this with the engine's own sequence object; the
    reference engine in ``fathom/engine.py`` provides a concrete implementation.
    """

    request_id: int

    def append_token(self, token_id: int) -> None: ...
    def in_thinking_phase(self) -> bool: ...


class DepthActuator:
    """Injects the end-of-thinking token to end a request's reasoning."""

    def __init__(self, end_of_thinking_id: int) -> None:
        self.end_of_thinking_id = end_of_thinking_id
        self._actuated: set[int] = set()

    def end_reasoning(self, seq: SequenceHandle) -> bool:
        """Force ``seq`` out of its thinking phase.

        Idempotent and safe: does nothing if the request has already been
        actuated or has already left the thinking phase on its own. Returns True
        iff this call performed the injection.
        """
        if seq.request_id in self._actuated:
            return False
        if not seq.in_thinking_phase():
            # the model finished thinking on its own; nothing to force
            return False
        seq.append_token(self.end_of_thinking_id)
        self._actuated.add(seq.request_id)
        return True

    def was_actuated(self, request_id: int) -> bool:
        return request_id in self._actuated

    def release(self, request_id: int) -> None:
        self._actuated.discard(request_id)
