"""Elastic-job scheduler.

The controller fixes *how much* memory to free this step (a target derived from
theta); the scheduler decides *which* flagged requests give up their tail.

Its input is the set of requests the detector has flagged, all of them past
their knee and therefore equally safe to end. The question is not which cut is
safe but which cuts relieve the binding resource soonest, so the rule is
deliberately coarse: each flagged request carries a priority

    rho_i = size of its KV cache (the memory it currently holds),

and the scheduler ends flagged requests in decreasing rho_i, freeing the largest
footprints first, stopping once it has released enough memory to meet the
target. Because every flagged request is already safe to end, the order costs no
quality.

The scheduler pointedly does *not* rank requests by the value of their remaining
reasoning: estimating that value online is the fragile judgment the detector was
built to avoid. It also never preempts a running request; it only ends reasoning
that was finished in all but name. Freed blocks return to the engine's own block
manager, which admits waiting requests and unblocks decoding ones through its
normal path.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List


@dataclass
class FlaggedRequest:
    request_id: int
    kv_blocks: int          # rho_i: KV footprint, in engine blocks


def reclaim_target_blocks(theta: float, flagged_blocks: int) -> int:
    """Memory to free this step: theta times the flagged footprint.

    At theta = 0 nothing is reclaimed; at theta = 1 the whole flagged footprint
    is eligible. Intermediate theta reclaims a proportional slice, which keeps
    the response graded rather than all-or-nothing.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    return int(round(theta * max(0, flagged_blocks)))


class ElasticJobScheduler:
    """Selects flagged requests to end, largest KV footprint first."""

    def select(self, flagged: Iterable[FlaggedRequest], theta: float) -> List[int]:
        """Return request ids to end this step.

        Greedy by decreasing rho_i until the freed blocks meet the theta target.
        Reclaiming the biggest holders first relieves the binding resource with
        the fewest interventions.
        """
        items = sorted(flagged, key=lambda r: r.kv_blocks, reverse=True)
        target = reclaim_target_blocks(theta, sum(r.kv_blocks for r in items))
        if target <= 0:
            return []

        freed = 0
        chosen: List[int] = []
        for r in items:
            if freed >= target:
                break
            chosen.append(r.request_id)
            freed += r.kv_blocks
        return chosen

    def reclaim(
        self,
        flagged: Iterable[FlaggedRequest],
        theta: float,
        end_reasoning: Callable[[int], None],
    ) -> int:
        """Convenience driver: select, then end each chosen request.

        ``end_reasoning`` is the actuator callback that performs the injection
        for one request id. Returns the number of requests ended.
        """
        chosen = self.select(flagged, theta)
        for rid in chosen:
            end_reasoning(rid)
        return len(chosen)
