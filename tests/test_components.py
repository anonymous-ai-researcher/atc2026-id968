"""Unit tests for FATHOM's control-plane logic.

These run on CPU in seconds and need no model or GPU. They check the behavior
the design relies on: the detector's threshold and debounce, the controller's
dual-trigger and smoothing, the scheduler's largest-footprint-first ordering,
and the calibration sweep picking the most aggressive tau under the beta target.

    python -m pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fathom.detector.regime_detector import RegimeDetector, DetectorConfig
from fathom.controller.load_controller import LoadAdaptiveController, ControllerConfig
from fathom.scheduler.elastic_scheduler import (
    ElasticJobScheduler, FlaggedRequest, reclaim_target_blocks,
)
from fathom.calibration.calibrate import RequestTrace, calibrate, evaluate_tau


def _probs(end_mass, vocab=8, end_id=0):
    p = [(1.0 - end_mass) / (vocab - 1)] * vocab
    p[end_id] = end_mass
    return p


# ----------------------------- detector --------------------------------------

def test_detector_requires_two_consecutive_steps():
    det = RegimeDetector(DetectorConfig(tau=0.6, end_of_thinking_id=0, debounce_steps=2))
    assert det.observe(1, _probs(0.7)) is False   # one step above tau: not yet
    assert det.observe(1, _probs(0.7)) is True     # second consecutive: flagged
    assert det.is_flagged(1)


def test_detector_debounce_resets_on_dip():
    det = RegimeDetector(DetectorConfig(tau=0.6, end_of_thinking_id=0, debounce_steps=2))
    assert det.observe(1, _probs(0.7)) is False    # above
    assert det.observe(1, _probs(0.2)) is False    # dip resets the streak
    assert det.observe(1, _probs(0.7)) is False    # only one above again
    assert det.observe(1, _probs(0.7)) is True     # now two consecutive


def test_detector_flags_each_request_once():
    det = RegimeDetector(DetectorConfig(tau=0.6, end_of_thinking_id=0, debounce_steps=1))
    assert det.observe(1, _probs(0.9)) is True
    assert det.observe(1, _probs(0.9)) is False    # already flagged


# ----------------------------- controller ------------------------------------

def test_controller_idle_gives_zero_theta():
    c = LoadAdaptiveController(ControllerConfig())
    theta = c.update(queue_depth=0, mem_fraction=0.1, slo_headroom=1.0)
    assert theta == 0.0


def test_controller_saturates_under_pressure():
    c = LoadAdaptiveController(ControllerConfig())
    for _ in range(20):  # let the EMA catch up to sustained high load
        theta = c.update(queue_depth=200, mem_fraction=0.99, slo_headroom=0.0)
    assert theta == 1.0


def test_controller_relaxes_after_load_clears():
    cfg = ControllerConfig(relax_rate=0.2)
    c = LoadAdaptiveController(cfg)
    for _ in range(20):
        c.update(200, 0.99, 0.0)        # drive to 1.0
    assert c.theta == 1.0
    # load vanishes; EMA-smoothed signals decay over several steps, and theta
    # relaxes by at most relax_rate per step rather than dropping instantly.
    thetas = [c.update(0, 0.1, 1.0) for _ in range(30)]
    assert thetas[-1] < 1.0                       # it does come down
    drops = [a - b for a, b in zip(thetas, thetas[1:])]
    assert all(d <= cfg.relax_rate + 1e-9 for d in drops)   # never faster than relax_rate


# ----------------------------- scheduler -------------------------------------

def test_scheduler_orders_by_footprint():
    sched = ElasticJobScheduler()
    flagged = [FlaggedRequest(1, 10), FlaggedRequest(2, 50), FlaggedRequest(3, 30)]
    chosen = sched.select(flagged, theta=1.0)
    assert chosen[0] == 2 and chosen[1] == 3   # largest footprints first


def test_scheduler_stops_at_target():
    sched = ElasticJobScheduler()
    flagged = [FlaggedRequest(1, 10), FlaggedRequest(2, 50), FlaggedRequest(3, 30)]
    # theta=0.5 of 90 blocks = 45 target; the 50-block request alone meets it
    chosen = sched.select(flagged, theta=0.5)
    assert chosen == [2]


def test_zero_theta_reclaims_nothing():
    assert reclaim_target_blocks(0.0, 100) == 0
    sched = ElasticJobScheduler()
    assert sched.select([FlaggedRequest(1, 10)], theta=0.0) == []


# ----------------------------- calibration -----------------------------------

def _trace(cross_step, knee_step, n=20):
    sig = [0.1] * n
    for t in range(cross_step, n):
        sig[t] = 0.9
    q_full = 1.0
    q_cut = [0.5 if t < knee_step else 0.99 for t in range(n)]
    return RequestTrace(sig, knee_step, q_full, q_cut)


def test_calibration_picks_aggressive_tau_under_target():
    # all requests cross at step 8 with knee at 5: every fire is past-knee, beta=0
    traces = [_trace(cross_step=8, knee_step=5) for _ in range(50)]
    r = calibrate(traces, beta_target=0.02, tau_grid=[0.5, 0.6, 0.7])
    assert r.beta <= 0.02
    assert r.tau == 0.5   # smallest (most aggressive) tau that satisfies the target


def test_calibration_flags_premature_cuts():
    # requests cross at step 3 but knee is at 8: every fire is premature, beta=1
    traces = [_trace(cross_step=3, knee_step=8) for _ in range(50)]
    r = evaluate_tau(traces, tau=0.6, debounce=2)
    assert r.beta == 1.0
