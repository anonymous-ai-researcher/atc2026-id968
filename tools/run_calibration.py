#!/usr/bin/env python3
"""Run offline calibration and write a server config.

Consumes per-request calibration traces (collected from the target model) and
produces the small JSON the server reads at startup. See the README for the
trace format and how to collect traces from a model.

Usage:
    python tools/run_calibration.py \
        --traces traces.jsonl \
        --end-of-thinking-id 128799 \
        --beta-target 0.02 \
        --out configs/my_model.json

Trace file: JSON lines, one object per request with fields
    signal              list[float]   readiness signal at each thinking step
    knee_step           int           index of the true knee
    full_quality        float         baseline (uninterrupted) quality
    quality_if_cut_at   list[float]   quality if cut at each step
"""

import argparse
import json
import sys

sys.path.insert(0, ".")
from fathom.calibration.calibrate import (  # noqa: E402
    RequestTrace, calibrate, write_config,
)


def load_traces(path):
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            traces.append(RequestTrace(
                signal=o["signal"],
                knee_step=o["knee_step"],
                full_quality=o["full_quality"],
                quality_if_cut_at=o["quality_if_cut_at"],
            ))
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="JSONL calibration traces")
    ap.add_argument("--end-of-thinking-id", type=int, required=True)
    ap.add_argument("--beta-target", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    traces = load_traces(args.traces)
    if not traces:
        print("no traces loaded", file=sys.stderr)
        return 1
    result = calibrate(traces, beta_target=args.beta_target)
    write_config(result, args.end_of_thinking_id, args.out)
    print(f"calibrated: tau={result.tau:.3f} beta={result.beta:.3f} "
          f"gamma={result.gamma:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
