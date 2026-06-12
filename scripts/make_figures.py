#!/usr/bin/env python3
"""Regenerate every figure in the paper.

Runs each figure script in ``figures/`` and writes its PDF next to it. Each
script is self-contained and prints the headline values it encodes, so the
output doubles as a check that the numbers match the paper.

Usage:
    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py fig5_pareto fig6_bursty   # a subset
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))

ALL = [
    "fig1_teaser", "fig2_motivation", "fig9_signal", "fig5_pareto",
    "fig10_scaling", "fig6_bursty", "fig12_decomp", "fig7_quality",
    "fig11_oracle", "fig8_calibration",
]


def main(argv):
    targets = argv[1:] or ALL
    env = dict(os.environ, MPLBACKEND="Agg")
    failed = []
    for name in targets:
        script = os.path.join(FIG_DIR, name + ".py")
        if not os.path.exists(script):
            print(f"skip (not found): {name}")
            continue
        print(f"=== {name} ===")
        r = subprocess.run([sys.executable, script], cwd=FIG_DIR, env=env)
        if r.returncode != 0:
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("All figures written to", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
