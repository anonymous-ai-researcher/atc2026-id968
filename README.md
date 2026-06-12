<div align="center">

# 🌊 FATHOM

### Measuring *how deep* a model thinks — and scheduling *how deep* it needs to

**Reasoning depth as a load-adaptive, system-scheduled resource, under a quality bound.**

<br>

![status](https://img.shields.io/badge/status-anonymous_artifact-6E9FD2?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square)
![control plane](https://img.shields.io/badge/control_plane-pure_stdlib-2E8B57?style=flat-square)
![tests](https://img.shields.io/badge/tests-CPU_only-2E8B57?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-555?style=flat-square)

<br>

[**Overview**](#-overview) ·
[**How it works**](#%EF%B8%8F-how-it-works) ·
[**Install**](#%EF%B8%8F-install) ·
[**Quickstart**](#-quickstart) ·
[**Reproduce**](#-reproduce-the-paper) ·
[**Layout**](#-repository-layout) ·
[**FAQ**](#-faq)

</div>

---

> [!NOTE]
> **Anonymized artifact.** This repository accompanies a double-blind submission.
> It contains **no author, affiliation, or citation information** by design.
> Paths, identifiers, and acknowledgments have been removed or replaced with
> neutral placeholders.

---

## 📖 Overview

Reasoning models spend most of their compute **thinking**: a long internal chain
of tokens the user never sees, often the overwhelming majority of a request's
cost. That thinking has two properties today's serving stacks ignore:

- 🫧 **It is elastic** — the same model on the same question thinks for wildly
  different lengths.
- ✂️ **It is largely discretionary** — much of it can be removed with no change
  to the final answer.

**FATHOM** treats reasoning depth as a first-class, schedulable resource. It sits
as a thin control plane around an **unmodified serving engine** and, under load,
ends the reasoning of requests that have stopped paying off, hands the freed
KV-cache memory to requests that need it, and does so **only inside a quality
bound the operator sets in advance**. Under light load it does nothing, and every
request runs to full length.

<div align="center">

### 📊 Headline result

| Metric | FATHOM at its operating point |
|:--|:--:|
| 🚀 **Goodput** | **up to 2.0×** an unmodified engine |
| ⏱️ **Tail latency (P99)** | **45–53% lower** under high load and bursts |
| 🎯 **Answer quality** | **within ~1%** of the unmodified baseline |
| 🛡️ **SLO violations** | **more than halved** |

<sub>Measured across nine reasoning models, three model families, 1.5B–70B parameters.</sub>

</div>

**What makes this hard**, and what FATHOM is built around: the signal that says
whether more thinking helps lives *inside the model*, while the decision that acts
on it lives *in the scheduler* — and at serving time the system can never see
whether the answer it is shaping is correct. FATHOM therefore acts only inside an
**offline-calibrated envelope** and defaults to inaction whenever it has no firm
reason to act.

---

## ⚙️ How it works

FATHOM adds four small components to the decode loop. Each step, they cooperate on
the running batch:

```text
                    ┌──────────────────── FATHOM control plane ────────────────────┐
  next-token probs ─▶│  Regime detector ─▶ flagged?   ┐                             │
  (from the sampler) │                                ├─▶ dual trigger ─▶ Depth     │
  queue / mem / SLO ─▶│  Load controller ─▶ θ ∈ [0,1]  ┘    (safe ∧ useful) actuator│
  (engine counters)  │                                          │  inject </think>  │
                    │  Elastic-job scheduler ◀── θ target ───────┘                  │
                    │     pick by KV footprint, recycle freed memory                │
                    └───────────────────────────────────────────────────────────────┘
                                 around an unmodified serving engine
```

| Component | Decides | Mechanism |
|:--|:--|:--|
| 🔍 **Regime detector** | *Is a cut safe?* | Reads `s_i(t)`, the probability mass the model puts on its end-of-thinking marker, off the distribution the sampler already produced. Flags a request once `s_i(t) ≥ τ` on two consecutive steps (debounce). **No extra forward pass.** |
| 🎚️ **Load-adaptive controller** | *Is a cut worth making?* | Folds queue depth, KV-memory use, and SLO headroom into one aggressiveness knob `θ = min(1, max(p_q, p_m, p_h))`, smoothed by an EMA so it tracks sustained load without chattering. `θ = 0` under slack. |
| ✂️ **Depth actuator** | *End the reasoning* | Injects the end-of-thinking token (*budget forcing*) — a one-token edit that makes the model commit to an answer from the reasoning so far. |
| 🗂️ **Elastic-job scheduler** | *Which requests give it up* | Among flagged requests (all past their knee, all safe), ends the largest KV footprints first until `θ`'s memory target is met, then recycles freed blocks to waiting / decoding requests. **Never preempts; never ranks by remaining value.** |

> [!IMPORTANT]
> **The dual trigger** is the heart of the safety argument: a request is ended
> only if the detector flagged it (*safe*) **and** `θ > 0` (*useful*). The
> threshold `τ` is fixed offline so that at most a fraction `β` of firings are
> premature; with the post-knee tail bounded by `γ`, the average quality loss
> over a window stays within `γ + β`, which the operator caps at `ε`.

---

## 🛠️ Install

```bash
git clone <ANONYMIZED-REPO-URL>
cd fathom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[figures,dev]"
```

The control plane and calibration are **pure standard library** — no GPU or heavy
dependencies needed to read, test, or extend them. `matplotlib` / `numpy` are only
for the figures; `pytest` only for the tests.

To run FATHOM *inside* a serving engine, install the engine separately (pinned in
`requirements.txt`):

```bash
pip install vllm==0.22.1
```

FATHOM attaches in the scheduling / sampling layer and adds no kernel code, so it
inherits whatever CUDA / torch the engine requires.

---

## 🚀 Quickstart

**Run the tests** (CPU, seconds, no model needed) — verifies the detector
threshold / debounce, the controller's dual-trigger and smoothing, the scheduler's
ordering, and the calibration sweep:

```bash
python -m pytest tests/ -q
```

**Use the control loop** programmatically:

```python
from fathom import FathomEngine
from fathom.detector.regime_detector import DetectorConfig
from fathom.engine import RequestState, LoadSnapshot

engine = FathomEngine(DetectorConfig(tau=0.60, end_of_thinking_id=128799))

running = [
    RequestState(request_id=1, next_token_probs=probs1, kv_blocks=120),
    RequestState(request_id=2, next_token_probs=probs2, kv_blocks=40),
]
load = LoadSnapshot(queue_depth=50, mem_fraction=0.93, slo_headroom=0.05)

result = engine.step(running, load, inject_fn=my_engine.append_end_of_thinking)
print(result.theta, result.ended)   # aggressiveness, and which requests were ended
```

---

## 🔬 Reproduce the paper

### Figures

Every figure is generated by a single self-contained script that prints the
headline numbers it encodes (so the output doubles as a check against the paper):

```bash
python scripts/make_figures.py                # all figures -> figures/*.pdf
python scripts/make_figures.py fig5_pareto    # a single figure
```

| Script | Paper figure |
|:--|:--|
| `fig1_teaser.py` | Quality as a bounded, schedulable dimension (teaser) |
| `fig2_motivation.py` | Accuracy vs reasoning length; reclaimable fraction |
| `fig9_signal.py` | The readiness signal `s_i(t)` along a trace |
| `fig5_pareto.py` | Quality vs goodput (Pareto frontier) |
| `fig10_scaling.py` | Goodput and P99 across the load range |
| `fig6_bursty.py` | Tail latency under a burst |
| `fig12_decomp.py` | P99 decomposed into queueing vs decode |
| `fig7_quality.py` | Per-request quality-loss distribution; ε sweep |
| `fig11_oracle.py` | FATHOM vs a knee-aware oracle |
| `fig8_calibration.py` | Sensitivity to β and to workload drift |

> [!TIP]
> The figure scripts encode the experimental results as the model that produced
> the paper's plots. They use TeX Gyre Heros if it is installed and fall back to
> the default sans font otherwise; set `TEXGYRE_DIR` to point at the font
> directory for pixel-identical output.

### End-to-end serving runs

The full evaluation serves nine open reasoning models (three families, 1.5B–70B)
on H100-class GPUs and replays requests as an arrival stream (a Poisson sweep and
bursty windows). To reproduce on your own hardware:

**1. Wire FATHOM into the engine.** `fathom/integration/vllm_adapter.py` documents
the four hook points and provides a thin wrapper. Wrap the sampler post-hook
(detector), the scheduler pre-step (controller + scheduler), and the sequence step
(actuator). **No model or kernel code changes.**

**2. Calibrate per model.** Collect per-request traces (the readiness signal at
each step, the true knee, and the quality if cut at each step), then:

```bash
python tools/run_calibration.py \
    --traces traces.jsonl \
    --end-of-thinking-id <YOUR_TOKENIZER_ID> \
    --beta-target 0.02 \
    --out configs/my_model.json
```

This sweeps `τ` and keeps the most aggressive threshold whose false-cut rate stays
under the target. Trace format is documented in `tools/run_calibration.py`.

**3. Serve.** Load the config at startup (`FathomVLLM.from_config(...)`), set the
latency SLO to a multiple of the unloaded median, and replay the arrival stream
while recording goodput, P99, and post-hoc quality.

The control-plane logic (detector / controller / actuator / scheduler /
calibration) is **fully provided and unit-tested** here. The engine binding is a
thin adapter because the exact engine symbols shift across versions; the README
and adapter pin a version and name the functions to wrap.

---

## 📁 Repository layout

```text
fathom/
├── detector/regime_detector.py     readiness signal + threshold + debounce
├── controller/load_controller.py   load -> aggressiveness θ (EMA-smoothed)
├── actuator/depth_actuator.py      end-of-thinking injection (budget forcing)
├── scheduler/elastic_scheduler.py  largest-KV-footprint-first reclamation
├── calibration/calibrate.py        offline τ sweep -> (τ, β, γ)
├── integration/vllm_adapter.py     the four engine hook points
└── engine.py                       the per-decode-step control loop

tools/run_calibration.py            CLI: traces -> server config
scripts/make_figures.py             regenerate every figure
figures/                            one self-contained script per figure
configs/example_calibration.json    example server config
tests/test_components.py            CPU unit tests for all four components
```

---

## ❓ FAQ

<details>
<summary><b>Does FATHOM change the model or the GPU kernels?</b></summary>

<br>

No. It lives entirely in the scheduling and sampling layer, adds no kernel code,
and inherits the engine's existing optimizations. The same hook points exist in
comparable engines.
</details>

<details>
<summary><b>What if the cut is wrong?</b></summary>

<br>

The detector is calibrated to keep premature cuts (those before the knee) under a
small target `β`, and the actuator fires only when load warrants it. The guarantee
is an *average* over a window, not a per-request promise; the tail of harder-hit
requests is examined directly in the paper's appendix.
</details>

<details>
<summary><b>Does it need a labeled dataset at serving time?</b></summary>

<br>

No. Calibration is entirely offline and produces a small static config. At serving
time FATHOM never observes correctness.
</details>

<details>
<summary><b>What does it assume about the model?</b></summary>

<br>

That the model exposes a readable end-of-thinking marker (e.g. a `</think>` token)
whose probability can be read off the sampler's distribution. Models that signal
the end of reasoning differently need their own readiness signal and calibration.
</details>

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

<div align="center">
<sub>Anonymized for double-blind review · no author, affiliation, or citation information by design.</sub>
</div>
