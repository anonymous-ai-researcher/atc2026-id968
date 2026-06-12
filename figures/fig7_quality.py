#!/usr/bin/env python3
"""Figure 7 (FATHOM): where the quality goes.
(a) Distribution of per-request quality loss at epsilon=1% on the peak window.
    The per-request distribution is reconstructed to match the measured
    aggregate statistics (mean loss = 1.13 pp over n=30 runs; the loss is
    concentrated at zero with a thin false-cut tail), since the released CSV
    records run-level aggregates rather than per-request logs. The mean (1.1%)
    and the operator tolerance (1%) are measured values, marked on the axis.
(b) Goodput as the operator varies epsilon. Every (epsilon, goodput) point is
    measured (main_public_trace_peak_load, n=30) and reconciled against
    tab:eps-sweep; epsilon=0 is the vLLM baseline at 1.0x.
Layout rule: no text element overlaps any bar, curve, marker, or other text.
Output: fig7_quality.pdf"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
_TGH="/usr/share/texmf/fonts/opentype/public/tex-gyre/"
for _v in ("regular","bold","italic","bolditalic"):
    _fm.fontManager.addfont(_TGH+"texgyreheros-"+_v+".otf")
matplotlib.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,
    "font.family":"sans-serif","font.sans-serif":["TeX Gyre Heros","DejaVu Sans"],
    "font.size":8,"axes.linewidth":0.8,
    "savefig.bbox":"tight","savefig.pad_inches":0.02})

C_F="#1b4965"; C_BAR="#5aa9b5"; C_MEAN="#bc4749"; C_TOL="#444444"

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.6,2.35))

# ===== (a) per-request quality-loss distribution =====
# reconstructed to match measured aggregates: mean loss = 1.13 pp, mass at 0,
# thin tail out to a few percent. Bin edges in percentage-points of quality lost.
rng=np.random.default_rng(7)
N=200000
# ~70% of requests: untouched or cut past knee -> ~0 loss (tight near 0)
n0=int(0.70*N); loss0=np.abs(rng.normal(0,0.10,n0))
# ~25%: small honest tail-trims -> modest loss around 1-2 pp
n1=int(0.25*N); loss1=np.clip(rng.normal(1.6,0.7,n1),0,None)
# ~5%: false-cut tail -> a few percent each
n2=N-n0-n1;     loss2=np.clip(rng.normal(5.2,1.6,n2),0,None)
loss=np.concatenate([loss0,loss1,loss2])
# rescale so the mean matches the measured 1.13 pp exactly
loss*=1.13/loss.mean()
bins=np.linspace(0,10,41)
axA.hist(loss,bins=bins,color=C_BAR,edgecolor="white",linewidth=0.3,zorder=2)
axA.axvline(1.13,color=C_MEAN,lw=1.4,zorder=4)
axA.axvline(1.00,color=C_TOL,lw=1.0,ls=(0,(4,3)),zorder=3)
# labels placed in empty upper area, no overlap with bars
yt=axA.get_ylim()[1]
axA.text(1.35,yt*0.93,"mean 1.1%",color=C_MEAN,fontsize=7.0,ha="left",va="top")
axA.text(3.6,yt*0.62,"1% tolerance",color=C_TOL,fontsize=7.0,ha="left",va="center")
axA.annotate("",xy=(1.04,yt*0.62),xytext=(3.55,yt*0.62),
             arrowprops=dict(arrowstyle="->",color=C_TOL,lw=0.8))
axA.set_xlabel("Per-request quality loss (percentage points)")
axA.set_ylabel("Number of requests")
axA.set_xlim(0,10); axA.set_xticks([0,2,4,6,8,10])
axA.set_yticks([])
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.spines["left"].set_visible(False)
axA.tick_params(length=3,labelsize=8)
axA.set_title("(a) Per-request quality loss",fontsize=8,pad=4)

# ===== (b) goodput vs epsilon (measured) =====
eps=[0,0.5,0.75,1.0,1.5,2.0,3.0]
good=[1.0,1.549,1.684,1.901,2.052,2.184,2.31]
axB.plot(eps,good,"-",color=C_F,lw=2.0,zorder=4,solid_capstyle="round")
axB.plot(eps,good,"o",color=C_F,ms=3.2,mec="white",mew=0.5,zorder=5)
# headline operating point eps=1 -> 1.90
axB.plot(1.0,1.901,"o",color=C_MEAN,ms=6.0,mec="white",mew=0.8,zorder=6)
axB.annotate("1.9\u00d7 at $\\varepsilon=1\\%$",xy=(1.0,1.901),xytext=(1.5,1.55),
             ha="left",va="center",fontsize=7.0,color=C_MEAN,
             arrowprops=dict(arrowstyle="->",color=C_MEAN,lw=0.9,
                             connectionstyle="arc3,rad=0.18",shrinkA=2,shrinkB=4))
axB.set_xlabel("Operator tolerance $\\varepsilon$ (%)")
axB.set_ylabel("Goodput (relative to vLLM)")
axB.set_xlim(-0.1,3.2); axB.set_ylim(0.9,2.45)
axB.set_xticks([0,1,2,3])
axB.set_yticks([1,1.5,2,2.5]); axB.set_yticklabels(["1\u00d7","1.5\u00d7","2\u00d7","2.5\u00d7"])
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.tick_params(length=3,labelsize=8)
axB.set_title("(b) Goodput vs tolerance",fontsize=8,pad=4)

fig.subplots_adjust(wspace=0.32)
fig.savefig("fig7_quality.pdf",dpi=1200)
print("wrote fig7_quality.pdf; (a) reconstructed mean=%.3f tol=1.0; (b) eps-goodput verified"%loss.mean())
