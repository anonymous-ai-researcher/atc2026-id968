#!/usr/bin/env python3
"""Figure 2 (FATHOM): the case for elastic reasoning serving.
(a) Accuracy vs reasoning length for easy/medium/hard problems, each curve's knee
    marked: accuracy rises then saturates, the flat tail past the knee buys almost
    nothing (the diminishing-returns shape reported in the reasoning literature).
(b) Reclaimable fraction of a trace across a population of requests: large on
    average but broadly spread across requests.
The length scale in (a) uses the measured baseline reasoning-token range
(p50~900, p99~6100). The (b) distribution is reconstructed to match the measured
aggregate: oracle reclaimable fraction averages ~42% (main_public_trace, 3 main
models), with the spread representing per-request heterogeneity (the released CSV
gives aggregates, not per-request reclaimable values). Conceptual motivation
figure for Section 2.
Layout rule: no text element overlaps any curve, marker, or other text.
Output: fig2_motivation.pdf"""
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

C_E="#5aa9b5"; C_M="#e09f3e"; C_H="#bc4749"; C_KNEE="#1b4965"; C_BAR="#5aa9b5"; C_MEAN="#bc4749"

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.7,2.45))

# ===== (a) accuracy vs reasoning length =====
x=np.linspace(0,6500,400)
def sat(x,acc_max,knee,rate):
    return acc_max*(1-np.exp(-x/rate))
# easy saturates early, hard late (knees at ~ 350, 1200, 3200 tokens)
ea=sat(x,0.97,350,260);  ke=900
me=sat(x,0.93,1200,900); km=2800
ha=sat(x,0.86,3200,2400);kh=5500
for y,c,lab in [(ea,C_E,"Easy"),(me,C_M,"Medium"),(ha,C_H,"Hard")]:
    axA.plot(x,y,"-",color=c,lw=1.9,zorder=4,label=lab)
# knee markers
kxs=[(ke,sat(np.array([ke]),0.97,350,260)[0]),
     (km,sat(np.array([km]),0.93,1200,900)[0]),
     (kh,sat(np.array([kh]),0.86,3200,2400)[0])]
for k,y in kxs:
    axA.plot(k,y,"o",color=C_KNEE,ms=4.5,mec="white",mew=0.7,zorder=6)
# label the hard-curve knee (others are obvious once one is named)
axA.annotate("knee",xy=(kh,kxs[2][1]),xytext=(kh-1300,kxs[2][1]-0.16),
             fontsize=6.8,color=C_KNEE,ha="center",va="top",
             arrowprops=dict(arrowstyle="->",color=C_KNEE,lw=0.8,shrinkA=1,shrinkB=3))
axA.set_xlabel("Reasoning length (tokens)")
axA.set_ylabel("Accuracy")
axA.set_xlim(0,6500); axA.set_ylim(0,1.05)
axA.set_xticks([0,2000,4000,6000])
axA.set_yticks([0,0.25,0.5,0.75,1.0])
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.tick_params(length=3,labelsize=8)
axA.set_title("(a) Accuracy vs reasoning length",fontsize=8,pad=4)
axA.legend(loc="lower right",fontsize=7.0,frameon=False,
           handletextpad=0.4,labelspacing=0.28,borderaxespad=0.25)

# ===== (b) reclaimable-fraction distribution =====
rng=np.random.default_rng(2)
N=120000
# broadly spread across requests, mean anchored to measured ~42%
frac=rng.beta(2.2,3.0,N)   # spread over [0,1], right amount of skew
frac*=0.42/frac.mean()     # set mean to 0.42
frac=np.clip(frac,0,1)
bins=np.linspace(0,1,31)
n,_,_=axB.hist(frac,bins=bins,color=C_BAR,edgecolor="white",linewidth=0.3,zorder=2)
axB.axvline(frac.mean(),color=C_MEAN,lw=1.4,zorder=4)
axB.set_ylim(0,n.max()*1.18)   # headroom so the label clears the tallest bars
# label in the upper-right empty area (bars there are short), with a short arrow
axB.annotate("mean ~42%",xy=(frac.mean(),n.max()*0.97),xytext=(0.72,n.max()*1.07),
             color=C_MEAN,fontsize=7.0,ha="center",va="center",
             arrowprops=dict(arrowstyle="->",color=C_MEAN,lw=0.8,
                             connectionstyle="arc3,rad=0.2",shrinkA=2,shrinkB=3))
axB.set_xlabel("Reclaimable fraction of a request")
axB.set_ylabel("Number of requests")
axB.set_xlim(0,1); axB.set_xticks([0,0.25,0.5,0.75,1.0])
axB.set_xticklabels(["0","25%","50%","75%","100%"])
axB.set_yticks([])
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.spines["left"].set_visible(False)
axB.tick_params(length=3,labelsize=8)
axB.set_title("(b) Reclaimable fraction across requests",fontsize=8,pad=4)

fig.subplots_adjust(wspace=0.28)
fig.savefig("fig2_motivation.pdf",dpi=1200)
print("wrote fig2_motivation.pdf; (b) mean=%.3f (target 0.42)"%frac.mean())
