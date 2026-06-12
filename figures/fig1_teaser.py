#!/usr/bin/env python3
"""Figure 1 (FATHOM teaser): quality as a bounded, schedulable dimension.
Answer quality (% of unmodified baseline) against serving goodput. A
quality-agnostic engine (vLLM) and always-on early exit each sit at one fixed
point; FATHOM traces the upper frontier the operator moves along, holding full
quality under slack and giving it up gently only under pressure. A simplified
preview of fig:pareto. All points measured (main_public_trace_peak_load, n=30),
reconciled against tab:baselines / tab:eps-sweep.
Layout rule: no text element overlaps any curve, marker, or other text.
Output: fig1_teaser.pdf"""
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

C_F="#1b4965"; C_V="#5c6b73"; C_E="#bc4749"

# FATHOM frontier (goodput, quality), anchored at vLLM origin; eps sweep
eps_pts=[(1.55,99.5),(1.68,99.2),(1.90,98.9),(2.05,98.4),(2.18,97.8),(2.31,96.4)]
fath=[(1.00,100.0)]+eps_pts
fx=[p[0] for p in fath]; fy=[p[1] for p in fath]
op=(1.90,98.9)
vllm=(1.00,100.0)
early=(1.58,93.3)

fig,ax=plt.subplots(figsize=(3.4,2.35))

# FATHOM frontier
ax.plot(fx,fy,"-",color=C_F,lw=2.3,zorder=5,solid_capstyle="round")
ax.plot([p[0] for p in eps_pts],[p[1] for p in eps_pts],"o",color=C_F,ms=3.0,
        zorder=6,mec="white",mew=0.5)
# fixed points
ax.plot(*vllm,"s",color=C_V,ms=6.5,mec="white",mew=0.8,zorder=7)
ax.plot(*early,"v",color=C_E,ms=7.0,mec="white",mew=0.8,zorder=7)

# labels next to each element, in clear space, no overlap
# unified arrow length: same data offset magnitude for all three,
# straight arrows (no curvature) so visual lengths match exactly.
# all three arrows have equal visual length (0.62 in); offsets precomputed
def lbl(text,xy,off,color,fs=7.0,ha="left"):
    ax.annotate(text,xy=xy,xytext=(xy[0]+off[0],xy[1]+off[1]),
                fontsize=fs,color=color,ha=ha,va="center",
                arrowprops=dict(arrowstyle="->",color=color,lw=0.85,
                                shrinkA=3,shrinkB=5))
lbl("Quality-agnostic\nengine (vLLM)",vllm,( 0.137,-2.213),C_V,ha="left")
ax.annotate("Always-on\nearly exit",xy=(early[0]+0.02,early[1]),
            xytext=(early[0]+0.02-0.292,early[1]),
            fontsize=7.0,color=C_E,ha="right",va="center",
            arrowprops=dict(arrowstyle="->",color=C_E,lw=0.85,shrinkA=3,shrinkB=5))
ax.annotate("FATHOM\nfrontier",xy=(1.90,98.9),xytext=(1.90+0.46,98.9+0.140),
            fontsize=7.2,color=C_F,ha="center",va="center",
            arrowprops=dict(arrowstyle="->",color=C_F,lw=0.85,shrinkA=3,shrinkB=5))

ax.set_xlabel("Serving goodput (relative to vLLM)")
ax.set_ylabel("Answer quality (%)")
ax.set_xlim(0.9,2.5); ax.set_ylim(91.5,101.0)
ax.set_xticks([1,1.5,2,2.5]); ax.set_xticklabels(["1\u00d7","1.5\u00d7","2\u00d7","2.5\u00d7"])
ax.set_yticks([92,94,96,98,100])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=3,labelsize=8)
fig.savefig("fig1_teaser.pdf",dpi=1200)
print("wrote fig1_teaser.pdf")
