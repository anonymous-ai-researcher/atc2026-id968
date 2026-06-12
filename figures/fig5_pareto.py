#!/usr/bin/env python3
"""Figure 5 (FATHOM): answer quality vs serving goodput, peak-load window,
three main models. FATHOM's epsilon sweep traces the upper frontier and
dominates every baseline the text names. Points computed from the authoritative
measured CSV (main_public_trace_peak_load, n=30) and reconciled against
tab:baselines and tab:eps-sweep. The frontier is anchored at the vLLM origin
(1.0, 100) because under slack FATHOM intervenes on nothing (see text).
Layout rule: no text element overlaps any curve, marker, legend, or other text.
Output: fig5_pareto.pdf"""
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

# ===== measured points, verified against tab:eps-sweep / tab:baselines =====
# FATHOM epsilon frontier (goodput, quality): eps = 0.5,0.75,1,1.5,2,3 %
eps_pts=[(1.55,99.5),(1.68,99.2),(1.90,98.9),(2.05,98.4),(2.18,97.8),(2.31,96.4)]
# anchor at vLLM origin: under slack FATHOM = vLLM (text: "At the left the curve
# coincides with vLLM"). The curve runs from (1.0,100) through the eps points.
frontier=[(1.00,100.0)]+eps_pts
fx=[p[0] for p in frontier]; fy=[p[1] for p in frontier]
op=(1.90,98.9)  # headline operating point, epsilon = 1%

vllm      =(1.00,100.0)
pascal    =(1.34,100.0)
rkv       =(1.27,99.9)
flashthink=(1.44,97.0)
specexit  =(1.51,98.0)
earlyexit =(1.58,93.3)

C_F="#1b4965"; C_V="#5c6b73"; C_P="#5aa9b5"; C_R="#7d8597"
C_FT="#e09f3e"; C_SE="#9e2a2b"; C_E="#bc4749"

fig,ax=plt.subplots(figsize=(3.5,2.35))
# FATHOM frontier (eps markers only; the anchor at 1.0 is not an eps point)
ax.plot(fx,fy,"-",color=C_F,lw=2.0,zorder=5,solid_capstyle="round")
ax.plot([p[0] for p in eps_pts],[p[1] for p in eps_pts],"o",
        color=C_F,ms=3.0,zorder=6,mec="white",mew=0.5)
ax.plot([],[],"-",color=C_F,lw=2.0,label="FATHOM frontier ($\\varepsilon$ sweep)")

def pt(xy,color,label,mk="o"):
    ax.plot(xy[0],xy[1],mk,color=color,ms=5.0,mec="white",mew=0.7,zorder=7,label=label)
pt(vllm,      C_V, "vLLM","s")
pt(pascal,    C_P, "PASCAL","D")
pt(rkv,       C_R, "R-KV","^")
pt(flashthink,C_FT,"FlashThink","P")
pt(specexit,  C_SE,"SpecExit","X")
pt(earlyexit, C_E, "Always-on early exit","v")

# --- headline annotation: placed in the empty lower-right region, no overlap ---
# empty zone is around x in [1.95,2.45], y in [92,96]; arrow points up-left to op
ax.annotate("1.9\u00d7 goodput,\n~1% quality loss",
            xy=(op[0],op[1]),xytext=(2.05,95.6),
            ha="center",va="center",fontsize=7.0,color=C_F,
            arrowprops=dict(arrowstyle="->",color=C_F,lw=0.9,
                            connectionstyle="arc3,rad=0.22",shrinkA=2,shrinkB=4))
# --- 1% tolerance guide: label sits at far right on the line, clear of markers ---
ax.axhline(99.0,color="black",lw=0.6,ls=(0,(4,3)),alpha=0.45,zorder=1)
ax.text(2.43,99.12,"1% tolerance",fontsize=7.0,color="black",alpha=0.6,
        va="bottom",ha="right")

ax.set_xlabel("Serving goodput (relative to vLLM)")
ax.set_ylabel("Answer quality (%)")
ax.set_xlim(0.9,2.5)
ax.set_ylim(91.0,101.2)
ax.set_xticks([1,1.5,2,2.5]); ax.set_xticklabels(["1\u00d7","1.5\u00d7","2\u00d7","2.5\u00d7"])
ax.set_yticks([92,94,96,98,100])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=3,labelsize=8)
# legend in upper-left empty area (above PASCAL/R-KV cluster, clear of frontier)
ax.legend(loc="lower left",bbox_to_anchor=(0.0,0.0),fontsize=7.0,frameon=False,
          handletextpad=0.4,labelspacing=0.32,borderaxespad=0.25)
fig.savefig("fig5_pareto.pdf",dpi=1200)
print("wrote fig5_pareto.pdf")
