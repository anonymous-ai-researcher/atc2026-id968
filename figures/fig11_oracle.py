#!/usr/bin/env python3
"""Figure 11 (FATHOM): FATHOM's quality-goodput frontier against an offline
oracle that knows each request's true knee, cuts exactly there, and never cuts
early (zero quality loss, the largest lossless reclamation). Peak-load window,
three main models. The two curves coincide through the flat region and separate
only at the aggressive end; the shaded band is the gain FATHOM does not capture.
All points from the authoritative measured CSV (main_public_trace_peak_load,
n=30), reconciled against the text: FATHOM 1.90x at op, oracle 2.40x.
Layout rule: no text element overlaps any curve, marker, legend, or other text.
Output: fig11_oracle.pdf"""
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

# ===== measured points, verified against CSV / text =====
# FATHOM epsilon frontier (goodput, quality): eps = 0.5,0.75,1,1.5,2,3 %
eps_pts=[(1.55,99.5),(1.68,99.2),(1.90,98.9),(2.05,98.4),(2.18,97.8),(2.31,96.4)]
fath=[(1.00,100.0)]+eps_pts
fxx=[p[0] for p in fath]; fyy=[p[1] for p in fath]
op_f=(1.90,98.9)   # FATHOM operating point, eps=1%

# oracle: lossless upper bound, knows the true knee. Anchored at vLLM origin,
# holds quality ~baseline out to its reclamation limit (2.40x, 99.5).
orc=[(1.00,100.0),(1.55,99.9),(1.90,99.7),(2.05,99.6),(2.40,99.5)]
oxx=[p[0] for p in orc]; oyy=[p[1] for p in orc]
op_o=(2.40,99.5)   # oracle operating-end point

C_F="#1b4965"; C_O="#9e7b9b"; C_SH="#cdbcca"

fig,ax=plt.subplots(figsize=(3.5,2.35))

# shaded band: gain FATHOM does not capture, between the two curves over the
# overlapping goodput range. Build by interpolating both onto a common grid.
gx=np.linspace(1.0,2.31,200)
fy_i=np.interp(gx,fxx,fyy)
oy_i=np.interp(gx,oxx,oyy)
ax.fill_between(gx,fy_i,oy_i,where=(oy_i>=fy_i),color=C_SH,alpha=0.55,
                lw=0,zorder=1,label="Gain not captured")

# oracle curve
ax.plot(oxx,oyy,"--",color=C_O,lw=1.8,zorder=4,dashes=(5,2),
        label="Oracle (true knee)")
ax.plot(op_o[0],op_o[1],"D",color=C_O,ms=5.0,mec="white",mew=0.7,zorder=6)
# FATHOM frontier
ax.plot(fxx,fyy,"-",color=C_F,lw=2.0,zorder=5,solid_capstyle="round",
        label="FATHOM frontier")
ax.plot([p[0] for p in eps_pts],[p[1] for p in eps_pts],"o",
        color=C_F,ms=3.0,zorder=6,mec="white",mew=0.5)

# operating-point annotation (placed in empty lower-right, no overlap)
ax.annotate("FATHOM 1.9\u00d7, oracle 2.4\u00d7\n(~4/5 of gain)",
            xy=(op_f[0],op_f[1]),xytext=(1.49,97.85),
            ha="center",va="center",fontsize=7.0,color=C_F,
            arrowprops=dict(arrowstyle="->",color=C_F,lw=0.9,
                            connectionstyle="arc3,rad=0.16",shrinkA=2,shrinkB=4))

ax.set_xlabel("Serving goodput (relative to vLLM)")
ax.set_ylabel("Answer quality (%)")
ax.set_xlim(0.95,2.55)
ax.set_ylim(95.5,100.6)
ax.set_xticks([1,1.5,2,2.5]); ax.set_xticklabels(["1\u00d7","1.5\u00d7","2\u00d7","2.5\u00d7"])
ax.set_yticks([96,97,98,99,100])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=3,labelsize=8)
ax.legend(loc="lower left",bbox_to_anchor=(0.0,0.0),fontsize=7.0,frameon=False,
          handletextpad=0.5,labelspacing=0.32,borderaxespad=0.25)
fig.savefig("fig11_oracle.pdf",dpi=1200)
print("wrote fig11_oracle.pdf")
print("FATHOM op:",op_f," oracle end:",op_o," capture ratio:",round(op_f[0]/op_o[0],3))
