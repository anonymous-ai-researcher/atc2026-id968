#!/usr/bin/env python3
"""Figure 9 (FATHOM, appendix): the readiness signal s_i(t) along one reasoning
trace. s_i(t) stays low while the marginal value v_i(t) of further reasoning is
high, then rises and holds once the answer converges, crossing the threshold tau
just after the true knee k_i. Conceptual schematic of the detector signal (no
per-token measurement in the released data); the shapes illustrate the mechanism
described in the text.
Layout rule: no text element overlaps any curve, marker, or other text.
Output: fig9_signal.pdf"""
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

C_S="#1b4965"   # readiness signal s
C_V="#e09f3e"   # marginal value v
C_TAU="#bc4749" # threshold
C_KNEE="#5c6b73"

t=np.linspace(0,1,400)
# s_i(t): low early, rises sigmoidally as answer converges, then holds high
s=1/(1+np.exp(-(t-0.55)/0.07)); s=s/s.max()
# v_i(t): high early, decays to ~0 as reasoning converges
v=1-1/(1+np.exp(-(t-0.45)/0.09)); v=v/v.max()
tau=0.5
# knee k_i: where v has effectively decayed (marginal value ~ small);
# s crosses tau just AFTER the knee
k=0.50
# s crosses tau at:
cross=t[np.argmin(np.abs(s-tau))]

fig,ax=plt.subplots(figsize=(3.5,2.3))
ax.plot(t,s,"-",color=C_S,lw=2.2,zorder=5,label="readiness signal $s_i(t)$")
ax.plot(t,v,color=C_V,lw=2.0,zorder=4,ls=(0,(5,2)),label="marginal value $v_i(t)$")
# threshold line
ax.axhline(tau,color=C_TAU,lw=1.0,ls=(0,(2,2)),zorder=3)
ax.text(0.015,tau+0.03,"threshold $\\tau$",fontsize=7.0,color=C_TAU,va="bottom")
# knee marker (vertical) and crossing marker
ax.axvline(k,color=C_KNEE,lw=0.9,ls=(0,(1,2)),zorder=2)
ax.text(k-0.02,0.80,"knee $k_i$",fontsize=7.0,color=C_KNEE,ha="right",va="center")
# crossing point of s and tau (just after knee)
ax.plot(cross,tau,"o",color=C_S,ms=5.0,mec="white",mew=0.8,zorder=6)
ax.annotate("$s_i$ crosses $\\tau$\nafter the knee",xy=(cross,tau),
            xytext=(0.74,0.30),fontsize=7.0,color=C_S,ha="center",va="center",
            arrowprops=dict(arrowstyle="->",color=C_S,lw=0.85,
                            connectionstyle="arc3,rad=0.2",shrinkA=2,shrinkB=5))

ax.set_xlabel("Reasoning progress")
ax.set_ylabel("Signal value")
ax.set_xlim(0,1); ax.set_ylim(0,1.12)
ax.set_xticks([0,0.5,1.0]); ax.set_xticklabels(["start","","end"])
ax.set_yticks([0,0.5,1.0]); ax.set_yticklabels(["0","",""])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=3,labelsize=8)
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.01),fontsize=7.0,frameon=False,
          ncol=2,handletextpad=0.5,columnspacing=1.4,borderaxespad=0.0)
fig.savefig("fig9_signal.pdf",dpi=1200)
print("wrote fig9_signal.pdf; s crosses tau at t=%.3f, knee at %.2f (crossing after knee: %s)"%(cross,k,cross>k))
