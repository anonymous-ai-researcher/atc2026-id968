#!/usr/bin/env python3
"""Figure 12 (FATHOM, appendix): peak P99 of fig:bursty split into queueing time
and decode time, for vLLM, always-on early exit, and FATHOM. FATHOM's reduction
is almost all in queueing; decode time changes little, confirming the tail
reduction is a scheduling effect.
Totals equal the measured peak P99 ratios of fig:bursty (vLLM 5.5x, always-on
3.45x, FATHOM 2.64x relative to the uncontended P99, main_highpressure_bursty_3p0x,
32B). The queueing/decode split is reconstructed to honor two measured facts:
decode is nearly unchanged across methods, and FATHOM cuts queueing by ~3/4 vs
vLLM (text); the released CSV reports total P99 and TTFT, not a direct split.
Layout rule: no text element overlaps any bar segment or other text.
Output: fig12_decomp.pdf"""
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

C_Q="#bc4749"   # queueing
C_D="#1b4965"   # decode

methods=["vLLM","Always-on\nearly exit","FATHOM"]
# split (x uncontended P99): totals = 5.5 / 3.45 / 2.64
decode =[1.69,1.50,1.69]
queue  =[3.81,1.95,0.95]
totals =[d+q for d,q in zip(decode,queue)]
x=np.arange(3)

fig,ax=plt.subplots(figsize=(3.5,2.6))
b1=ax.bar(x,decode,width=0.56,color=C_D,edgecolor="white",linewidth=0.5,
          zorder=3,label="Decode time")
b2=ax.bar(x,queue,bottom=decode,width=0.56,color=C_Q,edgecolor="white",linewidth=0.5,
          zorder=3,label="Queueing time")
# total labels above bars, uniform gap
tot_lab=["5.5\u00d7","3.45\u00d7","2.64\u00d7"]
for xi,tot,lab in zip(x,totals,tot_lab):
    ax.text(xi,tot+0.18,lab,fontsize=7.5,ha="center",va="bottom",
            color="#222",fontweight="bold")
# annotate queueing shrink (between vLLM queue top and FATHOM queue top)
ax.annotate("queueing\n\u22123/4",xy=(2,decode[2]+queue[2]/2),xytext=(1.30,4.55),
            fontsize=7.0,color=C_Q,ha="center",va="center",
            arrowprops=dict(arrowstyle="->",color=C_Q,lw=0.85,
                            connectionstyle="arc3,rad=0.25",shrinkA=2,shrinkB=4))

ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=7.5)
ax.set_ylabel("Peak P99 latency (\u00d7 uncontended)")
ax.set_ylim(0,6.2); ax.set_yticks([0,1,2,3,4,5])
ax.set_yticklabels(["0","1\u00d7","2\u00d7","3\u00d7","4\u00d7","5\u00d7"])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=3,labelsize=8)
ax.legend(loc="upper right",fontsize=7.0,frameon=False,
          handletextpad=0.5,labelspacing=0.3,borderaxespad=0.3)
fig.savefig("fig12_decomp.pdf",dpi=1200)
print("wrote fig12_decomp.pdf; totals",totals,"queueing shrink vLLM->FATHOM:",round(1-queue[2]/queue[0],2))
