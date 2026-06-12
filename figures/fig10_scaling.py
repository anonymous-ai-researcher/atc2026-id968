#!/usr/bin/env python3
"""Figure 10 (FATHOM): steady load sweep (contrast to the single burst of
fig:bursty). (a) Goodput (requests served within SLO) vs offered load.
(b) P99 latency vs offered load, against the SLO. Each curve flattens / knees at
its saturating load; FATHOM saturates last and keeps its tail under the SLO over
the widest load band. All points measured (controlled_load_sweep, 32B model,
n=30 per load), 16 load points 0.25x..3.0x. SLO = 3x the uncontended median
(~33-35 s). Layout rule: no text element overlaps any curve, marker, or text.
Output: fig10_scaling.pdf"""
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

C_F="#1b4965"; C_V="#bc4749"; C_R="#7d8597"; C_A="#e09f3e"; C_SLO="#444444"

load=[0.25,0.35,0.50,0.65,0.80,0.90,1.00,1.10,1.25,1.40,1.60,1.80,2.00,2.30,2.60,3.00]
# (a) goodput req/s (measured)
g_v=[5.54,5.49,5.49,5.58,5.60,5.45,5.19,4.67,3.96,3.18,2.39,1.88,1.47,1.09,0.81,0.57]
g_r=[6.37,6.49,6.42,6.52,6.44,6.44,5.99,5.57,4.70,3.80,2.97,2.38,1.91,1.44,1.12,0.82]
g_a=[7.74,7.63,7.72,7.73,7.70,7.63,7.34,6.66,5.80,4.62,3.61,2.86,2.35,1.79,1.40,1.03]
g_f=[8.18,8.27,8.20,8.28,8.17,8.33,7.75,7.21,6.37,5.24,4.24,3.44,2.83,2.27,1.84,1.46]
# (b) P99 latency s (measured)
p_v=[45.20,44.60,44.87,45.08,45.48,45.31,45.19,47.94,55.70,62.93,75.28,88.91,107.12,135.34,159.84,211.53]
p_r=[39.40,38.85,38.41,38.46,39.13,38.06,38.95,41.37,45.61,53.48,62.52,74.16,86.39,106.37,129.43,164.96]
p_a=[27.85,27.87,28.17,28.11,27.69,27.87,27.74,29.90,34.55,38.86,47.17,54.92,63.46,80.84,98.14,128.11]
p_f=[24.54,24.52,24.39,24.40,24.62,24.70,24.32,26.12,29.82,33.24,38.82,44.76,52.42,61.87,74.92,91.34]
SLO=35.0

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.7,2.5))

def series(ax,x,ys,labels,colors,markers):
    for y,lab,c,mk in zip(ys,labels,colors,markers):
        ax.plot(x,y,"-",color=c,lw=1.7,zorder=4)
        ax.plot(x,y,mk,color=c,ms=3.0,mec="white",mew=0.4,zorder=5,label=lab)

# ----- (a) goodput -----
series(axA,load,[g_v,g_r,g_a,g_f],
       ["vLLM","R-KV","Always-on","FATHOM"],
       [C_V,C_R,C_A,C_F],["s","^","v","o"])
axA.set_xlabel("Offered load (\u00d7 saturation)")
axA.set_ylabel("Goodput (req/s)")
axA.set_xlim(0.1,3.1); axA.set_ylim(0,9)
axA.set_xticks([0.5,1,1.5,2,2.5,3])
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.tick_params(length=3,labelsize=8)
axA.set_title("(a) Goodput",fontsize=8,pad=4)
axA.legend(loc="upper right",fontsize=6.8,frameon=False,
           handletextpad=0.4,labelspacing=0.28,borderaxespad=0.2)

# ----- (b) P99 latency -----
series(axB,load,[p_v,p_r,p_a,p_f],
       ["vLLM","R-KV","Always-on","FATHOM"],
       [C_V,C_R,C_A,C_F],["s","^","v","o"])
axB.axhline(SLO,color=C_SLO,lw=1.0,ls=(0,(4,3)),zorder=2)
axB.text(2.95,SLO+5,"SLO",fontsize=7.0,color=C_SLO,va="bottom",ha="right")
axB.set_xlabel("Offered load (\u00d7 saturation)")
axB.set_ylabel("P99 latency (s)")
axB.set_xlim(0.1,3.1); axB.set_ylim(0,220)
axB.set_xticks([0.5,1,1.5,2,2.5,3])
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.tick_params(length=3,labelsize=8)
axB.set_title("(b) P99 latency",fontsize=8,pad=4)
axB.legend(loc="upper left",fontsize=6.8,frameon=False,
           handletextpad=0.4,labelspacing=0.28,borderaxespad=0.2)

fig.subplots_adjust(wspace=0.30)
fig.savefig("fig10_scaling.pdf",dpi=1200)
print("wrote fig10_scaling.pdf")
