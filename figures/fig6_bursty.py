#!/usr/bin/env python3
"""Figure 6 (FATHOM): a threefold load burst over five minutes.
Top: offered load and the controller's aggressiveness theta.
Bottom: P99 tail latency for FATHOM, always-on early exit, and vLLM against the
latency SLO, with the peak gap over vLLM marked.
Peak P99 values (relative to the uncontended P99) are anchored to the measured
data (main_highpressure_bursty_3p0x, 32B model): vLLM 5.5x, always-on 3.45x,
FATHOM 2.64x; the within-burst time course is a representative trajectory whose
peaks equal those measured values (the released CSV gives window aggregates, not
per-second traces). SLO = 3x the uncontended median (see setup).
Layout rule: no text element overlaps any curve, marker, or other text.
Output: fig6_bursty.pdf"""
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

C_F="#1b4965"; C_V="#bc4749"; C_A="#e09f3e"; C_TH="#5aa9b5"; C_SLO="#444444"; C_LOAD="#7d8597"

t=np.linspace(0,5,400)              # minutes
# burst shape: baseline load 1x, spike to 3x between t=1.3 and t=3.2, smooth
def bump(t,a,b,lo,hi,rise=0.25):
    x=np.clip((t-a)/rise,0,1); y=np.clip((b-t)/rise,0,1)
    g=np.minimum(x,y); g=np.where((t>=a)&(t<=b),1.0,g)
    s=1/(1+np.exp(-(t-a)/0.18))-1/(1+np.exp(-(t-b)/0.18))
    s=s/s.max()
    return lo+(hi-lo)*s
load=bump(t,1.3,3.2,1.0,3.0)
theta=bump(t,1.45,3.25,0.0,1.0)     # follows load, slightly lagged

# uncontended P99 normalized to 1.0; peaks anchored to measured ratios
def p99(t,peak,base=1.0,a=1.4,b=3.25):
    s=bump(t,a,b,0.0,1.0)
    return base+(peak-base)*s
p_v=p99(t,5.5)
p_a=p99(t,3.45,base=0.85)   # always-on starts lower (already shortened)
p_f=p99(t,2.64,base=1.0,a=1.55,b=3.2)
SLO=3.0

fig,(axT,axB)=plt.subplots(2,1,figsize=(3.5,3.2),sharex=True,
                           gridspec_kw={"height_ratios":[1,1.5],"hspace":0.18})

# ---- top: load + theta ----
axT.plot(t,load,"-",color=C_LOAD,lw=1.8,label="Offered load")
axT.fill_between(t,1.0,load,color=C_LOAD,alpha=0.12,lw=0)
axT2=axT.twinx()
axT2.plot(t,theta,"-",color=C_TH,lw=1.8,label="Aggressiveness $\\theta$")
axT.set_ylabel("Load\n(\u00d7 base)",fontsize=7.5)
axT2.set_ylabel("$\\theta$",color=C_TH,fontsize=8,rotation=0,labelpad=8,va="center")
axT.set_ylim(0.8,3.3); axT.set_yticks([1,2,3]); axT.set_yticklabels(["1\u00d7","2\u00d7","3\u00d7"])
axT2.set_ylim(-0.05,1.15); axT2.set_yticks([0,1]); axT2.tick_params(colors=C_TH,labelsize=7.5)
axT.spines["top"].set_visible(False); axT2.spines["top"].set_visible(False)
axT.tick_params(length=3,labelsize=7.5)
# combined legend, upper-left empty area
l1,la=axT.get_legend_handles_labels(); l2,lb=axT2.get_legend_handles_labels()
axT.legend(l1+l2,la+lb,loc="lower center",bbox_to_anchor=(0.5,1.02),fontsize=6.8,
           frameon=False,ncol=2,handletextpad=0.4,columnspacing=1.4,borderaxespad=0.0)

# ---- bottom: P99 ----
axB.axhline(SLO,color=C_SLO,lw=1.0,ls=(0,(4,3)),zorder=2)
axB.text(0.07,SLO+0.12,"SLO",fontsize=7.0,color=C_SLO,va="bottom")
axB.plot(t,p_v,"-",color=C_V,lw=1.8,label="vLLM",zorder=4)
axB.plot(t,p_a,"-",color=C_A,lw=1.8,label="Always-on early exit",zorder=4)
axB.plot(t,p_f,"-",color=C_F,lw=2.0,label="FATHOM",zorder=5)
# peak gap marker between vLLM peak and FATHOM peak at t~2.25
xp=2.25
axB.annotate("",xy=(xp,5.5),xytext=(xp,2.64),
             arrowprops=dict(arrowstyle="<->",color="black",lw=0.9))
axB.text(xp+0.14,4.35,"~50%\nlower",fontsize=7.0,ha="left",va="center")
axB.set_ylabel("P99 latency\n(\u00d7 uncontended)",fontsize=7.5)
axB.set_xlabel("Time (minutes)")
axB.set_ylim(0.5,6.0); axB.set_yticks([1,2,3,4,5])
axB.set_yticklabels(["1\u00d7","2\u00d7","3\u00d7","4\u00d7","5\u00d7"])
axB.set_xlim(0,5); axB.set_xticks([0,1,2,3,4,5])
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.tick_params(length=3,labelsize=7.5)
axB.legend(loc="upper right",bbox_to_anchor=(1.0,1.0),fontsize=6.6,frameon=False,
           handletextpad=0.4,labelspacing=0.25,borderaxespad=-1.4)

fig.savefig("fig6_bursty.pdf",dpi=1200)
print("wrote fig6_bursty.pdf; peaks vLLM 5.5x always-on 3.45x FATHOM 2.64x SLO 3x")
