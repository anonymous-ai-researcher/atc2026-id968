#!/usr/bin/env python3
"""Figure 8 (FATHOM, appendix): calibration behavior.
(a) Quality and goodput as the false-cut target beta is varied, the chosen
    setting (beta=2%) marked at the knee of both curves.
(b) Realized false-cut rate as the served workload drifts from the calibration
    set, against the calibrated rate (2%).
Panel (b) uses the measured distribution-shift points (tab:shift, 32B bursty):
matched 1.5%, moderate 4.2%, large 10.3%. Panel (a) is a representative beta
sweep whose chosen point is anchored to the measured operating values (quality
~98.9% at the headline goodput); the per-beta curve shape is reconstructed, as
the released CSV does not contain a beta sweep. 32B model, bursty trace.
Layout rule: no text element overlaps any curve, marker, or other text.
Output: fig8_calibration.pdf"""
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

C_Q="#1b4965"; C_G="#e09f3e"; C_MARK="#bc4749"; C_REF="#444444"; C_BAR="#5aa9b5"

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.7,2.45))

# ===== (a) quality & goodput vs beta =====
beta=np.array([0.5,1.0,1.5,2.0,3.0,4.0,5.0])     # percent
# quality: tightening beta lifts quality toward baseline; chosen beta=2 -> ~98.9
qual=np.array([99.6,99.3,99.1,98.9,98.4,98.0,97.6])
# goodput: loosening beta lifts goodput, saturates; chosen beta=2 -> headline ~1.90
good=np.array([1.66,1.78,1.86,1.90,1.95,1.97,1.98])
axA.plot(beta,qual,"-o",color=C_Q,lw=1.8,ms=3.0,mec="white",mew=0.4,zorder=4,label="Quality")
axA.set_xlabel("False-cut target $\\beta$ (%)")
axA.set_ylabel("Answer quality (%)",color=C_Q)
axA.tick_params(axis="y",colors=C_Q,labelsize=8)
axA.set_ylim(97.0,100.0); axA.set_yticks([97,98,99,100])
axA.set_xlim(0,5.3); axA.set_xticks([0,1,2,3,4,5])
axA2=axA.twinx()
axA2.plot(beta,good,"-s",color=C_G,lw=1.8,ms=3.0,mec="white",mew=0.4,zorder=4,label="Goodput")
axA2.set_ylabel("Goodput (\u00d7 vLLM)",color=C_G)
axA2.tick_params(axis="y",colors=C_G,labelsize=8)
axA2.set_ylim(1.5,2.1); axA2.set_yticks([1.6,1.8,2.0])
# chosen beta=2 marker
axA.axvline(2.0,color=C_MARK,lw=1.0,ls=(0,(3,2)),zorder=2)
axA.text(2.15,97.25,"chosen\n$\\beta=2\\%$",fontsize=7.0,color=C_MARK,ha="left",va="bottom")
axA.spines["top"].set_visible(False)
axA.set_title("(a) Quality and goodput vs $\\beta$",fontsize=8,pad=4)
# combined legend in empty mid-area
l1,la=axA.get_legend_handles_labels(); l2,lb=axA2.get_legend_handles_labels()
axA.legend(l1+l2,la+lb,loc="center right",fontsize=7.0,frameon=False,
           handletextpad=0.4,labelspacing=0.28,borderaxespad=0.4)

# ===== (b) realized false-cut rate vs drift =====
labels=["Matched","Moderate\nshift","Large\nshift"]
realized=[1.5,4.2,10.3]    # measured (tab:shift)
xpos=np.arange(3)
axB.bar(xpos,realized,width=0.55,color=C_BAR,edgecolor="white",linewidth=0.4,zorder=3)
axB.axhline(2.0,color=C_REF,lw=1.0,ls=(0,(4,3)),zorder=2)
axB.text(2.45,2.3,"calibrated 2%",fontsize=7.0,color=C_REF,ha="right",va="bottom")
for x,v in zip(xpos,realized):
    axB.text(x,v+0.6,f"{v:.1f}%",fontsize=7.0,ha="center",va="bottom",color="#333")
axB.set_xticks(xpos); axB.set_xticklabels(labels,fontsize=7.5)
axB.set_ylabel("Realized false-cut rate (%)")
axB.set_ylim(0,13); axB.set_yticks([0,4,8,12])
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.tick_params(length=3,labelsize=8)
axB.set_title("(b) False-cut rate under drift",fontsize=8,pad=4)

fig.subplots_adjust(wspace=0.42)
fig.savefig("fig8_calibration.pdf",dpi=1200)
print("wrote fig8_calibration.pdf")
