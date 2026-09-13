#!/usr/bin/env python3
"""Freeway comparison figure (PROVISIONAL, seed 0): Ours vs the CKA-RL paper.

Panels:
  A) Table-1 PERF (final success rate) — Ours vs all paper methods (Freeway col).
  B) Table-1 FWT (forward transfer)    — Ours vs all paper methods.
  C) Ours per-mode final-policy RETENTION (score / local specialist).

Data provenance:
  - Ours PERF/FWT: process_results.py on data/Freeway/fw_s0 (Ours local-phase
    curve vs our Baseline), avg over 8 modes: PERF=0.7528, FWT=0.661.
  - Paper methods: CKA-RL (NeurIPS'25) Table 1, Freeway column (verbatim).
  - Ours retention: final-model 3-episode greedy scores / per-task local refs
    (retention_history.jsonl last entry) -> NOISY PROVISIONAL, mean 81%.
Caveats (in the figure): single seed; Ours PERF is the LOCAL-phase (plasticity)
curve; retention is 3-ep noisy (clean 100-ep GPU eval pending); Ours has live
past-task env access the paper methods do not.
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---- paper Table-1 Freeway (verbatim) + Ours (process_results) ----
METHODS = ["Baseline","FT-1","FT-N","ProgNet","PackNet","MaskNet","CReLUs",
           "CompoNet","CbpNet","CKA-RL","Ours"]
PERF = [0.1247,0.1512,0.7532,0.3125,0.2767,0.0644,0.7835,0.7629,0.7678,0.7923,0.7528]
FWT  = [0.0000,0.6935,0.6935,0.1938,0.1970,-0.0503,0.7303,0.7115,0.7201,0.7429,0.6610]

# ---- Ours per-mode retention (provisional, 3-ep greedy / local specialist) ----
MODES = list(range(8))
RET = [100,72,67,100,68,75,66,99]      # percent
RET_MEAN = float(np.mean(RET))

def barcolors(hl_ckA="#0072B2", hl_ours="#D55E00", other="#BBBBBB"):
    c=[]
    for m in METHODS:
        c.append(hl_ours if m=="Ours" else hl_ckA if m=="CKA-RL" else other)
    return c

plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "savefig.dpi":200,"figure.dpi":150})
fig, ax = plt.subplots(1,3, figsize=(15,4.2))

for a,(vals,name) in zip(ax[:2],[(PERF,"PERF (final success)"),(FWT,"FWT (forward transfer)")]):
    x=np.arange(len(METHODS))
    a.bar(x, vals, color=barcolors(), edgecolor="black", linewidth=0.4)
    a.axhline(0,color="black",lw=0.6)
    for xi,v in zip(x,vals):
        a.text(xi, v+(0.01 if v>=0 else -0.01), f"{v:.2f}", ha="center",
               va="bottom" if v>=0 else "top", fontsize=6.5, rotation=90)
    a.set_xticks(x); a.set_xticklabels(METHODS, rotation=45, ha="right")
    a.set_title(f"Freeway — {name}")
    a.set_ylim(min(0,min(vals))-0.08, 1.0)
# highlight legend note
ax[0].set_ylabel("success rate ∈ [0,1]")

# Panel C: Ours per-mode retention
x=np.arange(len(MODES))
ax[2].bar(x, RET, color="#D55E00", edgecolor="black", linewidth=0.4)
ax[2].axhline(70, color="0.4", ls="--", lw=1.0, label="0.7 retention bar")
ax[2].axhline(RET_MEAN, color="#009E73", ls="-", lw=1.2, label=f"mean {RET_MEAN:.0f}%")
for xi,v in zip(x,RET): ax[2].text(xi, v+1, f"{v}", ha="center", va="bottom", fontsize=7)
ax[2].set_xticks(x); ax[2].set_xticklabels([f"m{m}" for m in MODES])
ax[2].set_ylim(0,110); ax[2].set_ylabel("retention  (score / local specialist), %")
ax[2].set_title("Freeway — Ours per-mode retention"); ax[2].legend(fontsize=7, frameon=False)

fig.suptitle("Freeway: Ours vs CKA-RL (NeurIPS'25) — PROVISIONAL, seed 0", y=1.02, fontsize=12)
fig.text(0.5,-0.13,
  "PROVISIONAL/seed-0. Ours PERF/FWT from process_results (local-phase plasticity curve; FWT vs our Baseline). "
  "Paper methods = CKA-RL Table 1 (Freeway). Retention = 3-ep greedy / local specialist (NOISY; clean 100-ep GPU eval pending). "
  "Ours uses live past-task env access the paper methods do not.",
  ha="center", fontsize=7, style="italic", wrap=True)
fig.tight_layout()
import os
os.makedirs("png",exist_ok=True); os.makedirs("svg",exist_ok=True)
fig.savefig("png/freeway_ours_vs_paper.png", bbox_inches="tight")
fig.savefig("svg/freeway_ours_vs_paper.svg", bbox_inches="tight")
print("wrote png/freeway_ours_vs_paper.png + svg")
print("PERF Ours=%.4f (CKA-RL=%.4f) | FWT Ours=%.4f (CKA-RL=%.4f) | retention mean=%.0f%%"%(
    PERF[-1],PERF[9],FWT[-1],FWT[9],RET_MEAN))
