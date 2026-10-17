"""Рисунок 2 статьи: forest plot оценок MR (OR на +10 п.п. вероятности окончить колледж).

Читает results/mr_estimates.csv, mr_forward_rescaled.csv, mr_reverse.csv.
Запуск из корня репозитория: python article/mr_article/fig_forest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OUT = Path(__file__).parent

est = pd.read_csv(R / "mr_estimates.csv")
resc = pd.read_csv(R / "mr_forward_rescaled.csv")
names = {"IVW": "IVW (179 SNP)", "MR-Egger": "MR-Egger", "Weighted median": "Взвешенная медиана"}
rows = [(names[r.method], r.beta, r.se) for r in est.itertuples()]
st = resc.iloc[1]
rows.append((f"IVW после фильтрации Стайгера ({int(st.n_snp)} SNP)", st.beta, st.se))

lab = [r[0] for r in rows][::-1]
b = np.array([r[1] for r in rows])[::-1]
se = np.array([r[2] for r in rows])[::-1]
OR, lo, hi = np.exp(0.1 * b), np.exp(0.1 * (b - 1.96 * se)), np.exp(0.1 * (b + 1.96 * se))

fig, ax = plt.subplots(figsize=(9, 3.4))
y = np.arange(len(lab))
ax.errorbar(OR, y, xerr=[OR - lo, hi - OR], fmt="s", color="#1F4E79", ecolor="#1F4E79", capsize=3, ms=6)
ax.axvline(1, color="grey", ls="--", lw=1)
ax.set_yticks(y)
ax.set_yticklabels(lab)
for yi, o, l, h in zip(y, OR, lo, hi):
    ax.annotate(f"{o:.3f} ({l:.3f}–{h:.3f})".replace(".", ","), xy=(1.02, yi), xycoords=("axes fraction", "data"),
                va="center", fontsize=9)
ax.set_xlim(0.97, 1.33)
ax.set_xlabel("OR миопии на +10 п. п. вероятности окончить колледж (95% ДИ)")
fig.tight_layout()
fig.savefig(OUT / "fig2_forest.png", dpi=200, bbox_inches="tight")
print("saved", OUT / "fig2_forest.png")
