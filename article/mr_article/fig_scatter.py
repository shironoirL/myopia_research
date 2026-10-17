"""Рисунок 1 статьи: диаграмма рассеяния MR (SNP ориентированы по аллелю, повышающему экспозицию).

Читает results/mr_harmonised.csv, mr_estimates.csv. Запуск из корня репозитория.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OUT = Path(__file__).parent
h = pd.read_csv(R / "mr_harmonised.csv")
est = pd.read_csv(R / "mr_estimates.csv").set_index("method")

s = np.sign(h["beta_exp"]).replace(0, 1)
bx, by = h["beta_exp"] * s, h["beta_out"] * s

# интерсепт Egger для линии (в mr_estimates.csv хранится только наклон)
w = 1 / h["se_out"] ** 2
X = np.column_stack([np.ones(len(bx)), bx])
inter, slope = np.linalg.solve(X.T @ (X * w.values[:, None]), X.T @ (w * by))

fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.errorbar(bx, by, xerr=h["se_exp"], yerr=h["se_out"], fmt="o", ms=3, color="#4C72B0",
            alpha=0.55, elinewidth=0.5)
xr = np.linspace(0, bx.max() * 1.05, 50)
ax.plot(xr, est.loc["IVW", "beta"] * xr, "-", color="#C0392B", label="IVW")
ax.plot(xr, inter + slope * xr, "--", color="#117A65", label="MR-Egger")
ax.plot(xr, est.loc["Weighted median", "beta"] * xr, ":", color="#6C3483", lw=2, label="Взвешенная медиана")
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlim(left=0)
ax.set_xlabel("Эффект SNP на вероятность окончить колледж")
ax.set_ylabel("Эффект SNP на миопию (log OR)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_scatter.png", dpi=200)
print("saved", OUT / "fig1_scatter.png")
