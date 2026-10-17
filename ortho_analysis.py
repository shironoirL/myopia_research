"""
Orthokeratology + 0.01% Atropine (Kinoshita, 2-year RCT) —
оценка эффективности профилактики прогрессирования миопии у детей.

Блок диссертации: «оценка эффективности профилактики».
Сравнение торможения удлинения глаза:
    Комбинация (ОК-линзы + 0.01% атропин)  vs  Монотерапия (только ОК).

Что делает скрипт:
  1. Парсит Excel со сложной многоуровневой шапкой -> чистая таблица "1 строка = 1 глаз".
  2. Считает прирост длины оси (axial elongation) за 27 месяцев.
  3. Сравнивает группы: на ребёнка (среднее 2 глаз), t-test + Mann-Whitney.
  4. ANCOVA: поправка на исходную длину оси, возраст, рефракцию (ручной OLS + инференс).
  5. Строит траектории средней длины оси и распределение прироста.

Запуск:
    python ortho_analysis.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
XLSX = "datasets/Orthokeratology  0.01 Atropine/Dataset (Kinoshita).xlsx"
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# столбцы длины оси по визитам (0-indexed в сыром листе)
AL_COLS = {0: 8, 3: 10, 9: 13, 15: 16, 21: 19, 27: 22}
GROUPS = {"Combination (OK+Atropine)": (4, 80, 1),
          "Monotherapy (OK)": (84, 154, 0)}


def load_clean() -> pd.DataFrame:
    raw = pd.read_excel(XLSX, header=None)
    parts = []
    for label, (r0, r1, combo) in GROUPS.items():
        cols = [0, 2, 3, 4, 7] + list(AL_COLS.values())
        b = raw.iloc[r0:r1, cols].copy()
        b.columns = ["sid", "age", "sex", "SE0", "eye"] + [f"AL_{m}" for m in AL_COLS]
        for c in ["sid", "age", "sex"]:           # данные ребёнка только в строке R-глаза
            b[c] = b[c].ffill()
        b = b[b["eye"].isin(["R", "L"])].copy()
        b["group"] = label
        b["combo"] = combo
        num = ["age", "SE0"] + [f"AL_{m}" for m in AL_COLS]
        b[num] = b[num].apply(pd.to_numeric, errors="coerce")
        parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    df["elong"] = df["AL_27"] - df["AL_0"]        # прирост оси за 2 года
    return df


def per_child(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.dropna(subset=["elong"])
    agg = {f"AL_{m}": "mean" for m in AL_COLS}
    agg.update(elong=("mean"), AL0=("AL_0", "mean"), age=("age", "mean"), SE0=("SE0", "mean"))
    pc = ok.groupby(["group", "combo", "sid"]).agg(
        elong=("elong", "mean"), AL0=("AL_0", "mean"),
        age=("age", "mean"), SE0=("SE0", "mean")).reset_index()
    return pc


def ols_inference(y, X, names):
    """OLS с t-статистиками и p-значениями (без statsmodels)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = len(y) - Xd.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return pd.DataFrame({"term": ["intercept"] + names,
                         "coef": beta.round(4), "se": se.round(4),
                         "t": t.round(2), "p": p.round(4)})


def descriptive(df, pc):
    eyes = df.dropna(subset=["elong"]).groupby("group").agg(
        eyes=("elong", "size"), children=("sid", "nunique"),
        age=("age", "mean"), SE0=("SE0", "mean"),
        AL0=("AL_0", "mean"), elong_mm=("elong", "mean"),
        elong_sd=("elong", "std")).round(3)
    eyes.to_csv(OUT / "ortho_descriptive.csv")
    return eyes


def group_test(pc):
    a = pc[pc.combo == 1]["elong"]
    b = pc[pc.combo == 0]["elong"]
    diff = b.mean() - a.mean()
    _, p_t = stats.ttest_ind(a, b, equal_var=False)
    _, p_u = stats.mannwhitneyu(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    res = {"comb_mean": round(a.mean(), 3), "mono_mean": round(b.mean(), 3),
           "diff_mm": round(diff, 3), "rel_slowing_%": round(diff / b.mean() * 100),
           "t_p": round(p_t, 4), "mannwhitney_p": round(p_u, 4), "cohen_d": round(d, 2)}
    pd.Series(res).to_csv(OUT / "ortho_group_test.csv")
    return res


def ancova(pc):
    y = pc["elong"].values
    X = pc[["combo", "AL0", "age", "SE0"]].values
    tab = ols_inference(y, X, ["combo", "AL0", "age", "SE0"])
    tab.to_csv(OUT / "ortho_ancova.csv", index=False)
    return tab


def fig_trajectory(df):
    months = list(AL_COLS.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"Combination (OK+Atropine)": "#4C72B0", "Monotherapy (OK)": "#C44E52"}
    # средняя длина оси во времени
    for g, sub in df.groupby("group"):
        m = [sub[f"AL_{mo}"].mean() for mo in months]
        s = [sub[f"AL_{mo}"].sem() for mo in months]
        ax[0].errorbar(months, m, yerr=s, marker="o", capsize=3, color=colors[g], label=g)
    ax[0].set(xlabel="Месяцы", ylabel="Длина оси, мм",
              title="Средняя длина оси глаза во времени")
    ax[0].legend(fontsize=8)
    # распределение прироста за 2 года
    data = [df[df.group == g].dropna(subset=["elong"])["elong"] for g in colors]
    bp = ax[1].boxplot(data, labels=["Комбинация", "Монотерапия"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for i, d in enumerate(data, 1):
        ax[1].scatter(np.random.normal(i, 0.05, len(d)), d, s=12, color="grey", alpha=0.6, zorder=3)
    ax[1].set(ylabel="Прирост оси за 27 мес, мм",
              title="Удлинение глаза за 2 года (на глаз)")
    fig.suptitle("Kinoshita RCT: ОК+атропин тормозит удлинение глаза сильнее монотерапии")
    fig.tight_layout()
    fig.savefig(OUT / "ortho_trajectory.png", dpi=150)
    plt.close(fig)


def main():
    df = load_clean()
    pc = per_child(df)
    print("== Описательно (на глаз) ==")
    print(descriptive(df, pc).to_string(), "\n")

    print("== Прирост оси на ребёнка: сравнение групп ==")
    res = group_test(pc)
    print(f"Комбинация {res['comb_mean']} мм vs Монотерапия {res['mono_mean']} мм | "
          f"разница {res['diff_mm']} мм ({res['rel_slowing_%']}% медленнее)")
    print(f"t-test p={res['t_p']} | Mann-Whitney p={res['mannwhitney_p']} | "
          f"Cohen d={res['cohen_d']}\n")

    print("== ANCOVA (поправка на исходную AL, возраст, рефракцию) ==")
    tab = ancova(pc)
    print(tab.to_string(index=False))
    eff = tab[tab.term == "combo"].iloc[0]
    print(f"\nЭффект комбинации с поправкой: {eff['coef']} мм/2года (p={eff['p']})\n")

    fig_trajectory(df)
    print(f"Таблицы и график сохранены в {OUT}/")


if __name__ == "__main__":
    main()
