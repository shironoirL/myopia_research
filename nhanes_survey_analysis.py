"""
NHANES 2007-2008 с учётом сложного выборочного дизайна (к nhanes_analysis.py).

Тот же набор переменных и та же модель, но:
  - веса обследования MEC (WTMEC2YR);
  - дисперсия — линеаризация Тейлора (сэндвич-оценка) с учётом страт SDMVSTRA
    и первичных единиц отбора SDMVPSU, df = число PSU − число страт;
  - анализ подгруппы (12–19 лет) как domain analysis: наблюдения вне подгруппы
    дают нулевой вклад в оценочные уравнения, но все PSU остаются в дизайне.

Запуск: python nhanes_survey_analysis.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nhanes_analysis import load_merged, ND

warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
CONT = ["RIDAGEYR", "BMXBMI", "sed_hr", "INDFMPIR"]
BIN = ["female", "vigorous", "moderate"]
NICE = {"RIDAGEYR": "Возраст (1 SD)", "BMXBMI": "BMI (1 SD)", "sed_hr": "Сидячее время (1 SD)",
        "INDFMPIR": "Доход/бедность (1 SD)", "female": "Женский пол",
        "vigorous": "Интенсивная активность", "moderate": "Умеренная активность"}


def weighted_logit(X, y, w, n_iter=100):
    """Взвешенная логистическая регрессия (псевдоправдоподобие), Ньютон–Рафсон."""
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-X @ beta))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (w * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta, H


def linearized_cov(X, y, w, beta, H, strata, psu):
    """Сэндвич-оценка: H⁻¹ · B · H⁻¹, B — межPSU-ковариация сумм скоров внутри страт."""
    p = 1 / (1 + np.exp(-X @ beta))
    u = X * (w * (y - p))[:, None]
    B = np.zeros((X.shape[1], X.shape[1]))
    for h in np.unique(strata):
        mh = strata == h
        psus = np.unique(psu[mh])
        tot = np.array([u[mh & (psu == j)].sum(0) for j in psus])
        nh = len(psus)
        if nh > 1:
            d = tot - tot.mean(0)
            B += nh / (nh - 1) * d.T @ d
    Hi = np.linalg.inv(H)
    return Hi @ B @ Hi


def main():
    demo = pd.read_sas(f"{ND}/DEMO_E.xpt", format="xport")[["SEQN", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]]
    df = load_merged().merge(demo, on="SEQN")
    model = df.dropna(subset=CONT + BIN + ["myopia", "WTMEC2YR"]).copy()
    model = model[model["WTMEC2YR"] > 0]

    # стандартизация по взвешенным моментам подгруппы (OR на 1 SD популяции)
    w = model["WTMEC2YR"].values
    for c in CONT:
        m = np.average(model[c], weights=w)
        sd = np.sqrt(np.average((model[c] - m) ** 2, weights=w))
        model[c] = (model[c] - m) / sd

    # domain analysis: полный дизайн DEMO, вне подгруппы — нулевые скоры
    full = demo.dropna(subset=["SDMVSTRA", "SDMVPSU"]).merge(
        model[["SEQN", "myopia"] + CONT + BIN], on="SEQN", how="left")
    in_dom = full["myopia"].notna().values
    names = CONT + BIN
    X = np.column_stack([np.ones(len(full)), full[names].fillna(0).values])
    y = full["myopia"].fillna(0).values
    wd = np.where(in_dom, full["WTMEC2YR"].values, 0.0)

    beta, H = weighted_logit(X, y, wd)
    cov = linearized_cov(X, y, wd, beta, H, full["SDMVSTRA"].values, full["SDMVPSU"].values)
    se = np.sqrt(np.diag(cov))
    n_psu = full.groupby("SDMVSTRA")["SDMVPSU"].nunique().sum()
    dfree = n_psu - full["SDMVSTRA"].nunique()
    tcrit = stats.t.ppf(0.975, dfree)
    pval = 2 * stats.t.sf(np.abs(beta / se), dfree)
    tab = pd.DataFrame({"term": ["intercept"] + [NICE[n] for n in names],
                        "OR": np.exp(beta).round(3),
                        "CI_low": np.exp(beta - tcrit * se).round(3),
                        "CI_high": np.exp(beta + tcrit * se).round(3),
                        "p": pval.round(4)})
    tab.to_csv(OUT / "nhanes_survey_logit.csv", index=False)

    prev = np.average(model["myopia"], weights=w)
    print(f"Подростки 12–19: n = {len(model)}, взвешенная распространённость миопии {prev:.1%}, "
          f"df дизайна = {dfree}")
    print(tab.to_string(index=False))
    pd.Series({"n": len(model), "weighted_prevalence": prev, "design_df": dfree}).to_csv(
        OUT / "nhanes_survey_meta.csv")


if __name__ == "__main__":
    main()
