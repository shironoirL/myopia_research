"""
NHANES (2007-2008) — внешняя популяционная проверка факторов риска миопии.

Блок диссертации: внешняя валидация факторов, выявленных на OLSM, на большой
независимой популяции подростков 12-19 лет.

Исход: миопия (сфероэквивалент SE <= -0.5 D) по объективной авторефракции.
Объединение DEMO + VIX + BMX + PAQ по SEQN.

Ограничения (честно): NHANES — одномоментный срез (ассоциации, не развитие);
нет наследственности и часов чтения (проверяем подмножество факторов OLSM);
выборочные веса NHANES не применяются (нескорректированный анализ).

Запуск: python nhanes_analysis.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ND = "datasets/nhanes"
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def load_merged():
    demo = pd.read_sas(f"{ND}/DEMO_E.xpt", format="xport")[
        ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"]]
    vix = pd.read_sas(f"{ND}/VIX_E.xpt", format="xport")
    bmx = pd.read_sas(f"{ND}/BMX_E.xpt", format="xport")[["SEQN", "BMXBMI"]]
    paq = pd.read_sas(f"{ND}/PAQ_E.xpt", format="xport")[
        ["SEQN", "PAD680", "PAQ650", "PAQ665"]]

    vix["SE"] = (vix["VIXORSM"] + vix["VIXORCM"] / 2 +
                 vix["VIXOLSM"] + vix["VIXOLCM"] / 2) / 2
    df = demo.merge(vix[["SEQN", "SE"]], on="SEQN").merge(bmx, on="SEQN").merge(paq, on="SEQN")

    df = df[(df.RIDAGEYR >= 12) & (df.RIDAGEYR <= 19)].dropna(subset=["SE"]).copy()
    df["myopia"] = (df["SE"] <= -0.5).astype(int)
    df["female"] = (df["RIAGENDR"] == 2).astype(int)
    # сидячее время в часах; коды 7777/9999 -> пропуск
    df["sed_hr"] = df["PAD680"].where(df["PAD680"] < 7000) / 60
    df["vigorous"] = (df["PAQ650"] == 1).astype(float)   # интенсивная активность да
    df["moderate"] = (df["PAQ665"] == 1).astype(float)   # умеренная активность да
    return df


def logit_inference(X, y, names):
    """Логистическая регрессия с OR, 95% ДИ и p (Wald), без statsmodels."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    beta = np.concatenate([m.intercept_, m.coef_[0]])
    Xd = np.column_stack([np.ones(len(X)), X])
    p = m.predict_proba(X)[:, 1]
    W = p * (1 - p)
    cov = np.linalg.pinv(Xd.T * W @ Xd)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
    tab = pd.DataFrame({
        "term": ["intercept"] + names,
        "OR": np.exp(beta).round(3),
        "CI_low": np.exp(beta - 1.96 * se).round(3),
        "CI_high": np.exp(beta + 1.96 * se).round(3),
        "p": pval.round(4)})
    return tab


def main():
    df = load_merged()
    feats_cont = ["RIDAGEYR", "BMXBMI", "sed_hr", "INDFMPIR"]
    feats_bin = ["female", "vigorous", "moderate"]
    model_df = df.dropna(subset=feats_cont + feats_bin + ["myopia"]).copy()

    print(f"Подростки 12-19 с полными данными: n={len(model_df)}, "
          f"миопия {model_df.myopia.mean():.1%}\n")

    # стандартизуем непрерывные -> OR на 1 SD (сопоставимо с OLSM)
    Z = model_df.copy()
    for c in feats_cont:
        Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
    names = feats_cont + feats_bin
    X = Z[names].values
    y = Z["myopia"].values

    tab = logit_inference(X, y, names)
    nice = {"RIDAGEYR": "Возраст (1 SD)", "BMXBMI": "BMI (1 SD)",
            "sed_hr": "Сидячее время (1 SD)", "INDFMPIR": "Доход/бедность (1 SD)",
            "female": "Женский пол", "vigorous": "Интенсивная активность",
            "moderate": "Умеренная активность"}
    tab["term"] = tab["term"].map(lambda t: nice.get(t, t))
    tab.to_csv(OUT / "nhanes_logit.csv", index=False)
    print("== Логистическая регрессия: миопия у подростков NHANES ==")
    print(tab.to_string(index=False))

    # сопоставление направления с OLSM
    print("\n== Сопоставление с OLSM ==")
    print(" Возраст: OLSM — старше→ниже риск; NHANES — см. OR возраста.")
    print(" Активность/спорт: OLSM — защитный; NHANES — OR интенсивной/умеренной активности.")

    # forest plot
    plot = tab[tab.term != "intercept"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ypos = np.arange(len(plot))
    ax.errorbar(plot["OR"], ypos,
                xerr=[plot["OR"] - plot["CI_low"], plot["CI_high"] - plot["OR"]],
                fmt="o", color="#4C72B0", capsize=3)
    ax.axvline(1.0, ls="--", color="grey")
    ax.set_yticks(ypos); ax.set_yticklabels(plot["term"])
    ax.set(xlabel="Отношение шансов (OR) миопии, 95% ДИ",
           title=f"NHANES 12-19 лет (n={len(model_df)}): факторы миопии")
    fig.tight_layout()
    fig.savefig(OUT / "nhanes_forest.png", dpi=150)
    plt.close(fig)
    print(f"\nТаблица и forest-plot сохранены в {OUT}/")


if __name__ == "__main__":
    main()
