"""
Эмпирическая апробация гибридной схемы с SALF против базовых моделей
(OLS, GBM) по схеме и метрикам из статьи: MAE, R^2, WMAE и FNR в
клинически значимой зоне высокого риска. 5-кратная кросс-валидация.

Цели (непрерывное «прогрессирование / тяжесть», выше = хуже):
  - Kinoshita: прирост аксиальной длины за 27 мес (истинное прогрессирование);
  - myopia.csv: -SPHEQ как балл тяжести миопии (демонстрационный корпус).

Запуск: python run_salf_experiment.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from salf_hybrid import SALFHybrid

warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
RS = 42


def wmae(y, yhat, hr_mask, w_hr=3.0):
    w = np.where(hr_mask, w_hr, 1.0)
    return np.sum(w * np.abs(y - yhat)) / np.sum(w)


def fnr_highrisk(y_tr, y_te, yhat_te):
    """FNR для зоны быстрого прогрессирования (порог = q75 обучающей выборки)."""
    thr = np.quantile(y_tr, 0.75)
    true_pos = y_te >= thr
    pred_pos = yhat_te >= thr
    fn = np.sum(true_pos & ~pred_pos)
    tp = np.sum(true_pos & pred_pos)
    return fn / (tp + fn) if (tp + fn) else np.nan, thr


def evaluate(X, y, label):
    X = np.asarray(X, float); y = np.asarray(y, float)
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    delta = 0.5 * y.std()
    models = {
        "OLS": lambda: LinearRegression(),
        "GBM": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RS),
        "SALF-Hybrid": lambda: SALFHybrid(tau=0.75, delta=delta, l2=1.0, lam_decor=0.5),
    }
    rows = []
    for name, make in models.items():
        m = {"MAE": [], "R2": [], "WMAE": [], "FNR_highrisk": []}
        for tr, te in kf.split(X):
            mdl = make().fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            thr = np.quantile(y[tr], 0.75)
            hr = y[te] >= thr
            m["MAE"].append(mean_absolute_error(y[te], pred))
            m["R2"].append(r2_score(y[te], pred))
            m["WMAE"].append(wmae(y[te], pred, hr))
            fnr, _ = fnr_highrisk(y[tr], y[te], pred)
            m["FNR_highrisk"].append(fnr)
        rows.append({"dataset": label, "model": name,
                     **{f"{k}_mean": round(np.nanmean(v), 4) for k, v in m.items()},
                     **{f"{k}_std": round(np.nanstd(v), 4) for k, v in m.items()}})
    return pd.DataFrame(rows)


def load_olsm():
    df = pd.read_csv("myopia.csv", sep=";")
    y = -df["SPHEQ"].values                          # балл тяжести: выше = хуже
    X = df.drop(columns=["ID", "STUDYYEAR", "MYOPIC", "SPHEQ"]).values
    return X, y


def load_kinoshita():
    raw = pd.read_excel("datasets/Orthokeratology  0.01 Atropine/Dataset (Kinoshita).xlsx", header=None)
    AL = {0: 8, 3: 10, 9: 13, 15: 16, 21: 19, 27: 22}
    parts = []
    for r0, r1, combo in [(4, 80, 1), (84, 154, 0)]:
        b = raw.iloc[r0:r1, [0, 2, 3, 4, 7, 8, 22]].copy()
        b.columns = ["sid", "age", "sex", "SE0", "eye", "AL0", "AL27"]
        for c in ["sid", "age", "sex"]:
            b[c] = b[c].ffill()
        b = b[b["eye"].isin(["R", "L"])].copy()
        b["combo"] = combo
        b["sexF"] = (b["sex"] == "F").astype(int)
        for c in ["age", "SE0", "AL0", "AL27"]:
            b[c] = pd.to_numeric(b[c], errors="coerce")
        parts.append(b)
    df = pd.concat(parts, ignore_index=True).dropna(subset=["AL0", "AL27"])
    y = (df["AL27"] - df["AL0"]).values               # прирост оси: выше = хуже
    X = df[["age", "SE0", "AL0", "combo", "sexF"]].values
    return X, y


def main():
    res = []
    print("Прогон занимает ~1-2 мин (совместная оптимизация маршрутизатора)...\n")
    for loader, label in [(load_olsm, "OLSM (-SPHEQ)"),
                          (load_kinoshita, "Kinoshita (axial elongation)")]:
        X, y = loader()
        r = evaluate(X, y, label)
        res.append(r)
        print(f"=== {label}  (n={len(y)}) ===")
        cols = ["model", "MAE_mean", "R2_mean", "WMAE_mean", "FNR_highrisk_mean"]
        print(r[cols].to_string(index=False), "\n")
    out = pd.concat(res, ignore_index=True)
    out.to_csv(OUT / "salf_experiment.csv", index=False)
    print(f"Полная таблица сохранена в {OUT}/salf_experiment.csv")


if __name__ == "__main__":
    main()
