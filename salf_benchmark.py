"""
Расширенная проверка SALF-гибрида: даёт ли он больше, чем простые
асимметричные альтернативы?

Вопрос рецензента: снижение FNR в зоне высокого риска можно получить
и без новой модели — квантильной регрессией или простым сдвигом прогноза
вверх. Если SALF лишь «сдвигает порог», его ранжирующая способность
(AUC для зоны высокого риска) не лучше, чем у OLS/GBM.

Модели (для каждого tau):
  SALF-Hybrid        — метод автора (salf_hybrid.py);
  SALF-Linear        — абляция: только линейная ветвь по SALF (нужна ли гибридность?);
  ... λ=1e-3         — те же модели со слабой L2-регуляризацией (l2 = 1e-3 вместо 1.0);
  QuantReg           — линейная квантильная регрессия, квантиль tau;
  GBM-Quantile       — градиентный бустинг с квантильной функцией потерь, alpha = tau;
  OLS+shift          — OLS + tau-квантиль невязок обучения (сдвиг порога);
  GBM+shift          — то же для GBM.
Симметричные ориентиры: OLS, GBM (не зависят от tau).

Метрики: MAE, R2, FNR и FPR в зоне высокого риска (порог q75 обучения),
AUC_HR — пороговонезависимая способность прогноза ранжировать детей
по принадлежности к зоне высокого риска.

Валидация: повторная 5-блочная CV (10 повторов). Kinoshita — разбиение
по детям (оба глаза одного ребёнка всегда в одном блоке), иначе утечка.
Сравнение с SALF-Hybrid — скорректированный t-тест Nadeau–Bengio.

Запуск: python salf_benchmark.py   (~10-30 мин, параллельно по блокам)
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salf_hybrid import SALFHybrid, salf, salf_grad_yhat
from run_salf_experiment import load_olsm

warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
RS = 42
N_SPLITS, N_REPEATS = 5, 10
TAUS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
# l2 = 1.0 — настройка из статьи; при малом масштабе y штраф подавляет линейную
# ветвь почти до константы, поэтому проверяем и слабую регуляризацию.
L2_WEAK = 1e-3
SALF_MODELS = ["SALF-Hybrid", "SALF-Hybrid λ=1e-3"]
GBM_KW = dict(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RS)


# ---------- данные ----------
def load_kinoshita_grouped():
    """Как run_salf_experiment.load_kinoshita, но с id ребёнка для GroupKFold."""
    raw = pd.read_excel("datasets/Orthokeratology  0.01 Atropine/Dataset (Kinoshita).xlsx", header=None)
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
        b["child"] = f"{combo}_" + b["sid"].astype(str)
        parts.append(b)
    df = pd.concat(parts, ignore_index=True).dropna(subset=["AL0", "AL27"])
    y = (df["AL27"] - df["AL0"]).values
    X = df[["age", "SE0", "AL0", "combo", "sexF"]].values
    return X, y, df["child"].values


def splits(n, groups, repeat):
    if groups is None:
        return list(KFold(N_SPLITS, shuffle=True, random_state=RS + repeat).split(np.zeros(n)))
    # GroupKFold с перемешиванием групп: случайно перенумеровываем детей
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.default_rng(RS + repeat).permutation(len(uniq))))
    g = np.array([perm[x] for x in groups])
    return list(GroupKFold(N_SPLITS).split(np.zeros(n), groups=g))


# ---------- модели ----------
class SALFLinear:
    """Абляция: только линейная ветвь SALF-гибрида (этап 1)."""
    def __init__(self, tau, delta, l2=1.0):
        self.tau, self.delta, self.l2 = tau, delta, l2

    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        Xs = self.sc_.transform(X); p = Xs.shape[1]

        def obj(th):
            w, b = th[:p], th[p]
            f = Xs @ w + b
            g = salf_grad_yhat(y, f, self.tau, self.delta)
            return (salf(y, f, self.tau, self.delta) + self.l2 * w @ w,
                    np.append(Xs.T @ g + 2 * self.l2 * w, g.sum()))
        th = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B").x
        self.w_, self.b_ = th[:p], th[p]
        return self

    def predict(self, X):
        return self.sc_.transform(X) @ self.w_ + self.b_


class Shifted:
    """Базовая модель + tau-квантиль её невязок на обучении (сдвиг прогноза)."""
    def __init__(self, make, tau):
        self.make, self.tau = make, tau

    def fit(self, X, y):
        self.m_ = self.make().fit(X, y)
        self.shift_ = np.quantile(y - self.m_.predict(X), self.tau)
        return self

    def predict(self, X):
        return self.m_.predict(X) + self.shift_


def model_grid(delta):
    ols = lambda: LinearRegression()
    gbm = lambda: GradientBoostingRegressor(**GBM_KW)
    grid = [("OLS", None, ols), ("GBM", None, gbm)]
    for t in TAUS:
        grid += [
            ("SALF-Hybrid", t, lambda t=t: SALFHybrid(tau=t, delta=delta, l2=1.0, lam_decor=0.5)),
            ("SALF-Hybrid λ=1e-3", t, lambda t=t: SALFHybrid(tau=t, delta=delta, l2=L2_WEAK, lam_decor=0.5)),
            ("SALF-Linear", t, lambda t=t: SALFLinear(t, delta)),
            ("SALF-Linear λ=1e-3", t, lambda t=t: SALFLinear(t, delta, l2=L2_WEAK)),
            ("QuantReg", t, lambda t=t: make_pipeline(
                StandardScaler(), QuantileRegressor(quantile=t, alpha=0.0, solver="highs"))),
            ("GBM-Quantile", t, lambda t=t: GradientBoostingRegressor(loss="quantile", alpha=t, **GBM_KW)),
            ("OLS+shift", t, lambda t=t: Shifted(ols, t)),
            ("GBM+shift", t, lambda t=t: Shifted(gbm, t)),
        ]
    return grid


# ---------- оценка одного блока ----------
def run_fold(X, y, tr, te, repeat, fold):
    warnings.filterwarnings("ignore")          # воркеры joblib не наследуют фильтр
    np.seterr(over="ignore")
    delta = 0.5 * y[tr].std()
    thr = np.quantile(y[tr], 0.75)
    hr = y[te] >= thr
    rows = []
    for name, tau, make in model_grid(delta):
        pred = make().fit(X[tr], y[tr]).predict(X[te])
        flag = pred >= thr
        rows.append({
            "model": name, "tau": tau, "repeat": repeat, "fold": fold,
            "n_train": len(tr), "n_test": len(te),
            "MAE": mean_absolute_error(y[te], pred),
            "R2": r2_score(y[te], pred),
            "FNR_HR": np.mean(~flag[hr]) if hr.any() else np.nan,
            "FPR_HR": np.mean(flag[~hr]) if (~hr).any() else np.nan,
            "AUC_HR": roc_auc_score(hr, pred) if 0 < hr.sum() < len(hr) else np.nan,
        })
    return rows


def evaluate(X, y, groups, label):
    jobs = [(r, f, tr, te) for r in range(N_REPEATS)
            for f, (tr, te) in enumerate(splits(len(y), groups, r))]
    out = Parallel(n_jobs=-1)(delayed(run_fold)(X, y, tr, te, r, f) for r, f, tr, te in jobs)
    df = pd.DataFrame([row for rows in out for row in rows])
    df["dataset"] = label
    return df


# ---------- статистика ----------
def nb_ttest(d, n_train, n_test):
    """Скорректированный t-тест для повторной CV (Nadeau & Bengio, 2003)."""
    d = np.asarray(d); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return 2 * stats.t.sf(abs(t), k - 1)


def summarize(df):
    key = ["dataset", "model", "tau"]
    agg = (df.fillna({"tau": -1}).groupby(key)[["MAE", "R2", "FNR_HR", "FPR_HR", "AUC_HR"]]
           .agg(["mean", "std"]).round(4))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["tau"] = agg["tau"].replace(-1, np.nan)
    return agg


def paired_vs_salf(df):
    """AUC_HR и FNR/FPR: SALF-Hybrid против каждой альтернативы при том же tau."""
    rows = []
    for (ds, d), salf_m in [(g, m) for g in df.groupby("dataset") for m in SALF_MODELS]:
        n_tr, n_te = d["n_train"].mean(), d["n_test"].mean()
        for tau in TAUS:
            s = d[(d.model == salf_m) & (d.tau == tau)].sort_values(["repeat", "fold"])
            for alt in ["SALF-Linear λ=1e-3", "QuantReg", "GBM-Quantile", "OLS+shift", "GBM+shift"]:
                a = d[(d.model == alt) & (d.tau == tau)].sort_values(["repeat", "fold"])
                for m in ["AUC_HR", "FNR_HR", "FPR_HR", "MAE"]:
                    diff = s[m].values - a[m].values
                    ok = ~np.isnan(diff)
                    rows.append({"dataset": ds, "salf": salf_m, "tau": tau, "vs": alt, "metric": m,
                                 "salf_minus_alt": round(np.nanmean(diff), 4),
                                 "p_nb": round(nb_ttest(diff[ok], n_tr, n_te), 4)})
        # AUC симметричных ориентиров не зависит от tau
        s = d[(d.model == salf_m) & (d.tau == 0.75)].sort_values(["repeat", "fold"])
        for base in ["OLS", "GBM"]:
            a = d[d.model == base].sort_values(["repeat", "fold"])
            diff = s["AUC_HR"].values - a["AUC_HR"].values
            rows.append({"dataset": ds, "salf": salf_m, "tau": 0.75, "vs": base, "metric": "AUC_HR",
                         "salf_minus_alt": round(np.nanmean(diff), 4),
                         "p_nb": round(nb_ttest(diff[~np.isnan(diff)], n_tr, n_te), 4)})
    return pd.DataFrame(rows)


# ---------- графики ----------
STYLE = {"SALF-Hybrid": ("#C0392B", "o"), "SALF-Hybrid λ=1e-3": ("#8E44AD", "h"),
         "SALF-Linear λ=1e-3": ("#E67E22", "s"),
         "QuantReg": ("#2E86C1", "^"), "GBM-Quantile": ("#17A589", "v"),
         "OLS+shift": ("#7F8C8D", "D"), "GBM+shift": ("#34495E", "P")}


def plot(summary, fname="salf_benchmark.png"):
    dss = summary["dataset"].unique()
    fig, axes = plt.subplots(2, len(dss), figsize=(6.2 * len(dss), 10), squeeze=False)
    for j, ds in enumerate(dss):
        s = summary[summary.dataset == ds]
        ax1, ax2 = axes[0, j], axes[1, j]
        for m, (c, mk) in STYLE.items():
            r = s[s.model == m].sort_values("tau")
            ax1.plot(r["FPR_HR_mean"], r["FNR_HR_mean"], marker=mk, color=c, label=m)
            ax2.plot(r["tau"], r["AUC_HR_mean"], marker=mk, color=c, label=m)
        for base, mk in [("OLS", "x"), ("GBM", "*")]:
            b = s[s.model == base].iloc[0]
            ax1.scatter(b["FPR_HR_mean"], b["FNR_HR_mean"], marker=mk, s=110, color="k", zorder=5, label=base)
            ax2.axhline(b["AUC_HR_mean"], ls="--" if base == "OLS" else ":", color="k", lw=1, label=f"{base} (AUC)")
        ax1.set(xlabel="FPR (ложные тревоги вне зоны риска)", ylabel="FNR (пропуски в зоне риска)",
                title=f"{ds}: компромисс FNR–FPR по τ")
        ax2.set(xlabel="τ", ylabel="AUC для зоны высокого риска", title=f"{ds}: ранжирующая способность")
        ax1.grid(alpha=.3); ax2.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    Xo, yo = load_olsm()
    Xk, yk, gk = load_kinoshita_grouped()
    print(f"OLSM n={len(yo)}; Kinoshita n={len(yk)} глаз, {len(set(gk))} детей")
    df = pd.concat([evaluate(Xo, yo, None, "OLSM (-SPHEQ)"),
                    evaluate(Xk, yk, gk, "Kinoshita (axial elongation)")], ignore_index=True)
    df.to_csv(OUT / "salf_benchmark_folds.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT / "salf_benchmark_summary.csv", index=False)
    paired = paired_vs_salf(df)
    paired.to_csv(OUT / "salf_benchmark_paired.csv", index=False)
    plot(summary)

    pd.set_option("display.width", 200)
    cols = ["model", "tau", "MAE_mean", "R2_mean", "FNR_HR_mean", "FPR_HR_mean", "AUC_HR_mean"]
    for ds in summary.dataset.unique():
        print(f"\n=== {ds} ===")
        print(summary[summary.dataset == ds][cols].to_string(index=False))
    print("\n=== SALF-Hybrid минус альтернатива (AUC_HR), p — Nadeau–Bengio ===")
    print(paired[paired.metric == "AUC_HR"].to_string(index=False))
    print(f"\nСохранено: {OUT}/salf_benchmark_*.csv, salf_benchmark.png")


if __name__ == "__main__":
    main()
