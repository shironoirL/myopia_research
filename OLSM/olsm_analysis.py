"""
OLSM (myopia.csv) — воспроизводимый ML-эксперимент по прогнозу риска миопии.

Глава 1 диссертации: индивидуальный прогноз риска развития миопии у детей.

Что делает скрипт:
  1. Загружает myopia.csv (semicolon-separated).
  2. Сравнивает модели на повторной стратифицированной кросс-валидации.
  3. Считает out-of-fold вероятности для несмещённых ROC / PR / калибровки.
  4. Подбирает порог классификации (high-recall скрининговый режим).
  5. Интерпретирует факторы риска: коэффициенты, OR на 1 SD, permutation importance.
  6. Абляция: вклад SPHEQ vs образ жизни + наследственность.
  7. Сохраняет таблицы (results/*.csv) и графики (results/*.png).

Запуск:
    python olsm_analysis.py
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DROP_COLS = ["ID", "STUDYYEAR", "MYOPIC"]
# Исключаются только из инференциальной OR-модели (не из прогнозных):
#   AL = ACD+LT+VCD (VIF ~3e4 → нестабильные ДИ), DIOPTERHR — лин. комбинация нагрузки.
INFER_DROP = ["AL", "DIOPTERHR"]
BINARY = ["GENDER", "MOMMY", "DADMY"]
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def load_data(path: str = "myopia.csv"):
    df = pd.read_csv(path, sep=";")
    X = df.drop(columns=DROP_COLS)
    y = df["MYOPIC"].astype(int).values
    return df, X, y


def logreg_pipe():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def model_zoo():
    return {
        "LogReg (balanced)": logreg_pipe(),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "GradBoosting": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "HistGB": Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def compare_models(X, y):
    """Повторная стратифицированная CV: среднее ± std по метрикам."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "average_precision", "recall", "precision", "f1"]
    rows = []
    for name, model in model_zoo().items():
        r = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for k in scoring:
            a = r["test_" + k]
            row[k + "_mean"] = round(a.mean(), 3)
            row[k + "_std"] = round(a.std(), 3)
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "model_comparison.csv", index=False)
    return tab


def oof_probabilities(X, y):
    """Out-of-fold вероятности лучшей модели (LogReg) для несмещённых кривых."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(logreg_pipe(), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return proba


def threshold_analysis(y, proba):
    """Метрики при разных порогах + выбор high-recall режима."""
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        rows.append({"threshold": t, "recall": round(recall, 3),
                     "specificity": round(spec, 3), "precision": round(prec, 3),
                     "f1": round(f1, 3), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "threshold_analysis.csv", index=False)
    # high-recall скрининг: САМЫЙ ВЫСОКИЙ порог, ещё удерживающий recall >= 0.85
    # (recall падает с ростом порога, поэтому это максимум specificity при recall>=0.85)
    hr = tab[tab["recall"] >= 0.85]
    chosen = hr.iloc[-1] if len(hr) else tab.iloc[0]
    return tab, chosen


def risk_factors(X, y):
    """Коэффициенты, OR на 1 SD и out-of-fold permutation importance (по PR-AUC).

    DIOPTERHR исключён из интерпретации: это точная линейная комбинация
    READHR/STUDYHR/COMPHR/TVHR (DIOPTERHR = 3(READHR+STUDYHR)+2COMPHR+TVHR),
    из-за чего коэффициенты и permutation importance «размазываются» между
    дублирующими признаками. Для прогноза он остаётся в остальных функциях.
    """
    Xi = X.drop(columns=["DIOPTERHR"])

    # Коэффициенты и OR — на всей выборке (интерпретация знаков и величин эффекта).
    model = logreg_pipe().fit(Xi, y)
    coef = model.named_steps["clf"].coef_[0]

    # Permutation importance — out-of-fold: важность меряется на данных, не
    # видевших обучения, поэтому не смещена оптимистично.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_imp = []
    for tr, te in cv.split(Xi, y):
        m = logreg_pipe().fit(Xi.iloc[tr], y[tr])
        pi = permutation_importance(
            m, Xi.iloc[te], y[te], scoring="average_precision",
            n_repeats=30, random_state=RANDOM_STATE)
        fold_imp.append(pi.importances_mean)
    imp_mean = np.mean(fold_imp, axis=0)
    imp_std = np.std(fold_imp, axis=0)

    tab = pd.DataFrame({
        "feature": Xi.columns,
        "std_coef": coef.round(3),
        "OR_per_SD": np.exp(coef).round(2),
        "perm_importance": imp_mean.round(4),
        "perm_std": imp_std.round(4),
    }).sort_values("perm_importance", ascending=False)
    tab.to_csv(OUT / "risk_factors.csv", index=False)
    return tab


def ablation(df, y):
    """Вклад SPHEQ против образа жизни и наследственности."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    sc = ["roc_auc", "average_precision", "recall"]
    full = [c for c in df.columns if c not in DROP_COLS]
    sets = {
        "Full (with SPHEQ)": full,
        "Только SPHEQ": ["SPHEQ"],
        "Без SPHEQ (low-cost)": [c for c in full if c != "SPHEQ"],
        "Образ жизни + наследственность":
            ["MOMMY", "DADMY", "SPORTHR", "READHR", "COMPHR", "STUDYHR", "TVHR", "GENDER", "AGE"],
    }
    rows = []
    for label, cols in sets.items():
        r = cross_validate(logreg_pipe(), df[cols], y, cv=cv, scoring=sc, n_jobs=-1)
        rows.append({"feature_set": label, "n_features": len(cols),
                     **{k: round(r["test_" + k].mean(), 3) for k in sc}})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "ablation.csv", index=False)
    return tab


def _wald_logit(Xstd, y):
    """MLE-логит (без регуляризации) + Wald-ковари='(XᵀWX)⁻¹. Возвращает beta, se, p."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=np.inf, max_iter=10000).fit(Xstd, y)
    beta = np.concatenate([m.intercept_, m.coef_[0]])
    Xd = np.column_stack([np.ones(len(Xstd)), Xstd])
    p = m.predict_proba(Xstd)[:, 1]
    W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.pinv((Xd.T * W) @ Xd)))
    pval = 2 * (1 - stats.norm.cdf(np.abs(beta / se)))
    return beta, se, pval


def adjusted_or(df, y):
    """Одномерные и многомерные OR (на 1 SD для непрерывных) с 95% ДИ и p (Wald).

    Ядро инференциального раздела «факторы риска». Непрерывные признаки
    стандартизуются → OR интерпретируется как изменение шансов на +1 SD.
    AL и DIOPTERHR исключены (см. INFER_DROP): иначе мультиколлинеарность
    (VIF ~3e4) взрывает доверительные интервалы многомерной модели.
    """
    feats = [c for c in df.columns if c not in DROP_COLS + INFER_DROP]
    cont = [c for c in feats if c not in BINARY]
    Z = df[feats].copy().astype(float)
    for c in cont:
        Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()

    uni = []
    for c in feats:
        b, se, pv = _wald_logit(Z[[c]].values, y)
        uni.append((np.exp(b[1]), np.exp(b[1] - 1.96 * se[1]),
                    np.exp(b[1] + 1.96 * se[1]), pv[1]))
    b, se, pv = _wald_logit(Z[feats].values, y)

    tab = pd.DataFrame({
        "feature": feats,
        "OR_uni": [round(u[0], 3) for u in uni],
        "CI_uni": [f"{u[1]:.2f}-{u[2]:.2f}" for u in uni],
        "p_uni": [round(u[3], 4) for u in uni],
        "OR_adj": np.exp(b[1:]).round(3),
        "CI_adj": [f"{np.exp(b[i] - 1.96 * se[i]):.2f}-{np.exp(b[i] + 1.96 * se[i]):.2f}"
                   for i in range(1, len(b))],
        "p_adj": np.round(pv[1:], 4),
    })
    tab.to_csv(OUT / "adjusted_or.csv", index=False)
    return tab


# ---------- figures ----------

def fig_roc_pr(y, proba):
    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, lw=2, label=f"LogReg (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity (Recall)", title="ROC")
    ax[0].legend(loc="lower right")
    ax[1].plot(rec, prec, lw=2, label=f"LogReg (PR-AUC={ap:.3f})")
    ax[1].axhline(y.mean(), ls="--", color="grey", label=f"floor={y.mean():.3f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[1].legend(loc="upper right")
    fig.suptitle("OLSM: прогноз риска миопии (out-of-fold)")
    fig.tight_layout()
    fig.savefig(OUT / "roc_pr.png", dpi=150)
    plt.close(fig)


def oof_probabilities_plain(X, y):
    """OOF вероятности LogReg БЕЗ балансировки — естественно откалиброванные."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def fig_calibration(y, proba_bal, proba_plain):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="идеальная")
    for proba, lab, c in [(proba_bal, "LogReg balanced (для recall)", "#C44E52"),
                          (proba_plain, "LogReg natural (калибр.)", "#4C72B0")]:
        fp, mp = calibration_curve(y, proba, n_bins=8, strategy="quantile")
        ax.plot(mp, fp, "o-", color=c, label=lab)
    ax.set(xlabel="Предсказанная вероятность", ylabel="Доля фактических миопов",
           title="Калибровка (out-of-fold)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)
    plt.close(fig)


def fig_importance(rf_tab):
    top = rf_tab.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["perm_importance"], xerr=top["perm_std"], color="#4C72B0")
    ax.set(xlabel="Permutation importance (Δ PR-AUC)",
           title="Важность факторов риска (LogReg)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    df, X, y = load_data()
    print(f"Данные: {df.shape[0]} строк, {X.shape[1]} признаков, "
          f"миопов {y.sum()} ({y.mean():.1%})\n")

    print("== Сравнение моделей (CV 5x10) ==")
    print(compare_models(X, y).to_string(index=False), "\n")

    proba = oof_probabilities(X, y)
    print(f"== Out-of-fold LogReg: ROC-AUC={roc_auc_score(y, proba):.3f}, "
          f"PR-AUC={average_precision_score(y, proba):.3f} ==\n")

    thr_tab, chosen = threshold_analysis(y, proba)
    print("== Подбор порога (high-recall скрининг) ==")
    print(f"Выбран порог={chosen['threshold']}: recall={chosen['recall']}, "
          f"specificity={chosen['specificity']}, precision={chosen['precision']}\n")

    print("== Факторы риска (permutation importance, OOF) ==")
    rf_tab = risk_factors(X, y)
    print(rf_tab.to_string(index=False), "\n")

    print("== Отношения шансов: одномерные vs скорректированные (Wald) ==")
    print(adjusted_or(df, y).to_string(index=False), "\n")

    print("== Абляция ==")
    print(ablation(df, y).to_string(index=False), "\n")

    fig_roc_pr(y, proba)
    fig_calibration(y, proba, oof_probabilities_plain(X, y))
    fig_importance(rf_tab)
    print(f"Графики и таблицы сохранены в {OUT}/")


if __name__ == "__main__":
    main()
