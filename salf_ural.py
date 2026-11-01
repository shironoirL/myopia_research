"""
Внешняя проверка SALF на российской когорте: Ural Children Myopia Study (Уфа).

Данные: Bikbov M. M. et al. Myopic axial elongation in school children and the
COVID-19 lockdown in Russia: The Ural Children Myopia Study // PLoS ONE. 2023.
Vol. 18, No. 1. e0279020. Файл S2 (SPSS) из Supporting Information, CC BY 4.0.
461 школьник 6,8–16,8 года, два визита с интервалом 0,73–1,98 года (2019–2021),
дети на атропине и ортокератологии исключены авторами — естественное течение.

Задача — та же, что для Kinoshita в salf_benchmark.py: прогноз прироста осевой
длины по исходным признакам. Отличия:
  - исход — прирост в год (интервал между визитами у детей разный);
  - признаки: возраст, SE0, AL0, этнос (русские / нет); пола и лечения в файле нет;
  - возраст в файле отдельной колонкой не дан и восстанавливается точно
    из формулы поправки на возраст, записанной авторами в метке переменной.

Неправдоподобные записи (правило задано до моделирования): глаз исключается,
если суммарное изменение AL < -0,2 мм (укорочение сверх погрешности
биометрии) или > 2,0 мм. Анализ чувствительности — без исключений.

Модели, метрики и валидация (5-блочная CV по детям, 10 повторов) — из salf_benchmark.py.

Запуск: python salf_ural.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from salf_benchmark import OUT, evaluate, summarize, paired_vs_salf, plot

SAV = Path("datasets/ural_children_myopia/journal.pone.0279020.s002.sav")
AL_MIN, AL_MAX = -0.2, 2.0       # допустимое суммарное изменение AL, мм
EYES = {"R": "OD", "L": "OS"}


def read_ural():
    """Таблица «ребёнок × глаз» с исходными признаками и приростом AL."""
    raw, _ = pyreadstat.read_sav(SAV)
    # Метка переменной: CorrForAge = AL + 0.209 * (10.7 - Age_Baseline)
    age = 10.7 - (raw["Biometry_AL_IOL_master_OD_Bline_CorrForAge"]
                  - raw["Biometry_AL_IOL_master_OD_Bline"]) / 0.209
    parts = []
    for eye, code in EYES.items():
        parts.append(pd.DataFrame({
            "child": raw.index, "eye": eye, "age": age,
            "interval": raw["ExaminationInterval_Years"],
            "russian": raw["Ethnicity_Russian_Y1N0"],
            "SE0": raw[f"RefError_AutoRefr_SphpEq_Aftr_{code}_Bline"],
            "AL0": raw[f"Biometry_AL_IOL_master_{code}_Bline"],
            "AL1": raw[f"Biometry_AL_IOL_master_{code}_visit2"],
        }))
    df = pd.concat(parts, ignore_index=True)
    df["dAL"] = df["AL1"] - df["AL0"]
    df["dAL_year"] = df["dAL"] / df["interval"]
    return df


def load_ural_grouped(filter_implausible=True):
    df = read_ural().dropna(subset=["age", "SE0", "AL0", "AL1", "russian"])
    if filter_implausible:
        df = df[df["dAL"].between(AL_MIN, AL_MAX)]
    X = df[["age", "SE0", "AL0", "russian"]].values
    return X, df["dAL_year"].values, df["child"].values


def main():
    df = read_ural()
    bad = df[~df["dAL"].between(AL_MIN, AL_MAX)]
    print(f"Ural: {df['child'].nunique()} детей, {len(df)} глаз; "
          f"исключено как неправдоподобные: {len(bad)} глаз")
    print(bad[["child", "eye", "age", "AL0", "AL1", "dAL"]].round(2).to_string(index=False))

    runs = []
    for flt, label in [(True, "Ural (axial elongation/yr)"),
                       (False, "Ural, без исключений")]:
        X, y, g = load_ural_grouped(flt)
        print(f"{label}: n={len(y)} глаз, {len(set(g))} детей, "
              f"прирост {y.mean():.3f} ± {y.std():.3f} мм/год")
        runs.append(evaluate(X, y, g, label))
    folds = pd.concat(runs, ignore_index=True)
    folds.to_csv(OUT / "salf_ural_folds.csv", index=False)
    summary = summarize(folds)
    summary.to_csv(OUT / "salf_ural_summary.csv", index=False)
    paired = paired_vs_salf(folds)
    paired.to_csv(OUT / "salf_ural_paired.csv", index=False)
    plot(summary[summary.dataset == "Ural (axial elongation/yr)"], "salf_ural.png")

    pd.set_option("display.width", 200)
    cols = ["model", "tau", "MAE_mean", "R2_mean", "FNR_HR_mean", "FPR_HR_mean", "AUC_HR_mean"]
    for ds in summary.dataset.unique():
        print(f"\n=== {ds} ===")
        print(summary[summary.dataset == ds][cols].to_string(index=False))
    print("\n=== SALF-Hybrid минус альтернатива (AUC_HR), p — Nadeau–Bengio ===")
    print(paired[paired.metric == "AUC_HR"].to_string(index=False))
    print(f"\nСохранено: {OUT}/salf_ural_*.csv, salf_ural.png")


if __name__ == "__main__":
    main()
