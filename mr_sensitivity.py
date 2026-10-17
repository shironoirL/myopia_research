"""
Дополнительные проверки менделевской рандомизации (к mr_analysis.py).

Источники (сверено с GWAS Catalog, 2026-10):
  exposure GCST90029012 — Loh et al. 2018 (PMID 29892013), BOLT-LMM, UK Biobank,
           «Educational attainment (college completion)», бинарный признак,
           эффекты на наблюдаемой шкале (доля окончивших колледж), N = 454 813;
  outcome  GCST90044326 — Jiang et al. 2021 (PMID 34737426), fastGWA-GLMM, UK Biobank,
           поле 6147_1 «очки/линзы из-за близорукости» (самоотчёт), 36 623 / 419 031.
Обе GWAS — на одной когорте UK Biobank (почти полное пересечение выборок).

Что делает скрипт:
  1. Пересчёт оценки прямой MR в интерпретируемую шкалу:
     OR миопии на +10 п.п. генетически предсказанной вероятности окончить колледж.
  2. Тест направленности Стайгера + IVW после Steiger-фильтрации.
  3. Обратная MR: миопия -> окончание колледжа (инструменты миопии, PLINK-клампинг).

Запуск: python mr_sensitivity.py   (читает ~1 ГБ GWAS, несколько минут)
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mr_analysis import EA, MYO, PVAL_THR, PALINDROME, mr_ivw, mr_egger, mr_weighted_median

warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
PLINK = "tools/plink"
PANEL = Path("datasets/gwas/1kg/EUR")
N_EXP = 454_813
N_CASE, N_CTRL = 36_623, 419_031


# ---------- доля объяснённой дисперсии ----------
def r2_continuous(beta, se, n):
    """Наблюдаемая шкала (BOLT-LMM на 0/1-признаке): r2 ≈ t²/(t² + n)."""
    t2 = (beta / se) ** 2
    return t2 / (t2 + n)


def r2_logit_liability(lor, eaf):
    """Шкала подверженности для логистической модели: vg / (vg + π²/3)."""
    vg = 2 * eaf * (1 - eaf) * lor ** 2
    return vg / (vg + np.pi ** 2 / 3)


def steiger(r2_exp, r2_out, n_exp, n_out):
    """Тест Стайгера по суммарным r (Fisher z). Положительный z — направление exposure→outcome."""
    r_e, r_o = np.sqrt(r2_exp.sum()), np.sqrt(r2_out.sum())
    z = (np.arctanh(r_e) - np.arctanh(r_o)) / np.sqrt(1 / (n_exp - 3) + 1 / (n_out - 3))
    return r2_exp.sum(), r2_out.sum(), z, 2 * stats.norm.sf(abs(z))


# ---------- 1–2. прямая MR: шкала и Стайгер ----------
def forward_checks():
    h = pd.read_csv(OUT / "mr_harmonised.csv")
    bx, by, sx, sy = (h[c].values for c in ["beta_exp", "beta_out", "se_exp", "se_out"])
    b, se, p, *_ = mr_ivw(bx, by, sy)
    rows = [{"analysis": "IVW, все инструменты", "n_snp": len(h), "beta": b, "se": se, "p": p}]

    r2e = r2_continuous(bx, sx, N_EXP)
    r2o = r2_logit_liability(by, h["eaf_exp"].values)
    se_sum, so_sum, z, pz = steiger(r2e, r2o, N_EXP, N_CASE + N_CTRL)
    keep = r2e > r2o
    b2, se2, p2, *_ = mr_ivw(bx[keep], by[keep], sy[keep])
    rows.append({"analysis": "IVW после Steiger-фильтрации", "n_snp": int(keep.sum()),
                 "beta": b2, "se": se2, "p": p2})
    res = pd.DataFrame(rows)
    for k in [1, 10]:   # +1 = от 0 до 100% вероятности; +0.1 = +10 п.п.
        f = 1.0 if k == 1 else 0.1
        res[f"OR_per_{'100pp' if k == 1 else '10pp'}"] = np.exp(f * res["beta"])
        res[f"OR_lo_{'100pp' if k == 1 else '10pp'}"] = np.exp(f * (res["beta"] - 1.96 * res["se"]))
        res[f"OR_hi_{'100pp' if k == 1 else '10pp'}"] = np.exp(f * (res["beta"] + 1.96 * res["se"]))
    st = pd.DataFrame([{"r2_exposure_sum": se_sum, "r2_outcome_sum": so_sum,
                        "steiger_z": z, "steiger_p": pz,
                        "snps_wrong_direction": int((~keep).sum())}])
    return res, st


# ---------- 3. обратная MR ----------
def select_myopia_instruments():
    cols = ["variant_id", "effect_allele", "other_allele", "effect_allele_frequency",
            "beta", "standard_error", "p_value"]
    sig = []
    for ch in pd.read_csv(MYO, sep="\t", usecols=cols, chunksize=500_000, low_memory=False):
        ch["p_value"] = pd.to_numeric(ch["p_value"], errors="coerce")
        sig.append(ch[ch["p_value"] < PVAL_THR])
    sig = pd.concat(sig).dropna(subset=["variant_id", "beta", "standard_error"])
    sig = sig[sig["variant_id"].str.startswith("rs")]
    sig = sig[(sig["effect_allele_frequency"] > 0.01) & (sig["effect_allele_frequency"] < 0.99)]
    with tempfile.TemporaryDirectory() as td:
        assoc = os.path.join(td, "myo.assoc")
        sig.rename(columns={"variant_id": "SNP", "p_value": "P"})[["SNP", "P"]].to_csv(assoc, sep="\t", index=False)
        subprocess.run([PLINK, "--bfile", str(PANEL), "--clump", assoc,
                        "--clump-p1", "5e-8", "--clump-r2", "0.001", "--clump-kb", "10000",
                        "--clump-snp-field", "SNP", "--clump-field", "P",
                        "--out", os.path.join(td, "clp")], check=True, capture_output=True)
        keep = set(pd.read_csv(os.path.join(td, "clp.clumped"), sep=r"\s+")["SNP"])
    inst = sig[sig["variant_id"].isin(keep)].drop_duplicates("variant_id")
    inst = inst[~inst.apply(lambda r: frozenset((r["effect_allele"], r["other_allele"])) in PALINDROME, axis=1)]
    return len(sig), len(keep), inst


def lookup_ea(rsids):
    want = set(rsids)
    cols = ["hm_rsid", "hm_effect_allele", "hm_other_allele", "hm_beta", "standard_error"]
    found = [ch[ch["hm_rsid"].isin(want)]
             for ch in pd.read_csv(EA, sep="\t", usecols=cols, chunksize=500_000, low_memory=False)]
    return pd.concat(found).drop_duplicates("hm_rsid").rename(columns={"standard_error": "se_ea"})


def reverse_mr():
    n_sig, n_clumped, inst = select_myopia_instruments()
    ea = lookup_ea(inst["variant_id"])
    m = inst.merge(ea, left_on="variant_id", right_on="hm_rsid")
    rows = []
    for _, r in m.iterrows():
        e_x, o_x = str(r["effect_allele"]).upper(), str(r["other_allele"]).upper()
        e_y, o_y = str(r["hm_effect_allele"]).upper(), str(r["hm_other_allele"]).upper()
        by = r["hm_beta"]
        if (e_y, o_y) == (o_x, e_x):
            by = -by
        elif (e_y, o_y) != (e_x, o_x):
            continue
        rows.append({"rsid": r["variant_id"], "beta_exp": r["beta"], "se_exp": r["standard_error"],
                     "beta_out": by, "se_out": r["se_ea"]})
    h = pd.DataFrame(rows).dropna()
    h.to_csv(OUT / "mr_reverse_harmonised.csv", index=False)

    bx, by, sx, sy = (h[c].values for c in ["beta_exp", "beta_out", "se_exp", "se_out"])
    ivw = mr_ivw(bx, by, sy)
    eg = mr_egger(bx, by, sy)
    wm = mr_weighted_median(bx, by, sx, sy)
    F = (bx / sx) ** 2
    res = pd.DataFrame([
        {"method": "IVW", "beta": ivw[0], "se": ivw[1], "p": ivw[2]},
        {"method": "MR-Egger", "beta": eg[0], "se": eg[1], "p": eg[2]},
        {"method": "Weighted median", "beta": wm[0], "se": wm[1], "p": wm[2]},
    ])
    meta = {"significant_snps": n_sig, "clumped": n_clumped, "harmonised": len(h),
            "F_mean": F.mean(), "F_min": F.min(), "Q": ivw[3], "Q_df": ivw[4],
            "Q_p": stats.chi2.sf(ivw[3], ivw[4]), "egger_intercept": eg[3], "egger_intercept_p": eg[5]}
    return res, meta


def main():
    fwd, st = forward_checks()
    fwd.round(4).to_csv(OUT / "mr_forward_rescaled.csv", index=False)
    st.to_csv(OUT / "mr_steiger.csv", index=False)
    pd.set_option("display.width", 200)
    print("== Прямая MR: окончание колледжа -> миопия ==")
    print(fwd.round(4).to_string(index=False))
    print("\n== Тест Стайгера ==")
    print(st.to_string(index=False))

    rev, meta = reverse_mr()
    rev.round(5).to_csv(OUT / "mr_reverse.csv", index=False)
    pd.Series(meta).to_csv(OUT / "mr_reverse_meta.csv")
    print("\n== Обратная MR: миопия -> окончание колледжа (β: изменение доли на 1 log-OR миопии) ==")
    print(rev.round(5).to_string(index=False))
    print(pd.Series(meta).round(4).to_string())


if __name__ == "__main__":
    main()
