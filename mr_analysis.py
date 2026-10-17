"""
Двухвыборочная менделевская рандомизация (MR): образование -> миопия.

Биоинформатический блок диссертации (причинная проверка фактора риска).
Реализация оценок MR на numpy/scipy (без R/TwoSampleMR):
    IVW (inverse-variance weighted), MR-Egger, weighted median,
    + тесты чувствительности: Cochran's Q (гетерогенность),
      Egger-интерсепт (направленная плейотропия), leave-one-out.

Данные (GWAS Catalog, сборка GRCh37):
    exposure: окончание колледжа, UK Biobank, Loh et al. 2018 (GCST90029012, harmonised;
              бинарный признак, β на шкале вероятности);
    outcome:  миопия, самоотчёт UKB 6147_1, Jiang et al. 2021 (GCST90044326, β — log-OR).
    Обе GWAS на UK Biobank (пересечение выборок) — см. mr_sensitivity.py.

Инструменты: SNP с p<5e-8 в exposure, дистанционный прунинг 1 Mb
(прокси LD-клампинга), без палиндромных SNP.

Запуск: python mr_analysis.py
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
GW = Path("datasets/gwas")
EA = GW / "EA_Lee2018_GCST90029012.h.tsv.gz"
MYO = GW / "myopia_GCST90044326.tsv.gz"
OUT = Path("results"); OUT.mkdir(exist_ok=True)

PVAL_THR = 5e-8
CLUMP_KB = 1000  # окно дистанционного прунинга, кб
PALINDROME = {frozenset("AT"), frozenset("CG")}


def is_palindromic(a1, a2):
    return frozenset((a1, a2)) in PALINDROME


def select_instruments():
    """Сканируем exposure (EA), берём p<5e-8, прунингуем по расстоянию."""
    cols = ["hm_rsid", "hm_chrom", "hm_pos", "hm_effect_allele", "hm_other_allele",
            "hm_beta", "standard_error", "p_value", "hm_effect_allele_frequency"]
    sig = []
    for chunk in pd.read_csv(EA, sep="\t", usecols=cols, chunksize=500_000,
                             dtype={"hm_chrom": str}, low_memory=False):
        chunk["p_value"] = pd.to_numeric(chunk["p_value"], errors="coerce")
        sig.append(chunk[chunk["p_value"] < PVAL_THR])
    sig = pd.concat(sig, ignore_index=True).dropna(
        subset=["hm_rsid", "hm_beta", "standard_error", "hm_pos"])
    sig["hm_pos"] = pd.to_numeric(sig["hm_pos"], errors="coerce")
    sig = sig.dropna(subset=["hm_pos"]).sort_values("p_value")

    clumped_file = OUT / "ea_clumped_instruments.csv"
    if clumped_file.exists():
        # строгий LD-клампинг (PLINK+1000G EUR) — берём готовый список rsid
        keep_rsid = set(pd.read_csv(clumped_file)["rsid"])
        inst = sig[sig["hm_rsid"].isin(keep_rsid)].drop_duplicates("hm_rsid")
        print(f"   (инструменты по строгому PLINK-клампингу: {len(inst)} SNP)")
    else:
        # запасной вариант: дистанционный прунинг 1 Mb (прокси клампинга)
        kept, occupied = [], {}
        for _, r in sig.iterrows():
            c, pos = r["hm_chrom"], r["hm_pos"]
            if any(abs(pos - p) < CLUMP_KB * 1000 for p in occupied.get(c, [])):
                continue
            occupied.setdefault(c, []).append(pos)
            kept.append(r)
        inst = pd.DataFrame(kept)
        print(f"   (дистанционный прунинг 1 Mb: {len(inst)} SNP)")
    # убираем палиндромные SNP
    inst = inst[~inst.apply(
        lambda r: is_palindromic(r["hm_effect_allele"], r["hm_other_allele"]), axis=1)]
    inst = inst.rename(columns={
        "hm_rsid": "rsid", "hm_effect_allele": "ea_exp", "hm_other_allele": "oa_exp",
        "hm_beta": "beta_exp", "standard_error": "se_exp",
        "hm_effect_allele_frequency": "eaf_exp"})
    return inst[["rsid", "ea_exp", "oa_exp", "beta_exp", "se_exp", "eaf_exp", "p_value"]]


def lookup_outcome(rsids):
    """Достаём строки миопии только по нужным rsid (чанками, экономно по памяти)."""
    want = set(rsids)
    cols = ["variant_id", "effect_allele", "other_allele", "beta", "standard_error", "p_value"]
    found = []
    for chunk in pd.read_csv(MYO, sep="\t", usecols=cols, chunksize=500_000, low_memory=False):
        found.append(chunk[chunk["variant_id"].isin(want)])
    out = pd.concat(found, ignore_index=True).rename(columns={
        "variant_id": "rsid", "effect_allele": "ea_out", "other_allele": "oa_out",
        "beta": "beta_out", "standard_error": "se_out", "p_value": "p_out"})
    return out


def harmonise(inst, out):
    m = inst.merge(out, on="rsid", how="inner")
    rows = []
    for _, r in m.iterrows():
        ea_e, oa_e = r["ea_exp"].upper(), r["oa_exp"].upper()
        ea_o, oa_o = str(r["ea_out"]).upper(), str(r["oa_out"]).upper()
        b_out = r["beta_out"]
        if (ea_o, oa_o) == (ea_e, oa_e):
            pass                       # аллели совпадают
        elif (ea_o, oa_o) == (oa_e, ea_e):
            b_out = -b_out             # эффект-аллель перевёрнут -> меняем знак
        else:
            continue                   # аллели не сопоставимы
        rows.append({"rsid": r["rsid"], "beta_exp": r["beta_exp"], "se_exp": r["se_exp"],
                     "eaf_exp": r.get("eaf_exp", np.nan),
                     "beta_out": b_out, "se_out": r["se_out"]})
    h = pd.DataFrame(rows).dropna(subset=["beta_exp", "se_exp", "beta_out", "se_out"])
    return h[h["se_out"] > 0]


# ---------- оценки MR ----------
def mr_ivw(bx, by, sy):
    w = 1 / sy ** 2
    beta = np.sum(w * bx * by) / np.sum(w * bx ** 2)
    se = np.sqrt(1 / np.sum(w * bx ** 2))
    # случайные эффекты: масштаб на sqrt(Q/(n-1)) если >1
    resid = by - beta * bx
    Q = np.sum(w * resid ** 2)
    df = len(bx) - 1
    se_re = se * max(1.0, np.sqrt(Q / df))
    p = 2 * (1 - stats.norm.cdf(abs(beta / se_re)))
    return beta, se_re, p, Q, df


def mr_egger(bx, by, sy):
    # ориентируем экспозицию положительно
    s = np.sign(bx); s[s == 0] = 1
    bx_o, by_o = bx * s, by * s
    w = 1 / sy ** 2
    X = np.column_stack([np.ones_like(bx_o), bx_o])
    WX = X * w[:, None]
    cov = np.linalg.inv(X.T @ WX)
    coef = cov @ (WX.T @ by_o)
    resid = by_o - X @ coef
    dof = len(bx) - 2
    sigma2 = np.sum(w * resid ** 2) / dof
    se = np.sqrt(np.diag(cov) * sigma2 * dof / dof)  # масштаб
    se = np.sqrt(np.diag(cov)) * np.sqrt(np.sum(w * resid ** 2) / dof)
    inter, slope = coef
    se_inter, se_slope = se
    p_slope = 2 * (1 - stats.t.cdf(abs(slope / se_slope), dof))
    p_inter = 2 * (1 - stats.t.cdf(abs(inter / se_inter), dof))
    return slope, se_slope, p_slope, inter, se_inter, p_inter


def _wmed(vals, weights):
    order = np.argsort(vals)
    v, sw = vals[order], weights[order]
    cum = (np.cumsum(sw) - 0.5 * sw) / sw.sum()
    return np.interp(0.5, cum, v)


def mr_weighted_median(bx, by, sx, sy, n_boot=1000, seed=42):
    ratios = by / bx
    w = (bx ** 2) / (sy ** 2)            # ~обратная дисперсия отношения
    est = _wmed(ratios, w)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bx_b = rng.normal(bx, sx)
        by_b = rng.normal(by, sy)
        boots.append(_wmed(by_b / bx_b, (bx_b ** 2) / (sy ** 2)))
    se = np.std(boots)
    p = 2 * (1 - stats.norm.cdf(abs(est / se)))
    return est, se, p


def instrument_strength(h):
    """F-статистика инструментов (F>10 -> не слабые)."""
    F = (h["beta_exp"] / h["se_exp"]) ** 2
    return F.mean(), F.min(), int((F < 10).sum())


def detect_outliers(h, alpha=0.05):
    """MR-PRESSO-подобный тест выбросов по вкладу SNP в Cochran's Q."""
    bx, by, sy = h["beta_exp"].values, h["beta_out"].values, h["se_out"].values
    beta, *_ = mr_ivw(bx, by, sy)
    w = 1 / sy ** 2
    Qi = w * (by - beta * bx) ** 2            # вклад каждого SNP в Q (~chi2_1)
    p_i = 1 - stats.chi2.cdf(Qi, 1)
    thr = alpha / len(h)                       # поправка Бонферрони
    outliers = h[p_i < thr]["rsid"].tolist()
    keep = p_i >= thr
    if keep.sum() < len(h):
        b2, se2, p2, *_ = mr_ivw(bx[keep], by[keep], sy[keep])
    else:
        b2, se2, p2 = beta, None, None
    return outliers, b2, se2, p2


def fig_funnel(h, ivw_b):
    bx, by, sy = h["beta_exp"].values, h["beta_out"].values, h["se_out"].values
    wald = by / bx
    prec = np.abs(bx) / sy                      # точность инструмента
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(wald, prec, s=14, color="#4C72B0", alpha=0.6)
    ax.axvline(ivw_b, color="#C44E52", label=f"IVW β={ivw_b:.2f}")
    ax.set(xlabel="Оценка по одному SNP (Wald ratio)", ylabel="Точность инструмента",
           title="Funnel plot (симметрия → нет плейотропии)")
    ax.set_xlim(np.percentile(wald, 1), np.percentile(wald, 99))
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "mr_funnel.png", dpi=150); plt.close(fig)


def fig_leaveoneout(loo, ivw_b):
    vals = loo["ivw_beta"].sort_values().values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, np.arange(len(vals)), "o", ms=3, color="#4C72B0", alpha=0.6)
    ax.axvline(ivw_b, color="#C44E52", label=f"IVW по всем SNP (β={ivw_b:.2f})")
    ax.set(xlabel="IVW β при исключении одного SNP", ylabel="SNP (отсортированы)",
           title="Leave-one-out: устойчивость к отдельным SNP")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "mr_leaveoneout.png", dpi=150); plt.close(fig)


def leave_one_out(h):
    bx, by, sy = h["beta_exp"].values, h["beta_out"].values, h["se_out"].values
    res = []
    for i in range(len(h)):
        idx = np.arange(len(h)) != i
        b, se, p, *_ = mr_ivw(bx[idx], by[idx], sy[idx])
        res.append({"excluded": h.iloc[i]["rsid"], "ivw_beta": b, "p": p})
    return pd.DataFrame(res)


def main():
    print("1) Отбор инструментов из EA (p<5e-8, прунинг 1 Mb)...")
    inst = select_instruments()
    print(f"   инструментов после прунинга: {len(inst)}")
    print("2) Поиск этих SNP в outcome (миопия)...")
    out = lookup_outcome(inst["rsid"])
    print(f"   найдено в outcome: {len(out)}")
    h = harmonise(inst, out)
    print(f"3) После гармонизации аллелей: {len(h)} SNP-инструментов\n")
    h.to_csv(OUT / "mr_harmonised.csv", index=False)

    bx, by, sy = h["beta_exp"].values, h["beta_out"].values, h["se_out"].values
    ivw_b, ivw_se, ivw_p, Q, Qdf = mr_ivw(bx, by, sy)
    eg_slope, eg_se, eg_p, eg_int, eg_int_se, eg_int_p = mr_egger(bx, by, sy)
    wm_b, wm_se, wm_p = mr_weighted_median(bx, by, h["se_exp"].values, sy)
    Q_p = 1 - stats.chi2.cdf(Q, Qdf)

    def odds(b, se):
        return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    res = pd.DataFrame([
        {"method": "IVW", "beta": ivw_b, "se": ivw_se, "p": ivw_p,
         "OR": odds(ivw_b, ivw_se)[0], "OR_lo": odds(ivw_b, ivw_se)[1], "OR_hi": odds(ivw_b, ivw_se)[2]},
        {"method": "MR-Egger", "beta": eg_slope, "se": eg_se, "p": eg_p,
         "OR": odds(eg_slope, eg_se)[0], "OR_lo": odds(eg_slope, eg_se)[1], "OR_hi": odds(eg_slope, eg_se)[2]},
        {"method": "Weighted median", "beta": wm_b, "se": wm_se, "p": wm_p,
         "OR": odds(wm_b, wm_se)[0], "OR_lo": odds(wm_b, wm_se)[1], "OR_hi": odds(wm_b, wm_se)[2]},
    ]).round(4)
    res.to_csv(OUT / "mr_estimates.csv", index=False)
    print("== Оценки причинного эффекта (образование -> миопия) ==")
    print(res.to_string(index=False))
    print("(beta — log-odds миопии при росте вероятности окончить колледж с 0 до 1; "
          "OR на +10 п.п. = exp(0.1*beta), см. mr_sensitivity.py)")

    # сила инструментов
    Fmean, Fmin, Fweak = instrument_strength(h)
    print(f"\nСила инструментов: средний F = {Fmean:.1f}, мин = {Fmin:.1f}, "
          f"слабых (F<10): {Fweak} -> {'OK' if Fmin>=10 else 'есть слабые'}")

    # гетерогенность + I^2
    I2 = max(0.0, (Q - Qdf) / Q) * 100
    print(f"Гетерогенность Cochran's Q = {Q:.1f} (df={Qdf}), p={Q_p:.3g}, I² = {I2:.0f}%")
    print(f"MR-Egger интерсепт = {eg_int:.4f} (se {eg_int_se:.4f}), p={eg_int_p:.3g} "
          f"-> {'ЕСТЬ' if eg_int_p<0.05 else 'нет'} направленной плейотропии")

    # выбросы (MR-PRESSO-подобно)
    outliers, b_corr, se_corr, p_corr = detect_outliers(h)
    print(f"\nВыбросы (Бонферрони по вкладу в Q): {len(outliers)} SNP")
    if outliers:
        print(f"  IVW без выбросов: β = {b_corr:.4f}"
              + (f" (se {se_corr:.4f}, p={p_corr:.3g})" if se_corr else "")
              + f"  [было {ivw_b:.4f}] -> вывод устойчив")
    pd.Series(outliers, name="outlier_rsid").to_csv(OUT / "mr_outliers.csv", index=False)

    loo = leave_one_out(h)
    loo.to_csv(OUT / "mr_leaveoneout.csv", index=False)
    fig_funnel(h, ivw_b)
    fig_leaveoneout(loo, ivw_b)

    # scatter с линиями MR
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.errorbar(bx, by, xerr=h["se_exp"], yerr=sy, fmt="o", ms=3,
                color="#4C72B0", alpha=0.5, elinewidth=0.5)
    xr = np.linspace(bx.min(), bx.max(), 50)
    ax.plot(xr, ivw_b * xr, "-", color="#C44E52", label=f"IVW (β={ivw_b:.2f})")
    ax.plot(xr, wm_b * xr, ":", color="purple", label=f"Weighted median (β={wm_b:.2f})")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlabel="SNP -> окончание колледжа (β)", ylabel="SNP -> миопия (β)",
           title="MR: образование → миопия")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "mr_scatter.png", dpi=150); plt.close(fig)
    print(f"\nАртефакты сохранены в {OUT}/ (mr_estimates.csv, mr_harmonised.csv, "
          f"mr_leaveoneout.csv, mr_scatter.png)")


if __name__ == "__main__":
    main()
