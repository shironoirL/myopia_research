"""
Строгий LD-клампинг инструментов через PLINK + референс-панель 1000G EUR.
Запускать, КОГДА в datasets/gwas/ появится панель EUR.bed/EUR.bim/EUR.fam
(скачать через VPN: http://fileserve.mrcieu.ac.uk/ld/1kg.v3.tgz).

Заменяет дистанционный прунинг в mr_analysis.py на настоящий клампинг
(PLINK --clump, окно 10 Mb, r2<0.001 — стандарт TwoSampleMR).

Запуск: python clump_with_plink.py
Результат: results/ea_clumped_instruments.csv (rsid инструментов после клампинга).
"""
from __future__ import annotations
import subprocess, tempfile, os
from pathlib import Path
import pandas as pd
from mr_analysis import EA, PVAL_THR

GW = Path("datasets/gwas")
PLINK = "tools/plink"
PANEL = GW / "1kg" / "EUR"   # EUR.bed/.bim/.fam из 1kg.v3 (MRC-IEU)
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def main():
    if not (PANEL.with_suffix(".bed")).exists():
        raise SystemExit(f"Нет панели {PANEL}.bed — скачай 1kg.v3.tgz (EUR.*) в {GW}/")

    # 1) собрать значимые SNP exposure (rsid, p) для PLINK
    cols = ["hm_rsid", "p_value"]
    sig = []
    for ch in pd.read_csv(EA, sep="\t", usecols=cols, chunksize=500_000, low_memory=False):
        ch["p_value"] = pd.to_numeric(ch["p_value"], errors="coerce")
        sig.append(ch[ch["p_value"] < PVAL_THR])
    sig = pd.concat(sig).dropna().rename(columns={"hm_rsid": "SNP", "p_value": "P"})
    print(f"значимых SNP (p<{PVAL_THR}): {len(sig)}")

    with tempfile.TemporaryDirectory() as td:
        assoc = os.path.join(td, "ea.assoc")
        sig.to_csv(assoc, sep="\t", index=False)
        # 2) PLINK clump: окно 10000 кб, r2<0.001
        cmd = [PLINK, "--bfile", str(PANEL), "--clump", assoc,
               "--clump-p1", "5e-8", "--clump-r2", "0.001", "--clump-kb", "10000",
               "--clump-snp-field", "SNP", "--clump-field", "P",
               "--out", os.path.join(td, "clp")]
        subprocess.run(cmd, check=True)
        clumped = pd.read_csv(os.path.join(td, "clp.clumped"), sep=r"\s+")

    inst = clumped[["SNP"]].rename(columns={"SNP": "rsid"})
    inst.to_csv(OUT / "ea_clumped_instruments.csv", index=False)
    print(f"После строгого клампинга: {len(inst)} независимых инструментов")
    print(f"Сохранено в {OUT}/ea_clumped_instruments.csv — далее mr_analysis.py "
          f"может брать инструменты отсюда вместо дистанционного прунинга.")


if __name__ == "__main__":
    main()
