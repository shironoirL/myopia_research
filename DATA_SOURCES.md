# Источники данных и инструментов

Что лежит в `datasets/` и `tools/`, откуда скачано и как восстановить на новой машине.
Всё — открытые данные.

| Что | Путь | Размер | В git |
|---|---|---|---|
| OLSM (Hosmer–Lemeshow MYOPIA) | `myopia.csv`, `OLSM/myopia.csv` | 40 КБ | да |
| NHANES 2007–2008 | `datasets/nhanes/*.xpt` | 22 МБ | да |
| Kinoshita RCT (ОК + 0.01% атропин) | `datasets/Orthokeratology  0.01 Atropine/` | 120 КБ | да |
| Обзоры по образу жизни (PDF) | `datasets/Physical activity sedentary behaviour sleep myopia/` | 0.6 МБ | да |
| Kaggle-изображения (только ссылки) | `datasets/Kaggle Myopia Image Dataset/`, `datasets/Myopia images/` | — | да |
| GWAS summary statistics | `datasets/gwas/*.tsv.gz` | 0.9 ГБ | **нет** |
| Референс-панель 1000 Genomes (LD) | `datasets/gwas/1kg/` | 7.4 ГБ | **нет** |
| PLINK 1.9 | `tools/` | 49 МБ | **нет** |

## Быстрое восстановление (то, чего нет в git)

```bash
poetry install                      # окружение (.venv)

mkdir -p datasets/gwas/1kg tools
B=https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics

# Экспозиция MR: college completion, Loh et al. 2018 (harmonised, ~497 МБ)
curl -L -o datasets/gwas/EA_Lee2018_GCST90029012.h.tsv.gz \
  $B/GCST90029001-GCST90030000/GCST90029012/harmonised/29892013-GCST90029012-EFO_0011015.h.tsv.gz

# Исход MR: миопия, Jiang et al. 2021 (~457 МБ)
curl -L -o datasets/gwas/myopia_GCST90044326.tsv.gz \
  $B/GCST90044001-GCST90045000/GCST90044326/GCST90044326_buildGRCh37.tsv.gz

# LD-панель 1000 Genomes phase 3, MRC-IEU (архив ~1.6 ГБ, распакованный ~7.4 ГБ)
curl -L -o datasets/gwas/1kg.v3.tgz http://fileserve.mrcieu.ac.uk/ld/1kg.v3.tgz
tar -xzf datasets/gwas/1kg.v3.tgz -C datasets/gwas/1kg && rm datasets/gwas/1kg.v3.tgz

# PLINK 1.9 (v1.90b7.2, 11 Dec 2023), Linux x86_64
curl -L -o tools/plink.zip https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip
python -m zipfile -e tools/plink.zip tools/ && chmod +x tools/plink
```

Проверка целостности: `gzip -t datasets/gwas/*.tsv.gz`; md5 сверяются с `md5sum.txt`
в тех же папках FTP. Если не нужна вся панель 1000 Genomes, достаточно `EUR.bed/.bim/.fam`
(скрипты используют только EUR); остальные популяции после распаковки можно удалить.

Хосты `ftp.ebi.ac.uk` и `fileserve.mrcieu.ac.uk` из рабочей сети блокировались —
качать через VPN или из другой сети.

## Подробно по каждому источнику

### 1. OLSM — `myopia.csv`
- Набор **MYOPIA** (подвыборка Orinda Longitudinal Study of Myopia, n = 618) из книги
  Hosmer D. W., Lemeshow S., Sturdivant R. X. *Applied Logistic Regression*, 3rd ed.
  Wiley, 2013. Цитировать книгу.
- Разделитель `;`. Паспорт данных — `DATASET_myopia_olsm.md`.
- Скрипты: `olsm_analysis.py`, `run_salf_experiment.py`.

### 2. NHANES 2007–2008 — `datasets/nhanes/`
- CDC / NCHS, цикл 2007–2008 (суффикс `_E`), формат SAS XPORT.
- Файлы: `DEMO_E` (демография, веса), `VIX_E` (зрение, авторефракция), `BMX_E`
  (антропометрия), `PAQ_E` (физ. активность), `DR1TOT_E` (питание, день 1).
- Ссылка на файл: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/<ИМЯ>.xpt`,
  описание переменных: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2007
- Скрипты: `nhanes_analysis.py`, `nhanes_survey_analysis.py`.

### 3. Kinoshita RCT — `datasets/Orthokeratology  0.01 Atropine/Dataset (Kinoshita).xlsx`
- Открытый набор данных к статье: Kinoshita N., Konno Y., Hamada N., Kanda Y.,
  Shimmura-Tomita M., Kaburaki T., Kakehashi A. *Efficacy of combined orthokeratology
  and 0.01% atropine solution for slowing axial elongation in children with myopia:
  a 2-year randomised trial.* Scientific Reports, 2020.
- TODO: указать DOI/ссылку на репозиторий данных, откуда скачан xlsx.
- Скрипты: `ortho_analysis.py`, `salf_benchmark.py`, `run_salf_experiment.py`.

### 4. Обзоры — `datasets/Physical activity sedentary behaviour sleep myopia/`
- PDF систематических обзоров и метаанализов (физ. активность, near work, экраны, сон).
  Используются как литературная база, в коде не читаются.
- Если репозиторий публичный — проверить лицензии статей (не все в открытом доступе).

### 5. Изображения — только ссылки (сами изображения не скачивались)
- https://www.kaggle.com/datasets/kellysanderson/myopia-image-dataset
- https://www.kaggle.com/datasets/kellysanderson/myopia-images

### 6. GWAS для менделевской рандомизации — `datasets/gwas/`
Обе GWAS — UK Biobank, сборка GRCh37, скачаны из GWAS Catalog FTP (размеры файлов
совпадают с FTP до байта, проверено 2026-10-17).

- **Экспозиция — GCST90029012**: Loh P.-R. et al., 2018 (PMID 29892013, BOLT-LMM),
  окончание колледжа (бинарный признак), 454 813 европейцев. Harmonised-версия (`hm_*`).
  Внимание: имя файла `EA_Lee2018_…` историческое и неверное — это **не** Lee et al. 2018.
- **Исход — GCST90044326**: Jiang L. et al., 2021 (PMID 34737426, fastGWA-GLMM),
  UKB поле 6147_1 (очки/линзы из-за близорукости), 36 623 случая / 419 031 контроль.
- **LD-панель — 1kg.v3 (MRC-IEU)**: 1000 Genomes phase 3, PLINK-формат, по суперпопуляциям
  AFR/AMR/EAS/EUR/SAS. Используется EUR (503 человека).
- Скрипты (порядок запуска): `clump_with_plink.py` → `mr_analysis.py` → `mr_sensitivity.py`.

### 7. PLINK 1.9 — `tools/plink`
- https://www.cog-genomics.org/plink/ (v1.90b7.2, 64-bit, сборка 11 Dec 2023).
- Используется для LD-клампинга в `clump_with_plink.py` и `mr_sensitivity.py`.
