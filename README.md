# myopia_research

Код и материалы диссертации по моделированию, прогнозированию и оценке профилактики
миопии у детей.

## Запуск на новой машине

```bash
git clone https://github.com/shironoirL/myopia_research.git
cd myopia_research
poetry install
```

Мелкие датасеты лежат в репозитории. Крупные (GWAS, панель 1000 Genomes, ~8 ГБ) и PLINK
в git не входят — команды для их загрузки и описание всех источников в
[DATA_SOURCES.md](DATA_SOURCES.md). Без них не запустятся только MR-скрипты
(`clump_with_plink.py`, `mr_analysis.py`, `mr_sensitivity.py`).
