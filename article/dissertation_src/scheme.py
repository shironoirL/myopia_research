import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 11); ax.set_ylim(-0.25, 6.2); ax.axis("off")
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333", lw=1))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=9.5, wrap=True)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
D, M, R = "#DCE8F5", "#FDEBD3", "#E2F0D9"
ax.text(1.45, 5.95, "Открытые данные", ha="center", fontsize=11, weight="bold")
ax.text(5.5, 5.95, "Методы", ha="center", fontsize=11, weight="bold")
ax.text(9.45, 5.95, "Результат", ha="center", fontsize=11, weight="bold")
rows = [
 ("OLSM, n = 618\n(проспективная когорта)", "Классификация риска\nLogReg / RF / GB, CV 5×10,\nпорог, OR, абляция", "Модель скрининга\nи факторы риска (гл. 4)"),
 ("NHANES 2007–2008\nn = 990 (12–19 лет)", "Логистическая регрессия,\nOR на 1 SD", "Внешняя проверка\nфакторов (гл. 4)"),
 ("РКИ Kinoshita\n73 ребёнка, 146 глаз", "t / Манна–Уитни,\nANCOVA", "Эффективность\nОК + атропин (гл. 5)"),
 ("GWAS Catalog +\n1000 Genomes EUR", "Менделевская рандомизация\nIVW / Egger / WM, клампинг", "Причинный эффект\nобразования (гл. 5)"),
]
for i, (a, b, c) in enumerate(rows):
    y = 4.75 - i * 1.25
    box(0.2, y, 2.5, 0.95, a, D); box(4.0, y, 3.0, 0.95, b, M); box(8.2, y, 2.5, 0.95, c, R)
    arr(2.75, y + .47, 3.95, y + .47); arr(7.05, y + .47, 8.15, y + .47)
box(4.0, -0.05, 3.0, 0.62, "Гибридная схема SALF (гл. 3)\nриск-чувствительный прогноз", "#F8D7DA")
box(0.2, -0.05, 2.5, 0.62, "OLSM + Kinoshita\n(повторное использование)", D); arr(2.75, 0.26, 3.95, 0.26)
arr(7.05, 0.26, 8.15, 0.26); ax.text(9.45, 0.26, "Снижение пропуска\nбыстрых прогрессоров", ha="center", va="center", fontsize=9.5)
fig.tight_layout(); fig.savefig("pipeline.png", dpi=150)
