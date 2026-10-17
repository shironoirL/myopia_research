"""
Гибридная вычислительная схема с SALF — реализация метода из статьи
«Математическая модель прогноза прогрессирования миопии на основе
гибридной вычислительной схемы» (Халилов В.Р., Шулаев А.В.).

Структура модели (аддитивная композиция двух ветвей с маршрутизатором):

    y_hat(x) = f_lin(x) + g(x) * f_ens(x)

    f_lin  — линейная базовая ветвь (глобальный тренд);
    f_ens  — нелинейная корректирующая ветвь, обучается на невязках линейной;
    g(x)   — логистический маршрутизатор sigmoid(gamma·x + c) в (0,1),
             усиливает нелинейную ветвь там, где линейной недостаточно.

Функция потерь SALF — сглаженная асимметричная (квантильно-Хьюберова):
    e = y - y_hat
    a(e) = tau при e >= 0 (недооценка прогрессирования штрафуется сильнее),
           1 - tau при e < 0
    SALF = a(e) * Huber_delta(e)
непрерывна и гладкая в e=0 и e=±delta, выпукла по e.

Регуляризация: L2 на линейную ветвь + штраф косинусного сходства выходов
ветвей (декорреляция, снижает дублирование).

Поэтапное обучение: (1) линейная ветвь по SALF -> (2) ансамбль на невязках
-> (3) маршрутизатор по SALF + декорреляция -> (4) совместная донастройка.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin


# ---------- SALF ----------
def huber(e, delta):
    a = np.abs(e)
    return np.where(a <= delta, 0.5 * e ** 2, delta * (a - 0.5 * delta))


def huber_grad(e, delta):
    return np.where(np.abs(e) <= delta, e, delta * np.sign(e))


def salf(y, yhat, tau=0.7, delta=1.0):
    e = y - yhat
    w = np.where(e >= 0, tau, 1 - tau)
    return np.mean(w * huber(e, delta))


def salf_grad_yhat(y, yhat, tau=0.7, delta=1.0):
    """d SALF / d yhat (усреднённый по выборке)."""
    e = y - yhat
    w = np.where(e >= 0, tau, 1 - tau)
    return -(w * huber_grad(e, delta)) / len(y)


def cos_sim(a, b, eps=1e-8):
    ac, bc = a - a.mean(), b - b.mean()
    return float(ac @ bc / (np.linalg.norm(ac) * np.linalg.norm(bc) + eps))


class SALFHybrid(BaseEstimator, RegressorMixin):
    def __init__(self, tau=0.7, delta=1.0, l2=1.0, lam_decor=0.5,
                 gbm_estimators=200, gbm_depth=2, random_state=42):
        self.tau = tau
        self.delta = delta
        self.l2 = l2
        self.lam_decor = lam_decor
        self.gbm_estimators = gbm_estimators
        self.gbm_depth = gbm_depth
        self.random_state = random_state

    # стандартизация признаков
    def _scale(self, X, fit=False):
        if fit:
            self.mu_ = X.mean(0)
            self.sd_ = X.std(0) + 1e-8
        return (X - self.mu_) / self.sd_

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xs = self._scale(X, fit=True)
        n, p = Xs.shape

        # --- этап 1: линейная ветвь по SALF ---
        def lin_obj(theta):
            w, b = theta[:p], theta[p]
            f = Xs @ w + b
            loss = salf(y, f, self.tau, self.delta) + self.l2 * (w @ w)
            g = salf_grad_yhat(y, f, self.tau, self.delta)
            gw = Xs.T @ g + 2 * self.l2 * w
            gb = g.sum()
            return loss, np.append(gw, gb)
        r1 = minimize(lin_obj, np.zeros(p + 1), jac=True, method="L-BFGS-B")
        self.w_, self.b_ = r1.x[:p], r1.x[p]
        f_lin = Xs @ self.w_ + self.b_

        # --- этап 2: нелинейная ветвь на невязках ---
        resid = y - f_lin
        self.ens_ = GradientBoostingRegressor(
            n_estimators=self.gbm_estimators, max_depth=self.gbm_depth,
            learning_rate=0.05, random_state=self.random_state).fit(Xs, resid)
        f_ens = self.ens_.predict(Xs)

        # --- этап 3: маршрутизатор по SALF + декорреляция ---
        def route_obj(theta):
            gamma, c = theta[:p], theta[p]
            g = 1 / (1 + np.exp(-(Xs @ gamma + c)))
            yhat = f_lin + g * f_ens
            loss = salf(y, yhat, self.tau, self.delta)
            loss += self.lam_decor * cos_sim(f_lin, g * f_ens) ** 2
            return loss
        r3 = minimize(route_obj, np.zeros(p + 1), method="Powell",
                      options={"maxiter": 2000})
        self.gamma_, self.c_ = r3.x[:p], r3.x[p]

        # --- этап 4: совместная донастройка линейной ветви и маршрутизатора ---
        def joint_obj(theta):
            w, b, gamma, c = theta[:p], theta[p], theta[p + 1:2 * p + 1], theta[2 * p + 1]
            fl = Xs @ w + b
            g = 1 / (1 + np.exp(-(Xs @ gamma + c)))
            yhat = fl + g * f_ens
            loss = salf(y, yhat, self.tau, self.delta) + self.l2 * (w @ w)
            loss += self.lam_decor * cos_sim(fl, g * f_ens) ** 2
            return loss
        init = np.concatenate([self.w_, [self.b_], self.gamma_, [self.c_]])
        r4 = minimize(joint_obj, init, method="Powell", options={"maxiter": 3000})
        x = r4.x
        self.w_, self.b_ = x[:p], x[p]
        self.gamma_, self.c_ = x[p + 1:2 * p + 1], x[2 * p + 1]
        return self

    def predict(self, X):
        Xs = self._scale(np.asarray(X, float))
        f_lin = Xs @ self.w_ + self.b_
        f_ens = self.ens_.predict(Xs)
        g = 1 / (1 + np.exp(-(Xs @ self.gamma_ + self.c_)))
        return f_lin + g * f_ens
