import itertools
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


# 1. Просте ковзне середнє (SMA)
def sma(y, w):
    return np.convolve(y, np.ones(w) / w, mode='valid')


# 2. Зважене ковзне середнє (WMA): більші ваги останнім вимірам
def wma(y, w):
    weights = np.arange(1, w + 1) / (w * (w + 1) / 2)
    return np.convolve(y, weights[::-1], mode='valid')


# 3. Подвійне ковзне середнє Брауна (лінійний тренд)
def dma_forecast(y, w, horizon):
    m1 = sma(y, w)
    m2 = sma(m1, w)
    level = 2 * m1[-1] - m2[-1]
    trend = 2 / (w - 1) * (m1[-1] - m2[-1])
    return level + trend * np.arange(1, horizon + 1)


def forecast(method, y, w, horizon):
    if method == 'SMA':
        return np.full(horizon, sma(y, w)[-1])
    if method == 'WMA':
        return np.full(horizon, wma(y, w)[-1])
    return dma_forecast(y, w, horizon)


def fitted(method, y, w):
    f = np.full(len(y), np.nan)
    if method == 'SMA':
        f[w - 1:] = sma(y, w)
    elif method == 'WMA':
        f[w - 1:] = wma(y, w)
    else:
        m1 = sma(y, w)
        m2 = sma(m1, w)
        f[2 * w - 2:] = 2 * m1[w - 1:] - m2
    return f


# 4. Оптимізація вікна ковзною перевіркою (walk-forward) на інтервалі спостереження
def optimize_window(method, y, origins=(0.6, 0.7, 0.8), h_share=0.2):
    h = max(2, int(len(y) * h_share))
    best_w, best_err = None, np.inf
    for w in range(2, int(len(y) * 0.3)):
        errs = []
        for o in origins:
            n0 = int(len(y) * o)
            if n0 < 2 * w + 1 or n0 + h > len(y):
                continue
            errs.append(np.sqrt(np.mean((forecast(method, y[:n0], w, h) - y[n0:n0 + h]) ** 2)))
        if errs and np.mean(errs) < best_err:
            best_w, best_err = w, np.mean(errs)
    return best_w, best_err


# 5. ARIMA / SARIMA з вибором порядку за критерієм BIC
def sarima_fit(y, order, seasonal=(0, 0, 0, 0)):
    trend = 'c' if order[1] + seasonal[1] == 0 else ('t' if order[1] + seasonal[1] == 1 else 'n')
    return SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend).fit(disp=False)


def optimize_arima(y, season=None):
    best, best_bic = None, np.inf
    seasonal_opts = [(0, 0, 0, 0)]
    if season:
        seasonal_opts += [(1, 0, 0, season), (0, 0, 1, season), (1, 0, 1, season)]
    for order in itertools.product(range(3), range(3), range(3)):
        for so in seasonal_opts:
            try:
                bic = sarima_fit(y, order, so).bic
            except Exception:
                continue
            if bic < best_bic:
                best, best_bic = (order, so), bic
    return best, best_bic
