import numpy as np


# 1. МНК-поліном (статистичне навчання з Лр2)
def lsm(y, degree, horizon):
    t = np.arange(len(y)) / len(y)
    c = np.polyfit(t, y, degree)
    t_all = np.arange(len(y) + horizon) / len(y)
    p = np.polyval(c, t_all)
    return p[len(y):], p[:len(y)]


def optimize_degree(y, degrees=(1, 2, 3, 4)):
    n_fit = int(len(y) * 0.75)
    errs = [np.sqrt(np.mean((lsm(y[:n_fit], d, len(y) - n_fit)[0] - y[n_fit:]) ** 2)) for d in degrees]
    return degrees[int(np.argmin(errs))]


# 2. Рекурентний alfa-beta-gamma фільтр зі зростаючою пам'яттю (Лр3)
def abg(y, horizon):
    x = v = a = 0.0
    est = np.zeros(len(y))
    for k in range(len(y)):
        x_p, v_p = x + v + 0.5 * a, v + a
        r = y[k] - x_p
        d = (k + 1) * (k + 2) * (k + 3)
        x = x_p + 3 * (3 * k * k + 3 * k + 2) / d * r
        v = v_p + 18 * (2 * k + 1) / d * r
        a = a + 60 / d * r
        est[k] = x
    h = np.arange(1, horizon + 1)
    return x + v * h + 0.5 * a * h ** 2, est
