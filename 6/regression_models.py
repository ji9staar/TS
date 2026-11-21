import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


# 1. Ознаки: поліном за часом + гармоніки Фур'є для сезонності
def features(t, n, degree, season, harmonics):
    cols = [(t / n) ** d for d in range(1, degree + 1)]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * t / season), np.cos(2 * np.pi * k * t / season)]
    return np.column_stack(cols)


class TrendFourier:
    def __init__(self, degree, season=None, harmonics=0, alpha=0.0):
        self.degree, self.season, self.harmonics, self.alpha = degree, season, harmonics, alpha

    def fit(self, y):
        self.n = len(y)
        X = features(np.arange(self.n), self.n, self.degree, self.season, self.harmonics)
        self.model = (Ridge(alpha=self.alpha) if self.alpha else LinearRegression()).fit(X, y)
        return self

    def predict(self, start, stop):
        return self.model.predict(features(np.arange(start, stop), self.n, self.degree, self.season, self.harmonics))


# 2. Багатофакторна динамічна регресія: попереднє значення + сезонний лаг + день сезону
class LagRegression:
    def __init__(self, season=None):
        self.season = season

    def _x(self, hist, t):
        x = [hist[t - 1], t]
        if self.season:
            x += [hist[t - self.season]] + [float(t % self.season == s) for s in range(1, self.season)]
        return x

    def fit(self, y):
        self.y = list(np.asarray(y, dtype=float))
        start = self.season or 1
        X = [self._x(self.y, t) for t in range(start, len(self.y))]
        self.model = LinearRegression().fit(X, self.y[start:])
        self.fitted = np.array(self.y[:start] + list(self.model.predict(X)))
        return self

    def predict(self, start, stop):
        if stop <= len(self.y):
            return self.fitted[start:stop]
        hist = list(self.y)
        for t in range(len(self.y), stop):
            hist.append(self.model.predict([self._x(hist, t)])[0])
        return np.array(hist[start:stop])


def candidates(season):
    models = {}
    for d in (1, 2, 3):
        models[f'Регресія: поліном {d} ст.'] = TrendFourier(d)
        if season:
            models[f'Регресія: поліном {d} ст. + Фур\'є (2 гармоніки)'] = TrendFourier(d, season, 2)
            models[f'Регресія: поліном {d} ст. + Фур\'є (4 гармоніки), Ridge'] = TrendFourier(d, season, 4, 1.0)
    models['Регресія: динамічна (лаги)'] = LagRegression(season)
    return models


# 3. Вибір регресійної моделі за похибкою прогнозу на валідаційній частині
def select(train, season):
    n_fit = int(len(train) * 0.75)
    errors = {}
    for nm, m in candidates(season).items():
        try:
            p = m.fit(train[:n_fit]).predict(n_fit, len(train))
            errors[nm] = np.sqrt(np.mean((p - train[n_fit:]) ** 2))
        except Exception:
            continue
    best = min(errors, key=errors.get)
    return best, candidates(season)[best].fit(train), errors
