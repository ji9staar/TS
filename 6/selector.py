import numpy as np
from statsmodels.tsa.seasonal import STL
from regression_models import TrendFourier


# 1. Сила сезонності за STL-декомпозицією
def seasonal_strength(y, season):
    if not season or len(y) < 2 * season:
        return 0.0
    res = STL(y, period=season, robust=True).fit()
    return max(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))


# 2. Показник детермінованості: відношення СКВ залишків детермінованої моделі до СКВ перших різниць
#    тренд + білий шум: r ~ 1/sqrt(2) ~ 0.71;  випадкове блукання: r >> 1
def determinism_ratio(y, season, fs):
    use_season = season if fs > 0.5 else None
    m = TrendFourier(3, use_season, 2 if use_season else 0).fit(y)
    resid = y - m.predict(0, len(y))
    lag = use_season or 1
    diff = y[lag:] - y[:-lag]
    return np.std(resid) / np.std(diff)


# 3. Алгоритм вибору типу згладжування
def recommend(y, season):
    fs = seasonal_strength(y, season)
    r = determinism_ratio(y, season, fs)
    family = 'регресія' if r < 1.0 else 'експоненційне згладжування'
    print(f"сила сезонності Fs = {fs:.3f}, показник детермінованості r = {r:.3f}")
    print(f"Алгоритм вибору: r {'<' if r < 1 else '>='} 1 -> {family}")
    return family, fs, r
