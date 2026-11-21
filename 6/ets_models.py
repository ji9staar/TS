import itertools
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# 1. Перелік конфігурацій експоненційного згладжування (тип тренду x тип сезонності)
def configs(season):
    trends = [(None, False), ('add', False), ('add', True)]
    seasonals = [None] + (['add', 'mul'] if season else [])
    return list(itertools.product(trends, seasonals))


def name(cfg):
    (trend, damped), seasonal = cfg
    t = {None: 'без тренду', 'add': 'адитивний тренд'}[trend] + (' (згасаючий)' if damped else '')
    s = {None: 'без сезонності', 'add': 'адитивна сезонність', 'mul': 'мультиплікативна сезонність'}[seasonal]
    return f'ETS: {t}, {s}'


def fit(y, cfg, season):
    (trend, damped), seasonal = cfg
    return ExponentialSmoothing(y, trend=trend, damped_trend=damped, seasonal=seasonal,
                                seasonal_periods=season if seasonal else None,
                                initialization_method='estimated').fit()


# 2. Вибір конфігурації за похибкою прогнозу на валідаційній частині
def select(train, season):
    n_fit = int(len(train) * 0.75)
    results = {}
    for cfg in configs(season):
        if cfg[1] and n_fit < 2 * season:
            continue
        try:
            fc = fit(train[:n_fit], cfg, season).forecast(len(train) - n_fit)
            results[cfg] = np.sqrt(np.mean((fc - train[n_fit:]) ** 2))
        except Exception:
            continue
    best = min(results, key=results.get)
    return best, results
