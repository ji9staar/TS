import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from real_ts import ts_properties


# Параметри моделі курсу
def fit_model(ts):
    d = np.diff(ts.values)
    days = ts.index[1:].dayofweek
    work = days < 5
    dw, wd = d[work], days[work]
    mu = np.array([dw[wd == k].mean() for k in range(5)])
    e = dw - mu[wd]
    phi = np.corrcoef(e[:-1], e[1:])[0, 1]
    sigma = np.std(e[1:] - phi * e[:-1])
    params = {'x0': ts.values[0], 'mu': mu, 'phi': phi, 'sigma': sigma}
    names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт']
    print("\n--- Параметри синтетичної моделі ---")
    print("середня зміна за днями тижня: " + ', '.join(f"{n} {m:+.4f}" for n, m in zip(names, mu)))
    print(f"відхилення змін: e(t) = {phi:.4f}*e(t-1) + N(0, {sigma:.4f})")
    return params


# Генерація штучного курсу
def generate(params, index, rng):
    days = index[1:].dayofweek
    d = np.zeros(len(days))
    e = 0.0
    for i, k in enumerate(days):
        if k < 5:
            e = params['phi'] * e + rng.normal(0, params['sigma'])
            d[i] = params['mu'][k] + e
    return pd.Series(params['x0'] + np.concatenate([[0], np.cumsum(d)]), index=index)


# Порівняння штучних рядів зі справжнім
def compare(ts, params, n_sim=100):
    rng = np.random.default_rng(10)
    sims = [generate(params, ts.index, rng) for _ in range(n_sim)]
    real = ts_properties(ts.values)
    model = [ts_properties(s.values) for s in sims]
    print(f"\n--- Адекватність моделі: реальні дані vs {n_sim} синтетичних реалізацій ---")
    print(f"{'показник':<28}{'реальні':>10}{'модель':>10}{'СКВ':>9}  в межах 2 СКВ?")
    ok = 0
    for k in real:
        v = np.array([m[k] for m in model])
        inside = abs(real[k] - v.mean()) <= 2 * v.std() + 1e-9
        ok += inside
        print(f"{k:<28}{real[k]:10.4f}{v.mean():10.4f}{v.std():9.4f}  {'так' if inside else 'ні'}")
    print(f"збіг показників: {ok} з {len(real)}")
    d_real = np.diff(ts.values)
    d_sim = np.concatenate([np.diff(s.values) for s in sims])
    ks = stats.ks_2samp(d_real[d_real != 0], d_sim[d_sim != 0])
    print(f"КС-тест розподілу ненульових змін: p-value = {ks.pvalue:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts.values, color='black', linewidth=2, label='Реальний курс')
    for i in range(3):
        plt.plot(sims[i].index, sims[i].values, alpha=0.7, label=f'Синтетична реалізація {i + 1}')
    plt.title('Реальний та синтетичні ряди офіційного курсу USD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
