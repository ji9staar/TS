import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


# Завантаження курсу
def load_nbu(filename):
    d = pd.read_excel(filename)
    ts = pd.Series(d['Офіційний курс гривні, грн'].values,
                   index=pd.to_datetime(d['Дата'], dayfirst=True), name='USD')
    print(f"Офіційний курс USD (НБУ): {ts.index[0].date()} - {ts.index[-1].date()}, вимірів: {len(ts)}")
    return ts


# Характеристики ряду
def ts_properties(x):
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    return {
        'середнє': x.mean(),
        'СКВ': x.std(),
        'сер. денна зміна': d.mean(),
        'СКВ денної зміни': d.std(),
        'частка днів без змін': np.mean(d == 0),
        'асиметрія змін': stats.skew(d),
        'автокореляція змін (лаг 1)': acf(d, nlags=1)[1],
        'автокореляція змін (лаг 7)': acf(d, nlags=7)[7],
        'ADF p-value': adfuller(x)[1],
        'KPSS p-value': kpss(x, nlags='auto')[1],
    }


def print_properties(title, props):
    print(f"\n--- {title} ---")
    for k, v in props.items():
        print(f"{k:<28} {v:10.4f}")


# Розклад на тренд, сезонність і шум
def stl_decomposition(ts):
    res = STL(ts, period=7, robust=True).fit()
    var_r = np.var(res.resid)
    strength_trend = max(0, 1 - var_r / np.var(res.trend + res.resid))
    strength_season = max(0, 1 - var_r / np.var(res.seasonal + res.resid))
    print("\n--- STL-декомпозиція (період 7 днів) ---")
    print(f"зміна тренду за період = {res.trend.iloc[-1] - res.trend.iloc[0]:.3f} грн")
    print(f"сила тренду = {strength_trend:.3f}, сила тижневої сезонності = {strength_season:.3f}")
    weekday = res.seasonal.groupby(res.seasonal.index.dayofweek).mean()
    names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд']
    print("сезонна складова за днями тижня: " + ', '.join(f"{names[i]} {v:+.4f}" for i, v in weekday.items()))
    fig = res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('STL-декомпозиція офіційного курсу USD')
    fig.tight_layout()
    return res


# Кореляція курсу
def correlation_analysis(ts):
    x = ts.values
    print("\n--- Кореляція курсу з власними лагами ---")
    for lag in (1, 7, 14, 30):
        print(f"лаг {lag:2d} днів: r = {np.corrcoef(x[:-lag], x[lag:])[0, 1]:.4f}")
    d = np.diff(x)
    days = ts.index[1:].dayofweek
    print("середня зміна курсу за днями тижня:")
    print(pd.Series(d).groupby(days).mean().round(4).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(d, lags=21, ax=ax[0], title='ACF денних змін курсу')
    plot_pacf(d, lags=21, ax=ax[1], title='PACF денних змін курсу')
    plt.tight_layout()


# Кластеризація тижнів
def week_clustering(ts, n_clusters=3):
    weeks = [w for _, w in ts.resample('W') if len(w) == 7]
    feats = np.array([[w.iloc[-1] - w.iloc[0], np.std(np.diff(w.values)), w.mean()] for w in weeks])
    z = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    links = linkage(z[:, :2], method='ward')
    labels = fcluster(links, n_clusters, criterion='maxclust')
    print("\n--- Ієрархічна кластеризація тижнів (метод Уорда) ---")
    for c in range(1, n_clusters + 1):
        m = labels == c
        print(f"кластер {c}: тижнів = {m.sum():2d}   сер. зміна за тиждень = {feats[m, 0].mean():+.3f} грн   "
              f"волатильність = {feats[m, 1].mean():.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    dendrogram(links, ax=ax[0], no_labels=True, color_threshold=links[-n_clusters + 1, 2])
    ax[0].set_title('Дендрограма тижнів')
    colors = ['tab:green', 'tab:orange', 'tab:red', 'tab:blue']
    for w, c in zip(weeks, labels):
        ax[1].plot(w.index, w.values, color=colors[c - 1], linewidth=2)
    ax[1].set_title('Тижні курсу за кластерами')
    ax[1].grid(True)
    plt.tight_layout()
    return labels


# Графік курсу
def plot_series(ts):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].plot(ts.index, ts.values, label='Офіційний курс USD')
    ax[0].plot(ts.index, ts.rolling(30).mean(), label='Ковзне середнє 30 днів')
    ax[0].set_title('Офіційний курс USD (НБУ)')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(ts.index, ts.diff().rolling(14).std(), color='tab:red')
    ax[1].set_title('Ковзна волатильність денних змін (14 днів)')
    ax[1].grid(True)
    plt.tight_layout()
