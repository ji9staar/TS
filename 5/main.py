import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import r2_score
import window_methods as wm
import learning_methods as lm

warnings.filterwarnings('ignore')
HORIZONS = [0.5, 1.0, 1.5, 2.0]


# 1. Вхідні дані: синтетична модель варіанту 10 та офіційний курс USD (НБУ)
def synthetic_cubic(n=1500):
    np.random.seed(42)
    t = np.linspace(0, 15, n)
    return 0.5 * t ** 3 - 2 * t ** 2 + 3 * t + 15 + np.random.normal(0, 10, n)


def nbu_usd():
    d = pd.read_excel('data/usd_2023-2024_1.xlsx')
    return d['Офіційний курс гривні, грн'].values.astype(float)


# 2. Дослідження властивостей
def explore(y, title):
    d = np.diff(y)
    print(f"\n--- Властивості: {title} ---")
    print(f"вимірів = {len(y)}, середнє = {y.mean():.3f}, СКВ = {y.std():.3f}")
    print(f"ADF p: ряд = {adfuller(y)[1]:.4f}, 1-а різниця = {adfuller(d)[1]:.4f}, 2-а різниця = {adfuller(np.diff(d))[1]:.4f}")
    print(f"KPSS p (ряд) = {kpss(y, nlags='auto')[1]:.4f}")
    print(f"автокореляція 1-ї різниці: лаг 1 = {np.corrcoef(d[:-1], d[1:])[0, 1]:.3f}, "
          f"лаг 7 = {np.corrcoef(d[:-7], d[7:])[0, 1]:.3f}")


def metrics(test, fc, n_obs):
    res = []
    for h in HORIZONS:
        k = int(h * n_obs)
        e = fc[:k] - test[:k]
        res.append((np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e / test[:k])) * 100))
    return res


# 3. Обробка одного ряду
# 4. Ковзна перевірка короткострокового прогнозу по всьому ряду (оперативне використання)
def rolling_check(y, n_obs, windows, base_w, order, so, degree, step, h):
    names = [f'SMA w={base_w} (базова)', 'SMA опт.', 'WMA опт.', 'DMA опт.', 'ARIMA(1,1,0) (базова)', 'ARIMA опт.',
             'МНК', 'alfa-beta-gamma']
    errs = {n: [] for n in names}
    for o in range(n_obs, len(y) - h, step):
        hist, real = y[:o], y[o:o + h]
        fc = [wm.forecast('SMA', hist, base_w, h), wm.forecast('SMA', hist, windows['SMA'], h),
              wm.forecast('WMA', hist, windows['WMA'], h), wm.forecast('DMA', hist, windows['DMA'], h),
              wm.sarima_fit(hist, (1, 1, 0)).forecast(h), wm.sarima_fit(hist, order, so).forecast(h),
              lm.lsm(hist, degree, h)[0], lm.abg(hist, h)[0]]
        for n, f in zip(names, fc):
            errs[n].append(np.sqrt(np.mean((np.asarray(f) - real) ** 2)))
    print(f"\n--- Ковзна перевірка: прогноз на {h} вимірів з {len(errs[names[0]])} точок старту ---")
    for n in sorted(names, key=lambda k: np.mean(errs[k])):
        print(f"{n:<26} середня RMSE = {np.mean(errs[n]):9.4f}")


def process(title, y, n_obs, season=None, base_w=7, step=1, h_short=7):
    print(f"\n==================== {title} ====================")
    train, test = y[:n_obs], y[n_obs:]
    H = len(test)
    print(f"інтервал спостереження = {n_obs}, екстраполяція до {H} вимірів (2.0 інтервалу)")
    explore(train, title)

    print("\n--- Оптимізація параметрів (R&D) ---")
    windows = {}
    for m in ('SMA', 'WMA', 'DMA'):
        windows[m], err = wm.optimize_window(m, train)
        print(f"{m}: оптимальне вікно = {windows[m]} (RMSE walk-forward = {err:.3f})")
    (order, so), bic = wm.optimize_arima(train, season)
    print(f"ARIMA: оптимальна модель {order}x{so}, BIC = {bic:.2f}")
    degree = lm.optimize_degree(train)
    print(f"МНК: оптимальний степінь полінома = {degree}")

    models = {}
    models[f'SMA w={base_w} (базова)'] = (wm.forecast('SMA', train, base_w, H), wm.fitted('SMA', train, base_w))
    for m in ('SMA', 'WMA', 'DMA'):
        models[f'{m} w={windows[m]}'] = (wm.forecast(m, train, windows[m], H), wm.fitted(m, train, windows[m]))
    base = wm.sarima_fit(train, (1, 1, 0))
    models['ARIMA(1,1,0) (базова)'] = (base.forecast(H), base.fittedvalues)
    best = wm.sarima_fit(train, order, so)
    name = f'ARIMA{order}' + (f'x{so[:3]}s{so[3]}' if so[3] else '')
    models[name] = (best.forecast(H), best.fittedvalues)
    models[f'МНК, степінь {degree}'] = lm.lsm(train, degree, H)
    models['alfa-beta-gamma'] = lm.abg(train, H)

    print("\n--- KPI: R^2 на інтервалі спостереження, RMSE за горизонтами, MAPE на 2.0 ---")
    print(f"{'модель':<30}{'R^2':>8}" + ''.join(f"{'RMSE ' + str(h):>11}" for h in HORIZONS) + f"{'MAPE 2.0':>11}")
    table = {}
    for nm, (fc, fit) in models.items():
        fit = np.asarray(fit, dtype=float)
        mask = ~np.isnan(fit)
        mask[:3] = False
        r2 = r2_score(train[mask], fit[mask])
        m = metrics(test, np.asarray(fc), n_obs)
        table[nm] = [v[0] for v in m]
        print(f"{nm:<30}{r2:8.4f}" + ''.join(f"{v[0]:11.3f}" for v in m) + f"{m[-1][1]:10.2f}%")
    for i, h in enumerate(HORIZONS):
        print(f"найкраща модель на {h} інтервалу: {min(table, key=lambda k: table[k][i])}")

    rolling_check(y, n_obs, windows, base_w, order, so, degree, step, h_short)

    t = np.arange(len(y))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[3, 2]))
    ax[0].plot(t, y, color='lightgray', label='Дані')
    ax[0].axvline(n_obs, color='black', linestyle='--')
    for nm, (fc, _) in models.items():
        if 'базова' not in nm:
            ax[0].plot(t[n_obs:], fc, linewidth=1.8, label=nm)
    ax[0].set_title(f'{title}: екстраполяція')
    ax[0].legend(fontsize=8)
    ax[0].grid(True)
    names = list(table)
    im = ax[1].imshow(np.log10(np.array([table[n] for n in names])), cmap='Greys', aspect='auto')
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([f'{h}' for h in HORIZONS])
    for i, n in enumerate(names):
        for j in range(4):
            ax[1].text(j, i, f'{table[n][j]:.2f}', ha='center', va='center', color='red', fontsize=8)
    ax[1].set_title('RMSE за горизонтами (інтервали спостереження)')
    plt.tight_layout()


usd = nbu_usd()
plt.figure(figsize=(10, 3.5))
plot_acf(np.diff(usd[:53]), lags=21, ax=plt.gca(), title='ACF денних змін офіційного курсу (інтервал спостереження)')
plt.tight_layout()

process('Синтетичні дані варіанту 10 (кубічний тренд + нормальний шум)', synthetic_cubic(), 500, step=50, h_short=50)
process('Офіційний курс USD (НБУ)', usd, 53, season=7)
plt.show()
