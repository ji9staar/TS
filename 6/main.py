import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import ets_models as em
import regression_models as rm
import selector as sel

warnings.filterwarnings('ignore')


# 1. Вхідні дані: кубічна модель варіанту 10, офіційний курс USD (НБУ), продажі Superstore (Data_Set_10)
def synthetic_cubic(n=900):
    np.random.seed(42)
    t = np.linspace(0, 15, n)
    return 0.5 * t ** 3 - 2 * t ** 2 + 3 * t + 15 + np.random.normal(0, 10, n)


def nbu_usd():
    return pd.read_excel('data/usd_2023-2024_1.xlsx')['Офіційний курс гривні, грн'].values.astype(float)


def superstore_sales():
    d = pd.read_excel('data/Data_Set_10.xls')
    return d.set_index('Order Date').resample('MS')['Sales'].sum().values


def kpi(train, fit, test, fc):
    m = ~np.isnan(fit)
    return {'R2': r2_score(train[m], fit[m]), 'RMSE': np.sqrt(np.mean((fc - test) ** 2)),
            'MAE': mean_absolute_error(test, fc), 'MAPE': np.mean(np.abs((fc - test) / test)) * 100}


# 2. Обробка одного ряду обома підходами
def process(title, y, season, test_share=1 / 3):
    print(f"\n==================== {title} ====================")
    n_obs = int(len(y) * (1 - test_share))
    train, test = y[:n_obs], y[n_obs:]
    print(f"вимірів = {len(y)}, навчання = {n_obs}, екстраполяція = {len(test)}")
    print(f"ADF p (ряд) = {adfuller(train)[1]:.4f}, ADF p (1-а різниця) = {adfuller(np.diff(train))[1]:.4f}")
    family, fs, r = sel.recommend(train, season)

    cfg, ets_err = em.select(train, season)
    print("\n--- Експоненційне згладжування: RMSE на валідації ---")
    for c, e in sorted(ets_err.items(), key=lambda x: x[1]):
        print(f"{em.name(c):<62} {e:12.3f}")
    ets = em.fit(train, cfg, season)
    ets_fc = ets.forecast(len(test))
    ets_kpi = kpi(train, np.asarray(ets.fittedvalues), test, ets_fc)

    reg_name, reg, reg_err = rm.select(train, season)
    print("\n--- Регресійні моделі: RMSE на валідації ---")
    for nm, e in sorted(reg_err.items(), key=lambda x: x[1]):
        print(f"{nm:<62} {e:12.3f}")
    reg_fit = reg.predict(0, n_obs)
    reg_fc = reg.predict(n_obs, len(y))
    reg_kpi = kpi(train, reg_fit, test, reg_fc)

    print("\n--- KPI обраних моделей на інтервалі екстраполяції ---")
    for nm, k in [(em.name(cfg), ets_kpi), (reg_name, reg_kpi)]:
        print(f"{nm:<62} R2 = {k['R2']:.4f}  RMSE = {k['RMSE']:10.3f}  MAE = {k['MAE']:10.3f}  MAPE = {k['MAPE']:6.2f}%")
    votes = sum(reg_kpi[k] < ets_kpi[k] for k in ('RMSE', 'MAE', 'MAPE'))
    actual = 'регресія' if votes >= 2 else 'експоненційне згладжування'
    print(f"Регресія краща за {votes} з 3 KPI (RMSE, MAE, MAPE) -> фактично кращий: {actual} -> "
          f"{'рекомендацію підтверджено' if actual == family else 'рекомендацію не підтверджено'}")

    t = np.arange(len(y))
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw=dict(height_ratios=[3, 1.3]))
    ax[0].plot(t, y, color='lightgray', label='Дані')
    ax[0].plot(t[:n_obs], ets.fittedvalues, color='green', linewidth=1)
    ax[0].plot(t[n_obs:], ets_fc, color='green', linewidth=2, label=em.name(cfg))
    ax[0].plot(t[:n_obs], reg_fit, color='purple', linewidth=1)
    ax[0].plot(t[n_obs:], reg_fc, color='purple', linewidth=2, label=reg_name)
    ax[0].axvline(n_obs, color='black', linestyle='--')
    ax[0].set_title(f'{title}\nалгоритм вибору: {family} (r = {r:.2f})')
    ax[0].legend(fontsize=8)
    ax[0].grid(True)
    ax[1].plot(t[n_obs:], ets_fc - test, color='green', label='похибка ETS')
    ax[1].plot(t[n_obs:], reg_fc - test, color='purple', label='похибка регресії')
    ax[1].axhline(0, color='black', linewidth=0.8)
    ax[1].legend(fontsize=8)
    ax[1].grid(True)
    plt.tight_layout()
    return title, family, actual, r


summary = [process('Кубічна модель варіанту 10 (тренд + нормальний шум)', synthetic_cubic(), None),
           process('Офіційний курс USD (НБУ)', nbu_usd(), 7),
           process('Місячні продажі Superstore (Data_Set_10)', superstore_sales(), 12, test_share=0.25)]

print("\n==================== Перевірка алгоритму вибору ====================")
for title, rec, act, r in summary:
    print(f"{title:<52} r = {r:5.2f}  рекомендовано: {rec:<27} фактично: {act}")
plt.show()
