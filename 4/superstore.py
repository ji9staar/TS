import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans

MONTHS = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер', 'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']


# Завантаження продажів за місяцями
def load_superstore(filename):
    d = pd.read_excel(filename)
    monthly = d.set_index('Order Date').resample('MS')[['Sales', 'Profit', 'Quantity']].sum()
    print(f"Data_Set_10 (Superstore): замовлень = {len(d)}, період {d['Order Date'].min().date()} - "
          f"{d['Order Date'].max().date()}, місяців = {len(monthly)}")
    return d, monthly


# Характеристики продажів і сезонність
def properties_and_decomposition(monthly):
    s = monthly['Sales']
    yearly = s.groupby(s.index.year).sum()
    print("\n--- Властивості місячних продажів ---")
    print(f"середні продажі = {s.mean():.0f}, СКВ = {s.std():.0f}, коеф. варіації = {s.std() / s.mean():.2f}")
    print("річні продажі: " + ', '.join(f"{y}: {v:.0f}" for y, v in yearly.items()))
    print(f"середній річний приріст = {(yearly.iloc[-1] / yearly.iloc[0]) ** (1 / 3) * 100 - 100:.1f} %")
    print(f"ADF p-value = {adfuller(s)[1]:.4f}")
    print(f"середня рентабельність (Profit/Sales) = {monthly['Profit'].sum() / s.sum() * 100:.1f} %")

    res = seasonal_decompose(s, model='multiplicative', period=12)
    idx = res.seasonal.iloc[:12].values
    order = np.argsort(idx)
    print("сезонні індекси: " + ', '.join(f"{MONTHS[i]} {v:.2f}" for i, v in enumerate(idx)))
    print(f"пікові місяці: {', '.join(MONTHS[i] for i in order[-3:][::-1])}; "
          f"найслабші: {', '.join(MONTHS[i] for i in order[:2])}")
    fig = res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('Мультиплікативна декомпозиція продажів Superstore')
    fig.tight_layout()
    return res


# Кореляція категорій і регіонів
def correlation(d, monthly):
    cat = d.pivot_table(index=pd.Grouper(key='Order Date', freq='MS'), columns='Category', values='Sales', aggfunc='sum')
    print("\n--- Кореляція місячних продажів категорій ---")
    print(cat.corr().round(3))
    print(f"кореляція Sales - Profit = {monthly['Sales'].corr(monthly['Profit']):.3f}, "
          f"Sales - Quantity = {monthly['Sales'].corr(monthly['Quantity']):.3f}")
    reg = d.pivot_table(index=pd.Grouper(key='Order Date', freq='MS'), columns='Region', values='Sales', aggfunc='sum')
    print("середня кореляція продажів між регіонами = "
          f"{reg.corr().values[np.triu_indices(4, 1)].mean():.3f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for c in cat.columns:
        plt.plot(cat.index, cat[c], label=c)
    plt.title('Місячні продажі за категоріями')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.imshow(cat.corr(), cmap='gray_r', vmin=0, vmax=1)
    plt.xticks(range(3), cat.columns, rotation=15)
    plt.yticks(range(3), cat.columns)
    plt.title('Кореляція категорій')
    plt.tight_layout()


# Кластеризація товарів за кварталами
def subcategory_clustering(d, n_clusters=3):
    p = d.pivot_table(index='Sub-Category', columns=d['Order Date'].dt.quarter, values='Sales', aggfunc='sum').fillna(0)
    profile = p.div(p.sum(axis=1), axis=0)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(profile)
    print("\n--- Кластеризація 17 підкатегорій за часткою продажів у кварталах (KMeans) ---")
    plt.figure(figsize=(10, 4))
    for c in range(n_clusters):
        members = profile.index[km.labels_ == c]
        center = km.cluster_centers_[c]
        print(f"кластер {c}: частки Q1-Q4 = {', '.join(f'{v * 100:.1f}%' for v in center)}; {', '.join(members)}")
        plt.plot(['Q1', 'Q2', 'Q3', 'Q4'], center, marker='o', label=f'кластер {c} ({len(members)})')
    plt.title('Сезонні профілі кластерів підкатегорій')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# Штучні продажі: тренд x сезонність x шум
def synthetic(monthly, res, n_sim=100):
    s = monthly['Sales'].values
    t = np.arange(len(s))
    idx = res.seasonal.iloc[:12].values
    deseason = s / np.tile(idx, len(s) // 12)
    a, b = np.polyfit(t, deseason, 1)
    sigma = np.std(np.log(deseason / (a * t + b)))
    print(f"\n--- Синтетична модель: Sales(t) = ({b:.0f} + {a:.1f}*t) * S(міс) * exp(N(0, {sigma:.3f})) ---")
    rng = np.random.default_rng(10)
    stats_sim = []
    sims = []
    for _ in range(n_sim):
        y = (a * t + b) * np.tile(idx, len(s) // 12) * np.exp(rng.normal(0, sigma, len(s)))
        sims.append(y)
        yearly = y.reshape(4, 12).sum(axis=1)
        prof = y.reshape(4, 12).mean(axis=0)
        stats_sim.append([y.mean(), y.std(), (yearly[-1] / yearly[0]) ** (1 / 3) - 1,
                          np.corrcoef(prof, s.reshape(4, 12).mean(axis=0))[0, 1], np.argmax(prof) + 1])
    stats_sim = np.array(stats_sim)
    yr = s.reshape(4, 12).sum(axis=1)
    real = [s.mean(), s.std(), (yr[-1] / yr[0]) ** (1 / 3) - 1, 1.0, np.argmax(s.reshape(4, 12).mean(axis=0)) + 1]
    names = ['середні продажі', 'СКВ продажів', 'річний приріст', 'кореляція сезонного профілю', 'піковий місяць']
    print(f"{'показник':<30}{'реальні':>12}{'модель':>12}{'СКВ':>10}")
    for n, r, v in zip(names, real, stats_sim.T):
        print(f"{n:<30}{r:12.3f}{v.mean():12.3f}{v.std():10.3f}")

    plt.figure(figsize=(12, 4))
    plt.plot(monthly.index, s, color='black', linewidth=2, label='Реальні продажі')
    for i in range(2):
        plt.plot(monthly.index, sims[i], alpha=0.7, label=f'Синтетична реалізація {i + 1}')
    plt.title('Реальні та синтетичні місячні продажі')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
