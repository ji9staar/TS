import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# ФУНКЦІЯ парсингу реальних даних
def file_parsing(File_name, Data_names):
    d = pd.read_excel(File_name)
    results = []
    for name in Data_names:
        values = d[name].dropna().values
        results.append(values)
        print(f"--- Аналіз показника: {name} ---")
        Stat_characteristics(values, f"Реальні дані: {name}")
    return results

# Функція генерації нормального шуму
def random_norm(dm, dsig, iter):
    S = np.random.normal(dm, dsig, iter)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('--- статистичні характеристики НОРМАЛЬНОГО закону розподілу ---')
    print('Математичне сподівання =', mS)
    print('Дисперсія =', dS)
    print('СКВ =', scvS)
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Гістограма нормального розподілу")
    plt.show()
    return S

# Функція генерації рівномірного шуму
def random_uniform(low, high, iter):
    S = np.random.uniform(low, high, iter)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('--- статистичні характеристики РІВНОМІРНОГО закону розподілу ---')
    print('Математичне сподівання =', mS)
    print('Дисперсія =', dS)
    print('СКВ =', scvS)
    plt.hist(S, bins=20, facecolor="green", alpha=0.5)
    plt.title("Гістограма рівномірного розподілу")
    plt.show()
    return S

# Модель лінійного тренду
def Model_Linear(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.005 * i
    return S0

# Модель кубічного тренду
def Model_Cubic(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.000000001 * (i ** 3)
    return S0

# Адитивна модель: тренд + шум
def Model_Additive(Trend, Noise, n):
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = Trend[i] + Noise[i]
    return SV

# Адитивна модель з аномальними вимірами
def Model_Additive_AV(Trend, Noise, n, NAV, Q_AV, dsig):
    SV_AV = Model_Additive(Trend, Noise, n)
    anomaly_indices = np.random.randint(0, n, NAV)
    for idx in anomaly_indices:
        SV_AV[idx] += Q_AV * dsig * np.random.choice([-1, 1])
    return SV_AV

# Статистичні характеристики вибірки
def Stat_characteristics(SL, Text):
    mS = np.mean(SL)
    dS = np.var(SL)
    scvS = mt.sqrt(dS)
    print('---', Text, '---')
    print('Математичне сподівання =', mS)
    print('Дисперсія =', dS)
    print('СКВ =', scvS)
    print('-'*40)

# Графіки
def Plot_AV(Trend, SV_L, Text):
    plt.clf()
    plt.plot(SV_L, label="Вибірка", color="lightblue")
    plt.plot(Trend, label="Ідеальний тренд", color="orange")
    plt.title(Text)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    n = 10000
    iter = int(n)
    Q_AV = 5
    NAV = int(iter * 0.01) # 1% аномальних вимірів
    
    # Параметри нормального розподілу
    dm = 0
    dsig = 5
    
    # Моделі ідеального тренду
    Trend_Cubic = Model_Cubic(n)
    Trend_Linear = Model_Linear(n)
    
    # Моделі шуму
    Noise_Norm = random_norm(dm, dsig, iter)
    Noise_Unif = random_uniform(-8, 8, iter)
    
    # Комбінація 1: Кубічна модель + Нормальний шум
    SV_CN = Model_Additive(Trend_Cubic, Noise_Norm, n)
    Plot_AV(Trend_Cubic, SV_CN, 'Кубічна модель + Нормальний шум')
    Stat_characteristics(SV_CN, 'Вибірка (Кубічна + Нормальний)')
    
    # Комбінація 2: Лінійна модель + Рівномірний шум
    SV_LU = Model_Additive(Trend_Linear, Noise_Unif, n)
    Plot_AV(Trend_Linear, SV_LU, 'Лінійна модель + Рівномірний шум')
    Stat_characteristics(SV_LU, 'Вибірка (Лінійна + Рівномірний)')
    
    # Комбінація 3: Кубічна модель + Нормальний шум + Аномалії
    SV_CN_AV = Model_Additive_AV(Trend_Cubic, Noise_Norm, n, NAV, Q_AV, dsig)
    Plot_AV(Trend_Cubic, SV_CN_AV, 'Кубічна модель + Норм. шум + Аномальні виміри')
    Stat_characteristics(SV_CN_AV, 'Вибірка з Аномаліями (Кубічна + Норм)')
    
    # Обробка реальних даних
    try:
        file_parsing('Oschadbank (USD).xls', ['Продаж', 'Купівля', 'Курс НБУ'])
    except Exception as e:
        print("Файл реальних даних не знайдено або відсутні потрібні колонки.", e)

# --- ЗАВДАННЯ ІІІ РІВНЯ СКЛАДНОСТІ ---

# 1. Парсинг самостійно обраного сайту
def parse_custom_site_and_save():
    # Використовуємо відкритий API НБУ для отримання офіційного курсу валют
    url = 'https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json'
    print(f"Починаємо парсинг сайту: {url}")
    
    response = requests.get(url)
    
    parsed_data = []
    if response.status_code == 200:
        data = response.json()
        for item in data[:20]:
            parsed_data.append(item['rate'])
    else:
        print(f"Помилка з'єднання: {response.status_code}")
            
    # Збереження результатів у файл CSV
    df = pd.DataFrame({'Parsed_Rate': parsed_data})
    df.to_csv('parsed_custom_data.csv', index=False)
    print("Дані успішно збережено у файл 'parsed_custom_data.csv'")
    
    return np.array(parsed_data)

# 3, 4, 5. Аналіз реальних даних та синтез аналогічної моделі
def level_3_synthesis(real_data):
    if len(real_data) == 0:
        print("Немає даних для аналізу.")
        return

    # Оцінка динаміки та визначення статистичних характеристик
    mS_real = np.mean(real_data)
    dS_real = np.var(real_data)
    scvS_real = mt.sqrt(dS_real)
    
    print('--- Статистичні характеристики результатів парсингу ---')
    print('Математичне сподівання =', mS_real)
    print('Дисперсія =', dS_real)
    print('СКВ =', scvS_real)

    # Синтез моделі, аналогічної за характеристиками
    n = len(real_data)
    # Ідеальний тренд
    synthesized_trend = np.full(n, mS_real)
    # Шум з дисперсією реальних даних
    synthesized_noise = np.random.normal(0, scvS_real, n)
    
    # Синтезована вибірка
    synthesized_data = synthesized_trend + synthesized_noise

    # Візуалізація для порівняння
    plt.clf()
    plt.plot(real_data, label="Реальні дані (Парсинг)", marker='o')
    plt.plot(synthesized_data, label="Синтезована модель", linestyle='--')
    plt.title("Порівняння реальних (з парсингу) та синтезованих даних")
    plt.legend()
    plt.show()

# Виклик функцій ІІІ рівня
if __name__ == '__main__':    
    print("\n--- ВИКОНАННЯ 3 РІВНЯ СКЛАДНОСТІ ---")
    parsed_array = parse_custom_site_and_save()
    level_3_synthesis(parsed_array)