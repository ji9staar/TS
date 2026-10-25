import numpy as np
import matplotlib.pyplot as plt

# 1. Генерація даних
np.random.seed(42)
n = 400
t = np.arange(n)

ideal_trend = 0.005 * t**3 - 0.2 * t**2 + 5 * t + 50
noise = np.random.normal(0, 30, n)
data = ideal_trend + noise

# аномальні викиди
anomalies_idx = np.random.choice(n, 20, replace=False)
for idx in anomalies_idx:
    data[idx] += np.random.choice([-1, 1]) * 800

# 2. Очищення від аномалій
data_clean = data.copy()
window = 15
for i in range(n):
    start = max(0, i - window//2)
    end = min(n, i + window//2)
    local_med = np.median(data[start:end])
    local_std = np.std(data[start:end])
    if local_std > 0 and abs(data[i] - local_med) > 3 * local_std:
        data_clean[i] = local_med

# 3. Стандартний фільтр
alpha, beta, gamma = 0.01, 0.0001, 0.00001
x_s = np.zeros(n)
x_est, v_est, a_est = data_clean[0], 0.0, 0.0

for i in range(n):
    x_p = x_est + v_est + 0.5 * a_est
    v_p = v_est + a_est
    a_p = a_est
    
    r = data_clean[i] - x_p
    
    x_est = x_p + alpha * r
    v_est = v_p + beta * r
    a_est = a_p + 2 * gamma * r
    x_s[i] = x_est

# 4. Адаптивний фільтр (R&D)
x_ad = np.zeros(n)
x_est, v_est, a_est = data_clean[0], 0.0, 0.0

for i in range(n):
    x_p = x_est + v_est + 0.5 * a_est
    v_p = v_est + a_est
    a_p = a_est
    
    r = data_clean[i] - x_p
    
    # адаптація параметрів
    if abs(r) > 100:
        a_dyn, b_dyn, g_dyn = 0.85, 0.2, 0.05
    else:
        a_dyn, b_dyn, g_dyn = 0.05, 0.001, 0.0001
        
    x_est = x_p + a_dyn * r
    v_est = v_p + b_dyn * r
    a_est = a_p + 2 * g_dyn * r
    x_ad[i] = x_est

# 5. Візуалізація
plt.figure(figsize=(12, 7))
plt.plot(t, data, '.', color='lightcoral', alpha=0.5, label='Вхідні дані (з викидами)')
plt.plot(t, data_clean, '.', color='lightblue', alpha=0.7, label='Очищені дані')
plt.plot(t, ideal_trend, color='midnightblue', linewidth=3, label='Ідеальний кубічний тренд')
plt.plot(t, x_s, color='red', linestyle='--', linewidth=2, label='Стандартний фільтр (розходження)')
plt.plot(t, x_ad, color='dodgerblue', linewidth=3, label='Адаптивний фільтр (R&D)')

plt.title('Рекурентне згладжування: Стандартний vs Адаптивний фільтр')
plt.xlabel('Час (t)')
plt.ylabel('Амплітуда (y)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()