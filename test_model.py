import numpy as np
import pandas as pd

np.random.seed(42)  # для воспроизводимости

n_samples = 10000

# Корреляционная матрица для 5 признаков
cov = np.array([
    [1.0,  0.7,  0.5,  0.3,  0.1],
    [0.7,  1.0,  0.6,  0.4,  0.2],
    [0.5,  0.6,  1.0,  0.5,  0.3],
    [0.3,  0.4,  0.5,  1.0,  0.4],
    [0.1,  0.2,  0.3,  0.4,  1.0]
])

mean = np.array([0.5, 1.0, 1.5, 2.0, 0.8])  # примерные средние по признакам

# Генерируем признаки с заданной корреляцией
features = np.random.multivariate_normal(mean, cov, size=n_samples)

# Переводим признаки в DataFrame с названиями
df = pd.DataFrame(features, columns=['log_density', 'log_band_gap', 'log_eps_xx', 'log_eps_yy', 'log_eps_zz'])

# Ограничим признаки в разумных пределах (пример)
df = df.clip(lower=0.1)

# Создаём целевую переменную с логикой и шумом
# Примерная формула: коэффициент преломления зависит от суммы eps и обратного band_gap
noise = np.random.normal(0, 0.05, n_samples)
df['refractive_index'] = (
    1.5
    + 0.4 * df['log_eps_xx']
    + 0.3 * df['log_eps_yy']
    + 0.2 * df['log_eps_zz']
    - 0.5 * df['log_band_gap']
    + noise
)


def remove_outliers_iqr(df, features):
    df_clean = df.copy()
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Фильтруем строки, оставляя только значения внутри границ
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
    return df_clean

# Пример:
features = ['log_density', 'log_band_gap', 'log_eps_xx', 'log_eps_yy', 'log_eps_zz']
df_no_outliers = remove_outliers_iqr(df, features)

# Сохраняем в csv
df.to_csv('synthetic_realistic_data.csv', index=False)

print("synthetic_realistic_data.csv создан")

