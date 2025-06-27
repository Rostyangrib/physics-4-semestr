import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Изменено с StandardScaler на MinMaxScaler
import ast

# Загрузка данных
df = pd.read_csv("materials_refractive_index.csv")
print("Загруженные столбцы:", df.columns.tolist())

# Парсинг тензора диэлектрической проницаемости
def parse_matrix(x):
    try:
        matrix = ast.literal_eval(x)
        return pd.Series([matrix[0][0], matrix[1][1], matrix[2][2]])
    except:
        return pd.Series([np.nan, np.nan, np.nan])

df[["eps_xx", "eps_yy", "eps_zz"]] = df["electronic_dielectric"].apply(parse_matrix)

# Удаление строк с пропущенными значениями
df = df.dropna(subset=["eps_xx", "eps_yy", "eps_zz", "refractive_index"])

# Логарифмирование предикторов с защитой от log(0)
df["log_density"] = np.log(df["density"] + 1e-6)
df["log_band_gap"] = np.log(df["band_gap"] + 1e-6)
df["log_eps_xx"] = np.log(df["eps_xx"] + 1e-6)
df["log_eps_yy"] = np.log(df["eps_yy"] + 1e-6)
df["log_eps_zz"] = np.log(df["eps_zz"] + 1e-6)

# Нормализация признаков (приведение к [0, 1])
normalizer = MinMaxScaler()  # Используем MinMaxScaler вместо StandardScaler
features_to_normalize = ["log_density", "log_band_gap", "log_eps_xx", "log_eps_yy", "log_eps_zz"]
df[features_to_normalize] = normalizer.fit_transform(df[features_to_normalize])

# Построение модели
X = df[features_to_normalize]
X = sm.add_constant(X)  # Добавляем константу для intercept
y = df["refractive_index"]

model = sm.OLS(y, X).fit()
print(model.summary())

# Добавляем предсказанные значения и абсолютную разницу
df["predicted_n"] = model.predict(X)
df["n_difference"] = abs(df["refractive_index"] - df["predicted_n"])  # Абсолютная разница

# Сохранение результатов
output_columns = [
    "material_id", "formula", "density", "band_gap",
    "refractive_index", "predicted_n", "n_difference",
    "eps_xx", "eps_yy", "eps_zz", "crystal_system"
]
df[output_columns].to_csv("materials_with_predictions_normalized.csv", index=False)

# Вывод средней абсолютной ошибки (MAE)
mean_diff = df["n_difference"].mean()
print(f"\nСредняя абсолютная разница (MAE): {mean_diff:.4f}")
print(f"Относительная ошибка: {(mean_diff / df['refractive_index'].mean() * 100):.2f}%")