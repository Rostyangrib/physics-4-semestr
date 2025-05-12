import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_csv("output_with_refractive_index.csv")
print("Загруженные столбцы:", df.columns.tolist())

# Проверка наличия необходимых столбцов
required_columns = ['density', 'band_gap', 'crystal_code', 'refractive_index']

# X = df[['density', 'band_gap', 'crystal_code']]
# y = df['refractive_index']
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# print(model.summary())

# Логарифмирование предикторов
df["log_density"] = np.log(df["density"])
df["log_band_gap"] = np.log(df["band_gap"] + 1e-6)  # +1e-6 для избежания log(0)

# Новая модель с преобразованными предикторами
X = df[["log_density", "log_band_gap"]]
X = sm.add_constant(X)
y = df["refractive_index"]

model = sm.OLS(y, X).fit()
print(model.summary())

df['predicted_n'] = model.predict(X)
df.to_csv("materials_with_predictions.csv", index=False)
