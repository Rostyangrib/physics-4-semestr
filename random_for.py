import pandas as pd
import matplotlib.pyplot as plt
import joblib  # для сохранения модели
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
df = pd.read_csv("materials_with_predictions.csv")

# Выбор признаков и целевой переменной
features = ['log_density', 'log_band_gap', 'log_eps_xx', 'log_eps_yy', 'log_eps_zz']
target = 'predicted_n'

X = df[features]
y = df[target]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание и оценка
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse}")
print(f"Коэффициент детерминации (R²): {r2}")

# Визуализация важности признаков
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Важность признаков")
plt.title("Feature Importance (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Сохранение модели
joblib.dump(model, "random_forest_model.pkl")
print("Модель сохранена в файл: random_forest_model.pkl")
