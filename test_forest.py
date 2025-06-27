import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Загрузка данных
files = ['materials_refractive_index.csv', "synthetic_realistic_data.csv"]
df_list = [pd.read_csv(f) for f in files]
df_train = pd.concat(df_list, ignore_index=True)
# Признаки для обучения
features = ['log_density', 'log_band_gap']
# Необходимо определить коэффициент преломления
target = 'refractive_index'

X = df_train[features]
y = df_train[target]
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Обучение модели
model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)


# X_test = df_test[features]
# y_test = df_test[target]

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'true_n': y_test.values,
    'predicted_n': y_pred
})
comparison['absolute_error'] = abs(comparison['true_n'] - comparison['predicted_n'])
print(comparison)
comparison.to_csv("results.csv", index=False)

from sklearn.tree import plot_tree

# Выбираем одно дерево для визуализации
estimator = model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(estimator,
          feature_names=features,
          filled=True,
          rounded=True,
          max_depth=2)
plt.title("Пример дерева из случайного леса")
plt.show()

