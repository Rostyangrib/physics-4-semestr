import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.linear_model import Lasso

def parse_matrix(x):
    try:
        matrix = ast.literal_eval(x)
        return pd.Series([matrix[0][0], matrix[1][1], matrix[2][2]])
    except:
        return pd.Series([np.nan, np.nan, np.nan])

def regression():
    df = pd.read_csv("materials_refractive_index_5000.csv")
    print("Загруженные столбцы:", df.columns.tolist())

    # Парсинг тензора диэлектрической проницаемости


    df[["eps_xx", "eps_yy", "eps_zz"]] = df["electronic_dielectric"].apply(parse_matrix)
    df[["eps_ionic_xx", "eps_ionic_yy", "eps_ionic_zz"]] = df["ionic_dielectric"].apply(parse_matrix)
    df[["eps_total_xx", "eps_total_yy", "eps_total_zz"]] = df["total_dielectric"].apply(parse_matrix)

    # Удаление строк с пропущенными значениями
    df = df.dropna(subset=["eps_xx", "eps_yy", "eps_zz", "refractive_index"])

    tmp = df["band_gap"][0] * (df["eps_xx"][0] + df["eps_yy"][0] + df["eps_zz"][0])
    tmp_1 = df["band_gap"][9] * (df["eps_xx"][9] + df["eps_yy"][9] + df["eps_zz"][9])
    # Логарифмирование предикторов с защитой от log(0)
    df["log_density"] = np.log(df["density"] + 1e-6)
    df["log_band_gap"] = np.log(df["band_gap"] + 1e-6)
    df["log_eps_xx"] = np.log(df["eps_xx"] + 1e-6)
    df["log_eps_yy"] = np.log(df["eps_yy"] + 1e-6)
    df["log_eps_zz"] = np.log(df["eps_zz"] + 1e-6)

    df["log_ionic_xx"] = np.log(df["eps_ionic_xx"] + 1e-6)
    df["log_ionic_yy"] = np.log(df["eps_ionic_yy"] + 1e-6)
    df["log_ionic_zz"] = np.log(df["eps_ionic_zz"] + 1e-6)

    df["log_total_xx"] = np.log(df["eps_total_xx"] + 1e-6)
    df["log_total_yy"] = np.log(df["eps_total_yy"] + 1e-6)
    df["log_total_zz"] = np.log(df["eps_total_zz"] + 1e-6)

    df['avg_esp'] = (df["log_eps_xx"] * df["log_eps_yy"] * df["log_eps_zz"]) / 3
    #df['extra parameter'] = df["log_band_gap"] * (df["log_eps_xx"] * df["log_eps_yy"] * df["log_eps_zz"])
    df['extra parameter'] = df["log_band_gap"] * (df["log_eps_xx"] * df["log_eps_yy"] * df["log_eps_zz"])
    #df = df[df["extra parameter"] > 0.0001]
    df['band_gap_eps_xx'] = df['band_gap'] * df['eps_xx']
    df['band_gap^2'] = df['band_gap'] ** 2
    df['eps_xx^2'] = df['eps_xx'] ** 2
    features = [ "band_gap",
                         "log_eps_xx"
                         ]
    lasso = Lasso(alpha=0.1)
    # Построение модели
    X = df[features]
    X = sm.add_constant(X)  # Добавляем константу для intercept
    y = df["refractive_index"]

    lasso.fit(X, y)
    print("Коэффициенты Lasso:", lasso.coef_)

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Добавляем предсказанные значения и разницу
    df["predicted_n"] = model.predict(X)
    df["n_difference"] = round(abs(df["refractive_index"] - df["predicted_n"]), 6)

    # Сохранение результатов
    output_columns = [
        "material_id", "formula", "density", "band_gap",
        "refractive_index", "predicted_n", "n_difference",
        "eps_xx", "eps_yy", "eps_zz", "eps_ionic_xx", "eps_ionic_yy",
        "eps_ionic_zz"
    ]


    df[output_columns].to_csv("materials_with_predictions_log_not_scale.csv", index=False)
    mean_diff = df["n_difference"].mean()
    print(f"Средняя разница между реальным и предсказанным n: {mean_diff:.4f}")

    output_my_columns = [
        "refractive_index", "predicted_n", "n_difference",
    ]
    df[output_my_columns].to_csv("only_n.csv", index=False)


    return model

regression()
