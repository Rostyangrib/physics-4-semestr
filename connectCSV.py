import os
import pandas as pd
from refractive_index_script import RefractiveIndexMaterial

df = pd.read_csv("materials_with_crystal_codes.csv")

database_path = os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database")
data_nk_path = os.path.join(database_path, "data-nk")

# все .yml файлы в папке материала
def get_all_pages(formula):
    material_path = os.path.join(data_nk_path, "main", formula)
    if not os.path.exists(material_path):
        print(f"Папка для материала {formula} не найдена: {material_path}")
        return None

    yml_files = [f for f in os.listdir(material_path) if f.endswith(".yml")]
    if not yml_files:
        print(f"Нет .yml файлов для материала {formula}")
        return None

    pages = [os.path.splitext(f)[0] for f in yml_files]
    return pages

df["pages"] = df["formula"].apply(get_all_pages)

# Удаляем строки, где pages не определён
df = df.dropna(subset=["pages"])

# Функция для получения показателя преломления (перебирает все page)
def get_refractive_index(row, wavelength_nm=500):
    if not row["pages"]:
        return None

    for page in row["pages"]:
        try:
            rim = RefractiveIndexMaterial(shelf="main", book=row["formula"], page=page)
            n = rim.get_refractive_index(wavelength_nm)
            print(f"Успешно для {row['formula']}, page: {page}")
            return n
        except Exception as e:
            print(f"Ошибка для {row['formula']}, page {page}: {e}")
            continue

    print(f"Не удалось получить n для {row['formula']} после перебора всех page")
    return None

# столбец с refractive_index
df["refractive_index"] = df.apply(get_refractive_index, axis=1)
# Удаляем строки, где refractive_index равен null/NaN
df = df.dropna(subset=["refractive_index"])

df.to_csv("output_with_refractive_index.csv", index=False)