from mp_api.client import MPRester
import pandas as pd
from time import sleep

API_KEY = "gjS2rEd3EYKuyutybdW1marq1GAoL1db"
MAX_MATERIALS = 5000  # Желаемое количество материалов
CHUNK_SIZE = 500  # Размер одного запроса
DELAY = 1  # Задержка между запросами (секунды)

# Сопоставление кристаллических систем с числовыми кодами
crystal_system_mapping = {
    'Cubic': 1, 'Tetragonal': 2, 'Orthorhombic': 3,
    'Hexagonal': 4, 'Trigonal': 5, 'Monoclinic': 6, 'Triclinic': 7
}


def fetch_materials_with_dielectric():
    try:
        with MPRester(API_KEY) as mpr:
            all_docs = []
            all_electronic = []
            all_dielectric = []

            num_requests = (MAX_MATERIALS // CHUNK_SIZE) + 1

            print(f"Загрузка {MAX_MATERIALS} материалов в {num_requests} запросах...")

            for chunk_num in range(num_requests):
                print(f"Партия {chunk_num + 1}/{num_requests}...")

                # Базовые свойства
                docs = list(mpr.materials.search(
                    fields=["material_id", "formula_pretty", "density", "elements", "symmetry"],
                    num_chunks=1,
                    chunk_size=CHUNK_SIZE
                ))
                all_docs.extend(docs)

                # Электронные свойства (band_gap)
                electronic_docs = list(mpr.materials.summary.search(
                    fields=["material_id", "band_gap"],
                    num_chunks=1,
                    chunk_size=CHUNK_SIZE
                ))
                all_electronic.extend(electronic_docs)

                # Диэлектрические свойства (ионная и электронная части)
                dielectric_docs = list(mpr.materials.dielectric.search(
                    fields=["material_id", "total", "ionic", "electronic"],
                    num_chunks=1,
                    chunk_size=CHUNK_SIZE
                ))
                all_dielectric.extend(dielectric_docs)

                if len(all_docs) >= MAX_MATERIALS:
                    all_docs = all_docs[:MAX_MATERIALS]
                    all_electronic = all_electronic[:MAX_MATERIALS]
                    all_dielectric = all_dielectric[:MAX_MATERIALS]
                    break

                sleep(DELAY)

            # Создаем словари для быстрого доступа
            bandgap_dict = {doc.material_id: doc.band_gap for doc in all_electronic}
            dielectric_dict = {
                doc.material_id: {
                    "total_dielectric": doc.total,
                    "ionic_dielectric": doc.ionic,
                    "electronic_dielectric": doc.electronic
                }
                for doc in all_dielectric
            }

            # Обработка данных
            clean_data = []
            for doc in all_docs:
                crystal_system = getattr(doc.symmetry, 'crystal_system', None)
                if crystal_system:
                    crystal_system = crystal_system.value.capitalize()

                dielectric_data = dielectric_dict.get(doc.material_id, {})

                clean_data.append({
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "density": doc.density,
                    "elements": ", ".join([str(e) for e in doc.elements]),
                    "band_gap": bandgap_dict.get(doc.material_id),
                    "total_dielectric": dielectric_data.get("total_dielectric"),
                    "ionic_dielectric": dielectric_data.get("ionic_dielectric"),
                    "electronic_dielectric": dielectric_data.get("electronic_dielectric"),
                    "crystal_system": crystal_system,
                    "crystal_code": crystal_system_mapping.get(crystal_system, 0)
                })

            # Фильтрация данных
            df = pd.DataFrame(clean_data).dropna(subset=["band_gap", "electronic_dielectric"])
            df = df[df["band_gap"] != 0]

            return df

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return pd.DataFrame()


# Запуск сбора данных
df = fetch_materials_with_dielectric()

if not df.empty:
    df.to_csv("materials.csv", index=False)
    print(f"Сохранено {len(df)} материалов с диэлектрическими свойствами")
    print("\nПример данных:")
    print(df.head())
else:
    print("Не удалось получить данные")


# # Сохраняем результат в новый CSV
# df.to_csv("output.csv", index=False)
