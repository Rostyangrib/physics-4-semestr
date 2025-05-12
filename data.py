from mp_api.client import MPRester
import pandas as pd
from pymatgen.core import Element

API_KEY = "gjS2rEd3EYKuyutybdW1marq1GAoL1db"

crystal_system_mapping = {
    'Cubic': 1,
    'Tetragonal': 2,
    'Orthorhombic': 3,
    'Hexagonal': 4,
    'Trigonal': 5,
    'Monoclinic': 6,
    'Triclinic': 7
}

try:
    with MPRester(API_KEY) as mpr:
        print("Получаем базовые свойства материалов...")
        docs = list(mpr.materials.search(
            fields=["material_id", "formula_pretty", "density", "elements", "symmetry"],
            num_chunks=1,
            chunk_size=100
        ))

        print("Получаем электронные свойства...")
        # Получаем band_gap из другого эндпоинта
        electronic_docs = list(mpr.materials.summary.search(
            fields=["material_id", "band_gap"],
            num_chunks=1,
            chunk_size=100
        ))

        bandgap_dict = {doc.material_id: doc.band_gap for doc in electronic_docs}

        print("Обрабатываем данные...")
        clean_data = []
        for doc in docs:
            # Преобразуем элементы в строки
            elements_str = ", ".join([str(e) for e in doc.elements])

            crystal_system = getattr(doc.symmetry, 'crystal_system', None)
            if crystal_system:
                crystal_system = crystal_system.value.capitalize()

            crystal_code = crystal_system_mapping.get(crystal_system, 0)

            clean_data.append({
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "density": doc.density,
                "elements": elements_str,
                "band_gap": bandgap_dict.get(doc.material_id),
                "crystal_system": crystal_system,
                "crystal_code": crystal_code
            })

        df = pd.DataFrame(clean_data).dropna(subset=["band_gap"])

        df.to_csv("materials_with_crystal_codes.csv", index=False)
        print(f"\nУспешно сохранено {len(df)} материалов")
        print("\nПервые 5 строк:")
        print(df.head())

except Exception as e:
    print(f"\nКритическая ошибка: {str(e)}")
