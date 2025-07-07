import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

from typing import List, Dict, Optional, Any
import re
from configuration import Configuration

def generate_mace_fingerprints(
    configurations: List[Configuration],
    model_path: str,
    type_map: Dict[int, str],
    device: str = 'cpu',
    r_max: float = 6.0
) -> np.ndarray:
    """
    Превращает массив объектов Configuration в массив фингерпринтов.

    Args:
        configurations (List[Configuration]): Список конфигураций.
        model_path (str): Путь к файлу обученной модели MACE (*.model).
        type_map (Dict[int, str]): Словарь для сопоставления целочисленного типа
                                  атома из .cfg с его химическим символом (например, {0: 'C', 1: 'H'}).
        device (str): Устройство для вычислений ('cpu' или 'cuda').

    Returns:
        np.ndarray: Массив формы [total_atoms, fingerprint_length + 1],
                    где последний столбец - индекс конфигурации.
    """
    try:
        # Загружаем калькулятор MACE из файла модели
        calc = MACECalculator(model_paths=model_path, device=device, default_dtype='float64')
        calc.r_max = r_max
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Ошибка при загрузке модели MACE из '{model_path}': {e}")
        print("Пожалуйста, убедитесь, что путь к модели указан верно и файл не поврежден.")
        return np.array([])

    all_fingerprints = []

    print(f"Обработка {len(configurations)} конфигураций...")
    for config_idx, config in enumerate(configurations):
        if not config.atom_data:
            print(f"Пропуск конфигурации {config_idx}, так как в ней нет данных об атомах.")
            continue
            
        # Извлекаем данные для создания объекта ASE Atoms
        positions = np.array([[atom['cartes_x'], atom['cartes_y'], atom['cartes_z']] for atom in config.atom_data])
        # Преобразуем числовые типы в химические символы
        symbols = [type_map[atom['type']] for atom in config.atom_data]
        cell = np.array(config.supercell) if config.supercell else None
        pbc = True if config.supercell is not None else False

        # Создаем объект ASE Atoms
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
        
        # Вычисляем дескрипторы (фингерпринты) для всех атомов в структуре
        # 'mace_descriptors' - это ключ для получения фингерпринтов
        fingerprints = calc.get_descriptors(atoms) # -> shape [n_atoms, fingerprint_length]
        
        # Создаем столбец с индексом текущей конфигурации
        config_indices = np.full((fingerprints.shape[0], 1), config_idx)
        
        # Объединяем фингерпринты и индексы
        combined_data = np.hstack([fingerprints, config_indices])
        all_fingerprints.append(combined_data)
        
        if (config_idx + 1) % 10 == 0:
            print(f"  ...обработано {config_idx + 1}/{len(configurations)}")


    if not all_fingerprints:
        print("Не было создано ни одного фингерпринта.")
        return np.array([])

    print("Объединение результатов...")
    # Собираем все в один большой массив
    final_array = np.vstack(all_fingerprints)
    
    return final_array

