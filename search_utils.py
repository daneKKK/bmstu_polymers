# search_utils.py

import numpy as np
from typing import List, Dict, Optional, Tuple
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

# Импортируйте все зависимости, которые нужны классу
from ase import Atoms
from mace.calculators import MACECalculator
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# Ваши кастомные типы, если они есть (например, Configuration)
from configuration import Configuration 

# Вспомогательная функция для параллельной обработки
def _process_candidate_wrapper(args):
    """
    Распаковывает аргументы и вычисляет сходство для одного кандидата.
    """
    db_item, target_centroids = args
    similarity_to_target = ConfigurationSearcher._calculate_set_similarity(
        target_centroids, db_item["centroids"]
    )
    return {
        "config": db_item["config"],
        "centroids": db_item["centroids"],
        "relevance": similarity_to_target,
    }


class ConfigurationSearcher:
    """
    Выполняет поиск структурно похожих конфигураций на основе
    кластеризованных фингерпринтов MACE.
    """
    def __init__(
        self,
        configurations: List, # Замените на ваш тип Configuration
        type_to_atomic_num: Dict[int, int],
        # ##### ИЗМЕНЕНО: Удален n_clusters, добавлены параметры для DBSCAN #####
        eps: float = 0.1,
        min_samples: int = 2,
        model_path: str = "medium",
        device: str = 'cpu'
    ):
        """
        Инициализирует поисковик.

        Args:
            configurations: Список объектов Configuration для поиска.
            type_to_atomic_num: Словарь для сопоставления 'type' из .cfg с атомным номером.
            eps: Параметр DBSCAN. Максимальное расстояние между фингерпринтами для образования кластера.
                 Требует подбора под ваши данные.
            min_samples: Параметр DBSCAN. Минимальное количество атомов в кластере.
            model_path: Путь к модели MACE или название стандартной модели.
            device: Устройство для вычислений ('cpu' или 'cuda').
        """
        print(f"Загрузка модели MACE '{model_path}'...")
        self.calculator = MACECalculator(
            model_paths=model_path,
            device=device
        )
        self.device = device
        self.type_to_atomic_num = type_to_atomic_num

        # ##### ИЗМЕНЕНО: Сохраняем параметры DBSCAN #####
        self.dbscan_eps = eps
        self.dbscan_min_samples = min_samples

        self.db_fingerprints = []
        print(f"Предварительный расчет кластеризованных фингерпринтов для {len(configurations)} конфигураций...")
        for i, config in enumerate(configurations):
            fingerprint_centroids = self._get_clustered_fingerprints_from_config(config)
            if fingerprint_centroids is not None:
                self.db_fingerprints.append({"centroids": fingerprint_centroids, "config": config})
            print(f"  Обработано {i+1}/{len(configurations)}", end='\r')
        print("\nИнициализация завершена.")

    # ##### ИЗМЕНЕНО: Логика кластеризации полностью заменена #####
    def _cluster_fingerprints(self, per_atom_fp: np.ndarray) -> np.ndarray:
        """
        Кластеризует поатомные фингерпринты с помощью DBSCAN для поиска
        уникальных химических окружений.
        """
        n_atoms = per_atom_fp.shape[0]
        if n_atoms < self.dbscan_min_samples:
            # Если атомов слишком мало для кластеризации,
            # считаем каждый атом отдельным "кластером".
            return per_atom_fp

        # Инициализируем и обучаем DBSCAN
        # metric='euclidean' - стандарт, но можно попробовать 'cosine', если фингерпринты нормированы
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='euclidean').fit(per_atom_fp)
        
        # labels - массив, где каждому атому присвоен номер кластера (-1 для шума)
        labels = db.labels_

        # Находим уникальные номера кластеров (игнорируя шум)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # Вычисляем центроиды (средний фингерпринт) для каждого найденного кластера
        centroids = []
        for label in unique_labels:
            # Выбираем все фингерпринты, принадлежащие текущему кластеру
            class_member_mask = (labels == label)
            cluster_points = per_atom_fp[class_member_mask]
            
            # Вычисляем средний вектор (центроид) для этого кластера
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)

        # Атомы, помеченные как шум, также могут быть уникальными окружениями.
        # Добавляем фингерпринт каждого "шумового" атома как отдельный центроид.
        noise_points = per_atom_fp[labels == -1]
        if noise_points.shape[0] > 0:
            centroids.extend(noise_points)
        
        if not centroids:
            return np.array([])
            
        return np.array(centroids)

    # --- Остальные методы класса остаются без изменений ---

    def _config_to_ase(self, config: Configuration) -> Optional[Atoms]:
        if not config.atom_data: return None
        try:
            positions = [(atom['cartes_x'], atom['cartes_y'], atom['cartes_z']) for atom in config.atom_data]
            atomic_numbers = [self.type_to_atomic_num[atom['type']] for atom in config.atom_data]
            return Atoms(numbers=atomic_numbers, positions=positions)
        except KeyError as e:
            print(f"Ошибка: не найден атомный номер для типа {e}.")
            return None

    def _get_mace_fingerprint(self, ase_atoms: Atoms) -> np.ndarray:
        output = self.calculator.get_descriptors(ase_atoms)
        return output

    def _get_clustered_fingerprints_from_config(self, config: Configuration) -> Optional[np.ndarray]:
        ase_atoms = self._config_to_ase(config)
        if ase_atoms is None: return None
        per_atom_fp = self._get_mace_fingerprint(ase_atoms)
        return self._cluster_fingerprints(per_atom_fp)

    def _get_clustered_fingerprints_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1: return None
        AllChem.MMFFOptimizeMolecule(mol)
        positions = mol.GetConformer().GetPositions()
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        ase_atoms = Atoms(numbers=atomic_numbers, positions=positions)
        per_atom_fp = self._get_mace_fingerprint(ase_atoms)
        return self._cluster_fingerprints(per_atom_fp)
        
    @staticmethod
    def _calculate_set_similarity(centroids_a: np.ndarray, centroids_b: np.ndarray) -> float:
        if centroids_a.shape[0] == 0 or centroids_b.shape[0] == 0:
            return 0.0
        similarity_matrix = cosine_similarity(centroids_a, centroids_b)
        cost_matrix = 1 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        max_similarity_sum = similarity_matrix[row_ind, col_ind].sum()
        norm_factor = max(len(centroids_a), len(centroids_b))
        return max_similarity_sum / norm_factor

    def find_similar(
        self,
        smiles: str,
        top_k: int = 5,
        diversity_lambda: float = 0.7
    ) -> List[tuple[float, Configuration]]:
        print(f"Генерация 3D структуры и фингерпринтов для SMILES: {smiles}")
        target_centroids = self._get_clustered_fingerprints_from_smiles(smiles)
        if target_centroids is None or target_centroids.shape[0] == 0:
            print("Не удалось обработать SMILES."); return []

        print(f"Расчет релевантности для {len(self.db_fingerprints)} кандидатов (параллельно)...")
        with ProcessPoolExecutor() as executor:
            args_iterator = zip(self.db_fingerprints, repeat(target_centroids))
            candidate_pool = list(executor.map(_process_candidate_wrapper, args_iterator))
        
        print("Расчет релевантности завершен.")
        
        diverse_results = []
        candidate_pool.sort(key=lambda x: x['relevance'], reverse=True)

        if not candidate_pool: return []
        diverse_results.append(candidate_pool.pop(0))

        while len(diverse_results) < top_k and candidate_pool:
            mmr_scores = []
            for candidate in candidate_pool:
                max_similarity_to_results = 0
                for result in diverse_results:
                    sim = self._calculate_set_similarity(candidate["centroids"], result["centroids"])
                    if sim > max_similarity_to_results:
                        max_similarity_to_results = sim
                
                mmr_score = (diversity_lambda * candidate["relevance"] -
                             (1 - diversity_lambda) * max_similarity_to_results)
                mmr_scores.append(mmr_score)
            
            if not mmr_scores: break
            best_candidate_idx = np.argmax(mmr_scores)
            chosen_one = candidate_pool.pop(best_candidate_idx)
            diverse_results.append(chosen_one)

        final_output = [(item['relevance'], item['config']) for item in diverse_results]
        return final_output

    # Обязательно добавьте сюда все остальные методы, которые вы не показали
    # (_get_clustered_fingerprints_from_config, _get_mace_fingerprint, и т.д.)