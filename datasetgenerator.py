import numpy as np
import torch
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from mace.calculators import MACECalculator

# --- Вспомогательная функция для конвертации SMILES в ASE ---
def smiles_to_ase(smiles: str) -> Atoms:
    """
    Генерирует 3D структуру из SMILES и конвертирует ее в объект ase.Atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Не удалось обработать SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except (RuntimeError, ValueError):
        # Иногда оптимизация не сходится, это не критично
        pass
    
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    
    return Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0])

# --- Основная функция пайплайна ---
def find_similar_configs_from_smiles(
    polymer_smiles: str,
    reference_fingerprints: np.ndarray,
    mace_model_path: str,
    k_neighbors: int,
    cluster_method: str = 'dbscan',
    cluster_params: dict = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Выполняет полный пайплайн: от SMILES до списка похожих ID референсных конфигураций.

    Args:
        polymer_smiles (str): SMILES полимера в формате "[MONOMER_SMILES]".
        reference_fingerprints (np.ndarray): Референсный массив [N_ref, F_len + 1].
        mace_model_path (str): Путь к файлу модели MACE.
        k_neighbors (int): Количество ближайших соседей (K) для поиска.
        cluster_method (str): 'dbscan' или 'agglomerative'.
        cluster_params (dict): Параметры для кластеризации.
        device (str): 'cpu' или 'cuda'.

    Returns:
        np.ndarray: Отсортированный массив уникальных ID референсных конфигураций.
    """
    if cluster_params is None:
        cluster_params = {}

    # --- Шаг 1-2: Генерация ASE объектов для мономера и колец ---
    print("Шаг 1-2: Генерация структур из SMILES...")
    try:
        monomer_smiles = polymer_smiles.strip('[]')
        
        # Генерируем SMILES для колец. Примечание: предполагается, что первый и 
        # последний атомы в SMILES мономера являются точками соединения.
        ring_n2_smiles = monomer_smiles[0] + "1" + monomer_smiles[1:] + monomer_smiles + "1"
        ring_n3_smiles = monomer_smiles[0] + "1" + monomer_smiles[1:] + monomer_smiles * 2 + "1"
        
        atoms_monomer = smiles_to_ase(monomer_smiles)
        atoms_ring2 = smiles_to_ase(ring_n2_smiles)
        atoms_ring3 = smiles_to_ase(ring_n3_smiles)
        
        query_structures = [atoms_monomer, atoms_ring2, atoms_ring3]
        print(f"  Сгенерировано 3 структуры с {len(atoms_monomer)}, {len(atoms_ring2)}, {len(atoms_ring3)} атомами.")

    except Exception as e:
        print(f"Ошибка при генерации структур: {e}")
        return np.array([])

    # --- Шаг 3: Генерация MACE фингерпринтов для query-структур ---
    print("\nШаг 3: Вычисление MACE фингерпринтов для сгенерированных структур...")
    calc = MACECalculator(model_paths=mace_model_path, device=device, default_dtype='float64')
    
    query_fingerprints_list = []
    for i, atoms in enumerate(query_structures):
        descriptors = calc.get_descriptors(atoms)
        query_fingerprints_list.append(descriptors)
        print(f"  Структура {i+1}: получено {descriptors.shape[0]} фингерпринтов формы {descriptors.shape}")
        
    query_fingerprints = np.vstack(query_fingerprints_list)

    # --- Шаг 4: Кластеризация и поиск центроидов ---
    print(f"\nШаг 4: Кластеризация {query_fingerprints.shape[0]} фингерпринтов методом '{cluster_method}'...")
    
    # Нормализация перед кластеризацией
    scaler = StandardScaler()
    processed_fps = scaler.fit_transform(query_fingerprints)
    
    if cluster_method.lower() == 'dbscan':
        clusterer = DBSCAN(**cluster_params)
    elif cluster_method.lower() == 'agglomerative':
        clusterer = AgglomerativeClustering(**cluster_params)
    else:
        raise ValueError("Неизвестный метод кластеризации")
        
    labels = clusterer.fit_predict(processed_fps)
    
    # Находим центроиды для каждого кластера (исключая шум DBSCAN)
    cluster_centroids = []
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    
    for label in unique_labels:
        cluster_mask = (labels == label)
        # Центроид считаем по оригинальным (ненормализованным) фингерпринтам
        centroid = np.mean(query_fingerprints[cluster_mask], axis=0)
        cluster_centroids.append(centroid)
        
    if not cluster_centroids:
        print("Кластеризация не дала ни одного кластера (только шум). Поиск невозможен.")
        return np.array([])
        
    cluster_centroids = np.array(cluster_centroids)
    print(f"  Найдено {len(cluster_centroids)} кластеров (центроидов).")

    # --- Шаг 5: Поиск K ближайших соседей в референсном массиве ---
    print(f"\nШаг 5: Поиск {k_neighbors} ближайших соседей для каждого центроида...")
    
    # Готовим референсный массив
    ref_fps_only = reference_fingerprints[:, :-1]
    ref_config_ids = reference_fingerprints[:, -1].astype(int)
    
    # Нормализуем референсные данные тем же скейлером!
    ref_fps_processed = scaler.transform(ref_fps_only)
    
    # Обучаем модель поиска соседей
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', n_jobs=-1)
    nn_model.fit(ref_fps_processed)
    
    # Ищем соседей для наших центроидов (центроиды тоже нормализуем)
    centroids_processed = scaler.transform(cluster_centroids)
    distances, indices = nn_model.kneighbors(centroids_processed)
    
    print(f"  Найдены индексы соседей, форма: {indices.shape}")
    
    # Собираем все ID конфигураций, соответствующих найденным соседям
    found_indices = indices.flatten()
    corresponding_config_ids = ref_config_ids[found_indices]
    
    # Находим уникальные ID и сортируем их
    unique_ids = np.unique(corresponding_config_ids)
    
    return unique_ids, corresponding_config_ids

# --- Пример использования ---
if __name__ == '__main__':
    # === 1. Подготовка тестовых данных ===
    
    # !!! ЗАМЕНИТЕ ЭТО НА ВАШИ РЕАЛЬНЫЕ ДАННЫЕ !!!
    # Путь к вашей обученной MACE модели
    MACE_MODEL_PATH = "path/to/your/mace_model.model"
    # Количество соседей для поиска
    K_NEIGHBORS = 5
    # SMILES полимера для анализа
    # Пример: полиэтиленгликоль
    TARGET_SMILES = "[CCO]" 
    
    # --- Создадим фальшивый референсный массив фингерпринтов ---
    FINGERPRINT_LENGTH = 16 # Должна совпадать с выходом вашей MACE модели
    N_REF_ATOMS = 5000
    N_REF_CONFIGS = 100
    
    # Создадим 3 "облака" точек, имитирующих разные химические окружения
    fp_cloud1 = np.random.randn(2000, FINGERPRINT_LENGTH) * 0.5 + 5
    fp_cloud2 = np.random.randn(2000, FINGERPRINT_LENGTH) * 0.5 - 5
    # Этот кластер будет похож на наш query, так как SMILES тоже будет иметь C и O
    fp_cloud3_like_query = np.random.randn(1000, FINGERPRINT_LENGTH) * 0.5 

    mock_ref_fps = np.vstack([fp_cloud1, fp_cloud2, fp_cloud3_like_query])
    
    # Добавим ID конфигураций (от 0 до 99)
    mock_ref_ids = np.random.randint(0, N_REF_CONFIGS, size=(N_REF_ATOMS, 1))
    
    # Это наш итоговый референсный массив
    MOCK_REFERENCE_ARRAY = np.hstack([mock_ref_fps, mock_ref_ids])
    print(f"Создан тестовый референсный массив формы: {MOCK_REFERENCE_ARRAY.shape}\n")

    # === 2. Запуск пайплайна ===
    
    # Параметры для Agglomerative Clustering
    agg_params = {'n_clusters': 3, 'linkage': 'ward'}
    
    # Запускаем!
    try:
        similar_ids = find_similar_configs_from_smiles(
            polymer_smiles=TARGET_SMILES,
            reference_fingerprints=MOCK_REFERENCE_ARRAY,
            mace_model_path=MACE_MODEL_PATH,
            k_neighbors=K_NEIGHBORS,
            cluster_method='agglomerative',
            cluster_params=agg_params,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ:")
        print(f"Найдены следующие ID конфигураций, похожие на {TARGET_SMILES}:")
        print(similar_ids)
        print(f"Всего уникальных конфигураций: {len(similar_ids)}")

    except (FileNotFoundError, ValueError) as e:
        print("\n!!! ОШИБКА ВЫПОЛНЕНИЯ !!!")
        print(f"Не удалось выполнить пайплайн: {e}")
        print("Пожалуйста, убедитесь, что путь к модели MACE указан верно,")
        print("и все необходимые библиотеки установлены.")