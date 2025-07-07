import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering

def cluster_fingerprints(
    fingerprint_array: np.ndarray,
    method: str = 'dbscan',
    normalization: bool = True,
    **kwargs
) -> (np.ndarray, dict):
    """
    Нормализует, кластеризует фингерпринты и возвращает расширенный массив и статистику по кластерам.

    Args:
        fingerprint_array (np.ndarray): Исходный массив формы [N_atoms, F_length + 1],
                                        где последний столбец - индекс конфигурации.
        method (str): Метод кластеризации. 'dbscan' или 'agglomerative'.
        normalization (bool): Если True, фингерпринты будут нормализованы (StandardScaler).
        **kwargs: Параметры для алгоритма кластеризации.
                  Для DBSCAN: `eps` (default 0.5), `min_samples` (default 5).
                  Для Agglomerative: `n_clusters` (default 8), `linkage` (default 'ward').

    Returns:
        (np.ndarray, dict): Кортеж из двух элементов:
            1. Новый массив формы [N_atoms, F_length + 2], где последний столбец - метка кластера.
            2. Словарь со статистикой по каждому кластеру (размер, центроид, дисперсия).
    """
    if fingerprint_array.ndim != 2 or fingerprint_array.shape[1] < 2:
        raise ValueError("Входной массив должен быть 2D с как минимум двумя столбцами.")

    # --- a) Нормализация (и подготовка данных) ---
    # Отделяем фингерпринты от индекса конфигурации
    fingerprints = fingerprint_array[:, :-1]
    
    if normalization:
        print("Нормализация фингерпринтов (StandardScaler)...")
        scaler = StandardScaler()
        processed_fingerprints = scaler.fit_transform(fingerprints)
    else:
        processed_fingerprints = fingerprints.copy()

    # --- б) Кластеризация ---
    print(f"Кластеризация методом '{method}'...")
    if method.lower() == 'dbscan':
        # Получаем параметры или используем значения по умолчанию
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        print(f"  Параметры DBSCAN: eps={eps}, min_samples={min_samples}")
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        
    elif method.lower() == 'agglomerative':
        # Получаем параметры или используем значения по умолчанию
        n_clusters = kwargs.get('n_clusters', 8)
        linkage = kwargs.get('linkage', 'ward')
        print(f"  Параметры AgglomerativeClustering: n_clusters={n_clusters}, linkage='{linkage}'")

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        
    else:
        raise ValueError(f"Метод '{method}' не поддерживается. Используйте 'dbscan' или 'agglomerative'.")

    # Получаем метки кластеров для каждого фингерпринта
    labels = clusterer.fit_predict(processed_fingerprints)

    # --- в) Сохранение информации о кластерах ---
    print("Сбор статистики по кластерам...")
    cluster_info = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Для DBSCAN шум (-1) обрабатывается отдельно
        if label == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"cluster_{label}"

        # Находим все фингерпринты, принадлежащие этому кластеру
        cluster_mask = (labels == label)
        # Статистику считаем на оригинальных (ненормализованных) фингерпринтах для лучшей интерпретируемости
        original_cluster_points = fingerprints[cluster_mask]
        
        cluster_stats = {
            "size": original_cluster_points.shape[0],
            "centroid": np.mean(original_cluster_points, axis=0),
            "variance_per_axis": np.var(original_cluster_points, axis=0)
        }
        cluster_info[cluster_name] = cluster_stats

    # --- г) Добавление метки кластера в исходный массив ---
    # Преобразуем метки в столбец для конкатенации
    labels_column = labels.reshape(-1, 1)
    
    # Соединяем исходный массив с новым столбцом меток
    result_array = np.hstack([fingerprint_array, labels_column])
    
    print("Готово!")
    return result_array, cluster_info
