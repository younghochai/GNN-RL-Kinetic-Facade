from __future__ import annotations

"""
섹터 클러스터링: k-NN 그래프 생성 및 분할(S=128).
METIS가 없을 경우 스펙트럴 + k-means를 사용한다.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def build_module_graph(coords: np.ndarray, k_nn: int) -> csr_matrix:
    """모듈 좌표로 k-NN 그래프를 생성(csr_matrix)."""
    nbrs = NearestNeighbors(n_neighbors=min(k_nn + 1, len(coords)), algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # 자기 자신(열 0)을 제외
    indices = indices[:, 1:]
    indptr = [0]
    indices_list = []
    data = []
    for row in range(coords.shape[0]):
        for col in indices[row]:
            indices_list.append(int(col))
            data.append(1.0)
        indptr.append(len(indices_list))
    A = csr_matrix((np.array(data), np.array(indices_list), np.array(indptr)), shape=(coords.shape[0], coords.shape[0]))
    # 무방향화
    A = A.maximum(A.T)
    return A


def partition_graph_metis(A: csr_matrix, S: int, seed: int) -> np.ndarray:
    """그래프를 S 파트로 분할. pymetis 없으면 스펙트럴 k-means.
    간결성을 위해 여기서는 스펙트럴 대각화 대신 임베딩으로 k-means 적용.
    """
    # print(f"A: {A}")
    # print(f"S: {S}")
    # print(f"seed: {seed}")
    try:
        import pymetis  # type: ignore

        n_cuts, membership = pymetis.part_graph(S, adjacency=[A.indices[A.indptr[i]: A.indptr[i+1]].tolist() for i in range(A.shape[0])])
        return np.array(membership, dtype=np.int32)
    except Exception:
        pass

    # 스펙트럴 유사: 무작위 워크 기반 임베딩(간단화)
    # D^-1 A 의 상위 S 개 특성 벡터 대신, 몇 스텝 랜덤워크 특징을 사용
    deg = np.asarray(A.sum(axis=1)).ravel() + 1e-6
    D_inv = csr_matrix((1.0 / deg, (np.arange(A.shape[0]), np.arange(A.shape[0]))), shape=A.shape)
    P = D_inv @ A
    X = np.eye(A.shape[0], dtype=np.float32)
    steps = 3
    for _ in range(steps):
        X = P @ X
    rng = np.random.RandomState(seed)
    km = KMeans(n_clusters=S, n_init=10, random_state=rng)
    labels = km.fit_predict(X)
    return labels.astype(np.int32)


def cluster_modules(coords: np.ndarray, cfg) -> np.ndarray:
    """coords -> 섹터 라벨(0..S-1). cfg는 S, k_nn, seed를 포함."""
    S = int(getattr(cfg, "S_sectors", getattr(cfg, "S", 128)))
    k_nn = int(getattr(cfg, "k_nn", 8))
    seed = int(getattr(cfg, "seed", 7))

    # 좌표 기반 k-means 사용 (그래프 기반보다 균형잡힌 클러스터 생성)
    use_graph_clustering = getattr(cfg, "use_graph_clustering", False)

    if use_graph_clustering:
        A = build_module_graph(coords, k_nn=k_nn)
        labels = partition_graph_metis(A, S=S, seed=seed)
    else:
        # 단순 좌표 기반 k-means (더 균형잡힘)
        rng = np.random.RandomState(seed)
        km = KMeans(n_clusters=S, n_init=10, random_state=rng, max_iter=300)
        labels = km.fit_predict(coords).astype(np.int32)

    return labels
