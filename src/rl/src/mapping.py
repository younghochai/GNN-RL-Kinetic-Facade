from __future__ import annotations

"""
모듈-섹터 매핑 B, 섹터 평활화 M 구성.
요구 속성: B는 row-stochastic, 비음수, 희소. 경계 모듈 가우시안 페더링.
M = (1-α)I + α D^{-1} A_sec
"""

# from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from sklearn.neighbors import NearestNeighbors


def _sector_centers(coords: np.ndarray, labels: np.ndarray, S: int) -> np.ndarray:
    centers = np.zeros((S, coords.shape[1]), dtype=np.float32)
    # 전체 좌표의 중심 (fallback용)
    global_center = coords.mean(axis=0)

    for s in range(S):
        mask = labels == s
        if mask.any():
            centers[s] = coords[mask].mean(axis=0)
        else:
            # 빈 섹터는 전체 중심으로 설정 (원점이 아니라)
            centers[s] = global_center
    return centers


def build_B(coords: np.ndarray, labels: np.ndarray, cfg) -> csr_matrix:
    """Row-stochastic B (N x S) 생성.
    내부 모듈은 원-핫, 경계는 최대 3개 섹터로 가우시안 가중.
    """
    N = coords.shape[0]
    S = int(getattr(cfg, "S_sectors", getattr(cfg, "S", int(labels.max()) + 1)))
    tau = float(getattr(cfg, "tau", 1.5))
    k_nn = int(getattr(cfg, "k_nn", 8))

    centers = _sector_centers(coords, labels, S)
    nbrs = NearestNeighbors(n_neighbors=min(k_nn + 1, N)).fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)

    indptr = [0]
    indices = []
    data = []
    empty_rows = []

    for i in range(N):
        my_label = int(labels[i])
        neighbor_labels = set(int(labels[j]) for j in knn_idx[i, 1:])
        cand = {my_label}
        # 경계면에서 타 섹터 후보 추가(최대 3)
        for lbl in neighbor_labels:
            if len(cand) >= 3:
                break
            if lbl != my_label:
                cand.add(lbl)

        cand_list = sorted(list(cand))
        dists = np.linalg.norm(centers[cand_list] - coords[i][None, :], axis=1)
        weights = np.exp(- (dists / max(tau, 1e-6)) ** 2)
        weights_sum = weights.sum()

        # 가중치가 0이거나 매우 작은 경우 (섹터 중심이 (0,0,0)인 경우)
        if weights_sum < 1e-12:
            # 가장 가까운 유효한 섹터에 할당
            valid_sectors = [s for s in range(S) if np.any(centers[s] != 0)]
            if valid_sectors:
                best_s = min(valid_sectors, key=lambda s: np.linalg.norm(centers[s] - coords[i]))
                indices.append(int(best_s))
                data.append(1.0)
            else:
                # 모든 섹터가 비어있다면 첫 번째 섹터에 할당
                indices.append(0)
                data.append(1.0)
            empty_rows.append(i)
        else:
            weights = weights / weights_sum
            # 행 추가
            for s_idx, w in zip(cand_list, weights):
                indices.append(int(s_idx))
                data.append(float(w))
        indptr.append(len(indices))

    if empty_rows:
        print(f"⚠️  경고: {len(empty_rows)}개 모듈의 가우시안 가중치가 0이 되어 재배치됨 "
              f"({len(empty_rows)/N*100:.1f}%)")
        print(f"    → tau 값을 늘리면 개선될 수 있습니다 (현재 tau={tau:.1f})")

    B = csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(N, S))

    # 보정: 음수 금지, 행 합 1
    B.data = np.clip(B.data, 0.0, np.inf)
    row_sums = np.asarray(B.sum(axis=1)).ravel()

    # row sum이 0인 행 처리 (안전장치)
    zero_rows = np.where(row_sums < 1e-12)[0]
    if len(zero_rows) > 0:
        print(f"⚠️  긴급: {len(zero_rows)}개 행의 합이 0입니다. 균등 분포로 초기화합니다.")
        # 해당 행을 균등 분포로 설정
        B = B.tolil()
        for row_idx in zero_rows:
            B[row_idx, :] = 1.0 / S
        B = B.tocsr()
        row_sums = np.asarray(B.sum(axis=1)).ravel()

    inv_row = diags(1.0 / (row_sums + 1e-12))
    B = inv_row @ B
    return B


def sector_adjacency_from_labels(labels: np.ndarray, coords: np.ndarray, k_nn: int = 8) -> csr_matrix:
    """모듈 k-NN를 통해 섹터 간 인접을 유도한다."""
    N = coords.shape[0]
    S = int(labels.max()) + 1
    nbrs = NearestNeighbors(n_neighbors=min(k_nn + 1, N)).fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    edges = set()
    for i in range(N):
        si = int(labels[i])
        for j in knn_idx[i, 1:]:
            sj = int(labels[int(j)])
            if si != sj:
                a, b = (si, sj) if si < sj else (sj, si)
                edges.add((a, b))
    if not edges:
        return csr_matrix((S, S))
    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(S, S))
    A = A + A.T
    return A


def build_M(A_sec: csr_matrix, alpha: float) -> csr_matrix:
    """섹터 평활화 행렬 M = (1-α)I + α D^{-1} A_sec."""
    S = A_sec.shape[0]
    deg = np.asarray(A_sec.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    D_inv = diags(1.0 / deg)
    P = D_inv @ A_sec
    M = (1.0 - float(alpha)) * identity(S, format="csr") + float(alpha) * P
    return M


def validate_constant_preservation(B: csr_matrix, M: csr_matrix, S: int, N: int, tol: float = 1e-6) -> bool:
    """상수 보존: z=c·1 => a=BMz 가 상수인지 검사."""
    import numpy as np

    z = np.ones((S, 1), dtype=np.float64)
    a = B @ (M @ z)
    return bool(np.allclose(a, np.full((N, 1), a[0, 0]), atol=tol))


def boundary_step_response(B: csr_matrix, M: csr_matrix, S: int) -> float:
    """경계 스텝 응답의 부드러움(근접 모듈 간 분산)을 간단 수치로 반환(낮을수록 좋음)."""
    import numpy as np

    z = np.zeros((S, 1), dtype=np.float64)
    z[0, 0] = 1.0
    a = (B @ (M @ z)).ravel()
    # 1-최근접 차이 제곱 평균
    diffs = np.diff(np.sort(a))
    return float(np.mean(diffs ** 2))
