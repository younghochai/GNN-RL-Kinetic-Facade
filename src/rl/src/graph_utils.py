from __future__ import annotations

"""
그래프 구조 생성 유틸: 좌표 기반 edge_index, edge_attr 생성.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


def build_knn_graph(
    coords: np.ndarray,
    k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    k-NN 그래프 생성.
    
    Args:
        coords: (N, 3) 좌표
        k: 이웃 개수
    
    Returns:
        edge_index: (2, E) LongTensor
        edge_attr: (E, 2) FloatTensor [거리, 각도]
    """
    N = coords.shape[0]
    tree = cKDTree(coords)
    
    # k+1개 찾고 자기자신 제외
    distances, indices = tree.query(coords, k=k+1)
    distances = distances[:, 1:]  # 첫 번째는 자기자신
    indices = indices[:, 1:]
    
    # edge_index 구성
    source = np.repeat(np.arange(N), k)
    target = indices.ravel()
    edge_index = np.stack([source, target], axis=0)
    
    # edge_attr: 거리와 각도
    edge_distances = distances.ravel()
    
    # 각도 계산 (source와 target 노드 간 벡터의 각도)
    src_coords = coords[source]
    tgt_coords = coords[target]
    vec = tgt_coords - src_coords
    
    # xy 평면 각도 (atan2)
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    
    edge_attr = np.stack([edge_distances, angles], axis=1).astype(np.float32)
    
    return (
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_attr).float(),
    )


def build_radius_graph(
    coords: np.ndarray,
    radius: float = 10.0,
    max_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Radius 기반 그래프 생성.
    
    Args:
        coords: (N, 3) 좌표
        radius: 반경
        max_neighbors: 최대 이웃 수
    
    Returns:
        edge_index: (2, E) LongTensor
        edge_attr: (E, 2) FloatTensor [거리, 각도]
    """
    N = coords.shape[0]
    tree = cKDTree(coords)
    
    # 반경 내 이웃 찾기
    pairs = tree.query_pairs(radius, output_type='ndarray')
    
    # 양방향 엣지
    if len(pairs) == 0:
        # 엣지 없으면 self-loop
        edge_index = np.stack([np.arange(N), np.arange(N)], axis=0)
        edge_attr = np.zeros((N, 2), dtype=np.float32)
    else:
        edges_fwd = pairs
        edges_bwd = pairs[:, [1, 0]]
        edges = np.concatenate([edges_fwd, edges_bwd], axis=0)
        edge_index = edges.T
        
        # edge_attr
        src_coords = coords[edges[:, 0]]
        tgt_coords = coords[edges[:, 1]]
        vec = tgt_coords - src_coords
        distances = np.linalg.norm(vec, axis=1)
        angles = np.arctan2(vec[:, 1], vec[:, 0])
        edge_attr = np.stack([distances, angles], axis=1).astype(np.float32)
    
    return (
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_attr).float(),
    )

