from __future__ import annotations

"""
데이터셋 로더: 사전 생성된 PyG dataset 파일(.pt)을 로드한다.

파일 형태:
  - train_datapyg_dataset_*.pt
  - 각 샘플은 Data(x, edge_index, edge_attr, y, scaled_y, global_x) 형태
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from .utils import resolve_path


@dataclass
class SimpleScaler:
    """간단한 표준화 스케일러(mean/std)."""

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "SimpleScaler":
        return cls(mean=state.get("mean"), std=state.get("std"))


class PyGDataset:
    """사전 생성된 PyG dataset 파일 로더.

    파일 형태: torch.save([data1, data2, ...], path)
    각 data는 PyG Data 객체:
      - data.x: [N, 4] (x, y, z, theta)
      - data.edge_index: [2, E]
      - data.edge_attr: [E, 2]
      - data.y: [1, 1] (field_rad) - 실제 값
      - data.scaled_y: [1, 1] - 스케일된 값
      - data.global_x: [1, 7]
    """

    def __init__(self, path: str | Path, use_interpolation: bool = False) -> None:
        self.path = resolve_path(path)
        self.use_interpolation = use_interpolation

        if not self.path.exists():
            raise FileNotFoundError(f"PyG dataset file not found at: {self.path}")

        # 전체 데이터 로드
        self.data_list: List[Data] = torch.load(self.path, weights_only=False)

        if len(self.data_list) == 0:
            raise ValueError("PyG dataset is empty")

        # 샘플 정보 추출
        sample = self.data_list[0]
        self.num_samples = len(self.data_list)
        self.num_modules = sample.x.size(0)

        print(f"Loaded PyG dataset: {self.num_samples} samples, {self.num_modules} modules per sample")

        # 에피소드 쌍 생성 (연속된 샘플들)
        if use_interpolation:
            self.episode_pairs = self._build_episode_pairs()
            print(f"Built {len(self.episode_pairs)} episode pairs for interpolation")

    def __len__(self) -> int:
        return self.num_samples

    def get_item(self, i: int = 0) -> Data:
        """PyG Data 객체 반환."""
        idx = int(i % len(self.data_list))
        data = self.data_list[idx].clone()

        # batch 속성 추가 (단일 그래프)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        return data

    def extract_coords(self) -> np.ndarray:
        """모든 샘플의 평균 좌표를 추출 (x, y, z).

        Returns:
            coords: (N, 3) numpy array
        """
        # 첫 번째 샘플에서 좌표 추출 (좌표는 모든 샘플에서 동일하다고 가정)
        sample = self.data_list[0]
        coords = sample.x[:, :3].numpy()  # [N, 3]
        return coords

    def _build_episode_pairs(self) -> List[tuple]:
        """인접한 샘플 쌍을 에피소드로 사용 가능한 쌍으로 구성.

        현재는 모든 인접 샘플 쌍을 유효하다고 가정.
        향후 시간 정보가 있다면 실제 2시간 간격인지 검증 가능.

        Returns:
            [(i, i+1), ...] 형태의 쌍 리스트
        """
        pairs = []
        for i in range(len(self.data_list) - 1):
            pairs.append((i, i + 1))
        return pairs

    def _interpolate_global_x(
        self,
        global_x_start: torch.Tensor,
        global_x_end: torch.Tensor,
        num_steps: int,
    ) -> List[torch.Tensor]:
        """두 global_x 사이를 선형 보간.

        Args:
            global_x_start: [1, 7] 시작 global features
            global_x_end: [1, 7] 끝 global features
            num_steps: 중간 보간 개수 (n)

        Returns:
            [global_x_0, global_x_1, ..., global_x_{n+1}]
            총 n+2개 (시작, n개 중간, 끝)
        """
        result = []
        total_steps = num_steps + 2  # 시작 + 중간 n개 + 끝

        for t in range(total_steps):
            alpha = t / (total_steps - 1)  # 0 → 1

            # 선형 보간
            global_x_t = (1.0 - alpha) * global_x_start + alpha * global_x_end

            # Hour_sin, hour_cos (인덱스 3, 4) 재정규화
            # 순환 변수는 단위원 위에 있어야 함
            hour_sin = global_x_t[0, 3]
            hour_cos = global_x_t[0, 4]
            norm = torch.sqrt(hour_sin ** 2 + hour_cos ** 2) + 1e-8
            global_x_t[0, 3] = hour_sin / norm
            global_x_t[0, 4] = hour_cos / norm

            # DNI, DHI (인덱스 5, 6)는 음수가 될 수 없음
            global_x_t[0, 5] = torch.clamp(global_x_t[0, 5], min=0.0)
            global_x_t[0, 6] = torch.clamp(global_x_t[0, 6], min=0.0)

            result.append(global_x_t.clone())

        return result

    def sample_episode(self, num_interpolations: int = 8) -> List[Data]:
        """인접 샘플 쌍에서 보간된 에피소드 생성.

        Args:
            num_interpolations: 중간 보간 개수 (n)

        Returns:
            [data_0, data_1, ..., data_{n+1}]
            총 n+2개 Data 객체
        """
        if not self.use_interpolation:
            # 보간 사용 안 함: 랜덤 샘플 하나만 반환
            return [self.get_item(np.random.randint(len(self.data_list)))]

        # 랜덤 쌍 선택
        pair_idx = np.random.randint(len(self.episode_pairs))
        i_start, i_end = self.episode_pairs[pair_idx]

        data_start = self.data_list[i_start]
        data_end = self.data_list[i_end]

        # Global_x 보간
        global_xs = self._interpolate_global_x(
            data_start.global_x,
            data_end.global_x,
            num_interpolations,
        )

        # 각 타임스텝마다 Data 객체 생성
        episode = []
        for t, global_x_t in enumerate(global_xs):
            data_t = data_start.clone()  # 구조와 초기 theta 복사
            data_t.global_x = global_x_t

            # batch 속성 추가
            if not hasattr(data_t, 'batch') or data_t.batch is None:
                data_t.batch = torch.zeros(data_t.x.size(0), dtype=torch.long)

            episode.append(data_t)

        return episode
