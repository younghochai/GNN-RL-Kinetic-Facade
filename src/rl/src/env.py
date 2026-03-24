from __future__ import annotations

"""
섹터 환경(Environment): 행동 해석 → M → B → rate-limit → angle-clip → hard-override → surrogate 예측 → 보상.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

from .safety import angle_clip, hard_override, rate_limit
from .surrogate import SurrogateModel


@dataclass
class MultiDiscreteSpace:
    nvec: np.ndarray  # (S,)


@dataclass
class BoxSpace:
    shape: Tuple[int, ...]
    low: float
    high: float


class SectorEnv:
    def __init__(
        self,
        B: csr_matrix,
        M: csr_matrix,
        bins: List[float],
        surrogate: SurrogateModel,
        init_obs: Data,
        max_rate: float = 5.0,
        angle_bounds: Tuple[float, float] = (0.0, 90.0),
        continuous: bool = False,
        weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
        max_traj_steps: int = 10,
    ) -> None:
        self.B = B
        self.M = M
        self.bins = np.array(bins, dtype=np.float32)
        self.surrogate = surrogate
        self.max_rate = float(max_rate)
        self.angle_bounds = (float(angle_bounds[0]), float(angle_bounds[1]))
        self.continuous = bool(continuous)
        self.weights = weights  # (field, crowd, energy)
        assert len(self.weights) == 3, "weights must be a tuple of 3 floats"
        self.max_traj_steps = max_traj_steps
        self.traj_steps = 0

        # 에피소드 시퀀스 (보간 사용 시)
        self.episode_sequence = None
        self.episode_step = 0

        self.S = M.shape[0]
        self.N = B.shape[0]
        assert B.shape[1] == self.S, "B must be a matrix of shape (N, S)"

        # 관측 상태: PyG Data 객체
        assert isinstance(init_obs, Data), "init_obs must be a PyG Data object"
        self.data = init_obs.clone()

        # 그래프 구조는 고정 (edge_index, edge_attr, batch)
        self.edge_index = init_obs.edge_index.clone()
        self.edge_attr = init_obs.edge_attr.clone()
        self.batch = init_obs.batch.clone() if hasattr(init_obs, 'batch') else torch.zeros(self.N, dtype=torch.long)

        # 내부 상태(각도): x[:, 3]
        self.theta = self.data.x[:, 3].numpy().copy()

        # Baseline 계산: 초기 상태에서 대리모델로 예측
        # (데이터의 y가 아닌, 현재 각도 상태에서의 예측값 사용)
        self.baseline_field = None
        self.baseline_crowd = None
        self._compute_baseline()

        # action space
        if self.continuous:
            self.action_space = BoxSpace(shape=(self.S,), low=float(self.bins.min()), high=float(self.bins.max()))
        else:
            self.action_space = MultiDiscreteSpace(nvec=np.full((self.S,), len(self.bins), dtype=np.int64))

    def _compute_baseline(self) -> None:
        """현재 상태에서 대리모델로 baseline 계산."""
        pred = self.surrogate.predict(self.data)
        self.baseline_field = float(pred["field_rad"][0].item())
        self.baseline_crowd = float(pred["crowd_rad"][0].item())

    def reset(self, obs: Optional[Data] = None) -> Data:
        """환경 초기화.

        Args:
            obs: 새로운 초기 상태 (global_x 업데이트용). None이면 기존 유지.

        Returns:
            PyG Data 객체
        """
        # 행동 횟수 초기화
        self.traj_steps = 0

        # 에피소드 시퀀스 초기화
        self.episode_sequence = None
        self.episode_step = 0

        # 관측 업데이트 (global_x만 변경 가능, 좌표는 고정)
        if obs is not None:
            # global_x 업데이트 (태양 정보 등)
            self.data.global_x = obs.global_x.clone()
            # 각도 초기화
            self.theta = obs.x[:, 3].numpy().copy()

        # 각도 적용
        self.data.x[:, 3] = torch.from_numpy(self.theta).float()

        # batch 속성 유지
        self.data.batch = self.batch

        # Baseline 재계산: 현재 초기 상태에서 대리모델 예측
        self._compute_baseline()

        # 관측 반환 (복사본)
        return self.data.clone()

    def reset_episode(self, episode: List[Data]) -> Data:
        """에피소드 시퀀스로 초기화 (보간 사용 시).

        Args:
            episode: [data_0, data_1, ..., data_n] Data 객체 리스트

        Returns:
            PyG Data 객체 (첫 번째 관측)
        """
        # 에피소드 시퀀스 저장
        self.episode_sequence = episode
        self.episode_step = 0
        self.traj_steps = 0

        # 첫 번째 관측
        obs = episode[0]
        self.data.global_x = obs.global_x.clone()
        self.theta = obs.x[:, 3].numpy().copy()
        self.data.x[:, 3] = torch.from_numpy(self.theta).float()
        self.data.batch = self.batch

        # Baseline 재계산: 에피소드 초기 상태에서 대리모델 예측
        self._compute_baseline()

        return self.data.clone()

    def _action_to_delta(self, action: np.ndarray) -> np.ndarray:
        if self.continuous:
            z = np.asarray(action, dtype=np.float32)
        else:
            idx = np.asarray(action, dtype=np.int64)
            z = self.bins[idx]
        # z_sm = M @ z
        z_sm = self.M @ z.reshape(-1, 1)
        z_sm = np.asarray(z_sm).ravel()
        # a = B @ z_sm
        a = self.B @ z_sm.reshape(-1, 1)
        a = np.asarray(a).ravel()
        return a

    def compute_reward(self, field: float, crowd: float, energy: float) -> float:
        """보상 계산: baseline 대비 개선량 기반.

        Args:
            field: 현재 field radiation 예측값
            crowd: 현재 crowd radiation 예측값
            energy: 액션 에너지 비용

        Returns:
            보상 = w_f * (field - baseline_field) - w_c * (crowd - baseline_crowd) - w_e * energy
        """
        w_f, w_c, w_e = self.weights

        # Delta 계산: baseline이 없으면 절대값 사용 (fallback)
        if self.baseline_field is not None:
            delta_field = (field - self.baseline_field) / self.baseline_field
            if delta_field > 1.0:
                delta_field = 1.0
            elif delta_field < -1.0:
                delta_field = -1.0
            else:
                delta_field = delta_field
        else:
            delta_field = 0

        if self.baseline_crowd is not None:
            delta_crowd = (crowd - self.baseline_crowd) / self.baseline_crowd
            if delta_crowd > 1.0:
                delta_crowd = 1.0
            elif delta_crowd < -1.0:
                delta_crowd = -1.0
            else:
                delta_crowd = delta_crowd
        else:
            delta_crowd = 0
        # print(f"field: {field}, crowd: {crowd}, baseline_field: {self.baseline_field}, baseline_crowd: {self.baseline_crowd}")
        # print(f"delta_field: {delta_field}, delta_crowd: {delta_crowd}, energy: {energy}")

        return w_f * delta_field - w_c * delta_crowd - w_e * energy

    def step(
        self,
        action: np.ndarray,
        event_flags: Optional[Dict[str, float]] = None,
    ) -> Tuple[Data, float, bool, bool, Dict]:
        """환경 스텝 실행.

        Args:
            action: (S,) 섹터 액션
            event_flags: 하드 오버라이드 플래그

        Returns:
            (next_state, reward, done, truncated, info)
            - next_state: PyG Data 객체
        """
        # 행동 횟수 증가
        self.traj_steps += 1

        # 실행 체인
        a = self._action_to_delta(action)
        theta_next = rate_limit(self.theta, a, max_rate=self.max_rate)
        theta_next = angle_clip(theta_next, *self.angle_bounds)
        theta_next = hard_override(theta_next, event_flags)

        # 각도 업데이트
        self.theta = theta_next
        self.data.x[:, 3] = torch.from_numpy(self.theta).float()

        # 에피소드 시퀀스 사용 시: 다음 global_x로 업데이트 (시간 진행)
        if self.episode_sequence is not None:
            self.episode_step += 1
            # 에피소드 끝에 도달하지 않았으면 다음 global_x 적용
            if self.episode_step < len(self.episode_sequence):
                self.data.global_x = self.episode_sequence[self.episode_step].global_x.clone()

        # batch 속성 유지
        self.data.batch = self.batch

        # 대리모델 예측 → 보상
        pred = self.surrogate.predict(self.data)
        field = float(pred["field_rad"][0].item())
        crowd = float(pred["crowd_rad"][0].item())
        energy = float(np.mean(np.abs(a)))

        # 보상 계산
        reward = self.compute_reward(field, crowd, energy)

        info = {
            "field": field,
            "crowd": crowd,
            "energy": energy,
            "mean_action": float(np.mean(a)),
            "violation": 0.0,  # 체인에서 이미 제한/클립 적용으로 0% 목표
            "traj_steps": self.traj_steps,
            "episode_step": self.episode_step if self.episode_sequence else 0,
            "theta": self.theta,
            # Baseline 및 개선량 정보
            "baseline_field": self.baseline_field if self.baseline_field is not None else 0.0,
            "baseline_crowd": self.baseline_crowd if self.baseline_crowd is not None else 0.0,
            "delta_field": field - self.baseline_field if self.baseline_field is not None else field,
            "delta_crowd": crowd - self.baseline_crowd if self.baseline_crowd is not None else crowd,
        }

        # Done 조건
        if self.episode_sequence is not None:
            # 에피소드 시퀀스 사용: 에피소드 끝에 도달
            done = (self.episode_step >= len(self.episode_sequence) - 1)
            truncated = done
        else:
            # 기존 방식: max_traj_steps 도달
            done = (self.traj_steps >= self.max_traj_steps)
            truncated = done

        return self.data.clone(), float(reward), done, truncated, info
