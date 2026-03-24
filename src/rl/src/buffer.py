from __future__ import annotations

"""
Rollout 버퍼 + GAE 계산 및 미니배치 샘플러.
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class Transition:
    """단일 transition 저장.

    obs는 PyG Data로 저장하되, 메모리 효율을 위해 필요 시 tensor로 분리 가능.
    """
    obs_x: torch.Tensor          # node features [N, 4]
    obs_global_x: torch.Tensor   # global features [1, 4]
    action: np.ndarray
    reward: float
    done: bool
    logp: float
    value: float


class RolloutBuffer:
    """Rollout 버퍼: PyG Data 객체 저장.

    edge_index, edge_attr, batch는 고정이므로 별도 저장.
    """
    def __init__(self, capacity: int, edge_index: torch.Tensor = None, edge_attr: torch.Tensor = None, batch: torch.Tensor = None) -> None:
        self.capacity = int(capacity)
        self.storage: List[Transition] = []
        # 그래프 구조 (고정)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch

    def add(self, obs: Data, act: np.ndarray, rew: float, done: bool, logp: float, value: float) -> None:
        """Transition 추가.

        Args:
            obs: PyG Data 객체
            act: action array
            rew: reward
            done: done flag
            logp: log probability
            value: value estimate
        """
        if len(self.storage) >= self.capacity:
            return

        # 그래프 구조 저장 (첫 번째만)
        if self.edge_index is None and hasattr(obs, 'edge_index'):
            self.edge_index = obs.edge_index.clone()
            self.edge_attr = obs.edge_attr.clone() if hasattr(obs, 'edge_attr') else None
            self.batch = obs.batch.clone() if hasattr(obs, 'batch') else None

        self.storage.append(
            Transition(
                obs_x=obs.x.clone(),
                obs_global_x=obs.global_x.clone() if hasattr(obs, 'global_x') else None,
                action=act.copy(),
                reward=float(rew),
                done=bool(done),
                logp=float(logp),
                value=float(value),
            )
        )

    def get_data_list(self) -> List[Data]:
        """저장된 transition을 Data 리스트로 변환."""
        data_list = []
        for tr in self.storage:
            data = Data(
                x=tr.obs_x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                global_x=tr.obs_global_x,
                batch=self.batch,
            )
            data_list.append(data)
        return data_list

    def compute_gae(self, gamma: float, lam: float, last_value: float, normalize_rewards: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """GAE advantages와 returns를 계산.

        Args:
            gamma: discount factor
            lam: GAE lambda
            last_value: bootstrap value for last state
            normalize_rewards: reward를 정규화 (평균=0, 표준편차=1)

        Returns:
            (advantages, returns)
        """
        T = len(self.storage)
        values = np.array([tr.value for tr in self.storage] + [last_value], dtype=np.float32)
        rewards = np.array([tr.reward for tr in self.storage], dtype=np.float32)
        dones = np.array([tr.done for tr in self.storage], dtype=np.bool_)

        # Reward 정규화 (절댓값 무관하게 상대적 비교)
        if normalize_rewards and len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
            adv[t] = gae
        returns = adv + values[:-1]
        return adv, returns

    def iter_minibatches(self, batch_size: int, shuffle: bool = True) -> Iterator[np.ndarray]:
        idx = np.arange(len(self.storage))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]
