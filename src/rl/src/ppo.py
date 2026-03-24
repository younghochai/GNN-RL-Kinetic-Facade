from __future__ import annotations

"""
PPO 알고리즘(순수 구현): GAE, 클리핑 손실, 엔트로피 보너스, KL 측정, 그라디언트 클립.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from .policy import MLPPolicy, MultiCategorical, GNNPolicy


class PPOAgent:
    def __init__(
        self,
        policy: MLPPolicy | GNNPolicy,
        S: int,
        num_bins: int,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy
        self.S = S
        self.num_bins = num_bins
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = torch.device(device)

        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def _distribution(self, logits: torch.Tensor) -> MultiCategorical:
        return MultiCategorical(logits=logits)

    def evaluate(self, inputs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy 평가.

        Args:
            inputs: torch.Tensor (MLP) 또는 Data/Batch (GNN)
            actions: (B, S) 액션

        Returns:
            logp, entropy, values
        """
        logits, values = self.policy(inputs)
        dist = self._distribution(logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, values

    def update(self, batch: Dict, epochs: int, minibatch_size: int) -> Dict[str, float]:
        """PPO 업데이트.

        Args:
            batch: Dict with keys:
                - "data_list": List[Data] (GNN) or "features": np.ndarray (MLP)
                - "actions": np.ndarray [T, S]
                - "logp": np.ndarray [T]
                - "returns": np.ndarray [T]
                - "advantages": np.ndarray [T]
            epochs: 업데이트 epoch 수
            minibatch_size: 미니배치 크기

        Returns:
            학습 로그
        """
        # GNN vs MLP 분기
        use_gnn = "data_list" in batch

        if use_gnn:
            data_list = batch["data_list"]  # List[Data]
            num_samples = len(data_list)
        else:
            obs_feats = torch.from_numpy(batch["features"]).float().to(self.device)
            num_samples = obs_feats.shape[0]

        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        old_logp = torch.from_numpy(batch["logp"]).float().to(self.device)
        returns = torch.from_numpy(batch["returns"]).float().to(self.device)
        advantages = torch.from_numpy(batch["advantages"]).float().to(self.device)

        # advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # return 정규화 (value 학습 안정화)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        indices = np.arange(num_samples)
        logs = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}

        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                # 미니배치 준비
                if use_gnn:
                    mb_data_list = [data_list[i] for i in mb_idx]
                    # Batch로 합치기 (PyG 배치)
                    mb_batch = Batch.from_data_list(mb_data_list).to(self.device)
                    mb_inputs = mb_batch
                else:
                    mb_inputs = obs_feats[mb_idx]

                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                new_logp, entropy, values = self.evaluate(mb_inputs, mb_actions)
                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(log_ratio)

                # PPO clip objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = 0.5 * (log_ratio.pow(2).mean()).item()
                clipfrac = (torch.gt(torch.abs(ratio - 1.0), self.clip_range).float().mean()).item()

                logs["loss"] = float(loss.item())
                logs["policy_loss"] = float(policy_loss.item())
                logs["value_loss"] = float(value_loss.item())
                logs["entropy"] = float((-entropy_loss).item())
                logs["approx_kl"] = float(approx_kl)
                logs["clipfrac"] = float(clipfrac)

                if approx_kl > 1.5 * self.target_kl:
                    return logs

        return logs
