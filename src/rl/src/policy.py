from __future__ import annotations

"""
정책/가치망(MLP/GNN)과 MultiCategorical 분포 유틸.
입력:
  - MLPPolicy: global 4 + theta 통계(평균/표준편차/최소/최대) = 8차원 벡터
  - GNNPolicy: PyG Data 객체
출력: 정책 로짓 (S x B), 가치 V(s)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def build_features(obs: Dict[str, np.ndarray]) -> np.ndarray:
    g = obs["global"].astype(np.float32)
    theta = obs["modules"].astype(np.float32)[:, 3]
    stats = np.array([
        float(theta.mean()),
        float(theta.std() + 1e-8),
        float(theta.min()),
        float(theta.max()),
    ], dtype=np.float32)
    feat = np.concatenate([g, stats], axis=0)
    return feat


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, S: int, B: int) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.fc = nn.Linear(hidden_dim, S * B)
        # Xavier uniform 초기화 (작은 gain으로 높은 초기 엔트로피 유지)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.001)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        return logits.view(-1, self.S, self.B)


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        # Xavier uniform 초기화 (매우 작은 gain으로 초기 value 0 근처 유지)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.001)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, S: int, B: int, hidden_dims=(256, 128)) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.policy_head = PolicyHead(hidden_dims[-1], S=S, B=B)
        self.value_head = ValueHead(hidden_dims[-1])

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(feat)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


class GNNPolicy(nn.Module):
    """GNN 기반 정책/가치망.

    입력: PyG Data 객체
    출력: (logits, value)
    """
    def __init__(
        self,
        S: int,
        B: int,
        pretrained_model_path: str = None,
        freeze_backbone: bool = True,
        reinit_backbone: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        if pretrained_model_path is not None:
            # GNN backbone 로드
            self.backbone = torch.load(pretrained_model_path, weights_only=False, map_location=device)

            # 백본 재초기화 옵션 (테스트용)
            if reinit_backbone:
                self._reinitialize_backbone()
                print("  🔄 백본 가중치를 재초기화했습니다.")

            # 백본 freeze 옵션
            if freeze_backbone:
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                self.backbone.train()
                for param in self.backbone.parameters():
                    param.requires_grad = True
        else:
            raise ValueError("pretrained_model_path is required for GNNPolicy")

        # GNN backbone에서 출력 차원 추출
        hidden_channels = self.backbone.hidden_channels

        # get_hidden_state()의 실제 출력 차원 계산
        # use_global_features=True면 hidden_channels*2, 아니면 hidden_channels
        if self.backbone.use_global_features:
            output_dim = hidden_channels * 2
        else:
            output_dim = hidden_channels

        # 백본 출력 정규화 레이어 추가
        self.backbone_norm = nn.LayerNorm(output_dim)

        self.policy_head = PolicyHead(output_dim, S=S, B=B)
        self.value_head = ValueHead(output_dim)

    def _reinitialize_backbone(self) -> None:
        """백본의 모든 가중치를 재초기화합니다."""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 작은 gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyG Data로 forward.

        Args:
            data: PyG Data (x, edge_index, edge_attr, global_x, batch)

        Returns:
            logits: (1, S, B) 또는 (batch_size, S, B)
            value: (1,) 또는 (batch_size,)
        """
        # batch 속성이 없으면 추가 (단일 그래프)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # GNN backbone으로 hidden state 추출
        h = self.backbone.get_hidden_state(data)  # (batch_size, hidden_dim)

        # 백본 출력 정규화 (스케일 안정화)
        h = self.backbone_norm(h)

        logits = self.policy_head(h)  # (batch_size, S, B)
        value = self.value_head(h)    # (batch_size,)
        return logits, value


@dataclass
class MultiCategorical:
    """S개의 독립 Categorical의 합성 분포."""

    logits: torch.Tensor  # (B,S,Bins)

    def sample(self) -> torch.Tensor:
        B, S, K = self.logits.shape
        probs = F.softmax(self.logits, dim=-1)
        # torch.multinomial은 2D 지원 → reshape
        samples = []
        for s in range(S):
            idx = torch.multinomial(probs[:, s, :], num_samples=1)
            samples.append(idx)
        return torch.cat(samples, dim=1)  # (B,S)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        B, S, K = self.logits.shape
        logp_all = F.log_softmax(self.logits, dim=-1)
        # gather per sector
        logps = []
        for s in range(S):
            a_s = actions[:, s].long().unsqueeze(-1)
            logps.append(logp_all[:, s, :].gather(-1, a_s))
        return torch.cat(logps, dim=1).sum(dim=1)  # (B,)

    def entropy(self) -> torch.Tensor:
        p = F.softmax(self.logits, dim=-1)
        logp = F.log_softmax(self.logits, dim=-1)
        # 섹터별 entropy 평균 (합산 대신)
        ent = -(p * logp).sum(dim=-1).mean(dim=-1)  # (B,)
        return ent
