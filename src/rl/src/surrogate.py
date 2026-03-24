from __future__ import annotations

"""
대리모델 래퍼: TorchScript 로드, 미존재 시 단순 휴리스틱 더미 모델.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data

from .utils import resolve_path


class SurrogateModel:
    def __init__(
        self,
        field_model_path: str | Path | None,
        crowd_model_path: str | Path | None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.device = torch.device(device)
        self.field_model = None
        self.crowd_model = None
        if field_model_path:
            p = resolve_path(field_model_path)
            if p.exists():
                try:
                    self.field_model = torch.load(str(p), weights_only=False, map_location=self.device)
                    self.field_model.eval()
                    self.field_model.to(self.device)
                except Exception:
                    self.model = None
        if crowd_model_path:
            p = resolve_path(crowd_model_path)
            if p.exists():
                try:
                    self.crowd_model = torch.load(str(p), weights_only=False, map_location=self.device)
                    self.crowd_model.eval()
                    self.crowd_model.to(self.device)
                except Exception:
                    self.crowd_model = None

    @torch.no_grad()
    def predict(self, data: Data) -> Dict[str, torch.Tensor]:
        """PyG Data 객체로 예측.

        Args:
            data: PyG Data (x, edge_index, edge_attr, global_x, batch)

        Returns:
            {"field_rad": Tensor[1], "crowd_rad": Tensor[1]}
        """
        if self.field_model is not None and self.crowd_model is not None:
            # 원본을 보존하기 위해 복사본을 사용
            data_local = data.clone()
            if not hasattr(data_local, 'batch') or data_local.batch is None:
                data_local.batch = torch.zeros(data_local.x.size(0), dtype=torch.long, device=data_local.x.device)

            # 모델과 동일한 디바이스로 이동
            data_device = data_local.to(self.device)

            # GNN 모델에 Data 객체 전달
            field_out = self.field_model(data_device)
            crowd_out = self.crowd_model(data_device)
            # out은 (1, out_dim) 형태로 가정 (out_dim=2: field, crowd)
            if isinstance(field_out, torch.Tensor) and isinstance(crowd_out, torch.Tensor):
                # 출력이 단일 텐서면 [field, crowd]로 분리
                return {
                    "field_rad": field_out[:, 0:1].detach().cpu(),  # [1, 1]
                    "crowd_rad": crowd_out[:, 0:1].detach().cpu(),  # [1, 1]
                }
            else:
                # dict 형태로 반환하는 경우
                return {
                    "field_rad": field_out["field_rad"].detach().cpu(),
                    "crowd_rad": crowd_out["crowd_rad"].detach().cpu(),
                }

        # 더미 휴리스틱: 태양 고도와 각도 분포로 간단 근사
        if isinstance(data.global_x, torch.Tensor):
            global_x = data.global_x.detach().cpu().numpy()
        else:
            global_x = data.global_x
        if isinstance(data.x, torch.Tensor):
            x = data.x.detach().cpu().numpy()
        else:
            x = data.x

        sun_alt = float(global_x[0, 0])  # [1, 4] -> scalar
        theta = x[:, 3]  # [N]

        field = np.maximum(0.0, np.cos(np.deg2rad(theta))).mean() * np.maximum(0.0, np.sin(np.deg2rad(sun_alt)))
        crowd = np.maximum(0.0, np.sin(np.deg2rad(theta))).mean() * np.maximum(0.0, np.cos(np.deg2rad(sun_alt)))

        return {
            "field_rad": torch.tensor([[field]], dtype=torch.float32),  # [1, 1]
            "crowd_rad": torch.tensor([[crowd]], dtype=torch.float32),  # [1, 1]
        }
