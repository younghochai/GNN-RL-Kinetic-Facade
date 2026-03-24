from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .types import WeatherData, SunPosition, IrradianceData


def compute_time_features(ts_kst: pd.Timestamp) -> tuple[float, float]:
    """시간 인코딩 (학습 데이터와 동일 방식).

    Args:
        ts_kst: KST 시각

    Returns:
        (hour_sin, hour_cos): 시간의 sin/cos 인코딩

    Notes:
        학습 시 CyclicHourEmbedder 사용:
        sin(2π * hour / 24), cos(2π * hour / 24)
    """
    hour = ts_kst.hour + ts_kst.minute / 60.0  # 시간.분 형태
    w = 2.0 * np.pi / 24.0
    hour_sin = float(np.sin(w * hour))
    hour_cos = float(np.cos(w * hour))
    return hour_sin, hour_cos


def build_pyg_data(
    weather: WeatherData,
    sun: SunPosition,
    irradiance: IrradianceData,
    modules_xyz: np.ndarray,  # [N, 3]
    theta: np.ndarray,  # [N]
    edge_index: torch.Tensor,  # [2, E]
    edge_attr: torch.Tensor,  # [E, 2]
) -> Data:
    """PyG Data 객체 생성 (학습 데이터 형식과 동일).

    Args:
        weather: 기상 데이터
        sun: 태양 위치
        irradiance: DNI/DHI 데이터
        modules_xyz: 모듈 좌표 [N, 3]
        theta: 현재 개폐각 [N]
        edge_index: 그래프 엣지 인덱스 [2, E]
        edge_attr: 엣지 속성 [E, 2]

    Returns:
        PyG Data 객체:
            - x: [N, 4] (x, y, z, theta)
            - edge_index: [2, E]
            - edge_attr: [E, 2]
            - global_x: [1, 7] (Sun_X, Sun_Y, Sun_Z, hour_sin, hour_cos, DNI, DHI)
            - batch: [N] (모두 0)
    """
    print("  [STATE] PyG Data 생성 시작")

    # Node features: [x, y, z, theta]
    assert modules_xyz.shape[0] == theta.shape[0], "좌표와 각도 개수 불일치"
    n_modules = modules_xyz.shape[0]
    print(f"  [STATE] 노드 피처 생성: N={n_modules}")
    xyz_min = modules_xyz.min(axis=0)
    xyz_max = modules_xyz.max(axis=0)
    print(f"  [STATE]   - modules_xyz: shape={modules_xyz.shape}, range=[({xyz_min}), ({xyz_max})]")
    print(f"  [STATE]   - theta: shape={theta.shape}, range=[{theta.min():.2f}, {theta.max():.2f}]°")

    x = np.c_[modules_xyz, theta].astype(np.float32)
    x = torch.from_numpy(x).float()
    print(f"  [STATE] ✓ 노드 피처 (x): shape={x.shape}")

    # Global features: [Sun_X, Sun_Y, Sun_Z, hour_sin, hour_cos, DNI, DHI]
    hour_sin, hour_cos = compute_time_features(weather.ts)
    print(f"  [STATE] 시간 인코딩: hour_sin={hour_sin:.4f}, hour_cos={hour_cos:.4f}")

    global_x = torch.tensor([[
        sun.sun_x,
        sun.sun_y,
        sun.sun_z,
        hour_sin,
        hour_cos,
        irradiance.dni_wh_per_m2,
        irradiance.dhi_wh_per_m2,
    ]], dtype=torch.float)

    print(f"  [STATE] 글로벌 피처 (global_x): shape={global_x.shape}")
    print(f"  [STATE]   - Sun: (X={sun.sun_x:.4f}, Y={sun.sun_y:.4f}, Z={sun.sun_z:.4f})")
    print(f"  [STATE]   - Time: (sin={hour_sin:.4f}, cos={hour_cos:.4f})")
    dni_val = irradiance.dni_wh_per_m2
    dhi_val = irradiance.dhi_wh_per_m2
    print(f"  [STATE]   - Irradiance: DNI={dni_val:.2f}, DHI={dhi_val:.2f} Wh/m²")

    # Batch (단일 그래프)
    N = x.size(0)
    batch = torch.zeros(N, dtype=torch.long)

    # PyG Data 생성
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_x=global_x,
        batch=batch,
    )

    print("  [STATE] ✓ PyG Data 생성 완료:")
    print(f"  [STATE]   - x: {data.x.shape}")
    print(f"  [STATE]   - edge_index: {data.edge_index.shape}")
    print(f"  [STATE]   - edge_attr: {data.edge_attr.shape}")
    print(f"  [STATE]   - global_x: {data.global_x.shape}")
    print(f"  [STATE]   - batch: {data.batch.shape}")

    return data
