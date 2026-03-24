from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Tuple

import pandas as pd


@dataclass
class WeatherData:
    """기상청 API에서 받은 기상 데이터."""
    ts: "pd.Timestamp"  # KST 시각
    ghi_wh_per_m2: float    # 전천일사 (ICSR 변환값)
    T1H_degC: float         # 기온
    REH_pct: float          # 습도
    WSD_ms: float           # 풍속
    PTY_code: int           # 강수형태


@dataclass
class SunPosition:
    """태양 위치 정보 (pvlib 계산)."""
    sun_alt_deg: float      # 고도각
    sun_azi_deg: float      # 방위각
    theta_z_deg: float      # 천정각
    sun_x: float            # 태양 좌표 X (데카르트)
    sun_y: float            # 태양 좌표 Y
    sun_z: float            # 태양 좌표 Z


@dataclass
class IrradianceData:
    """분해된 일사량 데이터."""
    dni_wh_per_m2: float    # 직달일사
    dhi_wh_per_m2: float    # 산란일사


class RuntimeArtifacts(TypedDict):
    """Runtime 실행에 필요한 아티팩트 경로."""
    policy_checkpoint_path: str
    policy_backbone_path: str
    surrogate_path: str
    B_path: str
    M_path: str
    edge_index_path: str
    edge_attr_path: str
    modules_csv_path: str   # 모듈 좌표 + 초기 각도


class SafetyConfig(TypedDict):
    """안전 제약 설정."""
    max_rate_deg_per_step: float
    angle_bounds: Tuple[float, float]
    wind_close_ms: float
    rain_close_mmph: float
    pty_close_codes: list[int]
