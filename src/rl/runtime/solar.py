from __future__ import annotations

import numpy as np
import pandas as pd
from pvlib import solarposition

from .types import SunPosition


def compute_sun_position(ts_kst: pd.Timestamp, lat: float, lon: float, alt_m: float) -> SunPosition:
    """태양 위치 계산 (pvlib 기반).

    Args:
        ts_kst: KST 시각 (timezone-aware)
        lat: 위도
        lon: 경도
        alt_m: 고도 [m]

    Returns:
        SunPosition: 고도, 방위각, 천정각, 데카르트 좌표
    """
    print("  [SOLAR] 태양 위치 계산 시작")
    print(f"  [SOLAR] 입력: ts={ts_kst}, lat={lat}, lon={lon}, alt={alt_m}m")

    # pvlib로 태양 위치 계산
    df = solarposition.get_solarposition(
        time=ts_kst,
        latitude=lat,
        longitude=lon,
        altitude=alt_m
    )
    solar_dict = df[['zenith', 'azimuth']].to_dict('records')[0]
    print(f"  [SOLAR] ✓ pvlib 계산 완료: {solar_dict}")

    # 천정각 및 고도각
    theta_z = float(df["zenith"].iloc[0])
    sun_alt = float(90.0 - theta_z)
    sun_azi = float(df["azimuth"].iloc[0])
    print(f"  [SOLAR] 구면 좌표: 고도={sun_alt:.2f}°, 방위={sun_azi:.2f}°, 천정각={theta_z:.2f}°")

    # 태양 좌표 변환 (구면 → 데카르트)
    sun_x, sun_y, sun_z = _spherical_to_cartesian(sun_alt, sun_azi)
    print(f"  [SOLAR] 데카르트 좌표: X={sun_x:.4f}, Y={sun_y:.4f}, Z={sun_z:.4f}")

    result = SunPosition(
        sun_alt_deg=sun_alt,
        sun_azi_deg=sun_azi,
        theta_z_deg=theta_z,
        sun_x=sun_x,
        sun_y=sun_y,
        sun_z=sun_z,
    )
    print("  [SOLAR] ✓ SunPosition 생성 완료")

    return result


def _spherical_to_cartesian(altitude_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """태양 구면 좌표 → 데카르트 좌표 변환.

    **목업 함수**: 실제 변환 로직은 나중에 구현 필요.

    Args:
        altitude_deg: 태양 고도각 [도] (0=수평선, 90=천정)
        azimuth_deg: 태양 방위각 [도] (pvlib 규약: 북=0, 시계방향)

    Returns:
        (x, y, z): 태양 데카르트 좌표

    Notes:
        - 학습 데이터의 Sun_X, Sun_Y, Sun_Z와 일치하도록 구현 필요
        - 현재는 임시 변환식 사용

    TODO: 실제 EPW 파일의 태양 좌표 계산 방식 확인 후 구현
    """
    # 임시 변환 (표준 구면 → 데카르트)
    alt_rad = np.deg2rad(altitude_deg)
    azi_rad = np.deg2rad(azimuth_deg)

    # 표준 변환 (단위 구)
    # x: 동쪽 방향, y: 북쪽 방향, z: 위쪽 방향
    x = float(np.cos(alt_rad) * np.sin(azi_rad))
    y = float(np.cos(alt_rad) * np.cos(azi_rad))
    z = float(np.sin(alt_rad))

    return x, y, z
