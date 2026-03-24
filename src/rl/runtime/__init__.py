from __future__ import annotations

"""
실시간 운영(runtime) 패키지.

주요 모듈:
- controller: 실시간 제어 오케스트레이터
- weather: 기상청 API 연동
- solar: 태양 위치 계산 (pvlib)
- decompose: GHI → DNI/DHI 분해
- state: PyG Data 생성
- inference: GNN Policy 추론
- safety: 안전 제약
- control_io: 제어 신호 송신
- types: 데이터 타입 정의
"""

from .controller import RealTimeController
from .types import (
    WeatherData,
    SunPosition,
    IrradianceData,
    RuntimeArtifacts,
    SafetyConfig,
)

__all__ = [
    "RealTimeController",
    "WeatherData",
    "SunPosition",
    "IrradianceData",
    "RuntimeArtifacts",
    "SafetyConfig",
]
