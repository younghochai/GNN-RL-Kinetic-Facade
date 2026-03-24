from __future__ import annotations

"""
안전 계층: rate-limit, angle-clip, hard-override.
"""

from typing import Dict

import numpy as np


def rate_limit(theta: np.ndarray, delta: np.ndarray, max_rate: float) -> np.ndarray:
    """단계별 각속도 제한 적용.

    theta_next = theta + clip(delta, -max_rate, +max_rate)
    """
    delta_clipped = np.clip(delta, -max_rate, max_rate)
    return theta + delta_clipped


def angle_clip(theta: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """허용 각도 범위로 클리핑."""
    return np.clip(theta, lo, hi)


def hard_override(theta: np.ndarray, event_flags: Dict[str, float] | None) -> np.ndarray:
    """이벤트 플래그에 따른 하드 오버라이드.
    event_flags 예: {"force_close": 10.0}  # 섹터 인덱스 -> 강제각(도)
    theta 는 모듈 각도 배열(N,).
    실제 운영에서는 섹터별 마스크가 필요하지만, v1에서는 전체 적용 키만 제공.
    """
    if not event_flags:
        return theta
    theta_out = theta.copy()
    if "force_all" in event_flags:
        theta_out[:] = float(event_flags["force_all"])
    return theta_out
