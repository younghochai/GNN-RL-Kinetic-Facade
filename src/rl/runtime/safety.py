from __future__ import annotations

import numpy as np
from .types import WeatherData, SafetyConfig


def rate_limit(theta: np.ndarray, delta: np.ndarray, max_rate: float) -> np.ndarray:
    """변화율 제한.

    Args:
        theta: 현재 각도 [N]
        delta: 목표 각도 변화량 [N]
        max_rate: 최대 변화율 [도/스텝]

    Returns:
        theta_next: 제한된 다음 각도 [N]
    """
    print("  [SAFETY] Rate limit 적용")
    print(f"  [SAFETY]   - delta 범위 (전): [{delta.min():.2f}, {delta.max():.2f}]°")
    print(f"  [SAFETY]   - max_rate: ±{max_rate}°")
    delta_clipped = np.clip(delta, -max_rate, max_rate)
    print(f"  [SAFETY]   - delta 범위 (후): [{delta_clipped.min():.2f}, {delta_clipped.max():.2f}]°")
    n_limited = np.sum(np.abs(delta) > max_rate)
    print(f"  [SAFETY]   - 제한된 모듈 수: {n_limited}/{len(delta)}")
    return theta + delta_clipped


def angle_clip(theta: np.ndarray, min_angle: float, max_angle: float) -> np.ndarray:
    """각도 범위 제한.

    Args:
        theta: 각도 [N]
        min_angle: 최소 각도
        max_angle: 최대 각도

    Returns:
        clipped_theta: 제한된 각도 [N]
    """
    print("  [SAFETY] Angle clip 적용")
    print(f"  [SAFETY]   - theta 범위 (전): [{theta.min():.2f}, {theta.max():.2f}]°")
    print(f"  [SAFETY]   - 허용 범위: [{min_angle}, {max_angle}]°")
    clipped = np.clip(theta, min_angle, max_angle)
    print(f"  [SAFETY]   - theta 범위 (후): [{clipped.min():.2f}, {clipped.max():.2f}]°")
    n_clipped = np.sum((theta < min_angle) | (theta > max_angle))
    print(f"  [SAFETY]   - 제한된 모듈 수: {n_clipped}/{len(theta)}")
    return clipped


def hard_override(
    theta: np.ndarray,
    weather: WeatherData,
    safety_cfg: SafetyConfig
) -> np.ndarray:
    """기상 조건 기반 하드 오버라이드.

    Args:
        theta: 현재 각도 [N]
        weather: 기상 데이터
        safety_cfg: 안전 설정

    Returns:
        theta_final: 오버라이드 적용된 각도 [N]

    Notes:
        - 강풍/강우 시 전체 닫기 (0도)
        - 강수형태 코드에 따라 닫기
    """
    print("  [SAFETY] Hard override 체크")
    wind_threshold = safety_cfg.get("wind_close_ms", 15.0)
    pty_close_codes = safety_cfg.get("pty_close_codes", [1, 2, 3])

    wsd = weather.WSD_ms
    pty = weather.PTY_code
    print(f"  [SAFETY]   - 현재 풍속: {wsd:.1f} m/s (임계값: {wind_threshold} m/s)")
    print(f"  [SAFETY]   - 현재 강수형태: {pty} (닫기 코드: {pty_close_codes})")

    # 강풍 체크
    if wsd > wind_threshold:
        print("  [SAFETY] ⚠️  강풍 감지! 전체 모듈 닫기 (0°)")
        return np.zeros_like(theta)

    # 강수형태 체크
    if pty in pty_close_codes:
        print("  [SAFETY] ⚠️  강수 감지! 전체 모듈 닫기 (0°)")
        return np.zeros_like(theta)

    print("  [SAFETY] ✓ 기상 조건 정상, 오버라이드 없음")
    return theta
